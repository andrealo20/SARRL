import copy

import numpy as np
import pytest

from sarrl.controllers.adaptive_nominal import AdaptiveNominalConfig, AdaptiveNominalController
from sarrl.dynamics import PlanarArm
from sarrl.envs import DomainRandomization, PlanarReachEnv
from sarrl.envs.adaptive_projected import AdaptiveProjectedEnv

SEED = 9803300  # pilot range, outside every official block


def _env(sensor_noise=1e-3, plant="analytical"):
    randomization = DomainRandomization(
        mass_fraction=0.15,
        friction_fraction=0.30,
        motor_gain_fraction=0.15,
        payload_range=(0.0, 1.0),
        action_delay_max=2,
        sensor_noise_std=sensor_noise,
    )
    if plant == "mujoco":
        pytest.importorskip("mujoco")
        from sarrl.envs.mujoco_planar import MujocoPlanarReachEnv

        base = MujocoPlanarReachEnv(
            mode="torque",
            randomization=randomization,
            armature_range=(0.02, 0.08),
            actuator_time_constant_range=(0.01, 0.05),
        )
        config = AdaptiveNominalConfig(actuator_time_constants=(0.0, 0.01, 0.03, 0.06))
    else:
        base = PlanarReachEnv(mode="torque", randomization=randomization)
        config = AdaptiveNominalConfig()
    return AdaptiveProjectedEnv(base, config)


def _rollout(env, actions, seed=SEED):
    obs, _ = env.reset(seed=seed)
    trace = []
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        trace.append((obs.copy(), reward, terminated, truncated, info["baseline_torque"].copy()))
        if terminated or truncated:
            break
    return trace


def test_zero_residual_reproduces_the_identified_stack_and_reports_estimator_fields():
    env = _env()
    obs, _ = env.reset(seed=SEED)
    assert obs.shape == (8,)
    rng = np.random.default_rng(0)
    for _ in range(40):
        _, _, terminated, truncated, info = env.step(rng.uniform(-1.0, 1.0, 2))
        assert set(info) >= {
            "safety_infeasible",
            "safety_certified",
            "estimator_lag",
            "estimator_time_constant",
            "estimator_updates",
            "baseline_torque",
            "raw_residual",
            "projected_torque",
        }
        if terminated or truncated:
            break
    assert env.controller.updates == env.command_attempts == env.safety_certified_steps
    assert env.controller.updates == env.steps


def test_requires_a_torque_mode_plant():
    with pytest.raises(ValueError, match="torque-mode"):
        AdaptiveProjectedEnv(PlanarReachEnv(mode="residual"))


def test_checkpoint_mid_episode_continues_the_same_trajectory():
    env = _env()
    rng = np.random.default_rng(1)
    actions = rng.uniform(-1.0, 1.0, (60, 2))
    reference = _rollout(env, actions)

    env = _env()
    env.reset(seed=SEED)
    for action in actions[:25]:
        env.step(action)
    state = copy.deepcopy(env.state_dict())
    restored = AdaptiveProjectedEnv.from_state_dict(state)
    assert restored.constructor_config() == env.constructor_config()
    assert restored.controller.updates == 25
    for index, action in enumerate(actions[25:], start=25):
        obs_a, r_a, t_a, u_a, info_a = env.step(action)
        obs_b, r_b, t_b, u_b, info_b = restored.step(action)
        assert np.array_equal(obs_a, obs_b)
        assert r_a == r_b and t_a == t_b and u_a == u_b
        assert np.array_equal(info_a["baseline_torque"], info_b["baseline_torque"])
        assert np.array_equal(obs_a, reference[index][0])
        if t_a or u_a:
            break


def test_estimator_state_round_trips():
    controller = AdaptiveNominalController(PlanarArm(), AdaptiveNominalConfig())
    rng = np.random.default_rng(2)
    state = np.zeros(4)
    for _ in range(15):
        torque = rng.uniform(-5.0, 5.0, 2)
        after = state + 0.02 * np.concatenate([state[2:], torque])
        controller.begin_step(state[:2])
        controller.observe(state, after, torque)
        state = after
    twin = AdaptiveNominalController(PlanarArm(), AdaptiveNominalConfig())
    twin.load_state_dict(controller.state_dict())
    assert twin.hypothesis == controller.hypothesis
    assert np.array_equal(twin.parameters, controller.parameters)
    assert np.array_equal(twin.predict_state(state, 2), controller.predict_state(state, 2))
    assert twin.control_steps == controller.control_steps


def test_mujoco_plant_checkpoints_actuator_state():
    env = _env(plant="mujoco")
    rng = np.random.default_rng(3)
    actions = rng.uniform(-1.0, 1.0, (30, 2))
    env.reset(seed=SEED)
    for action in actions[:12]:
        env.step(action)
    state = copy.deepcopy(env.state_dict())
    restored = AdaptiveProjectedEnv.from_state_dict(state)
    assert restored.env.actuator_time_constant == env.env.actuator_time_constant
    assert restored.env.armature == env.env.armature
    for action in actions[12:]:
        obs_a, r_a, t_a, u_a, _ = env.step(action)
        obs_b, r_b, t_b, u_b, _ = restored.step(action)
        assert np.array_equal(obs_a, obs_b)
        assert r_a == r_b and t_a == t_b and u_a == u_b
        if t_a or u_a:
            break
