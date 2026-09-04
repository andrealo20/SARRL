from pathlib import Path

import numpy as np
import pytest

from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm
from sarrl.envs import DomainRandomization, PlanarReachEnv, SafetyProjectedEnv
from sarrl.rl import ReplayBuffer, SACAgent, SACConfig, load_training_session
from sarrl.rl.training_checkpoint import save_training_checkpoint
from sarrl.runtime import ControlStackConfig, SARRLControlStack
from sarrl.safety import HOCBFSafetyFilter, SafetyResult
from tools.train_sac import _validation_env


class FixedPolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def act(self, observation, deterministic=True):
        del observation, deterministic
        return self.action.copy()


def _randomization() -> DomainRandomization:
    return DomainRandomization(
        mass_fraction=0.15,
        friction_fraction=0.30,
        motor_gain_fraction=0.15,
        payload_range=(0.0, 1.0),
        action_delay_max=2,
    )


def test_projected_step_matches_deployment_stack_exactly():
    action = np.array([0.9, -0.8], dtype=np.float32)
    wrapped_base = PlanarReachEnv(mode="residual", randomization=_randomization())
    stack_base = PlanarReachEnv(mode="torque", randomization=_randomization())
    wrapped = SafetyProjectedEnv(wrapped_base)
    wrapped_obs, _ = wrapped.reset(seed=73)
    stack_obs, _ = stack_base.reset(seed=73)
    np.testing.assert_array_equal(wrapped_obs, stack_obs)

    nominal = PlanarArm()
    stack = SARRLControlStack(
        ComputedTorqueController(nominal),
        FixedPolicy(action),
        ControlStackConfig(require_safety=True),
        safety_filter=HOCBFSafetyFilter(nominal),
    )
    command = stack.command(stack_obs, stack_base.state, stack_base.q_des)
    expected = stack_base.step_torque(
        command.torque,
        baseline=command.baseline_torque,
    )
    got = wrapped.step(action)

    np.testing.assert_array_equal(got[0], expected[0])
    assert got[1:4] == expected[1:4]
    np.testing.assert_array_equal(wrapped.state, stack_base.state)
    np.testing.assert_allclose(got[4]["projected_torque"], command.torque)
    np.testing.assert_allclose(
        got[4]["safety_correction"], command.safety_correction
    )
    assert got[4]["safety_certified"]


@pytest.mark.parametrize("seed", [4, 37, 92])
def test_projected_trajectory_matches_deployment_stack(seed: int):
    wrapped_base = PlanarReachEnv(mode="residual", randomization=_randomization())
    stack_base = PlanarReachEnv(mode="torque", randomization=_randomization())
    wrapped = SafetyProjectedEnv(wrapped_base)
    wrapped_obs, _ = wrapped.reset(seed=seed)
    stack_obs, _ = stack_base.reset(seed=seed)
    np.testing.assert_array_equal(wrapped_obs, stack_obs)

    policy = FixedPolicy(np.zeros(2, dtype=np.float32))
    nominal = PlanarArm()
    stack = SARRLControlStack(
        ComputedTorqueController(nominal),
        policy,
        ControlStackConfig(require_safety=True),
        safety_filter=HOCBFSafetyFilter(nominal),
    )
    actions = np.random.default_rng(seed + 1000).uniform(-1.0, 1.0, size=(50, 2))
    for action in actions:
        policy.action = action.astype(np.float32)
        command = stack.command(stack_obs, stack_base.state, stack_base.q_des)
        state_before = stack_base.state.copy()
        got = wrapped.step(policy.action)
        if not command.executable:
            assert got[2] and not got[3]
            np.testing.assert_array_equal(wrapped.state, state_before)
            break

        expected = stack_base.step_torque(
            command.torque,
            baseline=command.baseline_torque,
        )
        np.testing.assert_array_equal(got[0], expected[0])
        assert got[1:4] == expected[1:4]
        np.testing.assert_array_equal(wrapped.state, stack_base.state)
        np.testing.assert_allclose(got[4]["projected_torque"], command.torque)
        stack_obs = expected[0]
        if got[2] or got[3]:
            break


def test_feasible_projection_does_not_shape_environment_reward():
    action = np.array([0.25, -0.4], dtype=np.float32)
    a = PlanarReachEnv(mode="residual")
    b = PlanarReachEnv(mode="torque")
    wrapped = SafetyProjectedEnv(a)
    obs_a, _ = wrapped.reset(seed=18)
    obs_b, _ = b.reset(seed=18)
    np.testing.assert_array_equal(obs_a, obs_b)

    baseline, _, candidate = wrapped._candidate(action)
    safety = wrapped.safety_filter.filter(wrapped.state, candidate)
    assert safety.success
    expected = b.step_torque(safety.torque, baseline=baseline)
    got = wrapped.step(action)
    assert got[1] == expected[1]


def test_infeasible_projection_aborts_without_advancing_plant(monkeypatch):
    wrapped = SafetyProjectedEnv(PlanarReachEnv(mode="residual"))
    observation, _ = wrapped.reset(seed=5)
    state = wrapped.state.copy()
    steps = wrapped.steps

    def fail_filter(current_state, candidate, obstacles=()):
        del current_state, obstacles
        return SafetyResult(
            torque=np.asarray(candidate, dtype=np.float64),
            success=False,
            correction_norm=0.0,
            min_margin=-1.0,
            active_constraints=(),
            current_safe=True,
        )

    monkeypatch.setattr(wrapped.safety_filter, "filter", fail_filter)
    next_obs, reward, terminated, truncated, info = wrapped.step(np.ones(2))

    np.testing.assert_array_equal(next_obs, observation)
    np.testing.assert_array_equal(wrapped.state, state)
    assert wrapped.steps == steps
    assert reward == -500.0
    assert terminated and not truncated
    assert not info["success"]
    assert info["safety_infeasible"]
    assert not info["safety_certified"]


def test_replay_retains_raw_policy_action():
    wrapped = SafetyProjectedEnv(PlanarReachEnv(mode="residual"))
    observation, _ = wrapped.reset(seed=9)
    action = np.array([0.95, -0.85], dtype=np.float32)
    next_obs, reward, terminated, _, _ = wrapped.step(action)
    replay = ReplayBuffer(8, 2, 10, seed=1)
    replay.add(observation, action, reward, next_obs, terminated)
    np.testing.assert_array_equal(replay.actions[0], action)


def test_projected_environment_checkpoint_continues_exactly(tmp_path: Path):
    wrapped = SafetyProjectedEnv(
        PlanarReachEnv(mode="residual", randomization=_randomization())
    )
    agent = SACAgent(8, 2, SACConfig(hidden=(16, 16)), seed=31)
    replay = ReplayBuffer(8, 2, 100, seed=31)
    observation, _ = wrapped.reset(seed=31)

    for _ in range(15):
        action = agent.act(observation)
        next_obs, reward, terminated, truncated, _ = wrapped.step(action)
        replay.add(observation, action, reward, next_obs, terminated)
        observation = next_obs
        if terminated or truncated:
            observation, _ = wrapped.reset()

    checkpoint = tmp_path / "projected_session.pt"
    save_training_checkpoint(
        checkpoint,
        agent,
        replay,
        wrapped,
        {"step": 15, "obs": observation},
    )

    expected_action = agent.act(observation)
    expected = wrapped.step(expected_action)
    loaded_agent, loaded_replay, loaded_env, loop = load_training_session(checkpoint)
    loaded_obs = np.asarray(loop["obs"], dtype=np.float32)
    got_action = loaded_agent.act(loaded_obs)
    got = loaded_env.step(got_action)

    assert isinstance(loaded_env, SafetyProjectedEnv)
    assert len(loaded_replay) == len(replay)
    np.testing.assert_array_equal(got_action, expected_action)
    np.testing.assert_array_equal(got[0], expected[0])
    assert got[1:4] == expected[1:4]
    np.testing.assert_array_equal(loaded_env.state, wrapped.state)


def test_projected_checkpoint_rejects_constructor_mismatch():
    original = SafetyProjectedEnv(PlanarReachEnv(mode="residual"))
    original.reset(seed=2)
    state = original.state_dict()
    other = SafetyProjectedEnv(
        PlanarReachEnv(mode="residual"),
        infeasible_reward=-250.0,
    )
    with pytest.raises(ValueError, match="constructor configuration"):
        other.load_state_dict(state)


def test_validation_projection_is_independent_of_training_projection():
    base = PlanarReachEnv(mode="residual", randomization=_randomization())
    projected_validation = _validation_env(
        base,
        safety_projected=True,
        infeasible_reward=-500.0,
    )
    assert isinstance(projected_validation, SafetyProjectedEnv)
    assert projected_validation.env.constructor_config() == base.constructor_config()

    projected_training = SafetyProjectedEnv(base)
    plain_validation = _validation_env(projected_training, safety_projected=False)
    assert isinstance(plain_validation, PlanarReachEnv)
    assert plain_validation.constructor_config() == base.constructor_config()
