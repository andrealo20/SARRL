import numpy as np
import pytest

from sarrl.envs import PlanarReachEnv
from sarrl.envs.planar_reach import DomainRandomization


def test_seeded_reset_is_reproducible():
    env = PlanarReachEnv(randomization=DomainRandomization(0.1, 0.2, 0.1))
    a, ia = env.reset(seed=42)
    gain_a = env.motor_gain.copy()
    params_a = env.arm.params
    b, ib = env.reset(seed=42)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ia["target"], ib["target"])
    np.testing.assert_array_equal(gain_a, env.motor_gain)
    assert params_a == env.arm.params


def test_environment_does_not_mutate_action():
    env = PlanarReachEnv()
    env.reset(seed=1)
    action = np.array([0.5, -0.25], dtype=np.float32)
    original = action.copy()
    env.step(action)
    np.testing.assert_array_equal(action, original)


def test_residual_zero_action_matches_nominal_candidate():
    env = PlanarReachEnv(mode="residual")
    env.reset(seed=7, target=np.array([1.1, 0.5]))
    baseline, candidate = env._candidate_torque(np.zeros(2))
    np.testing.assert_allclose(candidate, baseline)


def test_residual_action_is_bounded_by_actuator_limits():
    env = PlanarReachEnv(mode="residual", torque_limit=10.0, residual_limit=100.0)
    env.reset(seed=2)
    _, candidate = env._candidate_torque(np.array([1.0, -1.0]))
    assert np.all(np.abs(candidate) <= 10.0 + 1e-12)


def test_observation_and_step_contract():
    env = PlanarReachEnv()
    obs, _ = env.reset(seed=4)
    assert obs.shape == (8,)
    out = env.step(np.zeros(2))
    assert len(out) == 5
    next_obs, reward, terminated, truncated, info = out
    assert next_obs.shape == (8,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert np.isfinite(info["distance"])


def test_domain_context_and_delay_are_seed_reproducible():
    cfg = DomainRandomization(
        mass_fraction=0.2,
        friction_fraction=0.3,
        motor_gain_fraction=0.15,
        payload_range=(0.1, 1.0),
        sensor_noise_std=0.002,
        action_delay_max=3,
    )
    a = PlanarReachEnv(randomization=cfg)
    b = PlanarReachEnv(randomization=cfg)
    oa, _ = a.reset(seed=123)
    ob, _ = b.reset(seed=123)
    np.testing.assert_array_equal(oa, ob)
    np.testing.assert_array_equal(a.dynamics_context(), b.dynamics_context())
    assert a.action_delay == b.action_delay


def test_action_delay_delays_total_command_exactly():
    env = PlanarReachEnv(mode="torque", randomization=DomainRandomization(action_delay_max=2))
    # Find a deterministic reset with delay 2 rather than relying on one hard-coded RNG draw.
    for seed in range(100):
        env.reset(seed=seed)
        if env.action_delay == 2:
            break
    assert env.action_delay == 2
    action = np.array([0.5, -0.25])
    _, _, _, _, info0 = env.step(action)
    _, _, _, _, info1 = env.step(action)
    _, _, _, _, info2 = env.step(action)
    np.testing.assert_array_equal(info0["delayed_torque"], np.zeros(2))
    np.testing.assert_array_equal(info1["delayed_torque"], np.zeros(2))
    np.testing.assert_allclose(info2["delayed_torque"], action * env.torque_limit)


def test_fault_activates_on_requested_step_and_changes_dynamics():
    from sarrl.envs import FaultSpec

    env = PlanarReachEnv(
        mode="torque",
        fault=FaultSpec(start_step=2, motor_gain_multiplier=(0.5, 0.8), payload_delta=0.7),
    )
    env.reset(seed=3)
    original_gain = env.motor_gain.copy()
    original_payload = env.payload_mass
    env.step(np.zeros(2))
    env.step(np.zeros(2))
    assert not env._fault_active
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["fault_active"]
    np.testing.assert_allclose(env.motor_gain, original_gain * np.array([0.5, 0.8]))
    assert np.isclose(env.payload_mass, original_payload + 0.7)


def test_step_torque_matches_direct_torque_action_without_plant_disturbances():
    a = PlanarReachEnv(mode="torque")
    b = PlanarReachEnv(mode="torque")
    oa, _ = a.reset(seed=77)
    ob, _ = b.reset(seed=77)
    np.testing.assert_array_equal(oa, ob)
    action = np.array([0.3, -0.4], dtype=np.float32)
    out_a = a.step(action)
    out_b = b.step_torque(action * b.torque_limit)
    np.testing.assert_array_equal(out_a[0], out_b[0])
    np.testing.assert_allclose(out_a[1], out_b[1], atol=0.0)
    assert out_a[2:4] == out_b[2:4]


def test_step_torque_exposes_exact_preintegration_dynamics_invariant():
    env = PlanarReachEnv(
        mode="torque",
        randomization=DomainRandomization(
            mass_fraction=0.15,
            friction_fraction=0.30,
            motor_gain_fraction=0.15,
            payload_range=(0.0, 1.0),
            action_delay_max=2,
        ),
    )
    env.reset(seed=60_007)
    _, _, _, _, info = env.step_torque(np.array([17.123456789, -11.987654321]))

    recomputed = env.arm.forward_dynamics(
        info["pre_step_state"][:2],
        info["pre_step_state"][2:],
        info["plant_input_torque"],
    )
    np.testing.assert_allclose(info["pre_step_acceleration"], recomputed, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(
        info["actuator_scaled_torque"], info["plant_input_torque"]
    )


def test_environment_checkpoint_rejects_constructor_mismatch():
    env = PlanarReachEnv(mode="residual", randomization=DomainRandomization(mass_fraction=0.1))
    env.reset(seed=3)
    state = env.state_dict()
    other = PlanarReachEnv(mode="residual", randomization=DomainRandomization(mass_fraction=0.2))
    with pytest.raises(ValueError, match="constructor configuration"):
        other.load_state_dict(state)
