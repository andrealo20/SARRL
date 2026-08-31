import numpy as np

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
