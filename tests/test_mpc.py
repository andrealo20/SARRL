import numpy as np

from sarrl.controllers import MPCConfig, NonlinearMPC
from sarrl.dynamics import PlanarArm


def _small_mpc():
    return NonlinearMPC(
        PlanarArm(),
        MPCConfig(horizon=5, dt=0.02, max_iterations=60, joint_limit=(-3.1, 3.1)),
    )


def test_mpc_rollout_has_expected_shape_and_is_deterministic():
    mpc = _small_mpc()
    state = np.array([0.2, -0.3, 0.0, 0.0])
    sequence = np.zeros((5, 2))
    a = mpc.rollout(state, sequence)
    b = mpc.rollout(state, sequence)
    assert a.shape == (5, 4)
    np.testing.assert_array_equal(a, b)


def test_mpc_command_respects_torque_and_predicted_state_constraints():
    mpc = _small_mpc()
    state = np.array([0.2, 0.3, 0.0, 0.0])
    target = np.array([0.5, 0.6])
    result = mpc.command(state, target)
    assert result.success
    assert np.all(np.abs(result.torque) <= 40.0 + 1e-10)
    assert result.min_constraint_margin >= -2e-6


def test_mpc_optimisation_beats_its_initial_gravity_sequence():
    mpc = _small_mpc()
    state = np.array([-0.7, 0.5, 0.15, -0.1])
    target = np.array([0.45, 0.85])
    guess = mpc._initial_guess(state, target)
    initial_cost = mpc.objective(guess.ravel(), state, target)
    result = mpc.command(state, target)
    assert result.success
    assert result.cost < initial_cost - 1e-5


def test_mpc_warm_start_can_be_reset():
    mpc = _small_mpc()
    mpc.command(np.array([0.1, 0.2, 0.0, 0.0]), np.array([0.3, 0.4]))
    assert np.any(mpc._warm_start)
    mpc.reset()
    np.testing.assert_array_equal(mpc._warm_start, np.zeros((5, 2)))
