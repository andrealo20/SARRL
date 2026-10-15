from dataclasses import replace

import numpy as np
import pytest

from sarrl.safety.panda_backup import (
    PandaBackupConfig,
    PandaBackupWitness,
    backup_transition,
    terminal_metrics,
)


def witness(config=None):
    return PandaBackupWitness(np.full(7, -2.0), np.full(7, 2.0), np.full(7, 20.0), config)


@pytest.mark.parametrize("maximum", [0, 1, 4])
def test_augmented_map_matches_direct_fifo_dynamics(maximum):
    config = PandaBackupConfig(max_delay_steps=maximum)
    state = np.array([0.03, -0.1, *np.linspace(-0.2, 0.4, maximum)])
    for lag in range(maximum + 1):
        transition, _ = backup_transition(config, lag)
        sent = -25 * state[0] - 10 * state[1]
        delivered = state[-lag] if lag else sent
        q = state[0] + 0.01 * state[1] + 0.5 * 0.01**2 * delivered
        v = state[1] + 0.01 * delivered
        expected = [q, v, *state[3:], sent] if maximum else [q, v]
        np.testing.assert_allclose(transition @ state, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    "config",
    [
        PandaBackupConfig(max_delay_steps=True),
        PandaBackupConfig(max_delay_steps=5),
        PandaBackupConfig(control_dt=np.nan),
        PandaBackupConfig(kp=-1),
        PandaBackupConfig(horizon_steps=False),
        PandaBackupConfig(horizon_steps=0),
        PandaBackupConfig(kd=1000),
    ],
)
def test_invalid_or_unstable_backup_cannot_create_a_witness(config):
    with pytest.raises(ValueError):
        witness(config)


def test_equilibrium_with_coupled_mass_and_bias_is_accepted_for_every_lag():
    check = witness()
    mass = np.diag(np.arange(1.0, 8.0)) + 0.01 * np.ones((7, 7))
    matrix = np.linalg.inv(mass)
    equilibrium = np.linspace(-2, 2, 7)
    result = check.verify(
        np.zeros(7), np.zeros(7), equilibrium, matrix, -matrix @ equilibrium, [equilibrium] * 4
    )
    assert result.accepted
    assert [case.lag for case in result.cases] == list(range(5))
    assert all(case.checked_intervals == 1 for case in result.cases)
    np.testing.assert_allclose(
        result.policy.requested_torque(np.zeros(7), np.zeros(7)), equilibrium, rtol=0, atol=1e-14
    )


def test_new_candidate_cannot_erase_an_unsafe_command_in_flight():
    check = witness()
    v = np.zeros(7)
    v[0] = check.bounds.velocity[0] - 0.01
    pending = np.zeros(7)
    pending[0] = 20
    result = check.verify(np.zeros(7), v, -pending, np.eye(7), np.zeros(7), [pending])
    assert not result.accepted
    assert result.cases[1].reason == "pending_command_violation"
    assert result.cases[1].checked_intervals == 1


def test_safe_candidate_interval_without_a_terminal_witness_is_rejected():
    check = witness(PandaBackupConfig(max_delay_steps=0, horizon_steps=1))
    v = np.full(7, 1.0)
    assert check.bounds._inside_interval(np.zeros(7), v, np.zeros(7))
    result = check.verify(np.zeros(7), v, np.zeros(7), np.eye(7), np.zeros(7), [])
    assert not result.accepted
    assert result.cases[0].reason == "terminal_not_reached"
    longer = witness(PandaBackupConfig(max_delay_steps=0))
    assert longer.verify(np.zeros(7), v, np.zeros(7), np.eye(7), np.zeros(7), []).accepted


def test_positive_matrix_without_a_lyapunov_identity_is_rejected(monkeypatch):
    import sarrl.safety.panda_backup as module

    monkeypatch.setattr(
        module, "solve_discrete_lyapunov", lambda matrix, weight: np.eye(len(matrix))
    )
    with pytest.raises(ValueError, match="identity checks"):
        terminal_metrics(PandaBackupConfig(kd=10.2))


def test_terminal_ellipsoid_keeps_continuous_intervals_and_raw_torques_admissible():
    check = witness()
    mass = np.diag(np.linspace(0.8, 1.4, 7)) + 0.02 * np.ones((7, 7))
    policy = check.policy(np.zeros(7), np.linalg.inv(mass), np.linspace(-0.05, 0.05, 7))
    for lag, (_, _, metric, inverse) in enumerate(check.metrics):
        radius = check.terminal_radius_squared(policy, lag)
        assert radius > 0
        # Deterministic boundary directions exercise every state and joint.
        for index in range(len(metric)):
            state = np.zeros((len(metric), 7))
            state[:, index % 7] = inverse[:, index]
            state *= np.sqrt(radius / np.sum(state * (metric @ state))) * (1 - 1e-12)
            previous_energy = np.sum(state * (metric @ state))
            for _ in range(300):
                q, v = state[0], state[1]
                raw = policy.requested_torque(q, v)
                assert np.all(np.abs(raw) <= check.bounds.limit + 1e-10)
                acceleration = state[-lag] if lag else policy.matrix @ raw + policy.drift
                # Independent dense sampling supplements the exact interval check.
                times = np.linspace(0, 0.01, 21)[:, None]
                positions = q + times * v + 0.5 * times**2 * acceleration
                velocities = v + times * acceleration
                assert np.all(positions >= check.bounds.lower - 1e-10)
                assert np.all(positions <= check.bounds.upper + 1e-10)
                assert np.all(np.abs(velocities) <= check.bounds.velocity + 1e-10)
                next_q = q + 0.01 * v + 0.5 * 0.01**2 * acceleration
                next_v = v + 0.01 * acceleration
                sent = policy.matrix @ raw + policy.drift
                state = np.array([next_q, next_v, *state[3:], sent])
                energy = np.sum(state * (metric @ state))
                assert energy <= previous_energy + 1e-10
                previous_energy = energy


def test_policy_copies_model_and_uses_no_delay_or_delivered_command_input():
    check = witness()
    matrix, drift, anchor = np.eye(7), np.zeros(7), np.zeros(7)
    policy = check.policy(anchor, matrix, drift)
    matrix[:], drift[:], anchor[:] = 9, 9, 9
    np.testing.assert_array_equal(policy.requested_torque(np.ones(7), np.zeros(7)), np.full(7, -25))
    np.testing.assert_array_equal(policy.bounded_torque(np.ones(7), np.zeros(7)), np.full(7, -20))
    with pytest.raises(ValueError):
        policy.matrix[0, 0] = 2


@pytest.mark.parametrize("matrix", [np.zeros((7, 7)), -np.eye(7), np.ones((6, 6))])
def test_invalid_dynamics_are_rejected(matrix):
    with pytest.raises(ValueError):
        witness().verify(np.zeros(7), np.zeros(7), np.zeros(7), matrix, np.zeros(7), [])


def test_out_of_bounds_candidate_and_history_are_not_silently_clipped():
    check = witness()
    for candidate, history in [(np.full(7, 21), []), (np.zeros(7), [np.full(7, 21)])]:
        with pytest.raises(ValueError, match="bounded sent torques"):
            check.verify(np.zeros(7), np.zeros(7), candidate, np.eye(7), np.zeros(7), history)


def test_unbounded_equilibrium_produces_no_terminal_set():
    check = witness()
    result = check.verify(np.zeros(7), np.zeros(7), np.zeros(7), np.eye(7), np.full(7, 21), [])
    assert not result.accepted
    assert all(case.reason == "empty_terminal_set" for case in result.cases)


def test_missing_startup_commands_are_zero_torque_with_nonzero_drift():
    check = witness(PandaBackupConfig(max_delay_steps=1))
    v = np.zeros(7)
    v[0] = check.bounds.velocity[0] - 0.005
    drift = np.zeros(7)
    drift[0] = 1
    result = check.verify(np.zeros(7), v, -drift, np.eye(7), drift, [])
    assert not result.accepted
    assert result.cases[1].reason == "pending_command_violation"
    assert result.cases[1].checked_intervals == 1


def test_positive_command_can_reach_terminal_set_without_changing_feedback_between_lags():
    check = witness(replace(PandaBackupConfig(), horizon_steps=200))
    result = check.verify(np.zeros(7), np.zeros(7), np.full(7, 0.2), np.eye(7), np.zeros(7), [])
    assert result.accepted
    assert all(case.terminal_energy <= case.terminal_radius_squared for case in result.cases)
    assert result.policy.config == PandaBackupConfig()


@pytest.mark.parametrize("case", ["moving", "coupled_bias", "pending", "outside"])
def test_batched_prediction_matches_sequential_fifo_and_feedback(case):
    check = witness()
    q, v, candidate = np.zeros(7), np.full(7, 0.2), np.full(7, 0.2)
    mass, drift, history = np.eye(7), np.zeros(7), []
    if case == "coupled_bias":
        mass = np.diag(np.linspace(1, 2, 7)) + 0.02 * np.ones((7, 7))
        drift = np.linspace(-0.1, 0.1, 7)
        v, candidate = np.linspace(-0.3, 0.3, 7), np.linspace(-2, 2, 7)
        history = [np.full(7, value) for value in (-1, -0.5, 0.5, 1)]
    elif case == "pending":
        v[0], candidate[0] = check.bounds.velocity[0] - 0.01, -20
        history = [np.full(7, 20)]
    elif case == "outside":
        v[0] = 3
    matrix = np.linalg.inv(mass)
    result = check.verify(q, v, candidate, matrix, drift, history)
    for actual in result.cases:
        lag = actual.lag
        position, velocity = q.copy(), v.copy()
        # The reference executes one physical-model interval at a time, without powers.
        queue = [np.zeros(7)] * max(0, 4 - len(history)) + list(history[-4:])
        metric, radius = check.metrics[lag][2], check.terminal_radius_squared(result.policy, lag)
        expected_reason, expected_count = "terminal_not_reached", 200
        for step in range(200):
            sent = candidate if step == 0 else result.policy.requested_torque(position, velocity)
            if np.any(np.abs(sent) > check.bounds.limit):
                expected_reason, expected_count = "backup_torque_limit", step
                break
            delivered = queue[-lag] if lag else sent
            acceleration = matrix @ delivered + drift
            if not check.bounds._inside_interval(position, velocity, acceleration):
                expected_reason = (
                    "pending_command_violation"
                    if step < lag
                    else "candidate_interval_violation"
                    if step == lag
                    else "backup_interval_violation"
                )
                expected_count = step + 1
                break
            position = position + 0.01 * velocity + 0.5 * 0.01**2 * acceleration
            velocity = velocity + 0.01 * acceleration
            queue = queue[1:] + [sent]
            state = np.array([position - q, velocity] + [matrix @ u + drift for u in queue])
            energy = np.sum(state * (metric @ state))
            if energy <= radius:
                expected_reason, expected_count = "terminal_set_reached", step + 1
                break
        assert actual.reason == expected_reason
        assert actual.checked_intervals == expected_count
        assert actual.accepted == (expected_reason == "terminal_set_reached")
        if actual.accepted:
            np.testing.assert_allclose(actual.terminal_energy, energy, rtol=1e-10, atol=1e-12)
