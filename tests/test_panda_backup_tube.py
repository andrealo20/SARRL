from dataclasses import replace

import numpy as np
import pytest

from sarrl.safety.panda_backup import PandaBackupConfig, PandaBackupWitness
from sarrl.safety.panda_backup_tube import PandaBackupEnvelope, PandaBackupTube


def setup_tube(*, maximum=4, acceleration=1e-6, sensor=1e-7):
    witness = PandaBackupWitness(
        np.full(7, -2.0),
        np.full(7, 2.0),
        np.full(7, 20.0),
        PandaBackupConfig(max_delay_steps=maximum),
    )
    envelope = PandaBackupEnvelope(
        np.full(7, -1.0),
        np.full(7, 1.0),
        np.ones(7),
        np.full(7, acceleration),
        np.full(7, sensor),
        np.full(7, sensor),
        np.zeros(7),
        np.zeros(7),
        "Synthetic bounded double integrator; not a Panda plant envelope",
    )
    return witness, envelope, PandaBackupTube(witness, envelope)


@pytest.mark.parametrize("maximum", [0, 1, 4])
def test_small_nonzero_disturbance_has_a_conditional_terminal_set(maximum):
    _, _, tube = setup_tube(maximum=maximum)
    cases = tube.assess(np.zeros(7), np.eye(7), np.zeros(7))
    assert len(cases) == maximum + 1
    for case in cases:
        assert case.feasible
        assert 0 < case.contraction < 1
        assert 0 < case.minimum_radius < case.admissible_radius


def test_nominal_case_recovers_terminal_radius_with_the_model_domain_intersected():
    witness, _, tube = setup_tube(acceleration=0, sensor=0)
    domain_witness = PandaBackupWitness(
        np.full(7, -1.03), np.full(7, 1.03), np.full(7, 20.0), witness.config
    )
    domain_witness.bounds.velocity[:] = 1
    policy = domain_witness.policy(np.zeros(7), np.eye(7), np.zeros(7))
    for case in tube.assess(np.zeros(7), np.eye(7), np.zeros(7)):
        expected = np.sqrt(domain_witness.terminal_radius_squared(policy, case.lag))
        np.testing.assert_allclose(case.admissible_radius, expected, rtol=2e-12)
        assert case.disturbance_radius == case.minimum_radius == 0


@pytest.mark.parametrize("kind", ["dynamic", "measurement", "domain", "equilibrium"])
def test_unusable_contract_does_not_silently_return_feasible(kind):
    witness, envelope, _ = setup_tube()
    drift = np.zeros(7)
    if kind == "dynamic":
        envelope = replace(envelope, acceleration_error=np.full(7, 10.0))
    elif kind == "measurement":
        envelope = replace(envelope, position_error=np.ones(7))
    elif kind == "domain":
        envelope = replace(envelope, position_lower=np.full(7, 0.1))
    else:
        drift[:] = 21
    cases = PandaBackupTube(witness, envelope).assess(np.zeros(7), np.eye(7), drift)
    assert not any(case.feasible for case in cases)
    if kind == "dynamic":
        assert any(case.reason == "disturbance_exceeds_terminal_capacity" for case in cases)


@pytest.mark.parametrize("maximum", [0, 4])
def test_disturbed_true_fifo_stays_inside_terminal_bound_for_all_lags(maximum):
    witness, envelope, tube = setup_tube(maximum=maximum, acceleration=1e-5, sensor=1e-6)
    mass = np.diag(np.linspace(0.8, 1.4, 7)) + 0.02 * np.ones((7, 7))
    policy = witness.policy(np.zeros(7), np.linalg.inv(mass), np.linspace(-0.05, 0.05, 7))
    cases = tube.assess(policy.anchor, policy.matrix, policy.drift)
    for case in cases:
        assert case.feasible
        metric, inverse = tube.metrics[case.lag][2:]
        for direction in range(len(metric)):
            state = np.zeros((len(metric), 7))
            state[:, direction % 7] = inverse[:, direction]
            state *= case.admissible_radius / np.sqrt(np.sum(state * (metric @ state)))
            for step in range(80):
                signs = np.where((np.arange(7) + step) % 2, -1.0, 1.0)
                q, v = state[:2]
                sent = policy.requested_torque(
                    q + signs * envelope.position_error, v - signs * envelope.velocity_error
                )
                assert np.all(np.abs(sent) <= witness.bounds.limit + 1e-10)
                acceleration = state[-case.lag] if case.lag else policy.matrix @ sent + policy.drift
                acceleration = acceleration + signs * envelope.acceleration_error
                times = np.linspace(0, 0.01, 21)[:, None]
                positions = q + times * v + 0.5 * times**2 * acceleration
                velocities = v + times * acceleration
                assert np.all(np.abs(positions) <= 1 + 1e-10)
                assert np.all(np.abs(velocities) <= 1 + 1e-10)
                for queued in state[2:]:
                    assert np.all(np.abs(mass @ queued + policy.equilibrium_torque) <= 20 + 1e-10)
                previous_norm = np.sqrt(np.sum(state * (metric @ state)))
                next_q, next_v = positions[-1], velocities[-1]
                queue = [*state[3:], policy.matrix @ sent + policy.drift] if maximum else []
                state = np.array([next_q, next_v, *queue])
                next_norm = np.sqrt(np.sum(state * (metric @ state)))
                assert (
                    next_norm <= case.contraction * previous_norm + case.disturbance_radius + 1e-10
                )
                assert next_norm <= case.admissible_radius + 1e-10


@pytest.mark.parametrize("lag", [0, 2, 4])
def test_candidate_interval_uses_fixed_torque_not_backup_feedback(lag):
    _, envelope, tube = setup_tube(acceleration=0.2, sensor=0.001)
    metric = tube.metrics[lag][2]
    for q_sign in (-1, 1):
        for v_sign in (-1, 1):
            for w_sign in (-1, 1):
                # Identical delivered candidate/prefix cancels from the error equation.
                q = q_sign * envelope.position_error
                v = v_sign * envelope.velocity_error
                w = w_sign * envelope.acceleration_error
                state = np.array(
                    [q + 0.01 * v + 0.5 * 0.01**2 * w, v + 0.01 * w, *[np.zeros(7)] * 4]
                )
                assert (
                    np.sqrt(np.sum(state * (metric @ state))) <= tube.candidate_radius(lag) + 1e-12
                )


def test_tube_contains_errors_between_distinct_feedback_trajectories():
    _, envelope, tube = setup_tube(acceleration=0.1, sensor=0.001)
    for case in tube.assess(np.zeros(7), np.eye(7), np.zeros(7)):
        nominal, physical = np.zeros((6, 7)), np.zeros((6, 7))
        physical[0], physical[1] = envelope.position_error, -envelope.velocity_error
        metric = tube.metrics[case.lag][2]
        radius = np.sqrt(np.sum(physical * (metric @ physical)))
        radii = case.backup_radii(radius, 50)
        for step in range(50):
            sign = 1 if step % 2 else -1
            next_states = []
            for current, noisy in ((nominal, False), (physical, True)):
                q, v = current[:2]
                eq = sign * envelope.position_error if noisy else 0
                ev = -sign * envelope.velocity_error if noisy else 0
                sent = -25 * (q + eq) - 10 * (v + ev)
                delivered = current[-case.lag] if case.lag else sent
                w = sign * envelope.acceleration_error if noisy else 0
                acceleration = delivered + w
                next_states.append(
                    np.array(
                        [
                            q + 0.01 * v + 0.5 * 0.01**2 * acceleration,
                            v + 0.01 * acceleration,
                            *current[3:],
                            sent,
                        ]
                    )
                )
            nominal, physical = next_states
            error = physical - nominal
            assert np.sqrt(np.sum(error * (metric @ error))) <= radii[step + 1] + 1e-10


@pytest.mark.parametrize("lag", [0, 4])
def test_time_varying_acceleration_and_propagation_defects_fit_the_error_bound(lag):
    witness, envelope, _ = setup_tube(acceleration=0.1, sensor=0)
    envelope = replace(envelope, position_defect=np.full(7, 2e-6), velocity_defect=np.full(7, 3e-6))
    tube = PandaBackupTube(witness, envelope)
    case = tube.assess(np.zeros(7), np.eye(7), np.zeros(7))[lag]
    metric = tube.metrics[lag][2]
    # Start at an exact common equilibrium, then integrate two different half-step forces.
    for sign in (-1, 1):
        q, v = np.zeros(7), np.zeros(7)
        for force in (sign * envelope.acceleration_error, -sign * envelope.acceleration_error):
            q = q + 0.005 * v + 0.5 * 0.005**2 * force
            v = v + 0.005 * force
        for defect_sign in (-1, 1):
            state = np.array(
                [
                    q + defect_sign * envelope.position_defect,
                    v - defect_sign * envelope.velocity_defect,
                    *[np.zeros(7)] * 4,
                ]
            )
            assert np.sqrt(np.sum(state * (metric @ state))) <= case.disturbance_radius
            assert np.sqrt(np.sum(state * (metric @ state))) <= tube.candidate_radius(lag)


def test_coupled_torque_boundary_includes_aligned_sensor_errors():
    witness, envelope, tube = setup_tube(acceleration=1e-6, sensor=1e-5)
    direction = np.linspace(-0.5, 0.5, 7)
    mass = 5 * np.eye(7) + np.outer(direction, direction)
    policy = witness.policy(np.zeros(7), np.linalg.inv(mass), np.zeros(7))
    for case in tube.assess(policy.anchor, policy.matrix, policy.drift):
        assert case.feasible
        metric, inverse = tube.metrics[case.lag][2:]
        feedback = np.array([-25.0, -10.0, 0, 0, 0, 0])
        for row in mass:
            # This direction maximizes one actuator's feedback over the full-joint ball.
            state = np.outer(inverse @ feedback, row)
            state *= case.admissible_radius / np.sqrt(np.sum(state * (metric @ state)))
            signs = np.sign(row)
            raw = policy.requested_torque(
                state[0] - signs * envelope.position_error,
                state[1] - signs * envelope.velocity_error,
            )
            assert np.all(np.abs(raw) <= witness.bounds.limit + 1e-10)


@pytest.mark.parametrize(
    "field,value",
    [
        ("acceleration_error", np.full(7, -1)),
        ("velocity_error", np.full(7, np.inf)),
        ("position_error", np.zeros(6)),
        ("position_defect", np.full(7, -1)),
        ("velocity_defect", np.full(7, np.nan)),
        ("velocity_limit", np.zeros(7)),
        ("position_lower", np.ones(7)),
        ("assumption", ""),
    ],
)
def test_invalid_envelope_is_rejected(field, value):
    _, envelope, _ = setup_tube()
    with pytest.raises(ValueError):
        replace(envelope, **{field: value})


def test_envelope_copies_input_and_rejects_invalid_radius_requests():
    _, envelope, tube = setup_tube()
    source = np.ones(7)
    copied = replace(envelope, acceleration_error=source)
    source[:] = 9
    np.testing.assert_array_equal(copied.acceleration_error, np.ones(7))
    with pytest.raises(ValueError):
        copied.acceleration_error[0] = 3
    case = tube.assess(np.zeros(7), np.eye(7), np.zeros(7))[0]
    for radius, count in ((-1, 2), (np.inf, 2), (0, True), (0, -1)):
        with pytest.raises(ValueError):
            case.backup_radii(radius, count)
    for lag in (-1, True, 5):
        with pytest.raises(ValueError):
            tube.candidate_radius(lag)
