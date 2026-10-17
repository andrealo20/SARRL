from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from sarrl.safety.panda_state_region import PandaStateRegion, VectorBox
from sarrl.safety.panda_structured_backup import PandaParameterBox, PandaStructuredContext
from sarrl.safety.panda_structured_prefix import (
    PandaPrefixConfig,
    PandaPrefixErrors,
    PandaStructuredPrefix,
)


def errors(scale=0):
    return PandaPrefixErrors(*[np.full(7, scale) for _ in range(5)])


def region(*, nonlinear=False, bias=0):
    domain = VectorBox(np.full(7, -1.0), np.ones(7))
    m0, mp = np.eye(7), np.zeros((7, 7))
    h0, hp = np.full(7, bias), np.zeros(7)
    dm0, dmp = np.zeros((7, 7, 7)), np.zeros((7, 7, 7))
    dh0, dhp = np.zeros((7, 14)), np.zeros((7, 14))
    if nonlinear:
        direction = np.linspace(-0.1, 0.2, 7)
        m0 = np.diag(np.linspace(1.2, 1.6, 7)) + np.outer(direction, direction)
        mp = 0.1 * np.eye(7) + 0.2 * np.outer(direction, direction)
        h0, hp = np.linspace(0.02, -0.03, 7), np.full(7, 0.01)
        for i in range(7):
            dm0[i, i, i], dmp[i, i, i] = 0.06, 0.015
        dh0[:, :7] = np.abs(0.3 * np.eye(7) + 0.02 * np.ones((7, 7)))
        dhp[:, :7] = np.abs(0.1 * np.eye(7) + 0.01 * np.ones((7, 7)))
        dh0[:, 7:], dhp[:, 7:] = 0.08 * np.eye(7), 0.06 * np.eye(7)
    return PandaStateRegion(
        domain,
        domain,
        np.zeros(7),
        np.zeros(7),
        m0,
        mp,
        h0,
        hp,
        dm0,
        dmp,
        dh0,
        dhp,
        np.zeros((7, 7)),
        np.zeros(7),
        "Analytic derivative bounds for a manufactured seven-joint model, not Panda",
    )


def coefficients(model, q, v):
    return (
        model.mass0 + np.diag(0.06 * q),
        model.payload_mass + np.diag(0.015 * q),
        model.bias0 + (0.3 * np.eye(7) + 0.02 * np.ones((7, 7))) @ q + 0.04 * v * np.abs(v),
        model.payload_bias + (0.1 * np.eye(7) + 0.01 * np.ones((7, 7))) @ q + 0.03 * v**2,
    )


def checker(
    model, parameters=None, uncertainty=None, *, maximum=4, velocity=1, torque=20, config=None
):
    box = parameters or PandaParameterBox(np.zeros(8), np.zeros(8))
    return PandaStructuredPrefix(
        model,
        box,
        np.full(7, -1.0),
        np.ones(7),
        np.full(7, velocity),
        np.full(7, torque),
        uncertainty or errors(),
        config or PandaPrefixConfig(max_delay_steps=maximum),
    )


@pytest.mark.parametrize("payload,friction", [(0.5, 0.5), (1.5, 1.7), (2.5, 3.0)])
def test_nonlinear_uncertain_prefix_contains_independent_dense_trajectories_for_all_lags(
    payload, friction
):
    model = region(nonlinear=True)
    parameters = PandaParameterBox(np.array([0.5, *[0.5] * 7]), np.array([2.5, *[3.0] * 7]))
    noise = PandaPrefixErrors(
        np.full(7, 1e-5), np.full(7, 2e-5), np.zeros(7), np.zeros(7), np.full(7, 2e-6)
    )
    verify = checker(model, parameters, noise)
    context = PandaStructuredContext(np.zeros(7), np.array([0.9, *[1.5] * 7]))
    mq, mv = np.linspace(-0.02, 0.01, 7), np.linspace(-0.01, 0.03, 7)
    candidate = np.linspace(0.02, -0.01, 7)
    history = [np.full(7, value) for value in (0.02, -0.01, 0.01)]
    saved = [command.copy() for command in history]
    result = verify.verify_prefix(context, mq, mv, candidate, history, intervals=8)
    assert result.prefix_contained and not result.terminal_verified
    for case in result.cases:
        q, v = mq + noise.position_measurement, mv - noise.velocity_measurement
        queue = [np.zeros(7), *[command.copy() for command in history]]
        for step, record in enumerate(case.intervals):
            signs = np.where((np.arange(7) + step) % 2, -1.0, 1.0)
            if step:
                measured_q = q + signs * noise.position_measurement
                measured_v = v - signs * noise.velocity_measurement
                _, sent = context.point(
                    measured_q, measured_v, *coefficients(model, measured_q, measured_v)
                )
                sent += signs * noise.command_error
            else:
                sent = candidate
            delivered = queue[-case.lag] if case.lag else sent
            assert record.sent.contains(VectorBox.point(sent))
            assert record.delivered.contains(VectorBox.point(delivered))

            def dynamics(time, state, held=delivered):
                position, speed = state[:7], state[7:]
                m0, mp, h0, hp = coefficients(model, position, speed)
                acceleration = np.linalg.solve(
                    m0 + payload * mp, held - h0 - payload * hp - friction * speed
                )
                return np.concatenate((speed, acceleration))

            dense = solve_ivp(
                dynamics,
                (0, 0.01),
                np.concatenate((q, v)),
                t_eval=np.linspace(0, 0.01, 21),
                rtol=1e-11,
                atol=1e-13,
            )
            assert dense.success
            assert np.all(dense.y[:7] >= record.q_tube.lower[:, None] - 1e-10)
            assert np.all(dense.y[:7] <= record.q_tube.upper[:, None] + 1e-10)
            assert np.all(dense.y[7:] >= record.v_tube.lower[:, None] - 1e-10)
            assert np.all(dense.y[7:] <= record.v_tube.upper[:, None] + 1e-10)
            q, v = dense.y[:7, -1], dense.y[7:, -1]
            assert record.q_end.contains(VectorBox(q - 1e-14, q + 1e-14))
            assert record.v_end.contains(VectorBox(v - 1e-14, v + 1e-14))
            queue = [*queue[1:], sent.copy()]
    for old, new in zip(saved, history, strict=True):
        np.testing.assert_array_equal(old, new)


def test_position_violation_between_safe_endpoints_is_not_accepted():
    verify = checker(region(), maximum=0)
    q, v, candidate = np.zeros(7), np.zeros(7), np.zeros(7)
    q[0], v[0], candidate[0] = 1 - 1e-5, 0.01, -2
    assert q[0] + 0.01 * v[0] + 0.5 * 0.01**2 * candidate[0] < 1
    assert q[0] + 0.005 * v[0] + 0.5 * 0.005**2 * candidate[0] > 1
    result = verify.verify_prefix(
        PandaStructuredContext(np.zeros(7), np.zeros(8)), q, v, candidate, [], intervals=1
    )
    assert not result.prefix_contained
    assert result.cases[0].reason == "whole_interval_bounds_not_established"


def test_pending_command_cannot_be_erased_by_a_braking_candidate():
    verify = checker(region(), maximum=1, velocity=0.2)
    v, pending = np.zeros(7), np.zeros(7)
    v[0], pending[0] = 0.19, 20
    result = verify.verify_prefix(
        PandaStructuredContext(np.zeros(7), np.zeros(8)),
        np.zeros(7),
        v,
        -pending,
        [pending],
        intervals=1,
    )
    assert result.cases[0].prefix_contained
    assert not result.cases[1].prefix_contained
    assert result.cases[1].reason == "whole_interval_bounds_not_established"


def test_startup_queue_contains_zero_torque_not_zero_acceleration():
    verify = checker(region(bias=0.1), maximum=1)
    result = verify.verify_prefix(
        PandaStructuredContext(np.zeros(7), np.zeros(8)),
        np.zeros(7),
        np.zeros(7),
        np.full(7, 0.1),
        [],
        intervals=1,
    )
    assert result.prefix_contained
    np.testing.assert_array_equal(result.cases[1].intervals[0].delivered.lower, np.zeros(7))
    delayed_v = result.cases[1].intervals[0].v_end
    assert delayed_v.contains(VectorBox.point(np.full(7, -0.001)))
    assert not delayed_v.contains(VectorBox.point(np.zeros(7)))


def test_nonzero_partial_interval_propagation_defects_are_retained():
    uncertainty = replace(
        errors(), position_defect=np.full(7, 1e-4), velocity_defect=np.full(7, 2e-4)
    )
    verify = checker(region(), uncertainty=uncertainty)
    result = verify.verify_prefix(
        PandaStructuredContext(np.zeros(7), np.zeros(8)),
        np.zeros(7),
        np.zeros(7),
        np.zeros(7),
        [],
        intervals=1,
    )
    assert result.prefix_contained
    for case in result.cases:
        record = case.intervals[0]
        for fraction in np.linspace(0, 1, 21):
            q = fraction * uncertainty.position_defect
            v = -fraction * uncertainty.velocity_defect
            assert record.q_tube.contains(VectorBox.point(q))
            assert record.v_tube.contains(VectorBox.point(v))
        assert record.q_end.contains(VectorBox.point(uncertainty.position_defect))
        assert record.v_end.contains(VectorBox.point(-uncertainty.velocity_defect))


def test_region_acceleration_encloses_state_torque_and_physical_remainders():
    model = replace(
        region(nonlinear=True),
        mass_remainder=np.full((7, 7), 1e-3),
        bias_remainder=np.full(7, 2e-3),
    )
    parameters = PandaParameterBox(np.array([0.5, *[0.5] * 7]), np.array([2.5, *[3.0] * 7]))
    q_box, v_box = (
        VectorBox(np.full(7, -0.05), np.full(7, 0.05)),
        VectorBox(np.full(7, -0.03), np.full(7, 0.03)),
    )
    torque_box = VectorBox(np.full(7, 0.08), np.full(7, 0.12))
    enclosure = model.acceleration(parameters, q_box, v_box, torque_box)
    for mass in (0.5, 1.5, 2.5):
        for friction in (0.5, 3.0):
            for sign in (-1, 1):
                q, v = sign * np.linspace(-0.05, 0.05, 7), sign * np.linspace(0.03, -0.03, 7)
                m0, mp, h0, hp = coefficients(model, q, v)
                actual_mass = m0 + mass * mp + np.full((7, 7), 1e-3)
                rhs = np.full(7, 0.1 + sign * 0.02) - h0 - mass * hp - friction * v - 2e-3
                actual = np.linalg.solve(actual_mass, rhs)
                assert enclosure.contains(VectorBox.point(actual))


def test_singular_center_mass_is_reported_as_an_unverified_contract():
    model = replace(region(), payload_mass=-np.eye(7))
    parameters = PandaParameterBox(np.zeros(8), np.array([2.0, *np.zeros(7)]))
    result = checker(model, parameters).verify_prefix(
        PandaStructuredContext(np.zeros(7), np.zeros(8)),
        np.zeros(7),
        np.zeros(7),
        np.zeros(7),
        [],
        intervals=1,
    )
    assert not result.prefix_contained
    assert all(case.reason == "center_mass_family_unverified" for case in result.cases)


@pytest.mark.parametrize("failure", ["region", "mass", "torque", "budget"])
def test_unverified_contract_or_iteration_budget_fails_closed(failure):
    model, uncertainty, cfg, limit = region(), errors(), PandaPrefixConfig(max_delay_steps=0), 20
    if failure == "region":
        model = replace(model, q_domain=VectorBox(np.full(7, -0.001), np.full(7, 0.001)))
    elif failure == "mass":
        model = replace(model, mass_remainder=np.full((7, 7), 10.0))
    elif failure == "torque":
        uncertainty, limit = errors(1e-3), 0.01
    else:
        cfg = replace(cfg, picard_iterations=1)
    verify = checker(model, uncertainty=uncertainty, config=cfg, torque=limit)
    result = verify.verify_prefix(
        PandaStructuredContext(np.zeros(7), np.zeros(8)),
        np.zeros(7),
        np.full(7, 0.4),
        np.zeros(7),
        [],
        intervals=2,
    )
    assert not result.prefix_contained and not result.terminal_verified
    expected = {
        "region": "state_region_not_covered",
        "mass": "mass_variation_unverified",
        "torque": "raw_feedback_torque_bounds_not_established",
        "budget": "picard_budget_exhausted",
    }
    assert result.cases[0].reason == expected[failure]


@pytest.mark.parametrize(
    "field,value",
    [
        ("mass0_q_derivative", np.full((7, 7, 7), -1)),
        ("payload_bias_derivative", np.zeros((7, 13))),
        ("bias_remainder", np.full(7, np.inf)),
        ("assumption", ""),
        ("q_center", np.full(7, 2)),
    ],
)
def test_invalid_region_assumptions_are_rejected(field, value):
    with pytest.raises(ValueError):
        replace(region(), **{field: value})


@pytest.mark.parametrize(
    "configuration",
    [
        PandaPrefixConfig(control_dt=np.nan),
        PandaPrefixConfig(max_delay_steps=True),
        PandaPrefixConfig(max_delay_steps=5),
        PandaPrefixConfig(picard_iterations=0),
        PandaPrefixConfig(inflation_fraction=0),
    ],
)
def test_invalid_configuration_is_rejected(configuration):
    with pytest.raises(ValueError):
        checker(region(), config=configuration)


def test_invalid_measurement_and_command_inputs_are_rejected():
    with pytest.raises(ValueError):
        replace(errors(), velocity_measurement=np.full(7, np.inf))
    with pytest.raises(ValueError):
        replace(errors(), position_defect=np.full(7, -1))
    verify, context = checker(region()), PandaStructuredContext(np.zeros(7), np.zeros(8))
    for count in (0, True, 201):
        with pytest.raises(ValueError):
            verify.verify_prefix(
                context, np.zeros(7), np.zeros(7), np.zeros(7), [], intervals=count
            )
    with pytest.raises(ValueError):
        verify.verify_prefix(context, np.zeros(7), np.zeros(7), np.full(7, 21), [], intervals=1)


def test_region_and_error_arrays_are_copied_and_readonly():
    model, uncertainty = region(), errors()
    derivative = np.zeros((7, 14))
    copied = replace(model, bias0_derivative=derivative)
    derivative[:] = 2
    np.testing.assert_array_equal(copied.bias0_derivative, np.zeros((7, 14)))
    for array in (copied.bias0_derivative, model.q_domain.lower, uncertainty.command_error):
        with pytest.raises(ValueError):
            array.flat[0] = 2


def test_boxes_and_command_error_cannot_silently_shrink_assumptions():
    with pytest.raises(ValueError):
        VectorBox(np.ones(7), np.zeros(7))
    with pytest.raises(ValueError):
        VectorBox.point(np.zeros(7)).inflate(-np.ones(7))
    with pytest.raises(ValueError):
        region().torque(
            PandaStructuredContext(np.zeros(7), np.zeros(8)),
            VectorBox.point(np.zeros(7)),
            VectorBox.point(np.zeros(7)),
            -np.ones(7),
        )
