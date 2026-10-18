"""Held-flow linearization, isolated multipliers and conservative viscous compensation."""

from fractions import Fraction

import numpy as np
import pytest

from sarrl.safety.panda_center_arithmetic import ArrayEnclosure, _export
from sarrl.safety.panda_friction_compensation import bounded_friction_context
from sarrl.safety.panda_sampled_stability import (
    delayed_feedback_map,
    enclose_sampled_stability,
    held_feedback_map,
)
from sarrl.safety.panda_structured_backup import PandaParameterBox, PandaStructuredContext
from sarrl.safety.panda_validated_dynamics import arithmetic, point


def context():
    from sarrl.robots.panda import HOME

    return PandaStructuredContext(HOME, np.array([1.5, *[1.75] * 7]))


def parameters():
    return PandaParameterBox(np.array([0.5, *[0.5] * 7]), np.array([2.5, *[3.0] * 7]))


def test_friction_cap_preserves_retained_estimate_mass_anchor_gains_and_full_uncertainty():
    retained, physical = context(), parameters()
    saved = retained.parameters.copy()
    guarded = bounded_friction_context(retained, physical)
    np.testing.assert_array_equal(retained.parameters, saved)
    np.testing.assert_array_equal(guarded.anchor, retained.anchor)
    assert guarded.parameters[0] == retained.parameters[0]
    np.testing.assert_array_equal(guarded.parameters[1:], physical.lower[1:])
    assert (guarded.kp, guarded.kd) == (25, 10)
    assert np.all(physical.upper[1:] == 3)
    with pytest.raises(ValueError):
        guarded.parameters[0] = 0


def test_cap_never_increases_a_lower_estimate_or_depends_on_payload_cell():
    retained = PandaStructuredContext(np.zeros(7), np.array([1.5, 0.1, 0.3, 0.5, 0.7, 1, 2, 4]))
    first = bounded_friction_context(retained, parameters())
    lower = parameters().lower.copy()
    lower[0] = 1.5
    second = bounded_friction_context(retained, PandaParameterBox(lower, parameters().upper))
    np.testing.assert_array_equal(first.parameters, second.parameters)
    np.testing.assert_array_equal(first.parameters[1:], [0.1, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5])


@pytest.mark.parametrize("kind", ["context", "parameters"])
def test_cap_rejects_invalid_contract_types(kind):
    with pytest.raises(ValueError):
        bounded_friction_context(
            None if kind == "context" else context(), None if kind == "parameters" else parameters()
        )


@pytest.mark.parametrize("lag", [-1, 5, True, 1.5])
def test_stability_rejects_invalid_lag_before_numerics(lag):
    with pytest.raises(ValueError, match="lag"):
        enclose_sampled_stability(context(), np.array([1.5, *[0.5] * 7]), lag=lag)


@pytest.mark.parametrize("period", [0, -0.01, np.inf, np.nan])
def test_bad_period_is_rejected(period):
    with pytest.raises(ValueError, match="control_dt"):
        enclose_sampled_stability(context(), np.array([1.5, *[0.5] * 7]), control_dt=period)


def test_unmatched_payload_cannot_claim_the_anchor_is_an_equilibrium():
    with pytest.raises(ValueError, match="matched payload"):
        enclose_sampled_stability(context(), np.array([0.5, *[0.5] * 7]))


def test_held_flow_matches_exact_double_integrator_not_euler():
    pytest.importorskip("flint")
    identity = ArrayEnclosure(np.eye(7), np.eye(7))
    zero = ArrayEnclosure(np.zeros((7, 7)), np.zeros((7, 7)))
    with arithmetic(128) as backend:
        _, _, _, sampled = held_feedback_map(
            identity,
            zero,
            np.zeros(7),
            np.zeros(7),
            np.zeros(7),
            kp=25,
            kd=10,
            control_dt=0.01,
            backend=backend,
        )
        enclosed = _export(sampled)
    t = Fraction(0.01)
    expected = {
        (0, 0): 1 - 25 * t * t / 2,
        (0, 7): t - 10 * t * t / 2,
        (7, 0): -25 * t,
        (7, 7): 1 - 10 * t,
    }
    for index, value in expected.items():
        assert (
            Fraction(float(enclosed.lower[index]))
            <= value
            <= Fraction(float(enclosed.upper[index]))
        )
    assert enclosed.upper[0, 0] < 1


@pytest.mark.parametrize("lag", [1, 2, 4])
def test_augmented_map_keeps_pending_commands_and_sends_one_new_feedback(lag):
    pytest.importorskip("flint")
    with arithmetic(128) as backend:
        flow = point(np.eye(21), backend)
        flow[:14, 14:] = point(np.arange(98).reshape(14, 7) / 100, backend)
        feedback = point(np.arange(98).reshape(7, 14) / 10, backend)
        mapped = delayed_feedback_map(flow, feedback, lag, backend)
        state = point(np.arange(14 + 7 * lag), backend)
        answer = mapped @ state
        expected_state = flow[:14, :14] @ state[:14] + flow[:14, 14:] @ state[14:21]
        expected = np.concatenate((expected_state, state[21:], feedback @ state[:14]))
        a, b = _export(answer), _export(expected)
        assert np.all(a.lower <= b.upper) and np.all(b.lower <= a.upper)
        # The old queue head changes the physical update; it cannot be erased.
        assert not np.all(a.lower[:14] <= np.arange(14))


@pytest.fixture(scope="module")
def unstable():
    pytest.importorskip("flint")
    pytest.importorskip("mujoco")
    return enclose_sampled_stability(context(), np.array([1.5, *[0.5] * 7]))


def test_admissible_backup_has_certified_sampled_instability(unstable):
    assert unstable.instability_verified and not unstable.local_schur_stability_verified
    assert np.count_nonzero(unstable.eigen_modulus.lower > 1) == 2
    assert 1.009 < max(unstable.eigen_modulus.lower) < 1.010
    assert unstable.sampled_jacobian.lower.shape == (14, 14)


def test_gravity_derivative_and_sampled_map_agree_with_independent_float_kinematics(unstable):
    from scipy.linalg import expm

    from sarrl.safety.panda_kinematic_coefficients import packaged_kinematic_coefficients

    nominal, payload, damping = packaged_kinematic_coefficients(context().anchor)
    mass, gravity = nominal.mass + 1.5 * payload.mass, nominal.gravity_q + 1.5 * payload.gravity_q
    inverse = np.linalg.inv(mass)
    generator = np.zeros((21, 21))
    generator[:7, 7:14] = np.eye(7)
    generator[7:14, :7], generator[7:14, 7:14] = (
        -inverse @ gravity,
        -inverse @ np.diag(damping + 0.5),
    )
    generator[7:14, 14:] = inverse
    gain = np.column_stack((gravity - 25 * mass, np.diag(damping + 1.75) - 10 * mass))
    flow = expm(0.01 * generator)
    sampled = flow[:14, :14] + flow[:14, 14:] @ gain
    assert np.max(unstable.gravity_q.absolute_error(gravity)) < 1e-10
    assert np.max(unstable.sampled_jacobian.absolute_error(sampled)) < 1e-10


def test_guarded_compensation_removes_this_point_instability_at_largest_lag():
    pytest.importorskip("flint")
    pytest.importorskip("mujoco")
    guarded = bounded_friction_context(context(), parameters())
    result = enclose_sampled_stability(guarded, np.array([1.5, *[0.5] * 7]), lag=4)
    assert result.sampled_jacobian.lower.shape == (42, 42)
    assert result.local_schur_stability_verified and not result.instability_verified
    assert max(result.eigen_modulus.upper) < 0.965


def test_precision_refinement_restores_context_and_keeps_isolated_modes(unstable):
    import flint

    previous = flint.ctx.prec
    refined = enclose_sampled_stability(context(), np.array([1.5, *[0.5] * 7]), precision_bits=256)
    assert flint.ctx.prec == previous
    assert np.all(refined.eigen_modulus.lower <= unstable.eigen_modulus.upper)
    assert np.all(unstable.eigen_modulus.lower <= refined.eigen_modulus.upper)


def test_eigen_isolation_failure_restores_precision(monkeypatch):
    flint = pytest.importorskip("flint")
    pytest.importorskip("mujoco")
    import sarrl.safety.panda_sampled_stability as implementation

    previous = flint.ctx.prec

    def failure(*args):
        raise ValueError("injected matrix failure")

    monkeypatch.setattr(implementation, "_matrix", failure)
    with pytest.raises(ValueError, match="injected"):
        enclose_sampled_stability(context(), np.array([1.5, *[0.5] * 7]), precision_bits=256)
    assert flint.ctx.prec == previous
