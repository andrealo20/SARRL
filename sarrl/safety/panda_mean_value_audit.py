"""Causal replay and exact-rational checks of shared-coordinate Taylor endpoints."""

from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from sarrl.safety.panda_affine_state import AffineState
from sarrl.safety.panda_center_arithmetic import _export
from sarrl.safety.panda_mean_value_prefix import add_error, as_box, physical_coordinates
from sarrl.safety.panda_prefix_audit import _propagation, _require
from sarrl.safety.panda_state_region import VectorBox
from sarrl.safety.panda_validated_dynamics import arithmetic


def same_form(left, right):
    return all(
        np.array_equal(a, b)
        for a, b in (
            (left.center, right.center),
            (left.generators, right.generators),
            (left.residual.lower, right.residual.lower),
            (left.residual.upper, right.residual.upper),
        )
    )


def endpoint_inclusion(output, terms, error):
    """Check c/G rounding transfers and the residual inequality with exact dyadic inputs."""
    for i in range(7):
        center = sum(
            (weight * Fraction(float(form.center[i])) for weight, form in terms), Fraction(0)
        )
        low = sum(
            (weight * Fraction(float(form.residual.lower[i])) for weight, form in terms),
            Fraction(0),
        )
        high = sum(
            (weight * Fraction(float(form.residual.upper[i])) for weight, form in terms),
            Fraction(0),
        )
        discrepancy = sum(
            (
                abs(
                    sum(
                        (weight * Fraction(float(form.generators[i, j])) for weight, form in terms),
                        Fraction(0),
                    )
                    - Fraction(float(output.generators[i, j]))
                )
                for j in range(8)
            ),
            Fraction(0),
        )
        offset = center - Fraction(float(output.center[i]))
        _require(
            Fraction(float(output.residual.lower[i])) <= low + offset - discrepancy - error[i],
            "affine endpoint lower remainder",
        )
        _require(
            Fraction(float(output.residual.upper[i])) >= high + offset + discrepancy + error[i],
            "affine endpoint upper remainder",
        )


def audit_mean_value_prefix(
    checker, context, measured_q, measured_v, candidate, history, result, *, intervals, substeps=1
):
    _require(type(intervals) is int and 1 <= intervals <= 200, "invalid interval count")
    period = checker.substep_period(substeps)
    _require(len(history) == 4 and checker.config.max_delay_steps == 4, "known queue history")
    _require(
        all(checker.torques.contains(VectorBox.point(x)) for x in (candidate, *history)),
        "known input torque limit",
    )
    _require(not result.terminal_verified, "no terminal witness")
    _require(
        same_form(result.parameters, physical_coordinates(checker.parameters)), "parameter identity"
    )
    _require([c.lag for c in result.cases] == list(range(5)), "missing delay case")
    t, checked = Fraction(float(period)), 0
    with arithmetic(checker.precision_bits) as backend:
        for case in result.cases:
            required = intervals * substeps
            _require(len(case.intervals) <= required, "too many intervals")
            _require(case.prefix_contained == (len(case.intervals) == required), "completion flag")
            q = add_error(
                AffineState.constant(measured_q), checker.errors.position_measurement, backend
            )
            v = add_error(
                AffineState.constant(measured_v), checker.errors.velocity_measurement, backend
            )
            queue = [AffineState.constant(x) for x in history]
            for step, record in enumerate(case.intervals):
                _require(
                    same_form(q, record.q_start) and same_form(v, record.v_start), "start identity"
                )
                held, part = divmod(step, substeps)
                if part == 0:  # one send per held period, shared by its substeps
                    sent = (
                        AffineState.constant(candidate)
                        if held == 0
                        else add_error(
                            checker.region.feedback_affine(
                                context,
                                add_error(q, checker.errors.position_measurement, backend),
                                add_error(v, checker.errors.velocity_measurement, backend),
                                backend,
                            ),
                            checker.errors.command_error,
                            backend,
                        )
                    )
                _require(same_form(sent, record.sent), "feedback shared coordinates")
                delivered = queue[-case.lag] if case.lag else sent
                _require(same_form(delivered, record.delivered), "causal FIFO shared coordinates")
                _require(checker.torques.contains(as_box(sent, backend)), "raw torque limit")
                for physical, domain, tube, end in (
                    (
                        checker.positions,
                        checker.region.q_domain,
                        record.q_tube,
                        record.q_endpoint_bound,
                    ),
                    (
                        checker.velocities,
                        checker.region.v_domain,
                        record.v_tube,
                        record.v_endpoint_bound,
                    ),
                ):
                    _require(
                        physical.contains(tube) and domain.contains(tube) and tube.contains(end),
                        "tube domain or independent endpoint bound",
                    )
                acceleration = checker.region.acceleration(
                    checker.parameters, record.q_tube, record.v_tube, as_box(delivered, backend)
                )
                _require(record.acceleration_tube.contains(acceleration), "tube acceleration")
                _propagation(
                    as_box(q, backend),
                    as_box(v, backend),
                    record.acceleration_tube,
                    SimpleNamespace(
                        q_tube=record.q_tube,
                        v_tube=record.v_tube,
                        q_end=record.q_endpoint_bound,
                        v_end=record.v_endpoint_bound,
                    ),
                    period,
                    checker.errors,
                )
                start = checker.region.acceleration_affine(
                    result.parameters, q, v, delivered, backend
                )
                _require(
                    same_form(start, record.acceleration_start), "initial acceleration coordinates"
                )
                jerk = _export(
                    abs(
                        checker.region.jerk(
                            checker.parameters,
                            record.q_tube,
                            record.v_tube,
                            as_box(delivered, backend),
                            record.acceleration_tube,
                            backend,
                        )
                    )
                )
                _require(
                    np.all(record.jerk_bound.lower >= 0)
                    and np.all(record.jerk_bound.upper >= jerk.upper),
                    "held-flow derivative bound",
                )
                bounds = [Fraction(float(x)) for x in record.jerk_bound.upper]
                endpoint_inclusion(
                    record.q_end,
                    ((Fraction(1), q), (t, v), (t * t / 2, start)),
                    [t**3 / 6 * x for x in bounds],
                )
                endpoint_inclusion(
                    record.v_end, ((Fraction(1), v), (t, start)), [t * t / 2 * x for x in bounds]
                )
                _require(
                    type(record.picard_iterations) is int
                    and 1 <= record.picard_iterations <= checker.config.picard_iterations,
                    "iteration metadata",
                )
                q, v = record.q_end, record.v_end
                if part == substeps - 1:
                    queue = [*queue[1:], record.sent]
                checked += 1
    return {
        "checked_intervals": checked,
        "prefix_contained": result.prefix_contained,
        "terminal_verified": False,
        "physical_binding_verified": False,
    }
