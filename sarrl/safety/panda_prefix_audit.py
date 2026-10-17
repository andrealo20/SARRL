"""Replay causal queues and rational propagation inequalities without the Picard search."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from sarrl.safety.panda_center_arithmetic import _outward
from sarrl.safety.panda_state_region import VectorBox


def _fraction(value):
    return Fraction(float(value))


def _scale(low, high, a, b):
    products = (low * a, low * b, high * a, high * b)
    return min(products), max(products)


def _initial(value, error):
    return VectorBox(
        [
            _outward(_fraction(x) - _fraction(e), upper=False)
            for x, e in zip(value, error, strict=True)
        ],
        [
            _outward(_fraction(x) + _fraction(e), upper=True)
            for x, e in zip(value, error, strict=True)
        ],
    )


def _expanded(domain, width):
    return VectorBox(
        [
            _outward(_fraction(x) - _fraction(e), upper=False)
            for x, e in zip(domain.lower, width, strict=True)
        ],
        [
            _outward(_fraction(x) + _fraction(e), upper=True)
            for x, e in zip(domain.upper, width, strict=True)
        ],
    )


def _require(condition, message):
    if not condition:
        raise ValueError(f"prefix audit: {message}")


def _same(left, right):
    return np.array_equal(left.lower, right.lower) and np.array_equal(left.upper, right.upper)


def _propagation(q0, v0, acceleration, record, dt, errors):
    """Check propagation inequalities in exact rationals, without Arb image helpers."""
    t = _fraction(dt)
    for i in range(7):
        qlo, qhi = _fraction(q0.lower[i]), _fraction(q0.upper[i])
        vlo, vhi = _fraction(v0.lower[i]), _fraction(v0.upper[i])
        alo, ahi = _fraction(acceleration.lower[i]), _fraction(acceleration.upper[i])
        eq, ev = _fraction(errors.position_defect[i]), _fraction(errors.velocity_defect[i])
        for partial, qbox, vbox in (
            (True, record.q_tube, record.v_tube),
            (False, record.q_end, record.v_end),
        ):
            tl, th = (Fraction(0), t) if partial else (t, t)
            sl, sh = (Fraction(0), t * t / 2) if partial else (t * t / 2, t * t / 2)
            drift = _scale(vlo, vhi, tl, th)
            accel_q, accel_v = _scale(alo, ahi, sl, sh), _scale(alo, ahi, tl, th)
            qimage = (qlo + drift[0] + accel_q[0] - eq, qhi + drift[1] + accel_q[1] + eq)
            vimage = (vlo + accel_v[0] - ev, vhi + accel_v[1] + ev)
            for outer, image in ((qbox, qimage), (vbox, vimage)):
                low, high = _fraction(outer.lower[i]), _fraction(outer.upper[i])
                valid = (
                    low < image[0] and high > image[1]
                    if partial
                    else (low <= image[0] and high >= image[1])
                )
                _require(valid, "propagation image is not contained")


def audit_prefix(
    checker, context, measured_q, measured_v, candidate, history, result, *, intervals
):
    """Check retained intervals, not a terminal witness or physical model binding.

    Queue and propagation checks use separate code and exact rationals. Model
    and torque bounds are recomputed by the same trusted dynamics provider;
    this is not an independent implementation of the rigid-body equations.
    Failed suffixes remain unverified. No call to verify_prefix is made.
    """
    maximum, errors = checker.config.max_delay_steps, checker.errors
    _require(type(intervals) is int and 1 <= intervals <= 200, "invalid interval count")
    _require([case.lag for case in result.cases] == list(range(maximum + 1)), "missing delay case")
    candidate, history = VectorBox.point(candidate), [VectorBox.point(x) for x in history]
    _require(all(checker.torques.contains(x) for x in [candidate, *history]), "input torque limit")
    checked = 0
    for case in result.cases:
        _require(len(case.intervals) <= intervals, "too many intervals")
        _require(case.prefix_contained == (len(case.intervals) == intervals), "completion flag")
        queue = [VectorBox.point(np.zeros(7)) for _ in range(maximum - min(maximum, len(history)))]
        queue += history[-maximum:] if maximum else []
        q, v = (
            _initial(measured_q, errors.position_measurement),
            _initial(measured_v, errors.velocity_measurement),
        )
        for step, record in enumerate(case.intervals):
            if step == 0:
                _require(_same(record.sent, candidate), "first send differs from known candidate")
            else:
                sent = checker.region.torque(
                    context,
                    _expanded(q, errors.position_measurement),
                    _expanded(v, errors.velocity_measurement),
                    errors.command_error,
                )
                _require(record.sent.contains(sent), "feedback enclosure missing")
            delivered = queue[-case.lag] if case.lag else record.sent
            _require(_same(record.delivered, delivered), "causal FIFO mismatch")
            _require(checker.torques.contains(record.sent), "raw torque limit")
            for physical, domain, tube, end in (
                (checker.positions, checker.region.q_domain, record.q_tube, record.q_end),
                (checker.velocities, checker.region.v_domain, record.v_tube, record.v_end),
            ):
                _require(physical.contains(tube) and domain.contains(tube), "tube domain")
                _require(tube.contains(end), "endpoint outside tube")
            acceleration = checker.region.acceleration(
                checker.parameters, record.q_tube, record.v_tube, delivered
            )
            _require(record.acceleration.contains(acceleration), "acceleration enclosure missing")
            _propagation(q, v, record.acceleration, record, checker.config.control_dt, errors)
            _require(
                type(record.picard_iterations) is int
                and 1 <= record.picard_iterations <= checker.config.picard_iterations,
                "iteration metadata",
            )
            q, v = record.q_end, record.v_end
            if maximum:
                queue = [*queue[1:], record.sent]
            checked += 1
    _require(not result.terminal_verified, "finite prefix cannot certify terminal invariance")
    return {
        "checked_intervals": checked,
        "prefix_contained": result.prefix_contained,
        "terminal_verified": False,
        "physical_binding_verified": False,
    }
