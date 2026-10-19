"""Shared-parameter endpoints and validated second-order held-flow remainders."""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from sarrl.safety.panda_affine_state import AffineState, linear_combination
from sarrl.safety.panda_center_arithmetic import ArrayEnclosure, _export, _outward
from sarrl.safety.panda_state_region import EnclosureUnavailable, VectorBox
from sarrl.safety.panda_validated_dynamics import arithmetic, exact, interval, point
from sarrl.safety.panda_validated_prefix import PandaValidatedPrefix


def physical_coordinates(parameters):
    centers, radii = [], []
    for lo, hi in zip(parameters.lower, parameters.upper, strict=True):
        lo, hi = Fraction(float(lo)), Fraction(float(hi))
        center = float((lo + hi) / 2)
        centers.append(center)
        radii.append(_outward(max(Fraction(center) - lo, hi - Fraction(center)), upper=True))
    return AffineState(np.array(centers), np.diag(radii), ArrayEnclosure(np.zeros(8), np.zeros(8)))


def as_box(form, backend):
    result = form.hull(backend)
    return VectorBox(result.lower, result.upper)


def add_error(form, widths, backend):
    error = np.array([interval(-w, w, backend) for w in widths], dtype=object)
    return linear_combination(((1, form),), backend, extra=error)


def taylor_endpoint(q, v, acceleration, jerk, period, backend):
    """a(0) retains shared coordinates; integral remainders cover every ODE trajectory."""
    t = Fraction(float(period))
    qnext = linear_combination(
        ((1, q), (t, v), (t * t / 2, acceleration)),
        backend,
        extra=exact(t * t * t / 6, backend) * jerk * interval(-1, 1, backend),
    )
    vnext = linear_combination(
        ((1, v), (t, acceleration)),
        backend,
        extra=exact(t * t / 2, backend) * jerk * interval(-1, 1, backend),
    )
    return qnext, vnext


@dataclass(frozen=True)
class MeanValueInterval:
    q_start: AffineState
    v_start: AffineState
    sent: AffineState
    delivered: AffineState
    q_tube: VectorBox
    v_tube: VectorBox
    acceleration_tube: VectorBox
    acceleration_start: AffineState
    jerk_bound: ArrayEnclosure
    q_end: AffineState
    v_end: AffineState
    q_endpoint_bound: VectorBox
    v_endpoint_bound: VectorBox
    picard_iterations: int


@dataclass(frozen=True)
class MeanValueCase:
    lag: int
    prefix_contained: bool
    reason: str
    intervals: tuple[MeanValueInterval, ...]


@dataclass(frozen=True)
class MeanValueResult:
    parameters: AffineState
    cases: tuple[MeanValueCase, ...]
    terminal_verified: bool = False

    @property
    def prefix_contained(self):
        return bool(self.cases) and all(x.prefix_contained for x in self.cases)


class PandaMeanValuePrefix(PandaValidatedPrefix):
    """Offline ideal-ODE calculation, without unstructured physical/runtime defects."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if np.any(self.errors.position_defect) or np.any(self.errors.velocity_defect):
            raise ValueError("Taylor endpoints require zero partial-runtime defects")

    def verify_prefix(self, context, measured_q, measured_v, candidate, history, *, intervals):
        if type(intervals) is not int or not 1 <= intervals <= 200:
            raise ValueError("prefix interval count must be an integer from one through 200")
        if len(history) != 4 or self.config.max_delay_steps != 4:
            raise ValueError("the shared-coordinate screen requires four known prior sends")
        if any(not self.torques.contains(VectorBox.point(x)) for x in (candidate, *history)):
            raise ValueError("candidate and history must satisfy torque limits")
        parameters = physical_coordinates(self.parameters)
        cases = []
        with arithmetic(self.precision_bits) as backend:
            for lag in range(5):
                queue = [AffineState.constant(x) for x in history]
                q = add_error(
                    AffineState.constant(measured_q), self.errors.position_measurement, backend
                )
                v = add_error(
                    AffineState.constant(measured_v), self.errors.velocity_measurement, backend
                )
                records, reason = [], "normalized_model_prefix_contained_no_terminal"
                try:
                    for step in range(intervals):
                        sent = (
                            AffineState.constant(candidate)
                            if step == 0
                            else add_error(
                                self.region.feedback_affine(
                                    context,
                                    add_error(q, self.errors.position_measurement, backend),
                                    add_error(v, self.errors.velocity_measurement, backend),
                                    backend,
                                ),
                                self.errors.command_error,
                                backend,
                            )
                        )
                        if not self.torques.contains(as_box(sent, backend)):
                            raise EnclosureUnavailable("raw_feedback_torque_bounds_not_established")
                        delivered = queue[-lag] if lag else sent
                        tube = super()._interval(
                            as_box(q, backend),
                            as_box(v, backend),
                            as_box(sent, backend),
                            as_box(delivered, backend),
                        )
                        start = self.region.acceleration_affine(
                            parameters, q, v, delivered, backend
                        )
                        jerk = self.region.jerk(
                            self.parameters,
                            tube.q_tube,
                            tube.v_tube,
                            as_box(delivered, backend),
                            tube.acceleration,
                            backend,
                        )
                        magnitude = _export(abs(jerk))
                        jerk_bound = ArrayEnclosure(np.zeros(7), magnitude.upper)
                        qnext, vnext = taylor_endpoint(
                            q,
                            v,
                            start,
                            point(jerk_bound.upper, backend),
                            self.config.control_dt,
                            backend,
                        )
                        records.append(
                            MeanValueInterval(
                                q,
                                v,
                                sent,
                                delivered,
                                tube.q_tube,
                                tube.v_tube,
                                tube.acceleration,
                                start,
                                jerk_bound,
                                qnext,
                                vnext,
                                tube.q_end,
                                tube.v_end,
                                tube.picard_iterations,
                            )
                        )
                        q, v, queue = qnext, vnext, [*queue[1:], sent]
                except EnclosureUnavailable as failure:
                    reason = str(failure)
                cases.append(MeanValueCase(lag, len(records) == intervals, reason, tuple(records)))
        return MeanValueResult(parameters, tuple(cases))
