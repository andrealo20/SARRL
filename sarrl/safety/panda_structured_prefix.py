"""Finite held-input prefixes under explicit state-region and error assumptions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sarrl.robots.panda import vector7
from sarrl.safety.panda_backup import readonly
from sarrl.safety.panda_state_region import EnclosureUnavailable, VectorBox, padded_box


@dataclass(frozen=True)
class PandaPrefixErrors:
    position_measurement: np.ndarray
    velocity_measurement: np.ndarray
    position_defect: np.ndarray
    velocity_defect: np.ndarray
    command_error: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = vector7(getattr(self, name), name)
            if np.any(value < 0):
                raise ValueError("prefix error assumptions must be nonnegative")
            object.__setattr__(self, name, readonly(value))


@dataclass(frozen=True)
class PandaPrefixConfig:
    control_dt: float = 0.01
    max_delay_steps: int = 4
    picard_iterations: int = 20
    inflation_fraction: float = 0.1
    minimum_inflation: float = 1e-9

    def validate(self):
        values = np.asarray([self.control_dt, self.inflation_fraction, self.minimum_inflation])
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("prefix period and inflation settings must be finite and positive")
        if type(self.max_delay_steps) is not int or not 0 <= self.max_delay_steps <= 4:
            raise ValueError("prefix delay bound must be an integer from zero through four")
        if type(self.picard_iterations) is not int or not 1 <= self.picard_iterations <= 100:
            raise ValueError("Picard iteration budget must be an integer from one through 100")


@dataclass(frozen=True)
class PandaPrefixInterval:
    sent: VectorBox
    delivered: VectorBox
    q_tube: VectorBox
    v_tube: VectorBox
    acceleration: VectorBox
    q_end: VectorBox
    v_end: VectorBox
    picard_iterations: int


@dataclass(frozen=True)
class PandaPrefixCase:
    lag: int
    prefix_contained: bool
    reason: str
    intervals: tuple[PandaPrefixInterval, ...]


@dataclass(frozen=True)
class PandaPrefixResult:
    cases: tuple[PandaPrefixCase, ...]

    @property
    def prefix_contained(self):
        return all(case.prefix_contained for case in self.cases)

    @property
    def terminal_verified(self):
        return False


def scaled(box, factor):
    return VectorBox(factor * box.lower, factor * box.upper)


def summed(*boxes):
    return VectorBox(sum(box.lower for box in boxes), sum(box.upper for box in boxes))


def time_span(box, maximum):
    return VectorBox(np.minimum(0, maximum * box.lower), np.maximum(0, maximum * box.upper))


class PandaStructuredPrefix:
    """Conditional finite enclosure, without a terminal witness or task acceptance.

    Region derivatives/remainders and sensor/propagation assumptions are
    external contracts. They are not validated against Panda here. Numerical
    padding does not provide machine-checked interval arithmetic.
    """

    def __init__(
        self, region, parameters, lower, upper, velocity_limit, torque_limit, errors, config=None
    ):
        self.config = config or PandaPrefixConfig()
        self.config.validate()
        self.region, self.parameters, self.errors = region, parameters, errors
        self.positions = VectorBox(lower, upper)
        velocity, torque = (
            vector7(velocity_limit, "velocity_limit"),
            vector7(torque_limit, "torque_limit"),
        )
        if np.any(velocity <= 0) or np.any(torque <= 0):
            raise ValueError("prefix velocity and torque limits must be positive")
        self.velocities, self.torques = VectorBox(-velocity, velocity), VectorBox(-torque, torque)

    def _interval(self, q0, v0, sent, delivered):
        cfg, errors = self.config, self.errors
        q_tube = q0.inflate(np.full(7, cfg.minimum_inflation))
        v_tube = v0.inflate(np.full(7, cfg.minimum_inflation))
        for iteration in range(1, cfg.picard_iterations + 1):
            if not self.positions.contains(q_tube) or not self.velocities.contains(v_tube):
                raise EnclosureUnavailable("whole_interval_bounds_not_established")
            acceleration = self.region.acceleration(self.parameters, q_tube, v_tube, delivered)
            q_image = summed(
                q0, time_span(v0, cfg.control_dt), time_span(acceleration, 0.5 * cfg.control_dt**2)
            )
            v_image = summed(v0, time_span(acceleration, cfg.control_dt))
            q_image, v_image = (
                q_image.inflate(errors.position_defect),
                v_image.inflate(errors.velocity_defect),
            )
            if q_tube.contains(q_image, strict=True) and v_tube.contains(v_image, strict=True):
                q_end = summed(
                    q0, scaled(v0, cfg.control_dt), scaled(acceleration, 0.5 * cfg.control_dt**2)
                )
                v_end = summed(v0, scaled(acceleration, cfg.control_dt))
                q_end = padded_box(
                    q_end.lower - errors.position_defect, q_end.upper + errors.position_defect
                )
                v_end = padded_box(
                    v_end.lower - errors.velocity_defect, v_end.upper + errors.velocity_defect
                )
                # Endpoint roundoff padding must also stay inside the admitted tube.
                if not q_tube.contains(q_end) or not v_tube.contains(v_end):
                    raise EnclosureUnavailable("endpoint_padding_not_contained")
                return PandaPrefixInterval(
                    sent, delivered, q_tube, v_tube, acceleration, q_end, v_end, iteration
                )
            q_tube = self._expanded_trial(q_tube, q_image, self.positions, self.region.q_domain)
            v_tube = self._expanded_trial(v_tube, v_image, self.velocities, self.region.v_domain)
        raise EnclosureUnavailable("picard_budget_exhausted")

    def _expanded_trial(self, previous, image, physical, domain):
        # Reject an escaping image. Only surplus search inflation can be trimmed.
        if not physical.contains(image):
            raise EnclosureUnavailable("whole_interval_bounds_not_established")
        if not domain.contains(image):
            raise EnclosureUnavailable("state_region_not_covered")
        low = np.minimum(previous.lower, image.lower)
        high = np.maximum(previous.upper, image.upper)
        width = self.config.minimum_inflation + self.config.inflation_fraction * (high - low)
        return VectorBox(
            np.maximum(np.maximum(low - width, physical.lower), domain.lower),
            np.minimum(np.minimum(high + width, physical.upper), domain.upper),
        )

    def verify_prefix(self, context, measured_q, measured_v, candidate, history, *, intervals):
        if type(intervals) is not int or not 1 <= intervals <= 200:
            raise ValueError("prefix interval count must be an integer from one through 200")
        candidate = VectorBox.point(candidate)
        history = [VectorBox.point(command) for command in history]
        if any(not self.torques.contains(command) for command in [candidate, *history]):
            raise ValueError("candidate and sent history must satisfy torque limits")
        q0 = VectorBox.point(measured_q).inflate(self.errors.position_measurement)
        v0 = VectorBox.point(measured_v).inflate(self.errors.velocity_measurement)
        cases = []
        for lag in range(self.config.max_delay_steps + 1):
            maximum = self.config.max_delay_steps
            queue = [
                VectorBox.point(np.zeros(7)) for _ in range(maximum - min(maximum, len(history)))
            ]
            queue += history[-maximum:] if maximum else []
            q, v, records = q0, v0, []
            reason = "conditional_prefix_contained_no_terminal"
            try:
                for step in range(intervals):
                    if step == 0:
                        sent = candidate
                    else:
                        sent = self.region.torque(
                            context,
                            q.inflate(self.errors.position_measurement),
                            v.inflate(self.errors.velocity_measurement),
                            self.errors.command_error,
                        )
                    if not self.torques.contains(sent):
                        raise EnclosureUnavailable("raw_feedback_torque_bounds_not_established")
                    delivered = queue[-lag] if lag else sent
                    record = self._interval(q, v, sent, delivered)
                    records.append(record)
                    q, v = record.q_end, record.v_end
                    if maximum:
                        queue = [*queue[1:], sent]
            except EnclosureUnavailable as failure:
                reason = str(failure)
            cases.append(PandaPrefixCase(lag, len(records) == intervals, reason, tuple(records)))
        return PandaPrefixResult(tuple(cases))
