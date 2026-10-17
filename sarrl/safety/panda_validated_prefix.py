"""Offline, outward finite prefixes with explicit limits and persistent causal queues."""

from __future__ import annotations

import numpy as np

from sarrl.safety.panda_state_region import EnclosureUnavailable, VectorBox
from sarrl.safety.panda_structured_prefix import (
    PandaPrefixCase,
    PandaPrefixInterval,
    PandaPrefixResult,
    PandaStructuredPrefix,
)
from sarrl.safety.panda_validated_dynamics import (
    arithmetic,
    box,
    exact,
    inflate,
    interval,
    step_images,
)


class PandaValidatedPrefix(PandaStructuredPrefix):
    """Numerical containment for the declared model, never a production authorization.

    External physical, measurement and partial-runtime error assumptions remain
    required. The inherited terminal_verified property is always false.
    """

    def __init__(self, *args, precision_bits=128, **kwargs):
        super().__init__(*args, **kwargs)
        with arithmetic(precision_bits):
            pass
        self.precision_bits = precision_bits
        if getattr(self.region, "precision_bits", precision_bits) != precision_bits:
            raise ValueError("prefix and dynamics precision must match")

    def _inflate(self, domain, width):
        return inflate(domain, width, precision_bits=self.precision_bits)

    def _expanded_trial(self, previous, image, physical, domain):
        if not physical.contains(image):
            raise EnclosureUnavailable("whole_interval_bounds_not_established")
        if not domain.contains(image):
            raise EnclosureUnavailable("state_region_not_covered")
        low, high = np.minimum(previous.lower, image.lower), np.maximum(previous.upper, image.upper)
        with arithmetic(self.precision_bits) as backend:
            values = []
            for lo, hi in zip(low, high, strict=True):
                radius = exact(self.config.minimum_inflation, backend)
                radius += exact(self.config.inflation_fraction, backend) * (
                    exact(hi, backend) - exact(lo, backend)
                )
                values.append(interval(lo, hi, backend) + interval(-1, 1, backend) * radius)
            trial = box(np.array(values, dtype=object))
        # Only search surplus is trimmed, after checking the full reachable image.
        return VectorBox(
            np.maximum(np.maximum(trial.lower, physical.lower), domain.lower),
            np.minimum(np.minimum(trial.upper, physical.upper), domain.upper),
        )

    def _interval(self, q0, v0, sent, delivered):
        minimum = np.full(7, self.config.minimum_inflation)
        q_tube, v_tube = self._inflate(q0, minimum), self._inflate(v0, minimum)
        for iteration in range(1, self.config.picard_iterations + 1):
            if not self.positions.contains(q_tube) or not self.velocities.contains(v_tube):
                raise EnclosureUnavailable("whole_interval_bounds_not_established")
            acceleration = self.region.acceleration(self.parameters, q_tube, v_tube, delivered)
            qi, vi, qe, ve = step_images(
                q0,
                v0,
                acceleration,
                self.config.control_dt,
                self.errors,
                precision_bits=self.precision_bits,
            )
            if q_tube.contains(qi, strict=True) and v_tube.contains(vi, strict=True):
                if not q_tube.contains(qe) or not v_tube.contains(ve):
                    raise EnclosureUnavailable("endpoint_not_contained")
                return PandaPrefixInterval(
                    sent, delivered, q_tube, v_tube, acceleration, qe, ve, iteration
                )
            q_tube = self._expanded_trial(q_tube, qi, self.positions, self.region.q_domain)
            v_tube = self._expanded_trial(v_tube, vi, self.velocities, self.region.v_domain)
        raise EnclosureUnavailable("picard_budget_exhausted")

    def verify_prefix(self, context, measured_q, measured_v, candidate, history, *, intervals):
        if type(intervals) is not int or not 1 <= intervals <= 200:
            raise ValueError("prefix interval count must be an integer from one through 200")
        candidate, history = VectorBox.point(candidate), [VectorBox.point(x) for x in history]
        if any(not self.torques.contains(x) for x in [candidate, *history]):
            raise ValueError("candidate and history must satisfy torque limits")
        q0 = self._inflate(VectorBox.point(measured_q), self.errors.position_measurement)
        v0 = self._inflate(VectorBox.point(measured_v), self.errors.velocity_measurement)
        cases, maximum = [], self.config.max_delay_steps
        for lag in range(maximum + 1):
            queue = [
                VectorBox.point(np.zeros(7)) for _ in range(maximum - min(maximum, len(history)))
            ]
            queue += history[-maximum:] if maximum else []
            q, v, records = q0, v0, []
            reason = "normalized_model_prefix_contained_no_terminal"
            try:
                for step in range(intervals):
                    if step == 0:
                        sent = candidate  # Already known sent value, not a new feedback evaluation.
                    else:
                        sent = self.region.torque(
                            context,
                            self._inflate(q, self.errors.position_measurement),
                            self._inflate(v, self.errors.velocity_measurement),
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
