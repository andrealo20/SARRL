"""Conditional error tubes for the frozen backup model, without plant certification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigvalsh

from sarrl.robots.panda import vector7
from sarrl.safety.panda_backup import PandaBackupWitness, readonly


@dataclass(frozen=True)
class PandaBackupEnvelope:
    """Assumed physical domain and componentwise errors; no bound is inferred here.

    Acceleration error must hold throughout each interval for every bounded
    delivered torque and admissible plant parameter. Measurement errors apply
    at every backup send. Propagation defects bound additional q/v errors
    throughout every partial interval. A label is not validation evidence.
    """

    position_lower: np.ndarray
    position_upper: np.ndarray
    velocity_limit: np.ndarray
    acceleration_error: np.ndarray
    position_error: np.ndarray
    velocity_error: np.ndarray
    position_defect: np.ndarray
    velocity_defect: np.ndarray
    assumption: str

    def __post_init__(self):
        for name in (
            "position_lower",
            "position_upper",
            "velocity_limit",
            "acceleration_error",
            "position_error",
            "velocity_error",
            "position_defect",
            "velocity_defect",
        ):
            object.__setattr__(self, name, readonly(vector7(getattr(self, name), name)))
        if np.any(self.position_lower >= self.position_upper):
            raise ValueError("model domain must have nonempty position intervals")
        if np.any(self.velocity_limit <= 0):
            raise ValueError("model domain must have positive velocity limits")
        if any(
            np.any(getattr(self, name) < 0)
            for name in (
                "acceleration_error",
                "position_error",
                "velocity_error",
                "position_defect",
                "velocity_defect",
            )
        ):
            raise ValueError("error bounds must be nonnegative")
        if not isinstance(self.assumption, str) or not self.assumption.strip():
            raise ValueError("an explicit envelope assumption is required")


@dataclass(frozen=True)
class PandaBackupTubeCase:
    lag: int
    contraction: float
    disturbance_radius: float
    minimum_radius: float
    admissible_radius: float
    feasible: bool
    reason: str

    def backup_radii(self, initial_radius, intervals):
        """Backup intervals only; the candidate send has a different error map."""
        if not np.isfinite(initial_radius) or initial_radius < 0:
            raise ValueError("initial error radius must be finite and nonnegative")
        if type(intervals) is not int or intervals < 0:
            raise ValueError("interval count must be a nonnegative integer")
        radii = [float(initial_radius)]
        for _ in range(intervals):
            radii.append(self.contraction * radii[-1] + self.disturbance_radius)
            if not np.isfinite(radii[-1]):
                raise ValueError("error radius overflow")
        return np.asarray(radii)


class PandaBackupTube:
    """Assess terminal feasibility under supplied bounds, never accept a task command.

    A feasible case is conditional on its envelope. This class does not
    establish that the envelope describes MuJoCo, noise or a physical robot.
    It neither changes the supervisor nor performs a finite-prefix witness.
    """

    def __init__(self, witness: PandaBackupWitness, envelope: PandaBackupEnvelope):
        self.witness, self.envelope = witness, envelope
        self.config, self.metrics = witness.config, witness.metrics
        self.measurement_acceleration = readonly(
            self.config.kp * envelope.position_error + self.config.kd * envelope.velocity_error
        )

    def _metric(self, lag):
        if type(lag) is not int or not 0 <= lag <= self.config.max_delay_steps:
            raise ValueError("lag must belong to the configured constant-delay set")
        return self.metrics[lag]

    def _disturbance(self, lag):
        transition, _, metric, _ = self._metric(lag)
        dt, width = self.config.control_dt, len(metric)
        # Arbitrarily varying physical error has separate integral bounds for q and v.
        physical = np.sqrt(metric[0, 0]) * np.linalg.norm(
            0.5 * dt**2 * self.envelope.acceleration_error + self.envelope.position_defect
        )
        physical += np.sqrt(metric[1, 1]) * np.linalg.norm(
            dt * self.envelope.acceleration_error + self.envelope.velocity_defect
        )
        sensor_map = np.zeros(width)
        if lag == 0:
            sensor_map[:2] = [0.5 * dt**2, dt]
        if self.config.max_delay_steps:
            sensor_map[-1] = 1
        sensor = np.sqrt(sensor_map @ metric @ sensor_map)
        sensor *= np.linalg.norm(self.measurement_acceleration)
        # A small conservative numerical guard is not an interval-arithmetic proof.
        alpha = float(np.sqrt(max(0.0, eigvalsh(transition.T @ metric @ transition, metric)[-1])))
        return alpha + 1e-12, float((physical + sensor) * (1 + 1e-12))

    def candidate_radius(self, lag):
        """First send: same fixed candidate and exact recorded FIFO in both paths.

        Initial q/v offsets are bounded by the measurement envelope. No new
        sensor feedback term enters this interval, including at zero delay.
        """
        metric = self._metric(lag)[2]
        envelope, dt = self.envelope, self.config.control_dt
        q_width = envelope.position_error + dt * envelope.velocity_error
        q_width += 0.5 * dt**2 * envelope.acceleration_error + envelope.position_defect
        v_width = envelope.velocity_error + dt * envelope.acceleration_error
        v_width += envelope.velocity_defect
        squared = (
            metric[0, 0] * (q_width @ q_width)
            + metric[1, 1] * (v_width @ v_width)
            + 2 * abs(metric[0, 1]) * (q_width @ v_width)
        )
        if not np.isfinite(squared):
            raise ValueError("candidate error radius overflow")
        return float(np.sqrt(squared) * (1 + 1e-12))

    def _admissible_radius(self, policy, lag):
        _, acceleration, _, inverse = self._metric(lag)
        bounds, envelope, dt = self.witness.bounds, self.envelope, self.config.control_dt
        lower = np.maximum(bounds.lower, envelope.position_lower)
        upper = np.minimum(bounds.upper, envelope.position_upper)
        position_slack = np.minimum(policy.anchor - lower, upper - policy.anchor)
        velocity_slack = np.minimum(bounds.velocity, envelope.velocity_limit)
        interval_error = envelope.acceleration_error.copy()
        if lag == 0:
            interval_error += self.measurement_acceleration
        position_slack -= 0.5 * dt**2 * interval_error + envelope.position_defect
        velocity_slack -= dt * interval_error + envelope.velocity_defect
        torque_slack = bounds.limit - np.abs(policy.equilibrium_torque)
        feedback_slack = torque_slack - np.abs(policy.mass) @ self.measurement_acceleration
        if any(
            np.any(value <= 0)
            for value in (position_slack, velocity_slack, torque_slack, feedback_slack)
        ):
            return 0.0
        acceleration_support = np.sqrt(acceleration @ inverse @ acceleration)
        position_support = (
            np.sqrt(inverse[0, 0])
            + dt * np.sqrt(inverse[1, 1])
            + 0.5 * dt**2 * acceleration_support
        )
        velocity_support = np.sqrt(inverse[1, 1]) + dt * acceleration_support
        feedback = np.zeros(len(inverse))
        feedback[:2] = [-self.config.kp, -self.config.kd]
        mass_rows = np.linalg.norm(policy.mass, axis=1)
        feedback_support = mass_rows * np.sqrt(feedback @ inverse @ feedback)
        radii = [
            *list(position_slack / position_support),
            *list(velocity_slack / velocity_support),
            *list(feedback_slack / feedback_support),
        ]
        for index in range(2, len(inverse)):
            radii.extend(torque_slack / (mass_rows * np.sqrt(inverse[index, index])))
        return float(min(radii) * (1 - 1e-12))

    def assess(self, anchor, matrix, drift):
        """Return per-lag conditional radii; no plant bound or command is certified."""
        policy = self.witness.policy(anchor, matrix, drift)
        results = []
        for lag in range(self.config.max_delay_steps + 1):
            alpha, beta = self._disturbance(lag)
            radius = self._admissible_radius(policy, lag)
            if not np.all(np.isfinite([alpha, beta, radius])) or alpha >= 1:
                raise ValueError("terminal contraction or disturbance is invalid")
            minimum = beta / (1 - alpha)
            if not np.isfinite(minimum):
                raise ValueError("minimum terminal radius overflow")
            feasible = radius > 0 and minimum <= radius and alpha * radius + beta <= radius
            reason = (
                "conditional_terminal_feasible"
                if feasible
                else "empty_admissible_set"
                if radius <= 0
                else "disturbance_exceeds_terminal_capacity"
            )
            results.append(PandaBackupTubeCase(lag, alpha, beta, minimum, radius, feasible, reason))
        return tuple(results)
