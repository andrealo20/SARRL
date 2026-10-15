"""Queue-aware backup witnesses for a frozen estimated Panda acceleration model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from sarrl.robots.panda import vector7
from sarrl.safety.panda_joint_filter import PandaJointFilter, PandaJointFilterConfig


@dataclass(frozen=True)
class PandaBackupConfig:
    control_dt: float = 0.01
    max_delay_steps: int = 4
    horizon_steps: int = 200
    kp: float = 25.0
    kd: float = 10.0

    def validate(self):
        values = np.asarray([self.control_dt, self.kp, self.kd])
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("backup period and gains must be finite and positive")
        if type(self.max_delay_steps) is not int or not 0 <= self.max_delay_steps <= 4:
            raise ValueError("backup delay bound must be an integer from zero through four")
        if type(self.horizon_steps) is not int or self.horizon_steps < 1:
            raise ValueError("backup horizon must be a positive integer")


def backup_transition(config, lag):
    """Scalar-joint map for position, velocity and oldest-to-newest sent accelerations."""
    config.validate()
    if type(lag) is not int or not 0 <= lag <= config.max_delay_steps:
        raise ValueError("backup lag must belong to the configured constant-delay set")
    size, dt = 2 + config.max_delay_steps, config.control_dt
    acceleration = np.zeros(size)
    if lag:
        acceleration[size - lag] = 1
    else:
        acceleration[:2] = [-config.kp, -config.kd]
    transition = np.zeros((size, size))
    transition[0, :2], transition[1, 1] = [1, dt], 1
    transition[0] += 0.5 * dt**2 * acceleration
    transition[1] += dt * acceleration
    if config.max_delay_steps:
        transition[2:-1, 3:] = np.eye(config.max_delay_steps - 1)
        transition[-1, :2] = [-config.kp, -config.kd]
    return transition, acceleration


@lru_cache(maxsize=16)
def terminal_metrics(config):
    """Check stable maps and Lyapunov identities independently of solver status."""
    config.validate()
    weight = np.diag([config.kp, 1.0] + [config.control_dt**2] * config.max_delay_steps)
    metrics = []
    for lag in range(config.max_delay_steps + 1):
        transition, acceleration = backup_transition(config, lag)
        if np.max(np.abs(np.linalg.eigvals(transition))) >= 1 - 1e-12:
            raise ValueError("backup feedback is not stable for every constant delay")
        metric = solve_discrete_lyapunov(transition.T, weight)
        residual = transition.T @ metric @ transition - metric + weight
        if (
            not np.all(np.isfinite(metric))
            or not np.allclose(metric, metric.T, rtol=1e-10, atol=1e-10)
            or np.min(np.linalg.eigvalsh(metric)) <= 0
            or np.linalg.norm(residual) > 1e-9 * max(1.0, np.linalg.norm(metric))
        ):
            raise ValueError("backup terminal metric failed its numerical identity checks")
        inverse = np.linalg.solve(metric, np.eye(len(metric)))
        for array in (transition, acceleration, metric, inverse):
            array.setflags(write=False)
        metrics.append((transition, acceleration, metric, inverse))
    return tuple(metrics)


@lru_cache(maxsize=16)
def backup_powers(config):
    result = []
    for transition, _, _, _ in terminal_metrics(config):
        powers = [np.eye(len(transition))]
        for _ in range(config.horizon_steps - 1):
            powers.append(transition @ powers[-1])
        array = np.asarray(powers)
        array.setflags(write=False)
        result.append(array)
    return tuple(result)


def readonly(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PandaBackupPolicy:
    anchor: np.ndarray
    matrix: np.ndarray
    drift: np.ndarray
    mass: np.ndarray
    equilibrium_torque: np.ndarray
    torque_limit: np.ndarray
    config: PandaBackupConfig

    def requested_torque(self, q, v):
        displacement = vector7(q, "q") - self.anchor
        acceleration = -self.config.kp * displacement - self.config.kd * vector7(v, "v")
        return self.mass @ acceleration + self.equilibrium_torque

    def bounded_torque(self, q, v):
        """Physical clipping does not extend the unsaturated-model witness."""
        return np.clip(self.requested_torque(q, v), -self.torque_limit, self.torque_limit)


@dataclass(frozen=True)
class PandaBackupLagResult:
    lag: int
    accepted: bool
    reason: str
    checked_intervals: int
    terminal_energy: float | None = None
    terminal_radius_squared: float | None = None


@dataclass(frozen=True)
class PandaBackupResult:
    accepted: bool
    policy: PandaBackupPolicy
    cases: tuple[PandaBackupLagResult, ...]


class PandaBackupWitness:
    """One candidate interval followed by a common observable-state backup law."""

    def __init__(self, lower, upper, torque_limit, config=None):
        self.config = config or PandaBackupConfig()
        self.config.validate()
        self.metrics = terminal_metrics(self.config)
        self.powers = backup_powers(self.config)
        self.bounds = PandaJointFilter(
            lower,
            upper,
            torque_limit,
            PandaJointFilterConfig(
                control_dt=self.config.control_dt, max_delay_steps=self.config.max_delay_steps
            ),
        )

    def policy(self, anchor, matrix, drift):
        matrix = np.array(matrix, dtype=float, copy=True)
        if (
            matrix.shape != (7, 7)
            or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12)
        ):
            raise ValueError(
                "backup acceleration matrix must be finite and symmetric, shape (7, 7)"
            )
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError("backup acceleration matrix must be positive definite") from error
        mass = np.linalg.solve(matrix, np.eye(7))
        drift = vector7(drift, "drift")
        equilibrium = -mass @ drift
        if not np.all(np.isfinite(mass)) or not np.all(np.isfinite(equilibrium)):
            raise ValueError("backup inverse mass and equilibrium torque must be finite")
        return PandaBackupPolicy(
            readonly(vector7(anchor, "anchor")),
            readonly(matrix),
            readonly(drift),
            readonly(mass),
            readonly(equilibrium),
            readonly(self.bounds.limit),
            self.config,
        )

    def terminal_radius_squared(self, policy, lag):
        """Inscribed ellipsoid bounds whole intervals and all raw feedback/queue torques."""
        _, acceleration, _, inverse = self.metrics[lag]
        dt = self.config.control_dt
        position_slack = np.minimum(
            policy.anchor - self.bounds.lower, self.bounds.upper - policy.anchor
        )
        torque_slack = self.bounds.limit - np.abs(policy.equilibrium_torque)
        if np.any(position_slack <= 0) or np.any(torque_slack <= 0):
            return 0.0
        position_support = (
            np.sqrt(inverse[0, 0])
            + dt * np.sqrt(inverse[1, 1])
            + 0.5 * dt**2 * np.sqrt(acceleration @ inverse @ acceleration)
        )
        velocity_support = np.sqrt(inverse[1, 1]) + dt * np.sqrt(
            acceleration @ inverse @ acceleration
        )
        feedback = np.zeros(len(inverse))
        feedback[:2] = [-self.config.kp, -self.config.kd]
        torque_support = np.linalg.norm(policy.mass, axis=1) * np.sqrt(
            feedback @ inverse @ feedback
        )
        radii = [
            *list(position_slack / position_support),
            *list(self.bounds.velocity / velocity_support),
            *list(torque_slack / torque_support),
        ]
        for index in range(2, len(inverse)):
            queue_support = np.linalg.norm(policy.mass, axis=1) * np.sqrt(inverse[index, index])
            radii.extend(torque_slack / queue_support)
        return float(min(radii) ** 2)

    def _case(self, q, v, candidate, history, policy, lag):
        cfg, dt = self.config, self.config.control_dt
        queue = [np.zeros(7) for _ in range(max(0, cfg.max_delay_steps - len(history)))]
        queue += [u.copy() for u in history[-cfg.max_delay_steps :]] if cfg.max_delay_steps else []
        radius = self.terminal_radius_squared(policy, lag)
        if radius <= 0:
            return PandaBackupLagResult(lag, False, "empty_terminal_set", 0)
        metric = self.metrics[lag][2]
        initial = np.array(
            [q - policy.anchor, v] + [policy.matrix @ u + policy.drift for u in queue]
        )
        sent_acceleration = policy.matrix @ candidate + policy.drift
        first_acceleration = initial[-lag] if lag else sent_acceleration
        first = np.array(
            [initial[0] + dt * v + 0.5 * dt**2 * first_acceleration, v + dt * first_acceleration]
            + ([*initial[3:], sent_acceleration] if cfg.max_delay_steps else [])
        )
        after = self.powers[lag] @ first
        before = np.concatenate((initial[None], after[:-1]))
        requested_acceleration = -cfg.kp * before[:, 0] - cfg.kd * before[:, 1]
        torques = requested_acceleration @ policy.mass.T + policy.equilibrium_torque
        torques[0] = candidate
        accelerations = before[:, -lag] if lag else requested_acceleration.copy()
        if not lag:
            accelerations[0] = sent_acceleration
        starts, ends = before[:, 0] + policy.anchor, after[:, 0] + policy.anchor
        smallest, largest = np.minimum(starts, ends), np.maximum(starts, ends)
        stationary = np.divide(
            -before[:, 1],
            accelerations,
            out=np.zeros_like(accelerations),
            where=np.abs(accelerations) > 1e-14,
        )
        interior = (np.abs(accelerations) > 1e-14) & (stationary > 0) & (stationary < dt)
        extrema = starts + before[:, 1] * stationary + 0.5 * accelerations * stationary**2
        smallest = np.where(interior, np.minimum(smallest, extrema), smallest)
        largest = np.where(interior, np.maximum(largest, extrema), largest)
        tol = self.bounds.config.feasibility_tol
        valid_intervals = (
            np.all(smallest >= self.bounds.lower - tol, axis=1)
            & np.all(largest <= self.bounds.upper + tol, axis=1)
            & np.all(
                np.maximum(np.abs(before[:, 1]), np.abs(after[:, 1])) <= self.bounds.velocity + tol,
                axis=1,
            )
            & np.all(np.isfinite(after), axis=(1, 2))
        )
        valid_torques = np.all(
            np.isfinite(torques) & (np.abs(torques) <= self.bounds.limit), axis=1
        )
        energies = np.sum(after * (metric @ after), axis=(1, 2))
        failures = np.flatnonzero(~valid_intervals | ~valid_torques)
        terminals = np.flatnonzero(np.isfinite(energies) & (energies <= radius))
        if len(terminals) and (not len(failures) or terminals[0] < failures[0]):
            index = int(terminals[0])
            return PandaBackupLagResult(
                lag, True, "terminal_set_reached", index + 1, float(energies[index]), radius
            )
        if len(failures):
            step = int(failures[0])
            if not valid_torques[step]:
                return PandaBackupLagResult(lag, False, "backup_torque_limit", step)
            reason = (
                "pending_command_violation"
                if step < lag
                else "candidate_interval_violation"
                if step == lag
                else "backup_interval_violation"
            )
            return PandaBackupLagResult(lag, False, reason, step + 1)
        return PandaBackupLagResult(
            lag, False, "terminal_not_reached", cfg.horizon_steps, float(energies[-1]), radius
        )

    def verify(self, q, v, candidate, matrix, drift, history):
        q, v, candidate = (
            vector7(value, name) for value, name in ((q, "q"), (v, "v"), (candidate, "candidate"))
        )
        history = [vector7(u, "history") for u in history]
        if any(np.any(np.abs(u) > self.bounds.limit) for u in [candidate, *history]):
            raise ValueError("candidate and history must contain bounded sent torques")
        policy = self.policy(q, matrix, drift)
        cases = tuple(
            self._case(q.copy(), v.copy(), candidate, history, policy, lag)
            for lag in range(self.config.max_delay_steps + 1)
        )
        return PandaBackupResult(all(case.accepted for case in cases), policy, cases)
