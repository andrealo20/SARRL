"""Estimated-model joint constraints across a bounded constant command delay."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import OptimizeWarning, linprog, minimize, nnls

from sarrl.robots.panda import VELOCITY_LIMIT, vector7


@dataclass(frozen=True)
class PandaJointFilterConfig:
    control_dt: float = 0.01
    max_delay_steps: int = 4
    position_reserve_rad: float = 0.03
    velocity_reserve_rad_s: float = 0.10
    gamma: float = 5.0
    feasibility_tol: float = 1e-8
    stationarity_tol: float = 1e-6
    solver_iterations: int = 100

    def validate(self):
        values = (
            self.control_dt,
            self.position_reserve_rad,
            self.velocity_reserve_rad_s,
            self.gamma,
            self.feasibility_tol,
            self.stationarity_tol,
        )
        if not np.all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError("filter settings must be finite and positive")
        if self.gamma * self.control_dt >= 1:
            raise ValueError("barrier gain times control period must be less than one")
        if type(self.max_delay_steps) is not int or not 0 <= self.max_delay_steps <= 4:
            raise ValueError("delay bound must be an integer from zero through four")
        if type(self.solver_iterations) is not int or self.solver_iterations < 1:
            raise ValueError("solver iterations must be a positive integer")
        if self.velocity_reserve_rad_s >= min(VELOCITY_LIMIT):
            raise ValueError("velocity reserve leaves no admissible interval")


@dataclass(frozen=True)
class PandaProjectionResult:
    torque: np.ndarray
    predicted_feasible: bool
    reason: str
    correction_norm_nm: float
    minimum_scaled_margin: float | None
    stationarity_residual: float | None
    active_constraints: tuple[int, ...] = ()

    def diagnostics(self):
        return {
            **asdict(self),
            "torque": self.torque.tolist(),
            "active_constraints": list(self.active_constraints),
        }


def project_torque(candidate, A, b, limit, config=None):
    """Weighted torque projection with independent primal and nonnegative KKT checks."""
    cfg = config or PandaJointFilterConfig()
    cfg.validate()
    candidate, limit = vector7(candidate, "candidate"), vector7(limit, "limit")
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    if (
        A.ndim != 2
        or A.shape[1] != 7
        or b.shape != (len(A),)
        or not np.all(np.isfinite(A))
        or not np.all(np.isfinite(b))
        or np.any(limit <= 0)
    ):
        raise ValueError("projection requires finite seven-dimensional constraints and limits")
    # Torque bounds are explicit rows, so their dual contributions are checked too.
    matrix = np.vstack((A * limit, np.eye(7), -np.eye(7)))
    rhs = np.concatenate((b, -np.ones(14)))
    norm = np.linalg.norm(matrix, axis=1)
    zero = norm <= 1e-14
    fallback = np.clip(candidate, -limit, limit)
    if np.any(rhs[zero] > cfg.feasibility_tol):
        return PandaProjectionResult(fallback, False, "infeasible", 0.0, None, None)
    matrix, rhs = matrix[~zero], rhs[~zero]
    norm = norm[~zero]
    matrix, rhs = matrix / norm[:, None], rhs / norm
    center = candidate / limit
    if np.all(np.abs(candidate) <= limit) and np.min(matrix @ center - rhs) >= -cfg.feasibility_tol:
        return PandaProjectionResult(
            candidate.copy(), True, "unchanged", 0.0, float(np.min(matrix @ center - rhs)), 0.0
        )
    result = minimize(
        lambda x: 0.5 * float((x - center) @ (x - center)),
        np.clip(center, -1, 1),
        jac=lambda x: x - center,
        method="SLSQP",
        constraints={"type": "ineq", "fun": lambda x: matrix @ x - rhs, "jac": lambda x: matrix},
        options={"ftol": 1e-12, "maxiter": cfg.solver_iterations},
    )
    x = np.clip(np.asarray(result.x), -1, 1)
    margin = matrix @ x - rhs
    active = np.flatnonzero(margin <= 1e-7)
    try:
        if len(active):
            multipliers, residual = nnls(matrix[active].T, x - center, maxiter=1000)
            complementarity = np.max(np.abs(multipliers * margin[active]))
        else:
            residual, complementarity = np.linalg.norm(x - center), 0.0
    except (RuntimeError, ValueError):
        residual, complementarity = np.inf, np.inf
    valid = (
        np.all(np.isfinite(x))
        and min(margin) >= -cfg.feasibility_tol
        and residual <= cfg.stationarity_tol
        and complementarity <= cfg.stationarity_tol
    )
    if valid:
        return PandaProjectionResult(
            x * limit,
            True,
            "projected",
            float(np.linalg.norm(x * limit - candidate)),
            float(min(margin)),
            float(residual),
            tuple(int(i) for i in active),
        )
    with warnings.catch_warnings():
        # SciPy forwards this native HiGHS option while reporting it as unrecognised.
        warnings.filterwarnings(
            "ignore", message="Unrecognized options detected:.*threads.*", category=OptimizeWarning
        )
        feasibility = linprog(
            np.zeros(7),
            A_ub=-matrix,
            b_ub=-rhs,
            bounds=[(None, None)] * 7,
            method="highs",
            options={"threads": 1},
        )
    reason = "infeasible" if feasibility.status == 2 else "solver_failure"
    return PandaProjectionResult(
        fallback,
        False,
        reason,
        0.0,
        float(min(margin)) if np.all(np.isfinite(margin)) else None,
        float(residual) if np.isfinite(residual) else None,
    )


class PandaJointFilter:
    """Frozen affine prediction from measured state, estimated dynamics and sent commands."""

    def __init__(self, lower, upper, torque_limit, config=None):
        self.config = config or PandaJointFilterConfig()
        self.config.validate()
        reserve = self.config.position_reserve_rad
        self.lower, self.upper = (
            vector7(lower, "lower") + reserve,
            vector7(upper, "upper") - reserve,
        )
        self.limit = vector7(torque_limit, "torque_limit").copy()
        self.velocity = VELOCITY_LIMIT - self.config.velocity_reserve_rad_s
        if np.any(self.lower >= self.upper) or np.any(self.limit <= 0):
            raise ValueError("reserves and torque limits must leave admissible intervals")

    def _inside_interval(self, q, v, a):
        dt, tol = self.config.control_dt, self.config.feasibility_tol
        endpoint = q + dt * v + 0.5 * dt * dt * a
        smallest, largest = np.minimum(q, endpoint), np.maximum(q, endpoint)
        stationary = np.divide(-v, a, out=np.zeros(7), where=np.abs(a) > 1e-14)
        interior = (np.abs(a) > 1e-14) & (stationary > 0) & (stationary < dt)
        extremum = q + v * stationary + 0.5 * a * stationary**2
        smallest = np.where(interior, np.minimum(smallest, extremum), smallest)
        largest = np.where(interior, np.maximum(largest, extremum), largest)
        return bool(
            np.all(smallest >= self.lower - tol)
            and np.all(largest <= self.upper + tol)
            and np.all(np.maximum(np.abs(v), np.abs(v + dt * a)) <= self.velocity + tol)
        )

    def constraints(self, q, v, matrix, drift, history, ages=None):
        q, v, drift = vector7(q, "q"), vector7(v, "v"), vector7(drift, "drift")
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (7, 7) or not np.all(np.isfinite(matrix)):
            raise ValueError("acceleration matrix must be finite with shape (7, 7)")
        sent = [vector7(u, "history") for u in history]
        if any(np.any(np.abs(u) > self.limit + 1e-12) for u in sent):
            raise ValueError("history must contain sent bounded commands")
        dt, gamma, tol = self.config.control_dt, self.config.gamma, self.config.feasibility_tol
        rows, rhs, arrivals = [], [], []
        if ages is None:
            ages = range(self.config.max_delay_steps + 1)
        elif not ages or any(
            type(a) is not int or not 0 <= a <= self.config.max_delay_steps for a in ages
        ):
            raise ValueError("ages must be a nonempty subset of the admissible delays")
        for lag in ages:
            position, velocity = q.copy(), v.copy()
            for step in range(lag):
                index = len(sent) - lag + step
                command = sent[index] if index >= 0 else np.zeros(7)
                acceleration = matrix @ command + drift
                if not self._inside_interval(position, velocity, acceleration):
                    return None, None, "pending_command_violation", []
                position += velocity * dt + 0.5 * acceleration * dt * dt
                velocity += acceleration * dt
            lower_psi = velocity + gamma * (position - self.lower)
            upper_psi = -velocity + gamma * (self.upper - position)
            if (
                np.any(lower_psi < -tol)
                or np.any(upper_psi < -tol)
                or np.any(position < self.lower - tol)
                or np.any(position > self.upper + tol)
                or np.any(np.abs(velocity) > self.velocity + tol)
            ):
                return None, None, "outside_prediction_domain", []
            arrivals.append((position.copy(), velocity.copy()))
            for joint in range(7):
                rows.extend((matrix[joint], -matrix[joint], matrix[joint], -matrix[joint]))
                rhs.extend(
                    (
                        -drift[joint]
                        - 2 * gamma * velocity[joint]
                        - gamma**2 * (position[joint] - self.lower[joint]),
                        drift[joint]
                        + 2 * gamma * velocity[joint]
                        - gamma**2 * (self.upper[joint] - position[joint]),
                        (-self.velocity[joint] - velocity[joint]) / dt - drift[joint],
                        -(self.velocity[joint] - velocity[joint]) / dt + drift[joint],
                    )
                )
        return np.array(rows), np.array(rhs), "ready", arrivals

    def filter(self, q, v, candidate, matrix, drift, history, ages=None):
        candidate = vector7(candidate, "candidate")
        matrix, drift = np.asarray(matrix, dtype=float), vector7(drift, "drift")
        A, b, reason, arrivals = self.constraints(q, v, matrix, drift, history, ages)
        if A is None:
            return PandaProjectionResult(
                np.clip(candidate, -self.limit, self.limit), False, reason, 0.0, None, None
            )
        result = project_torque(candidate, A, b, self.limit, self.config)
        if result.predicted_feasible:
            acceleration = matrix @ result.torque + drift
            if not all(self._inside_interval(q0, v0, acceleration) for q0, v0 in arrivals):
                return PandaProjectionResult(
                    result.torque,
                    False,
                    "prediction_check_failed",
                    result.correction_norm_nm,
                    result.minimum_scaled_margin,
                    result.stationarity_residual,
                )
        return result
