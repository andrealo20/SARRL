"""Hard high-order CBF safety projection for the 2-DoF analytical arm.

The QP is a Euclidean projection in two torque variables.  Rather than depend
on a generic numerical QP solver, the implementation enumerates all possible
active sets of size 0, 1 and 2.  For a strictly convex quadratic objective in
R^2, at least one such active set contains the optimum whenever the polytope
is feasible.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from sarrl.dynamics import PlanarArm


@dataclass(frozen=True)
class CircularObstacle:
    center: tuple[float, float]
    radius: float
    margin: float = 0.08
    gamma1: float = 4.0
    gamma2: float = 4.0

    def validate(self) -> None:
        c = np.asarray(self.center, dtype=np.float64)
        if c.shape != (2,) or not np.all(np.isfinite(c)):
            raise ValueError("obstacle center must be a finite 2-vector")
        if self.radius <= 0.0 or self.margin < 0.0 or self.gamma1 <= 0.0 or self.gamma2 <= 0.0:
            raise ValueError("obstacle radius/gains must be positive and margin non-negative")


@dataclass(frozen=True)
class SafetyConfig:
    torque_limit: tuple[float, float] = (40.0, 40.0)
    joint_lower: tuple[float, float] = (-3.05, -3.05)
    joint_upper: tuple[float, float] = (3.05, 3.05)
    velocity_limit: tuple[float, float] = (7.0, 7.0)
    joint_gamma1: float = 5.0
    joint_gamma2: float = 5.0
    velocity_dt: float = 0.02
    feasibility_tol: float = 2e-8

    def validate(self) -> None:
        lo = np.asarray(self.joint_lower, dtype=np.float64)
        hi = np.asarray(self.joint_upper, dtype=np.float64)
        t = np.asarray(self.torque_limit, dtype=np.float64)
        v = np.asarray(self.velocity_limit, dtype=np.float64)
        if any(x.shape != (2,) for x in (lo, hi, t, v)):
            raise ValueError("safety limits must be pairs")
        if not all(np.all(np.isfinite(x)) for x in (lo, hi, t, v)):
            raise ValueError("safety limits must be finite")
        if np.any(lo >= hi) or np.any(t <= 0.0) or np.any(v <= 0.0):
            raise ValueError("invalid safety limits")
        if self.joint_gamma1 <= 0.0 or self.joint_gamma2 <= 0.0 or self.velocity_dt <= 0.0:
            raise ValueError("safety gains and dt must be positive")
        if self.feasibility_tol <= 0.0:
            raise ValueError("feasibility tolerance must be positive")


@dataclass(frozen=True)
class ProjectionResult:
    x: np.ndarray
    success: bool
    objective: float
    min_margin: float
    active: tuple[int, ...]


def project_polytope_2d(candidate, A, b, tol: float = 1e-9) -> ProjectionResult:
    """Exact projection of a point onto {x | A x >= b} in R^2."""
    c = np.asarray(candidate, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if c.shape != (2,) or A.ndim != 2 or A.shape[1] != 2 or b.shape != (A.shape[0],):
        raise ValueError("invalid projection dimensions")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("projection inputs must be finite")

    def feasible(x) -> tuple[bool, float]:
        margins = A @ x - b
        return bool(np.all(margins >= -tol)), float(np.min(margins)) if len(margins) else np.inf

    candidates: list[tuple[np.ndarray, tuple[int, ...]]] = []
    ok, _ = feasible(c)
    if ok:
        candidates.append((c.copy(), ()))

    for i, a in enumerate(A):
        norm2 = float(a @ a)
        if norm2 <= 1e-18:
            continue
        x = c + ((b[i] - float(a @ c)) / norm2) * a
        if feasible(x)[0]:
            candidates.append((x, (i,)))

    for i, j in combinations(range(A.shape[0]), 2):
        mat = np.stack([A[i], A[j]], axis=0)
        det = float(np.linalg.det(mat))
        if abs(det) <= 1e-12:
            continue
        x = np.linalg.solve(mat, np.array([b[i], b[j]], dtype=np.float64))
        if feasible(x)[0]:
            candidates.append((x, (i, j)))

    if not candidates:
        margins = A @ c - b
        return ProjectionResult(
            x=c.copy(),
            success=False,
            objective=0.0,
            min_margin=float(np.min(margins)) if len(margins) else np.inf,
            active=(),
        )
    x, active = min(candidates, key=lambda item: float(np.sum((item[0] - c) ** 2)))
    _, min_margin = feasible(x)
    return ProjectionResult(
        x=x,
        success=True,
        objective=0.5 * float(np.sum((x - c) ** 2)),
        min_margin=min_margin,
        active=active,
    )


@dataclass(frozen=True)
class SafetyResult:
    torque: np.ndarray
    success: bool
    correction_norm: float
    min_margin: float
    active_constraints: tuple[int, ...]
    current_safe: bool


class HOCBFSafetyFilter:
    """Project candidate torque onto hard joint/velocity/obstacle constraints."""

    def __init__(self, model: PlanarArm, config: SafetyConfig | None = None):
        self.model = model
        self.config = config or SafetyConfig()
        self.config.validate()

    def _affine_acceleration(self, q, qd) -> tuple[np.ndarray, np.ndarray]:
        M = self.model.mass_matrix(q)
        B = np.linalg.inv(M)
        drift = self.model.forward_dynamics(q, qd, np.zeros(2), include_friction=True)
        return B, drift

    def constraints(self, state, obstacles=()) -> tuple[np.ndarray, np.ndarray, bool]:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (4,) or not np.all(np.isfinite(state)):
            raise ValueError("state must be a finite vector of shape (4,)")
        q, qd = state[:2], state[2:]
        cfg = self.config
        B, drift = self._affine_acceleration(q, qd)
        lo = np.asarray(cfg.joint_lower)
        hi = np.asarray(cfg.joint_upper)
        vmax = np.asarray(cfg.velocity_limit)
        gsum = cfg.joint_gamma1 + cfg.joint_gamma2
        gprod = cfg.joint_gamma1 * cfg.joint_gamma2
        rows: list[np.ndarray] = []
        rhs: list[float] = []
        current_safe = bool(np.all(q >= lo) and np.all(q <= hi) and np.all(np.abs(qd) <= vmax))

        # Relative-degree-two joint-position barriers.
        for i in range(2):
            h_lo = q[i] - lo[i]
            rows.append(B[i].copy())
            rhs.append(float(-drift[i] - gsum * qd[i] - gprod * h_lo))

            h_hi = hi[i] - q[i]
            rows.append(-B[i].copy())
            rhs.append(float(drift[i] + gsum * qd[i] - gprod * h_hi))

            # One-step hard velocity bounds under the same affine dynamics.
            rows.append(B[i].copy())
            rhs.append(float((-vmax[i] - qd[i]) / cfg.velocity_dt - drift[i]))
            rows.append(-B[i].copy())
            rhs.append(float(-(vmax[i] - qd[i]) / cfg.velocity_dt + drift[i]))

        # Relative-degree-two Cartesian circular-obstacle barriers.
        p = self.model.forward_kinematics(q)
        J = self.model.jacobian(q)
        ee_vel = J @ qd
        jdot_qd = self.model.jacobian_dot_times_qd(q, qd)
        for obstacle in obstacles:
            obstacle.validate()
            center = np.asarray(obstacle.center, dtype=np.float64)
            r = p - center
            R = obstacle.radius + obstacle.margin
            h = float(r @ r - R * R)
            hdot = 2.0 * float(r @ ee_vel)
            current_safe = current_safe and h >= 0.0
            a = 2.0 * (r @ J @ B)
            constant = (
                2.0 * float(ee_vel @ ee_vel)
                + 2.0 * float(r @ (J @ drift + jdot_qd))
                + (obstacle.gamma1 + obstacle.gamma2) * hdot
                + obstacle.gamma1 * obstacle.gamma2 * h
            )
            rows.append(np.asarray(a, dtype=np.float64))
            rhs.append(-constant)

        # Torque box as half-spaces, kept inside the exact projection problem.
        limit = np.asarray(cfg.torque_limit)
        for i in range(2):
            e = np.zeros(2)
            e[i] = 1.0
            rows.append(e.copy())
            rhs.append(float(-limit[i]))
            rows.append(-e.copy())
            rhs.append(float(-limit[i]))
        return np.asarray(rows), np.asarray(rhs), current_safe

    def _project(self, state, candidate, A, b) -> ProjectionResult:
        """Projection onto the rows; subclasses may tighten them."""
        return project_polytope_2d(candidate, A, b, tol=self.config.feasibility_tol)

    def filter(self, state, candidate, obstacles=()) -> SafetyResult:
        candidate = np.asarray(candidate, dtype=np.float64)
        if candidate.shape != (2,) or not np.all(np.isfinite(candidate)):
            raise ValueError("candidate torque must be a finite vector of shape (2,)")
        A, b, current_safe = self.constraints(state, obstacles)
        projection = self._project(state, candidate, A, b)
        return SafetyResult(
            torque=projection.x,
            success=projection.success,
            correction_norm=float(np.linalg.norm(projection.x - candidate)),
            min_margin=projection.min_margin,
            active_constraints=projection.active,
            current_safe=current_safe,
        )


class MarginHOCBFSafetyFilter(HOCBFSafetyFilter):
    """HOCBF projection whose model-dependent rows carry a parameter-uncertainty margin.

    Every row except the four torque-box rows is affine in the acceleration the
    model predicts, qdd = B tau + drift, so its torque coefficients are a B for
    some acceleration coefficients a. If the command the model needs for that
    acceleration is wrong by du, the acceleration is wrong by -B du and the
    row by -(a B) du. Requiring the row for every |du_j| <= kappa sigma_j
    raises its bound by kappa * sum_j |A_rj| sigma_j.

    `uncertainty(state, command)` returns sigma, the per-joint standard
    deviation of the command the model needs for the acceleration it predicts
    for `command`, or None when the model carries no covariance for the step.
    Sigma grows with that acceleration, and the executed command is not known
    before the projection, so the margin uses the larger sigma of the candidate
    and of the projection without margin. When the tightened rows leave no
    feasible command, the margin is halved until they do, and dropped after
    `BACKOFF` halvings; the counters record both cases, and `applied` sums
    the per-joint command margin kappa * sigma actually enforced.
    """

    TORQUE_ROWS = 4
    BACKOFF = 8

    def __init__(self, model, config: SafetyConfig | None = None, *, kappa: float, uncertainty):
        super().__init__(model, config)
        if not np.isfinite(kappa) or kappa < 0.0:
            raise ValueError("kappa must be finite and non-negative")
        self.kappa = float(kappa)
        self.uncertainty = uncertainty
        self.reset_counters()

    def reset_counters(self) -> None:
        self.margin_steps = 0
        self.reduced_steps = 0
        self.dropped_steps = 0
        self.applied = np.zeros(2)

    def _sigma(self, state, command) -> np.ndarray | None:
        sigma = self.uncertainty(state, command)
        if sigma is None:
            return None
        sigma = np.asarray(sigma, dtype=np.float64)
        if sigma.shape != (2,) or not np.all(np.isfinite(sigma)) or np.any(sigma < 0.0):
            raise ValueError("uncertainty must be a finite non-negative pair")
        return sigma

    def _project(self, state, candidate, A, b) -> ProjectionResult:
        tol = self.config.feasibility_tol
        plain = project_polytope_2d(candidate, A, b, tol=tol)
        # Tightened rows cannot be feasible when the rows themselves are not.
        if self.kappa == 0.0 or not plain.success:
            return plain
        sigma = self._sigma(state, candidate)
        if sigma is None:
            return plain
        sigma = np.maximum(sigma, self._sigma(state, plain.x))
        rows = A.shape[0] - self.TORQUE_ROWS
        margin = np.zeros_like(b)
        margin[:rows] = self.kappa * (np.abs(A[:rows]) @ sigma)
        self.margin_steps += 1
        scale = 1.0
        for _ in range(self.BACKOFF + 1):
            projection = project_polytope_2d(candidate, A, b + scale * margin, tol=tol)
            if projection.success:
                if scale < 1.0:
                    self.reduced_steps += 1
                self.applied += scale * self.kappa * sigma
                return projection
            scale *= 0.5
        self.dropped_steps += 1
        return plain
