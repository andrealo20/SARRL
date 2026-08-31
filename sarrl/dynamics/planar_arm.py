"""Analytical rigid-body dynamics for a two-link planar manipulator."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class PlanarArmParams:
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    lc1: float = 0.5
    lc2: float = 0.5
    i1: float = 1.0 / 12.0
    i2: float = 1.0 / 12.0
    gravity: float = 9.81
    viscous: tuple[float, float] = (0.05, 0.05)
    coulomb: tuple[float, float] = (0.02, 0.02)
    friction_smoothing: float = 0.02
    payload_mass: float = 0.0

    def validate(self) -> None:
        positive = (self.m1, self.m2, self.l1, self.l2)
        if not all(np.isfinite(v) and v > 0.0 for v in positive):
            raise ValueError("masses and lengths must be positive and finite")
        if not (0.0 <= self.lc1 <= self.l1 and 0.0 <= self.lc2 <= self.l2):
            raise ValueError("centre-of-mass distances must lie on their links")
        if self.i1 < 0.0 or self.i2 < 0.0:
            raise ValueError("inertias must be non-negative")
        vals = (
            self.i1,
            self.i2,
            self.gravity,
            *self.viscous,
            *self.coulomb,
            self.friction_smoothing,
            self.payload_mass,
        )
        if not all(np.isfinite(v) for v in vals):
            raise ValueError("all parameters must be finite")
        if any(v < 0.0 for v in (*self.viscous, *self.coulomb)) or self.payload_mass < 0.0:
            raise ValueError("friction coefficients and payload mass must be non-negative")
        if self.friction_smoothing <= 0.0:
            raise ValueError("friction_smoothing must be positive")


class PlanarArm:
    """Two-link planar arm with RK4 integration and deterministic equations."""

    def __init__(self, params: PlanarArmParams | None = None):
        self.params = params or PlanarArmParams()
        self.params.validate()

    def with_params(self, **changes: float) -> "PlanarArm":
        return PlanarArm(replace(self.params, **changes))

    @staticmethod
    def _vec2(x: np.ndarray | list[float] | tuple[float, float], name: str) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape != (2,) or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be a finite vector of shape (2,)")
        return arr

    def mass_matrix(self, q) -> np.ndarray:
        q = self._vec2(q, "q")
        p = self.params
        c2 = np.cos(q[1])
        m11 = (
            p.i1
            + p.i2
            + p.m1 * p.lc1**2
            + p.m2 * (p.l1**2 + p.lc2**2 + 2.0 * p.l1 * p.lc2 * c2)
            + p.payload_mass * (p.l1**2 + p.l2**2 + 2.0 * p.l1 * p.l2 * c2)
        )
        m12 = (
            p.i2
            + p.m2 * (p.lc2**2 + p.l1 * p.lc2 * c2)
            + p.payload_mass * (p.l2**2 + p.l1 * p.l2 * c2)
        )
        m22 = p.i2 + p.m2 * p.lc2**2 + p.payload_mass * p.l2**2
        return np.array([[m11, m12], [m12, m22]], dtype=np.float64)

    def coriolis_matrix(self, q, qd) -> np.ndarray:
        q = self._vec2(q, "q")
        qd = self._vec2(qd, "qd")
        p = self.params
        h = (p.m2 * p.l1 * p.lc2 + p.payload_mass * p.l1 * p.l2) * np.sin(q[1])
        return np.array(
            [
                [-h * qd[1], -h * (qd[0] + qd[1])],
                [h * qd[0], 0.0],
            ],
            dtype=np.float64,
        )

    def gravity_vector(self, q) -> np.ndarray:
        q = self._vec2(q, "q")
        p = self.params
        q1, q2 = q
        g1 = (
            (p.m1 * p.lc1 + p.m2 * p.l1 + p.payload_mass * p.l1)
            * p.gravity
            * np.cos(q1)
            + (p.m2 * p.lc2 + p.payload_mass * p.l2)
            * p.gravity
            * np.cos(q1 + q2)
        )
        g2 = (p.m2 * p.lc2 + p.payload_mass * p.l2) * p.gravity * np.cos(q1 + q2)
        return np.array([g1, g2], dtype=np.float64)

    def friction(self, qd) -> np.ndarray:
        qd = self._vec2(qd, "qd")
        p = self.params
        visc = np.asarray(p.viscous, dtype=np.float64) * qd
        coul = np.asarray(p.coulomb, dtype=np.float64) * np.tanh(
            qd / p.friction_smoothing
        )
        return visc + coul

    def inverse_dynamics(self, q, qd, qdd, include_friction: bool = True) -> np.ndarray:
        q = self._vec2(q, "q")
        qd = self._vec2(qd, "qd")
        qdd = self._vec2(qdd, "qdd")
        tau = self.mass_matrix(q) @ qdd + self.coriolis_matrix(q, qd) @ qd
        tau = tau + self.gravity_vector(q)
        if include_friction:
            tau = tau + self.friction(qd)
        return tau

    def forward_dynamics(self, q, qd, tau, include_friction: bool = True) -> np.ndarray:
        q = self._vec2(q, "q")
        qd = self._vec2(qd, "qd")
        tau = self._vec2(tau, "tau")
        rhs = tau - self.coriolis_matrix(q, qd) @ qd - self.gravity_vector(q)
        if include_friction:
            rhs = rhs - self.friction(qd)
        return np.linalg.solve(self.mass_matrix(q), rhs)

    def state_derivative(self, state, tau, include_friction: bool = True) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (4,) or not np.all(np.isfinite(state)):
            raise ValueError("state must be a finite vector of shape (4,)")
        q, qd = state[:2], state[2:]
        qdd = self.forward_dynamics(q, qd, tau, include_friction=include_friction)
        return np.concatenate([qd, qdd])

    def step_rk4(self, state, tau, dt: float, include_friction: bool = True) -> np.ndarray:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        x = np.asarray(state, dtype=np.float64)
        tau = self._vec2(tau, "tau")
        k1 = self.state_derivative(x, tau, include_friction)
        k2 = self.state_derivative(x + 0.5 * dt * k1, tau, include_friction)
        k3 = self.state_derivative(x + 0.5 * dt * k2, tau, include_friction)
        k4 = self.state_derivative(x + dt * k3, tau, include_friction)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def forward_kinematics(self, q) -> np.ndarray:
        q = self._vec2(q, "q")
        p = self.params
        q1, q2 = q
        return np.array(
            [
                p.l1 * np.cos(q1) + p.l2 * np.cos(q1 + q2),
                p.l1 * np.sin(q1) + p.l2 * np.sin(q1 + q2),
            ],
            dtype=np.float64,
        )

    def jacobian(self, q) -> np.ndarray:
        q = self._vec2(q, "q")
        p = self.params
        q1, q2 = q
        s1, c1 = np.sin(q1), np.cos(q1)
        s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
        return np.array(
            [
                [-p.l1 * s1 - p.l2 * s12, -p.l2 * s12],
                [p.l1 * c1 + p.l2 * c12, p.l2 * c12],
            ],
            dtype=np.float64,
        )

    def inverse_kinematics(self, target, elbow_up: bool = False) -> np.ndarray:
        target = self._vec2(target, "target")
        p = self.params
        x, y = target
        r2 = x * x + y * y
        c2 = (r2 - p.l1**2 - p.l2**2) / (2.0 * p.l1 * p.l2)
        if c2 < -1.0 - 1e-10 or c2 > 1.0 + 1e-10:
            raise ValueError("target is outside the reachable workspace")
        c2 = float(np.clip(c2, -1.0, 1.0))
        s2_abs = np.sqrt(max(0.0, 1.0 - c2 * c2))
        s2 = -s2_abs if elbow_up else s2_abs
        q2 = np.arctan2(s2, c2)
        q1 = np.arctan2(y, x) - np.arctan2(p.l2 * s2, p.l1 + p.l2 * c2)
        return np.array([q1, q2], dtype=np.float64)

    def energy(self, q, qd) -> float:
        q = self._vec2(q, "q")
        qd = self._vec2(qd, "qd")
        p = self.params
        kinetic = 0.5 * float(qd @ self.mass_matrix(q) @ qd)
        q1, q2 = q
        potential = (
            p.m1 * p.gravity * p.lc1 * np.sin(q1)
            + p.m2
            * p.gravity
            * (p.l1 * np.sin(q1) + p.lc2 * np.sin(q1 + q2))
            + p.payload_mass
            * p.gravity
            * (p.l1 * np.sin(q1) + p.l2 * np.sin(q1 + q2))
        )
        return kinetic + float(potential)
