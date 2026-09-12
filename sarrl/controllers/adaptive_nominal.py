"""Computed torque on a command-space plant model identified online.

The planar arm is linear in its inertial and friction parameters, and the
actuator scales the delayed command by a per-joint motor gain. Dividing the
joint-i torque equation by that gain gives a relation that is still linear in
the unknowns and whose left-hand side is the command the controller sent:

    u_i(k - L) = Y_i(q, qd, qdd) (theta / g_i) + friction_i(qd) / g_i

Each joint therefore has its own parameter vector, already expressed in the
units the controller needs, and no separate gain estimate is required. The
unknown actuator delay L is handled by running one recursive least-squares
estimator per candidate lag and using the lag with the smallest recent
prediction error. Acceleration is the finite difference of the measured
velocity across one step, evaluated with the regressor at the midpoint state,
so the estimator only consumes quantities a real controller could observe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sarrl.dynamics import PlanarArm, PlanarArmParams

from .computed_torque import _angle_error

PARAMETER_NAMES = ("m1", "m2", "i1", "i2", "payload", "viscous", "coulomb")


def _vector(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite pair")
    return result


class CommandRegressor:
    """Rows of the torque equation that multiply the per-joint parameter vector."""

    def __init__(self, geometry: PlanarArmParams | None = None):
        self.geometry = geometry or PlanarArmParams()
        self.geometry.validate()

    def inertial_matrices(self, q):
        """Mass-matrix contribution of each inertial parameter, shape (5, 2, 2)."""
        p = self.geometry
        c2 = np.cos(q[1])
        m2_block = np.array(
            [
                [p.l1**2 + p.lc2**2 + 2.0 * p.l1 * p.lc2 * c2, p.lc2**2 + p.l1 * p.lc2 * c2],
                [p.lc2**2 + p.l1 * p.lc2 * c2, p.lc2**2],
            ]
        )
        payload_block = np.array(
            [
                [p.l1**2 + p.l2**2 + 2.0 * p.l1 * p.l2 * c2, p.l2**2 + p.l1 * p.l2 * c2],
                [p.l2**2 + p.l1 * p.l2 * c2, p.l2**2],
            ]
        )
        return np.stack(
            [
                np.array([[p.lc1**2, 0.0], [0.0, 0.0]]),
                m2_block,
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                np.ones((2, 2)),
                payload_block,
            ]
        )

    def rows(self, q, qd, qdd):
        """Regressor rows Y such that Y[i] @ theta_i equals the joint-i torque."""
        q = _vector(q, "q")
        qd = _vector(qd, "qd")
        qdd = _vector(qdd, "qdd")
        p = self.geometry
        s2 = np.sin(q[1])
        c1 = np.cos(q[0])
        c12 = np.cos(q[0] + q[1])
        inertia = self.inertial_matrices(q) @ qdd  # (5, 2)
        # Coriolis and gravity columns, indexed (parameter, joint).
        coriolis = np.zeros((5, 2))
        for index, coefficient in ((1, p.l1 * p.lc2), (4, p.l1 * p.l2)):
            h = coefficient * s2
            coriolis[index] = (-h * (2.0 * qd[0] * qd[1] + qd[1] ** 2), h * qd[0] ** 2)
        gravity = np.zeros((5, 2))
        gravity[0] = (p.lc1 * p.gravity * c1, 0.0)
        gravity[1] = (p.l1 * p.gravity * c1 + p.lc2 * p.gravity * c12, p.lc2 * p.gravity * c12)
        gravity[4] = (p.l1 * p.gravity * c1 + p.l2 * p.gravity * c12, p.l2 * p.gravity * c12)
        inertial = (inertia + coriolis + gravity).T  # (2, 5)
        friction = np.stack([qd, np.tanh(qd / p.friction_smoothing)], axis=1)  # (2, 2)
        return np.concatenate([inertial, friction], axis=1)

    @staticmethod
    def parameters(params: PlanarArmParams, motor_gain=(1.0, 1.0)):
        """Per-joint command-space parameter vectors of a known plant, shape (2, 7)."""
        gain = _vector(motor_gain, "motor gain")
        rows = []
        for joint in range(2):
            theta = np.array(
                [
                    params.m1,
                    params.m2,
                    params.i1,
                    params.i2,
                    params.payload_mass,
                    params.viscous[joint],
                    params.coulomb[joint],
                ]
            )
            rows.append(theta / gain[joint])
        return np.stack(rows)


@dataclass(frozen=True)
class AdaptiveNominalConfig:
    kp: tuple[float, float] = (36.0, 36.0)
    kd: tuple[float, float] = (12.0, 12.0)
    torque_limit: tuple[float, float] = (40.0, 40.0)
    dt: float = 0.02
    forgetting: float = 1.0
    process_noise: float = 0.02
    max_lag: int = 3
    selection_memory: float = 0.9
    prior_std: tuple[float, ...] = (1.0, 1.0, 0.1, 0.1, 2.0, 0.1, 0.04)
    lower: tuple[float, ...] = (0.2, 0.2, 0.01, 0.01, 0.0, 0.0, 0.0)
    upper: tuple[float, ...] = (5.0, 5.0, 0.5, 0.5, 6.0, 0.5, 0.2)

    def validate(self) -> None:
        if not 0.0 < self.forgetting <= 1.0 or not 0.0 <= self.selection_memory < 1.0:
            raise ValueError("forgetting must lie in (0, 1] and selection_memory in [0, 1)")
        if not np.isfinite(self.process_noise) or self.process_noise < 0.0:
            raise ValueError("process_noise must be finite and non-negative")
        if self.max_lag < 0 or self.dt <= 0.0:
            raise ValueError("max_lag must be non-negative and dt positive")
        for name in ("prior_std", "lower", "upper"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (7,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must hold seven finite values")
        if np.any(np.asarray(self.prior_std) <= 0.0):
            raise ValueError("prior_std must be positive")
        if np.any(np.asarray(self.lower) >= np.asarray(self.upper)):
            raise ValueError("lower bounds must be below upper bounds")


class AdaptiveNominalController:
    """Computed torque whose model is re-estimated from each executed command."""

    def __init__(self, model: PlanarArm, config: AdaptiveNominalConfig | None = None):
        self.model = model
        self.config = config or AdaptiveNominalConfig()
        self.config.validate()
        self.regressor = CommandRegressor(model.params)
        self.kp = np.asarray(self.config.kp, dtype=np.float64)
        self.kd = np.asarray(self.config.kd, dtype=np.float64)
        self.torque_limit = np.asarray(self.config.torque_limit, dtype=np.float64)
        self.prior = CommandRegressor.parameters(model.params)
        self._lower = np.asarray(self.config.lower, dtype=np.float64)
        self._upper = np.asarray(self.config.upper, dtype=np.float64)
        self._p0 = np.diag(np.asarray(self.config.prior_std, dtype=np.float64) ** 2)
        self.reset()

    # Estimator state -------------------------------------------------------

    def reset(self) -> None:
        lags = self.config.max_lag + 1
        self.theta = np.repeat(self.prior[None], lags, axis=0)  # (lags, 2, 7)
        self.covariance = np.repeat(np.stack([self._p0, self._p0])[None], lags, axis=0)
        self.error_score = np.zeros(lags)
        self.updates = 0
        self._sent: list[np.ndarray] = []

    @property
    def lag(self) -> int:
        if self.updates == 0:
            return 0
        return int(np.argmin(self.error_score))

    @property
    def parameters(self) -> np.ndarray:
        """Per-joint command-space parameters of the selected lag, shape (2, 7)."""
        return self.theta[self.lag].copy()

    def observe(self, state_before, state_after, sent_torque) -> dict:
        """Update every lag hypothesis with one executed transition."""
        before = np.asarray(state_before, dtype=np.float64)
        after = np.asarray(state_after, dtype=np.float64)
        if before.shape != (4,) or after.shape != (4,):
            raise ValueError("states must have shape (4,)")
        if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
            raise ValueError("states must be finite")
        sent = _vector(sent_torque, "sent torque")
        self._sent.append(sent.copy())
        cfg = self.config
        mid = 0.5 * (before + after)
        acceleration = (after[2:] - before[2:]) / cfg.dt
        phi = self.regressor.rows(mid[:2], mid[2:], acceleration)  # (2, 7)
        innovations = np.zeros((cfg.max_lag + 1, 2))
        for lag in range(cfg.max_lag + 1):
            index = len(self._sent) - 1 - lag
            # The plant queue starts with zero commands, exactly as the environment does.
            target = self._sent[index] if index >= 0 else np.zeros(2)
            for joint in range(2):
                row = phi[joint]
                theta = self.theta[lag, joint]
                cov = self.covariance[lag, joint]
                error = float(target[joint] - row @ theta)
                innovations[lag, joint] = error
                gain = cov @ row / (cfg.forgetting + row @ cov @ row)
                theta = np.clip(theta + gain * error, self._lower, self._upper)
                # Random-walk parameters: the additive term keeps the gain alive
                # after convergence so that an abrupt plant change is tracked
                # within a few excited steps instead of the slow 1/forgetting regrowth.
                cov = (cov - np.outer(gain, row @ cov)) / cfg.forgetting
                cov = 0.5 * (cov + cov.T) + cfg.process_noise * self._p0
                self.theta[lag, joint] = theta
                self.covariance[lag, joint] = cov
            squared = float(innovations[lag] @ innovations[lag])
            if self.updates == 0:
                self.error_score[lag] = squared
            else:
                self.error_score[lag] = (
                    cfg.selection_memory * self.error_score[lag]
                    + (1.0 - cfg.selection_memory) * squared
                )
        self.updates += 1
        return {
            "lag": self.lag,
            "innovation": innovations[self.lag].copy(),
            "error_score": self.error_score.copy(),
        }

    # Control law -----------------------------------------------------------

    def command(self, q, qd, q_des, qd_des=(0.0, 0.0), qdd_des=(0.0, 0.0)) -> np.ndarray:
        q = _vector(q, "q")
        qd = _vector(qd, "qd")
        q_des = _vector(q_des, "q_des")
        qd_des = _vector(qd_des, "qd_des")
        qdd_des = _vector(qdd_des, "qdd_des")
        qdd_cmd = qdd_des + self.kd * (qd_des - qd) + self.kp * _angle_error(q_des, q)
        rows = self.regressor.rows(q, qd, qdd_cmd)
        theta = self.theta[self.lag]
        tau = np.einsum("ij,ij->i", rows, theta)
        return np.clip(tau, -self.torque_limit, self.torque_limit)

    def estimated_model(self) -> EstimatedCommandModel:
        """Live model view for a safety filter, expressed in command units."""
        return EstimatedCommandModel(self)


class EstimatedCommandModel:
    """Plant model in command coordinates, read live from the estimator.

    Exposes the subset of the `PlanarArm` interface the HOCBF filter uses. The
    mass matrix has row i scaled by 1 / g_i, so `forward_dynamics` maps the
    command sent to the actuator to the joint acceleration it should produce.
    """

    def __init__(self, controller: AdaptiveNominalController):
        self.controller = controller
        self.kinematics = controller.model

    @property
    def params(self) -> PlanarArmParams:
        return self.kinematics.params

    def _theta(self) -> np.ndarray:
        return self.controller.theta[self.controller.lag]

    def mass_matrix(self, q) -> np.ndarray:
        q = _vector(q, "q")
        blocks = self.controller.regressor.inertial_matrices(q)  # (5, 2, 2)
        theta = self._theta()[:, :5]  # (2, 5)
        return np.stack(
            [np.tensordot(theta[joint], blocks[:, joint, :], axes=1) for joint in range(2)]
        )

    def bias(self, q, qd) -> np.ndarray:
        """Command needed at zero acceleration: Coriolis, gravity and friction."""
        rows = self.controller.regressor.rows(q, qd, np.zeros(2))
        return np.einsum("ij,ij->i", rows, self._theta())

    def inverse_dynamics(self, q, qd, qdd, include_friction: bool = True) -> np.ndarray:
        rows = self.controller.regressor.rows(q, qd, qdd)
        theta = self._theta()
        if not include_friction:
            theta = theta.copy()
            theta[:, 5:] = 0.0
        return np.einsum("ij,ij->i", rows, theta)

    def forward_dynamics(self, q, qd, tau, include_friction: bool = True) -> np.ndarray:
        tau = _vector(tau, "tau")
        bias = self.inverse_dynamics(q, qd, np.zeros(2), include_friction=include_friction)
        return np.linalg.solve(self.mass_matrix(q), tau - bias)

    def forward_kinematics(self, q):
        return self.kinematics.forward_kinematics(q)

    def jacobian(self, q):
        return self.kinematics.jacobian(q)

    def jacobian_dot_times_qd(self, q, qd):
        return self.kinematics.jacobian_dot_times_qd(q, qd)
