"""Online payload and viscous-friction identification for the seven-joint Panda."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sarrl.controllers.panda_nominal import PandaComputedTorqueController
from sarrl.robots.panda import PandaDynamics, PandaModelParameters, vector7


def parameters8(value) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (8,) or not np.all(np.isfinite(value)):
        raise ValueError("parameter vector must be finite with shape (8,)")
    return value


class PandaRegressor:
    """Known nominal dynamics plus a unit-payload column and seven damping columns."""

    def __init__(self):
        self.nominal = PandaDynamics()
        self.unit_payload = PandaDynamics(parameters=PandaModelParameters(payload_mass=1.0))

    def evaluate(self, q, qd, qdd) -> tuple[np.ndarray, np.ndarray]:
        q, qd, qdd = vector7(q, "q"), vector7(qd, "qd"), vector7(qdd, "qdd")
        nominal = self.nominal.inverse_dynamics(q, qd, qdd)
        unit = self.unit_payload.inverse_dynamics(q, qd, qdd)
        return nominal, np.column_stack([unit - nominal, np.diag(qd)])

    def inverse_dynamics(self, q, qd, qdd, parameters) -> np.ndarray:
        nominal, rows = self.evaluate(q, qd, qdd)
        return nominal + rows @ parameters8(parameters)

    def mass_matrix(self, q, parameters) -> np.ndarray:
        mass = parameters8(parameters)[0]
        nominal = self.nominal.mass_matrix(q)
        return nominal + mass * (self.unit_payload.mass_matrix(q) - nominal)


@dataclass(frozen=True)
class PandaEstimatorConfig:
    payload_prior_std: float = 3.0
    friction_prior_std: float = 5.0
    innovation_variance: float = 1.0
    max_payload_mass: float = 3.5
    max_viscous_extra: float = 5.0

    def validate(self) -> None:
        values = np.array(list(self.__dict__.values()))
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("estimator settings must be finite and positive")


class PandaRLS:
    """Bounded RLS; covariance is an inverse-information matrix, not calibrated uncertainty."""

    def __init__(self, config: PandaEstimatorConfig | None = None):
        self.config = config or PandaEstimatorConfig()
        self.config.validate()
        self.parameters = np.zeros(8)
        self.covariance = np.diag(
            [self.config.payload_prior_std**2] + [self.config.friction_prior_std**2] * 7
        )
        self.upper = np.array([self.config.max_payload_mass] + [self.config.max_viscous_extra] * 7)
        self.information = np.zeros((8, 8))
        self.updates = 0
        self.bound_hits = 0
        self.innovation_squared = np.zeros(7)

    def update(self, rows, residual) -> None:
        rows = np.asarray(rows, dtype=np.float64)
        residual = vector7(residual, "torque residual")
        if rows.shape != (7, 8) or not np.all(np.isfinite(rows)):
            raise ValueError("regressor must be finite with shape (7, 8)")
        self.innovation_squared += (residual - rows @ self.parameters) ** 2
        for row, value in zip(rows, residual, strict=True):
            gain = self.covariance @ row
            gain /= self.config.innovation_variance + row @ gain
            self.parameters += gain * (value - row @ self.parameters)
            contraction = np.eye(8) - np.outer(gain, row)
            self.covariance = (
                contraction @ self.covariance @ contraction.T
                + self.config.innovation_variance * np.outer(gain, gain)
            )
        bounded = np.clip(self.parameters, 0.0, self.upper)
        self.bound_hits += int(np.any(bounded != self.parameters))
        self.parameters = bounded
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.information += rows.T @ rows
        self.updates += 1

    def diagnostics(self) -> dict:
        eigenvalues = np.linalg.eigvalsh(self.information)
        rank = int(np.linalg.matrix_rank(self.information))
        return {
            "estimated_parameters": self.parameters.tolist(),
            "covariance_diagonal": np.diag(self.covariance).tolist(),
            "information_rank": rank,
            "information_condition": (
                float(eigenvalues[-1] / eigenvalues[0])
                if rank == 8 and eigenvalues[0] > 0.0
                else None
            ),
            "updates": self.updates,
            "bound_hits": self.bound_hits,
            "innovation_rms_nm": np.sqrt(self.innovation_squared / max(self.updates, 1)).tolist(),
        }


class PandaEstimatedDynamics:
    def __init__(self, regressor: PandaRegressor, estimator: PandaRLS):
        self.regressor, self.estimator = regressor, estimator
        self.torque_limit = regressor.nominal.torque_limit.copy()

    def inverse_dynamics(self, q, qd, qdd) -> np.ndarray:
        return self.regressor.inverse_dynamics(q, qd, qdd, self.estimator.parameters)

    def mass_matrix(self, q) -> np.ndarray:
        return self.regressor.mass_matrix(q, self.estimator.parameters)


class PandaAdaptiveController(PandaComputedTorqueController):
    """The nominal control law driven by estimates from past observed transitions only."""

    def __init__(self, config: PandaEstimatorConfig | None = None, *, kp=100.0, kd=20.0):
        self.regressor = PandaRegressor()
        self.estimator = PandaRLS(config)
        super().__init__(PandaEstimatedDynamics(self.regressor, self.estimator), kp=kp, kd=kd)

    def observe_transition(self, q, qd, q_next, qd_next, sent_command, dt: float) -> None:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("transition interval must be finite and positive")
        q, qd = vector7(q, "q"), vector7(qd, "qd")
        q_next, qd_next = vector7(q_next, "q_next"), vector7(qd_next, "qd_next")
        sent_command = vector7(sent_command, "sent_command")
        nominal, rows = self.regressor.evaluate(
            0.5 * (q + q_next), 0.5 * (qd + qd_next), (qd_next - qd) / dt
        )
        self.estimator.update(rows, sent_command - nominal)
