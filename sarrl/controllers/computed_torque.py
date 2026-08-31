"""Computed-torque baseline controller."""

from __future__ import annotations

import numpy as np

from sarrl.dynamics import PlanarArm


def _angle_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    delta = target - current
    return np.arctan2(np.sin(delta), np.cos(delta))


class ComputedTorqueController:
    def __init__(
        self,
        model: PlanarArm,
        kp: tuple[float, float] = (36.0, 36.0),
        kd: tuple[float, float] = (12.0, 12.0),
        torque_limit: tuple[float, float] = (40.0, 40.0),
    ):
        self.model = model
        self.kp = np.asarray(kp, dtype=np.float64)
        self.kd = np.asarray(kd, dtype=np.float64)
        self.torque_limit = np.asarray(torque_limit, dtype=np.float64)
        if self.kp.shape != (2,) or self.kd.shape != (2,):
            raise ValueError("kp and kd must have two components")
        if np.any(self.kp <= 0.0) or np.any(self.kd <= 0.0):
            raise ValueError("controller gains must be positive")
        if self.torque_limit.shape != (2,) or np.any(self.torque_limit <= 0.0):
            raise ValueError("torque limits must be positive")

    def command(
        self,
        q,
        qd,
        q_des,
        qd_des=(0.0, 0.0),
        qdd_des=(0.0, 0.0),
    ) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        q_des = np.asarray(q_des, dtype=np.float64)
        qd_des = np.asarray(qd_des, dtype=np.float64)
        qdd_des = np.asarray(qdd_des, dtype=np.float64)
        if any(v.shape != (2,) for v in (q, qd, q_des, qd_des, qdd_des)):
            raise ValueError("all controller vectors must have shape (2,)")
        qdd_cmd = qdd_des + self.kd * (qd_des - qd) + self.kp * _angle_error(q_des, q)
        tau = self.model.inverse_dynamics(q, qd, qdd_cmd, include_friction=True)
        return np.clip(tau, -self.torque_limit, self.torque_limit)
