"""Computed torque on the Panda's independent nominal model."""

from __future__ import annotations

import numpy as np

from sarrl.robots.panda import PandaDynamics, vector7


class PandaComputedTorqueController:
    def __init__(self, dynamics: PandaDynamics | None = None, *, kp=100.0, kd=20.0):
        if not np.isfinite(kp) or not np.isfinite(kd) or kp <= 0.0 or kd <= 0.0:
            raise ValueError("controller gains must be finite and positive")
        self.dynamics = dynamics or PandaDynamics()
        self.kp = float(kp)
        self.kd = float(kd)
        self.last_raw_torque = np.zeros(7)

    def command(self, q, qd, q_ref, qd_ref=None, qdd_ref=None) -> np.ndarray:
        q, qd = vector7(q, "q"), vector7(qd, "qd")
        q_ref = vector7(q_ref, "q_ref")
        qd_ref = np.zeros(7) if qd_ref is None else vector7(qd_ref, "qd_ref")
        qdd_ref = np.zeros(7) if qdd_ref is None else vector7(qdd_ref, "qdd_ref")
        # These joints have finite ranges; wrapping the error could cross a joint stop.
        acceleration = qdd_ref + self.kp * (q_ref - q) + self.kd * (qd_ref - qd)
        self.last_raw_torque = self.dynamics.inverse_dynamics(q, qd, acceleration)
        limit = self.dynamics.torque_limit
        return np.clip(self.last_raw_torque, -limit, limit)
