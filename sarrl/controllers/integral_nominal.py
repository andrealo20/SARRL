"""Computed torque with transactional integral acceleration and back-calculation."""

from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

import numpy as np

from .computed_torque import ComputedTorqueController, _angle_error


def _vector(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite pair")
    return result.copy()


class IntegralNominalController(ComputedTorqueController):
    """Keep pre-command data until a successful transition explicitly commits it."""

    def __init__(self, model):
        super().__init__(model)
        self.ki = np.full(2, 36.0)
        self.kaw = 4.0
        self.dt = 0.02
        self.z_limit = np.full(2, 36.0)
        self.reset()

    def reset(self):
        self._episode_id = uuid4().hex
        self._attempt_version = 0
        self._z = np.zeros(2, dtype=np.float64)
        self._pending = None
        self._aborted = False

    @property
    def z(self):
        return self._z.copy()

    @property
    def pending(self):
        return deepcopy(self._pending)

    def state_dict(self):
        """Diagnostic snapshot only; mid-episode restoration is unsupported."""
        return {
            "episode_id": self._episode_id,
            "attempt_version": self._attempt_version,
            "z": self.z,
            "pending": self.pending,
            "aborted": self._aborted,
        }

    def command(self, q, qd, q_des, qd_des=(0.0, 0.0), qdd_des=(0.0, 0.0)):
        if self._pending is not None or self._aborted:
            raise ValueError("pending or aborted transaction requires resolution or reset")
        q, qd, q_des, qd_des, qdd_des = (
            _vector(value, name)
            for value, name in zip(
                (q, qd, q_des, qd_des, qdd_des),
                ("q", "qd", "q_des", "qd_des", "qdd_des"),
                strict=True,
            )
        )
        error = _angle_error(q_des, q)
        acceleration = qdd_des + self.kd * (qd_des - qd) + self.kp * error
        mass = np.asarray(self.model.mass_matrix(q), dtype=np.float64).copy()
        if mass.shape != (2, 2) or not np.all(np.isfinite(mass)):
            raise ValueError("invalid nominal mass matrix")
        pd = _vector(
            self.model.inverse_dynamics(q, qd, acceleration, include_friction=True), "PD torque"
        )
        integral = mass @ self._z
        raw = _vector(pd + integral, "raw torque")
        b = np.clip(raw, -self.torque_limit, self.torque_limit)
        self._pending = {
            "token": (self._episode_id, self._attempt_version),
            "z_before": self.z,
            "e_pre": error,
            "M_nominal_pre": mass,
            "nominal_pd_unclipped": pd,
            "integral_torque": integral,
            "tau_raw": raw,
            "b": b.copy(),
            "ki_error": self.ki * error,
        }
        self._attempt_version += 1
        return b

    def _transaction(self, token):
        if self._pending is None or tuple(token) != self._pending["token"]:
            raise ValueError("stale, duplicate or out-of-order transaction token")
        return self.pending

    def commit(self, token, filtered_torque, sent_torque):
        """Commit once after execution, using only cached nominal data and commands."""
        row = self._transaction(token)
        h = _vector(filtered_torque, "filtered torque")
        sent = _vector(sent_torque, "sent torque")
        if not np.array_equal(sent, np.clip(h, -self.torque_limit, self.torque_limit)):
            raise ValueError("sent torque differs from command clipping")
        mass = row["M_nominal_pre"]
        row["u_sent"] = sent
        row["hocbf_correction"] = h - row["b"]
        for name, delta in (
            ("aw_clip", row["b"] - row["tau_raw"]),
            ("aw_hocbf", h - row["b"]),
            ("aw_sendclip", sent - h),
            ("back_calculation", sent - row["tau_raw"]),
        ):
            row[name] = _vector(self.kaw * np.linalg.solve(mass, delta), name)
        if not np.allclose(
            row["aw_clip"] + row["aw_hocbf"] + row["aw_sendclip"],
            row["back_calculation"],
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ValueError("anti-windup decomposition mismatch")
        row["dz"] = _vector(row["ki_error"] + row["back_calculation"], "dz")
        row["z_unclipped_next"] = _vector(self._z + self.dt * row["dz"], "next integral")
        row["z_after"] = np.clip(row["z_unclipped_next"], -self.z_limit, self.z_limit)
        row["integral_clipping"] = row["z_after"] - row["z_unclipped_next"]
        row["commit"] = True
        self._z = row["z_after"].copy()
        self._pending = None
        return row

    def abort(self, token):
        """Close a rejected attempt without updating the integral; reset before reuse."""
        row = self._transaction(token)
        for field in (
            "u_sent",
            "hocbf_correction",
            "aw_clip",
            "aw_hocbf",
            "aw_sendclip",
            "back_calculation",
            "dz",
            "z_unclipped_next",
            "integral_clipping",
        ):
            row[field] = None
        row.update(commit=False, z_after=self.z)
        self._pending = None
        self._aborted = True
        return row
