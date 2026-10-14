"""Retain physical substep states with the unchanged Panda integration sequence."""

from __future__ import annotations

import numpy as np

from sarrl.envs.panda_reach import PandaReachEnv
from sarrl.robots.panda import VELOCITY_LIMIT, mujoco, vector7


class PandaJointReachEnv(PandaReachEnv):
    def step_torque(self, torque):
        if self._done:
            raise RuntimeError("reset the Panda episode before stepping")
        command = vector7(torque, "torque")
        applied = np.clip(command, -self.torque_limit, self.torque_limit)
        self.saturated_steps += int(np.any(np.abs(command - applied) > 1e-12))
        self.data.ctrl[:] = applied
        self.max_torque = np.maximum(self.max_torque, np.abs(applied))
        self.last_substeps = []
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            if not np.all(np.isfinite(self.state)):
                raise FloatingPointError("non-finite Panda state")
            self.position_excess = max(
                self.position_excess,
                float(np.max(np.maximum(self.lower - self.data.qpos, self.data.qpos - self.upper))),
            )
            self.velocity_excess = max(
                self.velocity_excess, float(np.max(np.abs(self.data.qvel) - VELOCITY_LIMIT))
            )
            self.max_speed = np.maximum(self.max_speed, np.abs(self.data.qvel))
            self.contact_substeps += int(self.data.ncon > 0)
            self.last_substeps.append(
                {
                    "q": self.data.qpos.tolist(),
                    "qd": self.data.qvel.tolist(),
                    "contacts": int(self.data.ncon),
                }
            )
        self.steps += 1
        self._done = self.steps >= self.horizon
        info = self.diagnostics()
        return self._observation(), -info["tip_error_m"], False, self._done, info
