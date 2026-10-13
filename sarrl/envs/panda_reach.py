"""Nominal Panda joint-trajectory benchmark, without learning or a safety certificate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sarrl.robots.panda import (
    HOME,
    VELOCITY_LIMIT,
    PandaModelParameters,
    load_panda_model,
    mujoco,
    vector7,
)


@dataclass(frozen=True)
class PandaReachConfig:
    timestep: float = 0.002
    control_dt: float = 0.01
    move_seconds: float = 2.0
    settle_seconds: float = 1.0
    joint_tolerance: float = 0.01
    speed_tolerance: float = 0.05
    tip_tolerance: float = 0.01

    def validate(self) -> None:
        values = np.array(list(self.__dict__.values()))
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("Panda timing and tolerances must be finite and positive")
        for interval in (
            self.control_dt / self.timestep,
            self.move_seconds / self.control_dt,
            self.settle_seconds / self.control_dt,
        ):
            if not np.isclose(interval, round(interval), atol=1e-10, rtol=0.0):
                raise ValueError("Panda timing intervals must be integer multiples")


def quintic_reference(start, goal, time: float, duration: float):
    """Position, velocity and acceleration with zero derivatives at either endpoint."""
    start, goal = vector7(start, "start"), vector7(goal, "goal")
    if not np.isfinite(time) or not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("reference time must be finite and duration positive")
    s = np.clip(time / duration, 0.0, 1.0)
    delta = goal - start
    blend = 10 * s**3 - 15 * s**4 + 6 * s**5
    speed = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
    acceleration = (60 * s - 180 * s**2 + 120 * s**3) / duration**2
    return start + blend * delta, speed * delta, acceleration * delta


class PandaReachEnv:
    """Seven torque controls, fixed-horizon episodes and substep constraint observation."""

    def __init__(
        self,
        config: PandaReachConfig | None = None,
        *,
        parameters: PandaModelParameters | None = None,
    ):
        self.config = config or PandaReachConfig()
        self.config.validate()
        self.parameters = parameters or PandaModelParameters()
        self.model = load_panda_model(self.config.timestep, parameters=self.parameters)
        self.data = mujoco.MjData(self.model)
        self.substeps = round(self.config.control_dt / self.config.timestep)
        self.horizon = round(
            (self.config.move_seconds + self.config.settle_seconds) / self.config.control_dt
        )
        self.lower = self.model.jnt_range[:, 0].copy()
        self.upper = self.model.jnt_range[:, 1].copy()
        self.torque_limit = self.model.actuator_ctrlrange[:, 1].copy()
        self.site = self.model.site("attachment_site").id
        self._done = True

    @property
    def state(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel])

    def _observation(self) -> np.ndarray:
        return np.concatenate([self.state, self.goal])

    def _path_clear(self, start, goal) -> bool:
        probe = mujoco.MjData(self.model)
        for fraction in np.linspace(0.0, 1.0, 21):
            probe.qpos[:] = start + fraction * (goal - start)
            mujoco.mj_forward(self.model, probe)
            if probe.ncon:
                return False
        return True

    def reset(self, *, seed: int | None = None, start=None, goal=None):
        rng = np.random.default_rng(seed)
        amplitude = np.array([0.4, 0.25, 0.4, 0.3, 0.4, 0.3, 0.4])
        for _attempt in range(100):
            initial = (
                HOME + rng.uniform(-0.12, 0.12, 7) if start is None else vector7(start, "start")
            )
            target = (
                HOME + rng.uniform(-amplitude, amplitude) if goal is None else vector7(goal, "goal")
            )
            inside = all(
                np.all((q > self.lower + 0.15) & (q < self.upper - 0.15)) for q in (initial, target)
            )
            if inside and self._path_clear(initial, target):
                break
            if start is not None and goal is not None:
                raise ValueError("explicit Panda path violates joint clearance or has contacts")
        else:
            raise RuntimeError("cannot sample a collision-free Panda reference path")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = initial
        mujoco.mj_forward(self.model, self.data)
        self.start, self.goal = initial.copy(), target.copy()
        probe = mujoco.MjData(self.model)
        probe.qpos[:] = target
        mujoco.mj_forward(self.model, probe)
        self.target = probe.site_xpos[self.site].copy()
        self.steps = 0
        self._done = False
        self.position_excess = 0.0
        self.velocity_excess = 0.0
        self.contact_substeps = 0
        self.max_torque = np.zeros(7)
        self.max_speed = np.zeros(7)
        self.saturated_steps = 0
        return self._observation(), {"seed": seed, "sampling_attempts": _attempt + 1}

    def reference(self):
        return quintic_reference(
            self.start, self.goal, self.steps * self.config.control_dt, self.config.move_seconds
        )

    def step_torque(self, torque):
        if self._done:
            raise RuntimeError("reset the Panda episode before stepping")
        command = vector7(torque, "torque")
        applied = np.clip(command, -self.torque_limit, self.torque_limit)
        self.saturated_steps += int(np.any(np.abs(command - applied) > 1e-12))
        self.data.ctrl[:] = applied
        self.max_torque = np.maximum(self.max_torque, np.abs(applied))
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            # mj_step advances qpos after its position-stage fields; refresh them before scoring.
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
        self.steps += 1
        self._done = self.steps >= self.horizon
        info = self.diagnostics()
        return self._observation(), -info["tip_error_m"], False, self._done, info

    def diagnostics(self) -> dict:
        joint_error = float(np.max(np.abs(self.goal - self.data.qpos)))
        tip_error = float(np.linalg.norm(self.target - self.data.site_xpos[self.site]))
        speed = float(np.max(np.abs(self.data.qvel)))
        unsafe = (
            self.position_excess > 0.0 or self.velocity_excess > 0.0 or self.contact_substeps > 0
        )
        reached = (
            joint_error <= self.config.joint_tolerance
            and speed <= self.config.speed_tolerance
            and tip_error <= self.config.tip_tolerance
        )
        return {
            "joint_error_max_rad": joint_error,
            "tip_error_m": tip_error,
            "final_speed_max_rad_s": speed,
            "joint_position_excess_rad": self.position_excess,
            "joint_velocity_excess_rad_s": self.velocity_excess,
            "contact_substeps": self.contact_substeps,
            "unsafe_episode": bool(unsafe),
            "success": bool(self._done and reached and not unsafe),
            "steps": self.steps,
            "max_torque_nm": self.max_torque.tolist(),
            "max_speed_rad_s": self.max_speed.tolist(),
            "saturated_steps": self.saturated_steps,
        }
