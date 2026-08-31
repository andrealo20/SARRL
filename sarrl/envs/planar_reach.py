"""Self-contained 2-DoF Cartesian reaching environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm, PlanarArmParams
from sarrl.utils.spaces import BoxSpace


@dataclass(frozen=True)
class DomainRandomization:
    mass_fraction: float = 0.0
    friction_fraction: float = 0.0
    motor_gain_fraction: float = 0.0

    def validate(self) -> None:
        for value in (self.mass_fraction, self.friction_fraction, self.motor_gain_fraction):
            if not np.isfinite(value) or value < 0.0 or value >= 1.0:
                raise ValueError("randomization fractions must be in [0, 1)")


class PlanarReachEnv:
    """Torque or residual-torque reaching task with deterministic seeded resets."""

    def __init__(
        self,
        mode: str = "torque",
        dt: float = 0.02,
        max_steps: int = 250,
        torque_limit: float = 40.0,
        residual_limit: float = 8.0,
        success_radius: float = 0.05,
        randomization: DomainRandomization | None = None,
    ):
        if mode not in {"torque", "residual"}:
            raise ValueError("mode must be 'torque' or 'residual'")
        if dt <= 0.0 or max_steps <= 0 or torque_limit <= 0.0 or residual_limit <= 0.0:
            raise ValueError("invalid environment limits")
        self.mode = mode
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.torque_limit = float(torque_limit)
        self.residual_limit = float(residual_limit)
        self.success_radius = float(success_radius)
        self.randomization = randomization or DomainRandomization()
        self.randomization.validate()

        self.nominal_arm = PlanarArm()
        self.arm = self.nominal_arm
        self.controller = ComputedTorqueController(
            self.nominal_arm, torque_limit=(self.torque_limit, self.torque_limit)
        )
        self.action_space = BoxSpace(-np.ones(2), np.ones(2))
        self.observation_space = BoxSpace(-np.ones(8), np.ones(8))
        self._rng = np.random.default_rng(0)
        self.state = np.zeros(4, dtype=np.float64)
        self.target = np.array([1.0, 0.0], dtype=np.float64)
        self.q_des = np.zeros(2, dtype=np.float64)
        self.motor_gain = np.ones(2, dtype=np.float64)
        self.steps = 0

    def _sample_arm(self) -> PlanarArm:
        r = self.randomization
        p = PlanarArmParams()
        mf = self._rng.uniform(1.0 - r.mass_fraction, 1.0 + r.mass_fraction, size=2)
        ff = self._rng.uniform(
            1.0 - r.friction_fraction, 1.0 + r.friction_fraction, size=2
        )
        params = PlanarArmParams(
            m1=p.m1 * mf[0],
            m2=p.m2 * mf[1],
            l1=p.l1,
            l2=p.l2,
            lc1=p.lc1,
            lc2=p.lc2,
            i1=p.i1 * mf[0],
            i2=p.i2 * mf[1],
            gravity=p.gravity,
            viscous=(p.viscous[0] * ff[0], p.viscous[1] * ff[1]),
            coulomb=(p.coulomb[0] * ff[0], p.coulomb[1] * ff[1]),
            friction_smoothing=p.friction_smoothing,
        )
        return PlanarArm(params)

    def _sample_target(self) -> np.ndarray:
        # Avoid the singular centre and exact full extension.
        radius = self._rng.uniform(0.45, 1.75)
        angle = self._rng.uniform(-0.9 * np.pi, 0.9 * np.pi)
        return radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)

    def reset(self, seed: int | None = None, target=None):
        if seed is not None:
            if seed < 0:
                raise ValueError("seed must be non-negative")
            self._rng = np.random.default_rng(seed)
        self.arm = self._sample_arm()
        gain_frac = self.randomization.motor_gain_fraction
        self.motor_gain = self._rng.uniform(1.0 - gain_frac, 1.0 + gain_frac, size=2)
        self.state = np.concatenate(
            [
                self._rng.uniform(-0.25, 0.25, size=2),
                self._rng.uniform(-0.05, 0.05, size=2),
            ]
        ).astype(np.float64)
        self.target = self._sample_target() if target is None else np.asarray(target, dtype=np.float64)
        if self.target.shape != (2,) or not np.all(np.isfinite(self.target)):
            raise ValueError("target must be a finite vector of shape (2,)")
        self.q_des = self.nominal_arm.inverse_kinematics(self.target)
        self.steps = 0
        obs = self._observation()
        return obs, {"target": self.target.copy()}

    def _observation(self) -> np.ndarray:
        q = self.state[:2]
        qd = self.state[2:]
        ee = self.arm.forward_kinematics(q)
        error = self.target - ee
        obs = np.concatenate(
            [
                q / np.pi,
                np.clip(qd / 8.0, -1.0, 1.0),
                self.target / 2.0,
                error / 2.0,
            ]
        )
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _candidate_torque(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite vector of shape (2,)")
        action = np.clip(action, -1.0, 1.0)
        q, qd = self.state[:2], self.state[2:]
        if self.mode == "torque":
            baseline = np.zeros(2, dtype=np.float64)
            candidate = action * self.torque_limit
        else:
            baseline = self.controller.command(q, qd, self.q_des)
            candidate = baseline + action * self.residual_limit
        return baseline, np.clip(candidate, -self.torque_limit, self.torque_limit)

    def step(self, action):
        baseline, commanded = self._candidate_torque(action)
        applied = commanded * self.motor_gain
        self.state = self.arm.step_rk4(self.state, applied, self.dt)
        self.steps += 1

        q, qd = self.state[:2], self.state[2:]
        ee = self.arm.forward_kinematics(q)
        distance = float(np.linalg.norm(self.target - ee))
        success = distance <= self.success_radius and float(np.linalg.norm(qd)) <= 0.35
        terminated = bool(success)
        truncated = self.steps >= self.max_steps and not terminated
        reward = -distance - 0.01 * float(qd @ qd) - 0.0002 * float(commanded @ commanded)
        if success:
            reward += 10.0

        info = {
            "distance": distance,
            "success": success,
            "baseline_torque": baseline.astype(np.float32),
            "commanded_torque": commanded.astype(np.float32),
            "applied_torque": applied.astype(np.float32),
            "motor_gain": self.motor_gain.astype(np.float32),
        }
        return self._observation(), float(reward), terminated, truncated, info
