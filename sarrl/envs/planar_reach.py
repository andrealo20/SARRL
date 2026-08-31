"""Self-contained 2-DoF Cartesian reaching environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm, PlanarArmParams
from sarrl.utils.spaces import BoxSpace


@dataclass(frozen=True)
class DomainRandomization:
    mass_fraction: float = 0.0
    friction_fraction: float = 0.0
    motor_gain_fraction: float = 0.0
    payload_range: tuple[float, float] = (0.0, 0.0)
    sensor_noise_std: float = 0.0
    action_delay_max: int = 0

    def validate(self) -> None:
        for value in (self.mass_fraction, self.friction_fraction, self.motor_gain_fraction):
            if not np.isfinite(value) or value < 0.0 or value >= 1.0:
                raise ValueError("randomization fractions must be in [0, 1)")
        lo, hi = self.payload_range
        if not np.isfinite(lo) or not np.isfinite(hi) or lo < 0.0 or hi < lo:
            raise ValueError("payload_range must be finite, non-negative and ordered")
        if not np.isfinite(self.sensor_noise_std) or self.sensor_noise_std < 0.0:
            raise ValueError("sensor_noise_std must be finite and non-negative")
        if not isinstance(self.action_delay_max, int) or self.action_delay_max < 0:
            raise ValueError("action_delay_max must be a non-negative integer")


@dataclass(frozen=True)
class FaultSpec:
    """Abrupt in-episode dynamics change used only for controlled fault studies."""

    start_step: int
    motor_gain_multiplier: tuple[float, float] = (1.0, 1.0)
    payload_delta: float = 0.0

    def validate(self) -> None:
        if self.start_step < 0:
            raise ValueError("fault start_step must be non-negative")
        gains = np.asarray(self.motor_gain_multiplier, dtype=np.float64)
        if gains.shape != (2,) or not np.all(np.isfinite(gains)) or np.any(gains <= 0.0):
            raise ValueError("fault motor gains must be a positive finite pair")
        if not np.isfinite(self.payload_delta):
            raise ValueError("fault payload_delta must be finite")


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
        fault: FaultSpec | None = None,
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
        self.fault = fault
        if self.fault is not None:
            self.fault.validate()

        self.nominal_arm = PlanarArm()
        self.arm = self.nominal_arm
        self.controller = ComputedTorqueController(
            self.nominal_arm, torque_limit=(self.torque_limit, self.torque_limit)
        )
        self.action_space = BoxSpace(-np.ones(2), np.ones(2))
        self.observation_space = BoxSpace(-np.ones(8), np.ones(8))
        self._rng = np.random.default_rng(0)
        self._noise_rng = np.random.default_rng(1)
        self.state = np.zeros(4, dtype=np.float64)
        self.target = np.array([1.0, 0.0], dtype=np.float64)
        self.q_des = np.zeros(2, dtype=np.float64)
        self.motor_gain = np.ones(2, dtype=np.float64)
        self.mass_scale = np.ones(2, dtype=np.float64)
        self.friction_scale = np.ones(2, dtype=np.float64)
        self.payload_mass = 0.0
        self.action_delay = 0
        self._command_queue: list[np.ndarray] = []
        self._fault_active = False
        self.steps = 0

    def _sample_arm(self) -> PlanarArm:
        r = self.randomization
        p = PlanarArmParams()
        self.mass_scale = self._rng.uniform(
            1.0 - r.mass_fraction, 1.0 + r.mass_fraction, size=2
        )
        self.friction_scale = self._rng.uniform(
            1.0 - r.friction_fraction, 1.0 + r.friction_fraction, size=2
        )
        self.payload_mass = float(self._rng.uniform(*r.payload_range))
        params = PlanarArmParams(
            m1=p.m1 * self.mass_scale[0],
            m2=p.m2 * self.mass_scale[1],
            l1=p.l1,
            l2=p.l2,
            lc1=p.lc1,
            lc2=p.lc2,
            i1=p.i1 * self.mass_scale[0],
            i2=p.i2 * self.mass_scale[1],
            gravity=p.gravity,
            viscous=(
                p.viscous[0] * self.friction_scale[0],
                p.viscous[1] * self.friction_scale[1],
            ),
            coulomb=(
                p.coulomb[0] * self.friction_scale[0],
                p.coulomb[1] * self.friction_scale[1],
            ),
            friction_smoothing=p.friction_smoothing,
            payload_mass=self.payload_mass,
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
            self._noise_rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
        self.arm = self._sample_arm()
        gain_frac = self.randomization.motor_gain_fraction
        self.motor_gain = self._rng.uniform(1.0 - gain_frac, 1.0 + gain_frac, size=2)
        self.action_delay = int(self._rng.integers(0, self.randomization.action_delay_max + 1))
        self.state = np.concatenate(
            [
                self._rng.uniform(-0.25, 0.25, size=2),
                self._rng.uniform(-0.05, 0.05, size=2),
            ]
        ).astype(np.float64)
        self.target = (
            self._sample_target() if target is None else np.asarray(target, dtype=np.float64)
        )
        if self.target.shape != (2,) or not np.all(np.isfinite(self.target)):
            raise ValueError("target must be a finite vector of shape (2,)")
        self.q_des = self.nominal_arm.inverse_kinematics(self.target)
        self._command_queue = [np.zeros(2, dtype=np.float64) for _ in range(self.action_delay)]
        self._fault_active = False
        self.steps = 0
        obs = self._observation()
        return obs, self._info_base()

    def _sensed_state(self) -> np.ndarray:
        if self.randomization.sensor_noise_std == 0.0:
            return self.state.copy()
        noise = self._noise_rng.normal(0.0, self.randomization.sensor_noise_std, size=4)
        return self.state + noise

    def _observation(self) -> np.ndarray:
        sensed = self._sensed_state()
        q = sensed[:2]
        qd = sensed[2:]
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

    def _activate_fault_if_due(self) -> None:
        if self.fault is None or self._fault_active or self.steps < self.fault.start_step:
            return
        self.motor_gain = self.motor_gain * np.asarray(self.fault.motor_gain_multiplier)
        new_payload = self.arm.params.payload_mass + self.fault.payload_delta
        if new_payload < 0.0:
            raise ValueError("fault would make payload mass negative")
        self.arm = self.arm.with_params(payload_mass=new_payload)
        self.payload_mass = float(new_payload)
        self._fault_active = True

    def _delayed_command(self, commanded: np.ndarray) -> np.ndarray:
        if self.action_delay == 0:
            return commanded
        self._command_queue.append(commanded.copy())
        return self._command_queue.pop(0)

    def _info_base(self) -> dict:
        return {
            "target": self.target.copy(),
            "mass_scale": self.mass_scale.astype(np.float32),
            "friction_scale": self.friction_scale.astype(np.float32),
            "motor_gain": self.motor_gain.astype(np.float32),
            "payload_mass": float(self.payload_mass),
            "action_delay": int(self.action_delay),
            "fault_active": bool(self._fault_active),
        }

    def dynamics_context(self) -> np.ndarray:
        """Ground-truth context for diagnostics and auxiliary training only."""
        return np.array(
            [
                *self.mass_scale,
                *self.friction_scale,
                *self.motor_gain,
                self.payload_mass,
                float(self.action_delay),
            ],
            dtype=np.float32,
        )

    def constructor_config(self) -> dict:
        return {
            "mode": self.mode,
            "dt": self.dt,
            "max_steps": self.max_steps,
            "torque_limit": self.torque_limit,
            "residual_limit": self.residual_limit,
            "success_radius": self.success_radius,
            "randomization": asdict(self.randomization),
            "fault": asdict(self.fault) if self.fault is not None else None,
        }

    def state_dict(self) -> dict:
        return {
            "constructor_config": self.constructor_config(),
            "mode": self.mode,
            "dt": self.dt,
            "max_steps": self.max_steps,
            "state": self.state.copy(),
            "target": self.target.copy(),
            "q_des": self.q_des.copy(),
            "motor_gain": self.motor_gain.copy(),
            "mass_scale": self.mass_scale.copy(),
            "friction_scale": self.friction_scale.copy(),
            "payload_mass": self.payload_mass,
            "action_delay": self.action_delay,
            "command_queue": [x.copy() for x in self._command_queue],
            "fault_active": self._fault_active,
            "steps": self.steps,
            "arm_params": asdict(self.arm.params),
            "rng_state": self._rng.bit_generator.state,
            "noise_rng_state": self._noise_rng.bit_generator.state,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> PlanarReachEnv:
        cfg = state.get("constructor_config")
        if cfg is None:
            raise ValueError("environment checkpoint lacks constructor configuration")
        cfg = dict(cfg)
        randomization = DomainRandomization(**dict(cfg.pop("randomization")))
        fault_data = cfg.pop("fault")
        fault = FaultSpec(**dict(fault_data)) if fault_data is not None else None
        env = cls(randomization=randomization, fault=fault, **cfg)
        env.load_state_dict(state)
        return env

    def load_state_dict(self, state: dict) -> None:
        stored_cfg = state.get("constructor_config")
        if stored_cfg is not None and stored_cfg != self.constructor_config():
            raise ValueError("environment checkpoint constructor configuration does not match")
        if state["mode"] != self.mode or float(state["dt"]) != self.dt:
            raise ValueError("environment checkpoint configuration does not match")
        if int(state["max_steps"]) != self.max_steps:
            raise ValueError("environment checkpoint max_steps does not match")
        for name, shape in (
            ("state", (4,)),
            ("target", (2,)),
            ("q_des", (2,)),
            ("motor_gain", (2,)),
            ("mass_scale", (2,)),
            ("friction_scale", (2,)),
        ):
            value = np.asarray(state[name], dtype=np.float64)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid environment checkpoint field {name}")
            setattr(self, name, value.copy())
        self.payload_mass = float(state["payload_mass"])
        self.action_delay = int(state["action_delay"])
        self._command_queue = [
            np.asarray(x, dtype=np.float64).copy() for x in state["command_queue"]
        ]
        if any(x.shape != (2,) for x in self._command_queue):
            raise ValueError("invalid delayed-command queue")
        self._fault_active = bool(state["fault_active"])
        self.steps = int(state["steps"])
        self.arm = PlanarArm(PlanarArmParams(**state["arm_params"]))
        self._rng.bit_generator.state = state["rng_state"]
        self._noise_rng.bit_generator.state = state["noise_rng_state"]

    def step_torque(self, commanded, baseline=None):
        """Advance the plant with a physical torque command in N m.

        This bypasses the environment's action-to-torque mapping and is the
        execution entry point for composed controllers such as SARRLControlStack.
        Actuator delay, faults and motor gain still belong to the plant and are
        therefore applied here.
        """
        self._activate_fault_if_due()
        commanded = np.asarray(commanded, dtype=np.float64)
        if commanded.shape != (2,) or not np.all(np.isfinite(commanded)):
            raise ValueError("commanded torque must be a finite vector of shape (2,)")
        commanded = np.clip(commanded, -self.torque_limit, self.torque_limit)
        if baseline is None:
            baseline = np.zeros(2, dtype=np.float64)
        baseline = np.asarray(baseline, dtype=np.float64)
        if baseline.shape != (2,) or not np.all(np.isfinite(baseline)):
            raise ValueError("baseline torque must be a finite vector of shape (2,)")

        delayed = self._delayed_command(commanded)
        applied = delayed * self.motor_gain
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

        info = self._info_base()
        info.update(
            {
                "distance": distance,
                "success": success,
                "baseline_torque": baseline.astype(np.float32),
                "commanded_torque": commanded.astype(np.float32),
                "delayed_torque": delayed.astype(np.float32),
                "applied_torque": applied.astype(np.float32),
            }
        )
        return self._observation(), float(reward), terminated, truncated, info

    def sample_action(self) -> np.ndarray:
        """Sample an exploratory action from the environment-owned RNG."""
        return self.action_space.sample(self._rng)

    def step(self, action):
        baseline, commanded = self._candidate_torque(action)
        return self.step_torque(commanded, baseline=baseline)
