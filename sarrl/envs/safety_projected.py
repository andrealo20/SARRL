"""Training environment with a required HOCBF torque projection."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from sarrl.safety import HOCBFSafetyFilter, SafetyConfig

from .planar_reach import PlanarReachEnv


class SafetyProjectedEnv:
    """Apply the deployment HOCBF inside the residual-policy transition."""

    def __init__(
        self,
        env: PlanarReachEnv,
        safety_config: SafetyConfig | None = None,
        *,
        infeasible_reward: float = -500.0,
        intervention_tolerance: float = 1e-9,
    ):
        if env.mode != "residual":
            raise ValueError("SafetyProjectedEnv requires a residual-mode environment")
        if not np.isfinite(infeasible_reward) or infeasible_reward >= 0.0:
            raise ValueError("infeasible_reward must be finite and negative")
        if not np.isfinite(intervention_tolerance) or intervention_tolerance < 0.0:
            raise ValueError("intervention_tolerance must be finite and non-negative")

        self.env = env
        self.safety_config = safety_config or SafetyConfig()
        self.safety_config.validate()
        expected_limit = np.full(2, env.torque_limit, dtype=np.float64)
        configured_limit = np.asarray(self.safety_config.torque_limit, dtype=np.float64)
        if not np.array_equal(configured_limit, expected_limit):
            raise ValueError("safety and environment torque limits must match")

        self.safety_filter = HOCBFSafetyFilter(env.nominal_arm, self.safety_config)
        self.infeasible_reward = float(infeasible_reward)
        self.intervention_tolerance = float(intervention_tolerance)
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._last_observation: np.ndarray | None = None
        self._reset_episode_diagnostics()

    def _reset_episode_diagnostics(self) -> None:
        self.command_attempts = 0
        self.safety_certified_steps = 0
        self.safety_intervention_steps = 0
        self.safety_infeasible = False
        self.safety_correction_sum = 0.0
        self.safety_correction_max = 0.0

    @property
    def mode(self) -> str:
        return self.env.mode

    @property
    def dt(self) -> float:
        return self.env.dt

    @property
    def max_steps(self) -> int:
        return self.env.max_steps

    @property
    def torque_limit(self) -> float:
        return self.env.torque_limit

    @property
    def residual_limit(self) -> float:
        return self.env.residual_limit

    @property
    def success_radius(self) -> float:
        return self.env.success_radius

    @property
    def randomization(self):
        return self.env.randomization

    @property
    def fault(self):
        return self.env.fault

    @property
    def state(self) -> np.ndarray:
        return self.env.state

    @property
    def target(self) -> np.ndarray:
        return self.env.target

    @property
    def q_des(self) -> np.ndarray:
        return self.env.q_des

    @property
    def arm(self):
        return self.env.arm

    @property
    def nominal_arm(self):
        return self.env.nominal_arm

    @property
    def steps(self) -> int:
        return self.env.steps

    def _distance(self) -> float:
        position = self.env.arm.forward_kinematics(self.env.state[:2])
        return float(np.linalg.norm(self.env.target - position))

    def _candidate(self, action) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite vector of shape (2,)")
        action = np.clip(action, -1.0, 1.0)
        baseline = self.env.controller.command(
            self.env.state[:2], self.env.state[2:], self.env.q_des
        )
        raw_residual = action * self.env.residual_limit
        return baseline, raw_residual, baseline + raw_residual

    def _safety_info(
        self,
        *,
        action: np.ndarray,
        baseline: np.ndarray,
        raw_residual: np.ndarray,
        candidate: np.ndarray,
        projected: np.ndarray,
        correction: float,
        min_margin: float,
        active_constraints: tuple[int, ...],
        certified: bool,
        current_safe: bool,
    ) -> dict:
        return {
            "policy_action": action.astype(np.float32),
            "raw_residual": raw_residual.astype(np.float32),
            "candidate_torque": candidate.astype(np.float32),
            "projected_torque": projected.astype(np.float32),
            "safety_correction": float(correction),
            "safety_min_margin": float(min_margin),
            "safety_active_constraints": tuple(active_constraints),
            "safety_certified": bool(certified),
            "safety_current_safe": bool(current_safe),
            "safety_infeasible": bool(self.safety_infeasible),
            "safety_intervened": bool(correction > self.intervention_tolerance),
            "safety_command_attempts": int(self.command_attempts),
            "safety_certified_steps": int(self.safety_certified_steps),
            "safety_intervention_steps": int(self.safety_intervention_steps),
            "safety_correction_sum": float(self.safety_correction_sum),
            "safety_correction_max": float(self.safety_correction_max),
            "baseline_torque": baseline.astype(np.float32),
        }

    def reset(self, seed: int | None = None, target=None):
        observation, info = self.env.reset(seed=seed, target=target)
        self._last_observation = observation.copy()
        self._reset_episode_diagnostics()
        return observation, info

    def sample_action(self) -> np.ndarray:
        return self.env.sample_action()

    def step(self, action):
        if self._last_observation is None:
            raise RuntimeError("reset must be called before step")

        clipped_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        baseline, raw_residual, candidate = self._candidate(action)
        result = self.safety_filter.filter(self.env.state, candidate)
        self.command_attempts += 1
        self.safety_correction_sum += result.correction_norm
        self.safety_correction_max = max(
            self.safety_correction_max, result.correction_norm
        )
        self.safety_intervention_steps += int(
            result.correction_norm > self.intervention_tolerance
        )

        if not result.success:
            self.safety_infeasible = True
            info = self.env._info_base()
            info.update(
                {
                    "distance": self._distance(),
                    "success": False,
                    "fault_active": bool(self.env._fault_active),
                    "commanded_torque": candidate.astype(np.float32),
                }
            )
            info.update(
                self._safety_info(
                    action=clipped_action,
                    baseline=baseline,
                    raw_residual=raw_residual,
                    candidate=candidate,
                    projected=result.torque,
                    correction=result.correction_norm,
                    min_margin=result.min_margin,
                    active_constraints=result.active_constraints,
                    certified=False,
                    current_safe=result.current_safe,
                )
            )
            return (
                self._last_observation.copy(),
                self.infeasible_reward,
                True,
                False,
                info,
            )

        self.safety_certified_steps += 1
        observation, reward, terminated, truncated, info = self.env.step_torque(
            result.torque,
            baseline=baseline,
        )
        self._last_observation = observation.copy()
        info.update(
            self._safety_info(
                action=clipped_action,
                baseline=baseline,
                raw_residual=raw_residual,
                candidate=candidate,
                projected=result.torque,
                correction=result.correction_norm,
                min_margin=result.min_margin,
                active_constraints=result.active_constraints,
                certified=True,
                current_safe=result.current_safe,
            )
        )
        return observation, reward, terminated, truncated, info

    def constructor_config(self) -> dict:
        return {
            "environment_type": "safety_projected",
            "base_environment": self.env.constructor_config(),
            "safety_config": asdict(self.safety_config),
            "infeasible_reward": self.infeasible_reward,
            "intervention_tolerance": self.intervention_tolerance,
        }

    def state_dict(self) -> dict:
        return {
            "environment_type": "safety_projected",
            "constructor_config": self.constructor_config(),
            "environment": self.env.state_dict(),
            "last_observation": (
                None if self._last_observation is None else self._last_observation.copy()
            ),
            "episode_diagnostics": {
                "command_attempts": self.command_attempts,
                "safety_certified_steps": self.safety_certified_steps,
                "safety_intervention_steps": self.safety_intervention_steps,
                "safety_infeasible": self.safety_infeasible,
                "safety_correction_sum": self.safety_correction_sum,
                "safety_correction_max": self.safety_correction_max,
            },
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> SafetyProjectedEnv:
        if state.get("environment_type") != "safety_projected":
            raise ValueError("not a safety-projected environment checkpoint")
        config = dict(state["constructor_config"])
        config.pop("environment_type", None)
        config.pop("base_environment")
        safety_config = SafetyConfig(**dict(config.pop("safety_config")))
        env = PlanarReachEnv.from_state_dict(state["environment"])
        wrapped = cls(env, safety_config=safety_config, **config)
        wrapped.load_state_dict(state)
        return wrapped

    def load_state_dict(self, state: dict) -> None:
        if state.get("environment_type") != "safety_projected":
            raise ValueError("not a safety-projected environment checkpoint")
        if state.get("constructor_config") != self.constructor_config():
            raise ValueError(
                "safety-projected checkpoint constructor configuration does not match"
            )
        self.env.load_state_dict(state["environment"])
        last_observation = state.get("last_observation")
        if last_observation is None:
            self._last_observation = None
        else:
            value = np.asarray(last_observation, dtype=np.float32)
            if value.shape != self.observation_space.shape or not np.all(np.isfinite(value)):
                raise ValueError("invalid safety-projected last observation")
            self._last_observation = value.copy()

        diagnostics = dict(state["episode_diagnostics"])
        self.command_attempts = int(diagnostics["command_attempts"])
        self.safety_certified_steps = int(diagnostics["safety_certified_steps"])
        self.safety_intervention_steps = int(diagnostics["safety_intervention_steps"])
        self.safety_infeasible = bool(diagnostics["safety_infeasible"])
        self.safety_correction_sum = float(diagnostics["safety_correction_sum"])
        self.safety_correction_max = float(diagnostics["safety_correction_max"])
