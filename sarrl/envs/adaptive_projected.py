"""Residual training environment whose nominal and certificate are identified online.

The wrapper composes, inside every transition, the stack the v2.0 campaign
evaluated: the adaptive nominal controller computes the baseline torque from
its current estimate, the residual policy adds a bounded correction, the
HOCBF filter projects the sum against the estimated model, the plant executes
the projected command and the estimator observes the measured transition.
The residual policy sees the plant observation only; the estimate is not part
of its input.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from sarrl.controllers.adaptive_nominal import AdaptiveNominalConfig, AdaptiveNominalController
from sarrl.dynamics import PlanarArm
from sarrl.safety import HOCBFSafetyFilter, SafetyConfig

from .planar_reach import PlanarReachEnv


def plant_from_state_dict(state: dict, plant: str):
    """Rebuild the plant a checkpoint describes, analytical or MuJoCo."""
    if plant == "mujoco":
        from .mujoco_planar import MujocoPlanarReachEnv

        return MujocoPlanarReachEnv.from_state_dict(state)
    if plant != "analytical":
        raise ValueError(f"unknown plant {plant}")
    return PlanarReachEnv.from_state_dict(state)


class AdaptiveProjectedEnv:
    """Identified nominal + residual action + HOCBF on the estimate, per step."""

    def __init__(
        self,
        env: PlanarReachEnv,
        config: AdaptiveNominalConfig | None = None,
        safety_config: SafetyConfig | None = None,
        *,
        infeasible_reward: float = -500.0,
        intervention_tolerance: float = 1e-9,
        compensate_delay: bool = True,
    ):
        if env.mode != "torque":
            raise ValueError("AdaptiveProjectedEnv drives a torque-mode plant")
        if not np.isfinite(infeasible_reward) or infeasible_reward >= 0.0:
            raise ValueError("infeasible_reward must be finite and negative")
        if not np.isfinite(intervention_tolerance) or intervention_tolerance < 0.0:
            raise ValueError("intervention_tolerance must be finite and non-negative")
        self.env = env
        self.config = config or AdaptiveNominalConfig()
        self.config.validate()
        self.safety_config = safety_config or SafetyConfig()
        self.safety_config.validate()
        expected_limit = np.full(2, env.torque_limit, dtype=np.float64)
        configured_limit = np.asarray(self.safety_config.torque_limit, dtype=np.float64)
        if not np.array_equal(configured_limit, expected_limit):
            raise ValueError("safety and environment torque limits must match")
        if float(self.config.dt) != float(env.dt):
            raise ValueError("estimator and plant control periods must match")

        self.controller = AdaptiveNominalController(PlanarArm(), self.config)
        self.safety_filter = HOCBFSafetyFilter(
            self.controller.estimated_model(), self.safety_config
        )
        self.infeasible_reward = float(infeasible_reward)
        self.intervention_tolerance = float(intervention_tolerance)
        self.compensate_delay = bool(compensate_delay)
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

    # Plant pass-through ------------------------------------------------------

    @property
    def mode(self) -> str:
        return "residual"

    @property
    def plant(self) -> str:
        return "mujoco" if hasattr(self.env, "substeps") else "analytical"

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

    # Transition ---------------------------------------------------------------

    def _safety_info(self, **fields) -> dict:
        info = {
            "safety_infeasible": bool(self.safety_infeasible),
            "safety_command_attempts": int(self.command_attempts),
            "safety_certified_steps": int(self.safety_certified_steps),
            "safety_intervention_steps": int(self.safety_intervention_steps),
            "safety_correction_sum": float(self.safety_correction_sum),
            "safety_correction_max": float(self.safety_correction_max),
            "estimator_lag": int(self.controller.lag),
            "estimator_time_constant": float(self.controller.time_constant),
            "estimator_updates": int(self.controller.updates),
            "estimator_model_fallback_steps": int(self.controller.model_fallback_steps),
            "estimator_prediction_fallbacks": int(self.controller.prediction_fallbacks),
        }
        info.update(fields)
        return info

    def reset(self, seed: int | None = None, target=None):
        observation, info = self.env.reset(seed=seed, target=target)
        self.controller.reset()
        self._last_observation = observation.copy()
        self._reset_episode_diagnostics()
        return observation, info

    def sample_action(self) -> np.ndarray:
        return self.env.sample_action()

    def step(self, action):
        if self._last_observation is None:
            raise RuntimeError("reset must be called before step")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite vector of shape (2,)")
        action = np.clip(action, -1.0, 1.0)

        # The measured state the controller, filter and estimator all see.
        measured = np.asarray(self.env._sensed_state(), dtype=np.float64)
        self.controller.begin_step(measured[:2])
        state = self.controller.predict_state(measured) if self.compensate_delay else measured
        baseline = self.controller.command(state[:2], state[2:], self.env.q_des)
        raw_residual = action * self.env.residual_limit
        candidate = baseline + raw_residual
        result = self.safety_filter.filter(state, candidate)
        self.command_attempts += 1
        self.safety_correction_sum += result.correction_norm
        self.safety_correction_max = max(self.safety_correction_max, result.correction_norm)
        self.safety_intervention_steps += int(result.correction_norm > self.intervention_tolerance)
        common = {
            "policy_action": action.astype(np.float32),
            "raw_residual": raw_residual.astype(np.float32),
            "candidate_torque": candidate.astype(np.float32),
            "projected_torque": np.asarray(result.torque, dtype=np.float32),
            "safety_correction": float(result.correction_norm),
            "safety_min_margin": float(result.min_margin),
            "safety_active_constraints": tuple(result.active_constraints),
            "safety_current_safe": bool(result.current_safe),
            "safety_intervened": bool(result.correction_norm > self.intervention_tolerance),
            "baseline_torque": baseline.astype(np.float32),
        }

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
            info.update(self._safety_info(safety_certified=False, **common))
            return self._last_observation.copy(), self.infeasible_reward, True, False, info

        self.safety_certified_steps += 1
        observation, reward, terminated, truncated, info = self.env.step_torque(
            result.torque, baseline=baseline
        )
        after = np.asarray(self.env._sensed_state(), dtype=np.float64)
        self.controller.observe(measured, after, result.torque)
        self._last_observation = observation.copy()
        info.update(self._safety_info(safety_certified=True, **common))
        return observation, reward, terminated, truncated, info

    # Checkpointing ------------------------------------------------------------

    def constructor_config(self) -> dict:
        return {
            "environment_type": "adaptive_projected",
            "plant": self.plant,
            "base_environment": self.env.constructor_config(),
            "adaptive_config": asdict(self.config),
            "safety_config": asdict(self.safety_config),
            "infeasible_reward": self.infeasible_reward,
            "intervention_tolerance": self.intervention_tolerance,
            "compensate_delay": self.compensate_delay,
        }

    def state_dict(self) -> dict:
        return {
            "environment_type": "adaptive_projected",
            "constructor_config": self.constructor_config(),
            "environment": self.env.state_dict(),
            "controller": self.controller.state_dict(),
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
    def from_state_dict(cls, state: dict) -> AdaptiveProjectedEnv:
        if state.get("environment_type") != "adaptive_projected":
            raise ValueError("not an adaptive-projected environment checkpoint")
        config = dict(state["constructor_config"])
        config.pop("environment_type", None)
        config.pop("base_environment")
        env = plant_from_state_dict(state["environment"], config.pop("plant"))
        adaptive = _adaptive_config(config.pop("adaptive_config"))
        safety_config = SafetyConfig(**dict(config.pop("safety_config")))
        wrapped = cls(env, adaptive, safety_config=safety_config, **config)
        wrapped.load_state_dict(state)
        return wrapped

    def load_state_dict(self, state: dict) -> None:
        if state.get("environment_type") != "adaptive_projected":
            raise ValueError("not an adaptive-projected environment checkpoint")
        if state.get("constructor_config") != self.constructor_config():
            raise ValueError(
                "adaptive-projected checkpoint constructor configuration does not match"
            )
        self.env.load_state_dict(state["environment"])
        self.controller.load_state_dict(state["controller"])
        last_observation = state.get("last_observation")
        if last_observation is None:
            self._last_observation = None
        else:
            value = np.asarray(last_observation, dtype=np.float32)
            if value.shape != self.observation_space.shape or not np.all(np.isfinite(value)):
                raise ValueError("invalid adaptive-projected last observation")
            self._last_observation = value.copy()
        diagnostics = dict(state["episode_diagnostics"])
        self.command_attempts = int(diagnostics["command_attempts"])
        self.safety_certified_steps = int(diagnostics["safety_certified_steps"])
        self.safety_intervention_steps = int(diagnostics["safety_intervention_steps"])
        self.safety_infeasible = bool(diagnostics["safety_infeasible"])
        self.safety_correction_sum = float(diagnostics["safety_correction_sum"])
        self.safety_correction_max = float(diagnostics["safety_correction_max"])


def _adaptive_config(data: dict) -> AdaptiveNominalConfig:
    """Rebuild the estimator configuration from its dataclass dictionary."""
    fields = {}
    for name, value in dict(data).items():
        fields[name] = tuple(value) if isinstance(value, list) else value
    return AdaptiveNominalConfig(**fields)
