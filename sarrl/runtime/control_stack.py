"""Composable runtime controller for the complete SARRL planar stack."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sarrl.controllers import ComputedTorqueController
from sarrl.models import ResidualDynamicsEnsemble, UncertaintyGate
from sarrl.safety import HOCBFSafetyFilter


@dataclass(frozen=True)
class ControlStackConfig:
    residual_limit: tuple[float, float] = (8.0, 8.0)
    torque_limit: tuple[float, float] = (40.0, 40.0)
    require_safety: bool = False
    clip_ensemble_query: bool = False

    def validate(self) -> None:
        r = np.asarray(self.residual_limit, dtype=np.float64)
        t = np.asarray(self.torque_limit, dtype=np.float64)
        if r.shape != (2,) or t.shape != (2,) or np.any(r <= 0.0) or np.any(t <= 0.0):
            raise ValueError("runtime residual/torque limits must be positive pairs")
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(t)):
            raise ValueError("runtime limits must be finite")


@dataclass(frozen=True)
class ControlStackResult:
    torque: np.ndarray
    baseline_torque: np.ndarray
    raw_residual: np.ndarray
    gated_residual: np.ndarray
    uncertainty: np.ndarray
    uncertainty_scale: float
    safety_correction: float
    safety_certified: bool
    executable: bool
    ensemble_mean: np.ndarray
    ensemble_query_torque: np.ndarray


class SARRLControlStack:
    """Physics baseline + residual policy + uncertainty gate + hard safety projection."""

    def __init__(
        self,
        baseline: ComputedTorqueController,
        policy,
        config: ControlStackConfig | None = None,
        safety_filter: HOCBFSafetyFilter | None = None,
        dynamics_ensemble: ResidualDynamicsEnsemble | None = None,
        uncertainty_gate: UncertaintyGate | None = None,
        device="cpu",
    ):
        self.baseline = baseline
        self.policy = policy
        self.config = config or ControlStackConfig()
        self.config.validate()
        self.safety_filter = safety_filter
        self.dynamics_ensemble = dynamics_ensemble
        self.uncertainty_gate = uncertainty_gate
        self.device = device
        if (dynamics_ensemble is None) != (uncertainty_gate is None):
            raise ValueError("dynamics ensemble and uncertainty gate must be supplied together")
        if self.config.require_safety and self.safety_filter is None:
            raise ValueError("require_safety=True needs a safety filter")

    def command(self, observation, state, q_des, obstacles=(), deterministic: bool = True):
        state = np.asarray(state, dtype=np.float64)
        q_des = np.asarray(q_des, dtype=np.float64)
        if state.shape != (4,) or q_des.shape != (2,):
            raise ValueError("runtime expects state(4) and q_des(2)")
        baseline = self.baseline.command(state[:2], state[2:], q_des)
        action = np.asarray(
            self.policy.act(observation, deterministic=deterministic), dtype=np.float64
        )
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("policy must return a finite 2-vector")
        action = np.clip(action, -1.0, 1.0)
        raw_residual = action * np.asarray(self.config.residual_limit)

        uncertainty = np.zeros(2, dtype=np.float64)
        ensemble_mean = np.zeros(2, dtype=np.float64)
        probe_torque = baseline + raw_residual
        scale = 1.0
        gated_residual = raw_residual.copy()
        if self.dynamics_ensemble is not None:
            if self.config.clip_ensemble_query:
                limit = np.asarray(self.config.torque_limit)
                probe_torque = np.clip(probe_torque, -limit, limit)
            ensemble_mean, uncertainty = self.dynamics_ensemble.predict(
                state.astype(np.float32), probe_torque.astype(np.float32), device=self.device
            )
            gated_residual, scale = self.uncertainty_gate.apply(raw_residual, uncertainty)

        candidate = baseline + gated_residual
        limit = np.asarray(self.config.torque_limit)
        if self.safety_filter is None:
            torque = np.clip(candidate, -limit, limit)
            return ControlStackResult(
                torque=torque,
                baseline_torque=baseline,
                raw_residual=raw_residual,
                gated_residual=gated_residual,
                uncertainty=np.asarray(uncertainty, dtype=np.float64),
                uncertainty_scale=float(scale),
                safety_correction=0.0,
                safety_certified=False,
                executable=True,
                ensemble_mean=np.asarray(ensemble_mean, dtype=np.float64),
                ensemble_query_torque=np.asarray(probe_torque, dtype=np.float64),
            )

        safety = self.safety_filter.filter(state, candidate, obstacles)
        if not safety.success:
            # The torque field is diagnostic only in this case. Callers must
            # inspect executable; require_safety deliberately has no fallback
            # that could be mistaken for a certified command.
            return ControlStackResult(
                torque=np.clip(candidate, -limit, limit),
                baseline_torque=baseline,
                raw_residual=raw_residual,
                gated_residual=gated_residual,
                uncertainty=np.asarray(uncertainty, dtype=np.float64),
                uncertainty_scale=float(scale),
                safety_correction=safety.correction_norm,
                safety_certified=False,
                executable=not self.config.require_safety,
                ensemble_mean=np.asarray(ensemble_mean, dtype=np.float64),
                ensemble_query_torque=np.asarray(probe_torque, dtype=np.float64),
            )
        return ControlStackResult(
            torque=safety.torque,
            baseline_torque=baseline,
            raw_residual=raw_residual,
            gated_residual=gated_residual,
            uncertainty=np.asarray(uncertainty, dtype=np.float64),
            uncertainty_scale=float(scale),
            safety_correction=safety.correction_norm,
            safety_certified=True,
            executable=True,
            ensemble_mean=np.asarray(ensemble_mean, dtype=np.float64),
            ensemble_query_torque=np.asarray(probe_torque, dtype=np.float64),
        )
