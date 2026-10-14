"""Residual training on the identified nominal with the retained covariance margin."""

from __future__ import annotations

import numpy as np

from sarrl.envs.adaptive_projected import AdaptiveProjectedEnv
from sarrl.safety import MarginHOCBFSafetyFilter


class AdaptiveMarginResidualEnv(AdaptiveProjectedEnv):
    def __init__(self, env, config=None, safety_config=None, *, certificate_kappa=0.25, **kwargs):
        if not np.isfinite(certificate_kappa) or certificate_kappa < 0:
            raise ValueError("certificate kappa must be finite and nonnegative")
        super().__init__(env, config, safety_config, **kwargs)
        self.certificate_kappa = float(certificate_kappa)
        self.safety_filter = MarginHOCBFSafetyFilter(
            self.controller.estimated_model(),
            self.safety_config,
            kappa=self.certificate_kappa,
            uncertainty=self.controller.command_uncertainty,
        )

    def reset(self, seed=None, target=None):
        result = super().reset(seed=seed, target=target)
        self.safety_filter.reset_counters()
        return result

    def constructor_config(self):
        return {
            **super().constructor_config(),
            "environment_type": "adaptive_margin_residual",
            "certificate_kappa": self.certificate_kappa,
        }

    def state_dict(self):
        margin = self.safety_filter
        return {
            **super().state_dict(),
            "environment_type": "adaptive_margin_residual",
            "margin_counters": {
                "margin_steps": margin.margin_steps,
                "reduced_steps": margin.reduced_steps,
                "dropped_steps": margin.dropped_steps,
                "applied": margin.applied.copy(),
            },
        }

    def load_state_dict(self, state):
        if state.get("environment_type") != "adaptive_margin_residual":
            raise ValueError("not an adaptive-margin residual checkpoint")
        super().load_state_dict({**state, "environment_type": "adaptive_projected"})
        counters = state["margin_counters"]
        for name in ("margin_steps", "reduced_steps", "dropped_steps"):
            value = counters[name]
            if type(value) is not int or value < 0:
                raise ValueError("margin counters must be nonnegative integers")
            setattr(self.safety_filter, name, value)
        applied = np.asarray(counters["applied"], dtype=float)
        if applied.shape != (2,) or not np.all(np.isfinite(applied)) or np.any(applied < 0):
            raise ValueError("applied margins must be a finite nonnegative pair")
        self.safety_filter.applied = applied.copy()
