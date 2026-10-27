"""Cumulative Panda alignment with an estimated-model joint projection."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict

import numpy as np

from sarrl.controllers.panda_delay import PandaDelayController
from sarrl.robots.panda import vector7
from sarrl.safety.panda_joint_filter import (
    PandaJointFilter,
    PandaJointFilterConfig,
    PandaProjectionResult,
)


def selected_age_set(scores, lags):
    """Ages whose accumulated score equals the minimum exactly; empty if any score is not finite."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(lags),) or not np.all(np.isfinite(scores)):
        return []
    return [int(lag) for lag, score in zip(lags, scores, strict=True) if score == scores.min()]


class PandaFilteredController(PandaDelayController):
    """Retain causal RLS; latch bounded model braking after a failed prediction/QP."""

    def __init__(self, filter_config=None, selected_ages=False):
        super().__init__(predict=False)
        self.selected_ages = bool(selected_ages)
        self.age_sets = []
        cfg = filter_config or PandaJointFilterConfig()
        ranges = self.regressor.nominal.model.jnt_range
        self.joint_filter = PandaJointFilter(
            ranges[:, 0], ranges[:, 1], self.dynamics.torque_limit, cfg
        )
        self.aborted = False
        self.abort_step = None
        self.attempts = self.projected_steps = self.braking_steps = 0
        self.reasons = Counter()
        self.correction_sum = self.correction_max = 0.0
        self.last_filter = None
        self.last_nominal_command = None

    def command(self, q, qd, q_ref, qd_ref=None, qdd_ref=None):
        q, qd = vector7(q, "q"), vector7(qd, "qd")
        nominal = super().command(q, qd, q_ref, qd_ref, qdd_ref)
        self.last_nominal_command = nominal.copy()
        theta = self.estimator.parameters.copy()
        bias = self.regressor.inverse_dynamics(q, qd, np.zeros(7), theta)
        mass = self.regressor.mass_matrix(q, theta)
        np.linalg.cholesky(mass)
        B = np.linalg.solve(mass, np.eye(7))
        drift = -B @ bias
        ages = None
        if self.selected_ages:
            ages = selected_age_set(self.scores, self.lags)
            self.age_sets.append(ages)
        if not self.aborted:
            if self.selected_ages and not ages:
                limit = self.joint_filter.limit
                result = PandaProjectionResult(
                    np.clip(nominal, -limit, limit), False, "invalid_age_set", 0.0, None, None
                )
            else:
                result = self.joint_filter.filter(
                    q, qd, nominal, B, drift, self.sent_history[-4:], ages
                )
            self.attempts += 1
            self.last_filter = result.diagnostics()
            self.reasons[result.reason] += 1
            self.projected_steps += int(result.reason == "projected")
            self.correction_sum += result.correction_norm_nm
            self.correction_max = max(self.correction_max, result.correction_norm_nm)
            if result.predicted_feasible:
                return result.torque.copy()
            self.aborted = True
            self.abort_step = len(self.selected_history)
        self.braking_steps += 1
        braking = np.clip(
            bias - 20 * mass @ qd, -self.dynamics.torque_limit, self.dynamics.torque_limit
        )
        self.last_filter = {**self.last_filter, "braking": True, "torque": braking.tolist()}
        return braking

    def filter_diagnostics(self):
        diagnostics = {
            "config": asdict(self.joint_filter.config),
            "aborted": self.aborted,
            "abort_step": self.abort_step,
            "attempts": self.attempts,
            "projected_steps": self.projected_steps,
            "braking_steps": self.braking_steps,
            "reasons": dict(self.reasons),
            "correction_sum_nm": self.correction_sum,
            "correction_max_nm": self.correction_max,
            "claim": "estimated frozen-model feasibility; no physical safety certificate",
        }
        if self.selected_ages:
            diagnostics["age_sets"] = [list(ages) for ages in self.age_sets]
            diagnostics["age_set_at_abort"] = (
                None if self.abort_step is None else list(self.age_sets[self.abort_step])
            )
        return diagnostics
