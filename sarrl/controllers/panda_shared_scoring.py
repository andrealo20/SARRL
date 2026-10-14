"""Score delay hypotheses with one causally acquired physical parameter vector."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from sarrl.controllers.panda_score_memory import PandaScoreMemoryController
from sarrl.robots.panda import vector7


@dataclass(frozen=True)
class PandaSharedScoreConfig:
    bootstrap_steps: int = 50
    stable_steps: int = 25

    def validate(self):
        if (
            type(self.bootstrap_steps) is not int
            or self.bootstrap_steps < 2
            or type(self.stable_steps) is not int
            or self.stable_steps < 1
        ):
            raise ValueError("bootstrap and stability intervals must be positive integers")


class PandaSharedScoreController(PandaScoreMemoryController):
    """Change scoring parameters only; keep all candidate control RLS updates."""

    def __init__(self, shared_config=None):
        self.shared_config = shared_config or PandaSharedScoreConfig()
        self.shared_config.validate()
        super().__init__()
        self.stable_updates = 0
        self.shared_parameters = None
        self.calibrated_at_update = None
        self.calibrated_delay_steps = None
        self.calibration_information_rank = None
        self.scoring_parameter_history = []

    def observe_transition(self, q, qd, q_next, qd_next, sent_command, dt):
        if self.shared_parameters is None:
            previous = self.selected_delay_steps
            super().observe_transition(q, qd, q_next, qd_next, sent_command, dt)
            self.scoring_parameter_history.append(None)
            self.stable_updates = (
                self.stable_updates + 1 if previous == self.selected_delay_steps else 1
            )
            rank = int(np.linalg.matrix_rank(self.estimator.information))
            if (
                self.score_memory.updates >= self.shared_config.bootstrap_steps
                and self.stable_updates >= self.shared_config.stable_steps
                and rank == 8
            ):
                self.shared_parameters = self.estimator.parameters.copy()
                self.shared_parameters.setflags(write=False)
                self.calibrated_at_update = self.score_memory.updates
                self.calibrated_delay_steps = self.selected_delay_steps
                self.calibration_information_rank = rank
            return
        if not np.isfinite(dt) or not np.isclose(
            dt, self.delay_config.control_dt, rtol=0, atol=1e-12
        ):
            raise ValueError("transition interval must match the configured control period")
        q, qd = vector7(q, "q"), vector7(qd, "qd")
        q_next, qd_next = vector7(q_next, "q_next"), vector7(qd_next, "qd_next")
        sent = vector7(sent_command, "sent_command")
        if np.any(np.abs(sent) > self.dynamics.torque_limit + 1e-12):
            raise ValueError("the command history requires sent clipped commands")
        nominal, rows = self.regressor.evaluate(
            (q + q_next) / 2, (qd + qd_next) / 2, (qd_next - qd) / dt
        )
        if not np.all(np.isfinite(nominal)) or not np.all(np.isfinite(rows)):
            raise ValueError("shared scoring requires a finite regressor")
        self.sent_history.append(sent.copy())
        innovations = np.empty(len(self.lags))
        for index, (lag, estimator) in enumerate(zip(self.lags, self.estimators, strict=True)):
            history_index = len(self.sent_history) - 1 - lag
            aligned = self.sent_history[history_index] if history_index >= 0 else np.zeros(7)
            residual = aligned - nominal
            innovation = residual - rows @ self.shared_parameters
            innovations[index] = float(innovation @ innovation)
            estimator.update(rows, residual)
        self.score_memory.update(innovations)
        self.innovation_history.append(innovations.tolist())
        self.scoring_parameter_history.append(self.shared_parameters.tolist())
        selected = int(np.argmin(self.scores))
        self.selection_switches += int(selected != self.selected)
        self.selected = selected
        self.estimator = self.estimators[selected]
        self.dynamics.estimator = self.estimator
        self.selected_history.append(self.selected_delay_steps)

    def delay_diagnostics(self):
        return {
            **super().delay_diagnostics(),
            "shared_score_config": asdict(self.shared_config),
            "shared_scoring_active": self.shared_parameters is not None,
            "shared_parameters": None
            if self.shared_parameters is None
            else self.shared_parameters.tolist(),
            "calibrated_at_update": self.calibrated_at_update,
            "calibrated_delay_steps": self.calibrated_delay_steps,
            "calibration_information_rank": self.calibration_information_rank,
            "candidate_control_parameters": "unchanged cumulative bounded RLS",
        }
