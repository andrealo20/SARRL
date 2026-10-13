"""Finite-memory innovation scoring with the original Panda delay-bank estimators."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from sarrl.controllers.panda_delay import PandaDelayController
from sarrl.robots.panda import vector7


@dataclass(frozen=True)
class PandaScoreMemoryConfig:
    rule: str = "exponential"
    memory_steps: int = 50

    def validate(self):
        if self.rule not in ("exponential", "window"):
            raise ValueError("score rule must be exponential or window")
        if type(self.memory_steps) is not int or self.memory_steps < 2:
            raise ValueError("score memory must be an integer of at least two intervals")

    @property
    def decay(self):
        return (self.memory_steps - 1) / (self.memory_steps + 1)


class InnovationScoreMemory:
    """Sum recent nonnegative pre-update squared innovations for each candidate."""

    def __init__(self, candidates, config=None):
        self.config = config or PandaScoreMemoryConfig()
        self.config.validate()
        if type(candidates) is not int or candidates < 1:
            raise ValueError("candidate count must be a positive integer")
        self.scores = np.zeros(candidates)
        self.recent = deque(maxlen=self.config.memory_steps)
        self.updates = 0

    def update(self, innovations):
        values = np.asarray(innovations, dtype=float)
        if (
            values.shape != self.scores.shape
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
        ):
            raise ValueError("squared innovations must be finite, nonnegative and match candidates")
        if self.config.rule == "exponential":
            self.scores *= self.config.decay
            self.scores += values
        else:
            self.recent.append(values.copy())
            self.scores[:] = np.sum(self.recent, axis=0)
        self.updates += 1


class PandaScoreMemoryController(PandaDelayController):
    """Change score memory only; retain the original per-candidate RLS updates."""

    def __init__(self, score_config=None):
        super().__init__(predict=False)
        self.score_memory = InnovationScoreMemory(len(self.lags), score_config)
        self.scores = self.score_memory.scores
        self.innovation_history = []

    def observe_transition(self, q, qd, q_next, qd_next, sent_command, dt):
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
            0.5 * (q + q_next), 0.5 * (qd + qd_next), (qd_next - qd) / dt
        )
        self.sent_history.append(sent.copy())
        innovations = np.empty(len(self.lags))
        for index, (lag, estimator) in enumerate(zip(self.lags, self.estimators, strict=True)):
            history_index = len(self.sent_history) - 1 - lag
            aligned = self.sent_history[history_index] if history_index >= 0 else np.zeros(7)
            residual = aligned - nominal
            innovation = residual - rows @ estimator.parameters
            innovations[index] = float(innovation @ innovation)
            estimator.update(rows, residual)
        self.score_memory.update(innovations)
        self.innovation_history.append(innovations.tolist())
        selected = int(np.argmin(self.scores))
        self.selection_switches += int(selected != self.selected)
        self.selected = selected
        self.estimator = self.estimators[selected]
        self.dynamics.estimator = self.estimator
        self.selected_history.append(self.selected_delay_steps)

    def control_diagnostics(self):
        return {**super().control_diagnostics(), "delay_scores": self.scores.tolist()}

    def delay_diagnostics(self):
        return {
            **super().delay_diagnostics(),
            "score_memory": asdict(self.score_memory.config),
            "score_decay": self.score_memory.config.decay
            if self.score_memory.config.rule == "exponential"
            else None,
            "score_updates": self.score_memory.updates,
        }
