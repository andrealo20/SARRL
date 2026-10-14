"""Per-transition forgetting for the Panda delay bank's parameter estimators."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from sarrl.controllers.panda_adaptive import PandaRLS
from sarrl.controllers.panda_score_memory import PandaScoreMemoryConfig, PandaScoreMemoryController
from sarrl.robots.panda import vector7


@dataclass(frozen=True)
class PandaEstimatorMemoryConfig:
    memory_steps: int | None = 50

    def validate(self):
        if self.memory_steps is not None and (
            type(self.memory_steps) is not int or self.memory_steps < 2
        ):
            raise ValueError(
                "parameter memory must be None or an integer of at least two intervals"
            )

    @property
    def decay(self):
        return (
            1.0 if self.memory_steps is None else (self.memory_steps - 1) / (self.memory_steps + 1)
        )


class PandaForgettingRLS(PandaRLS):
    """Inflate inverse information once before all seven observations in an interval."""

    def __init__(self, config=None, memory_config=None):
        self.memory_config = memory_config or PandaEstimatorMemoryConfig()
        self.memory_config.validate()
        super().__init__(config)

    def update(self, rows, residual):
        design = np.asarray(rows, dtype=np.float64)
        values = vector7(residual, "torque residual")
        if design.shape != (7, 8) or not np.all(np.isfinite(design)):
            raise ValueError("regressor must be finite with shape (7, 8)")
        fields = ("parameters", "covariance", "information", "innovation_squared")
        saved = {name: getattr(self, name).copy() for name in fields}
        counts = self.updates, self.bound_hits
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if self.memory_config.decay != 1.0:
                    self.covariance /= self.memory_config.decay
                    self.information *= self.memory_config.decay
                super().update(design, values)
                if not all(np.all(np.isfinite(getattr(self, name))) for name in fields):
                    raise FloatingPointError("nonfinite parameter-memory update")
        except FloatingPointError:
            for name, value in saved.items():
                setattr(self, name, value)
            self.updates, self.bound_hits = counts
            raise

    def diagnostics(self):
        return {
            **super().diagnostics(),
            "parameter_memory": asdict(self.memory_config),
            "parameter_decay": self.memory_config.decay,
        }


class PandaJointMemoryController(PandaScoreMemoryController):
    """Keep exponential score memory fixed; change only candidate parameter memory."""

    def __init__(self, memory_config=None):
        super().__init__(PandaScoreMemoryConfig("exponential", 50))
        self.parameter_memory = memory_config or PandaEstimatorMemoryConfig()
        self.parameter_memory.validate()
        self.estimators = [
            PandaForgettingRLS(estimator.config, self.parameter_memory)
            for estimator in self.estimators
        ]
        self.estimator = self.estimators[self.selected]
        self.dynamics.estimator = self.estimator

    def delay_diagnostics(self):
        return {
            **super().delay_diagnostics(),
            "parameter_memory": asdict(self.parameter_memory),
            "parameter_decay": self.parameter_memory.decay,
            "candidate_information_ranks": [
                int(np.linalg.matrix_rank(estimator.information)) for estimator in self.estimators
            ],
        }
