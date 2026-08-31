"""Tiny finite Box-space abstraction so the analytical core has no Gym dependency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoxSpace:
    low: np.ndarray
    high: np.ndarray

    def __post_init__(self) -> None:
        low = np.asarray(self.low, dtype=np.float32)
        high = np.asarray(self.high, dtype=np.float32)
        if low.shape != high.shape or low.ndim != 1:
            raise ValueError("BoxSpace requires one-dimensional matching bounds")
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("BoxSpace bounds must be finite")
        if np.any(high <= low):
            raise ValueError("every high bound must exceed its low bound")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.low.shape

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high).astype(np.float32)
