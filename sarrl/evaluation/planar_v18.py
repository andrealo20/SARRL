"""Prospective penalty ablation, with crossed paired operational intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .planar_v17 import PairedInterval

CONDITIONS = {"P0_inloop_reference": -500.0, "P1_inloop_half_penalty": -250.0}
METRICS = (
    "success",
    "unsafe_episode",
    "safety_infeasible",
    "normalized_violation_integral",
    "unsafe_state_fraction",
    "safety_intervention_fraction",
    "safety_correction_mean",
)


@dataclass(frozen=True)
class Protocol:
    engineering: bool = False

    @property
    def seeds(self):
        return (99018, 99019) if self.engineering else (30, 31, 32, 33, 34)

    @property
    def steps(self):
        return 1000 if self.engineering else 200_000

    @property
    def warmup(self):
        return 32 if self.engineering else 5000

    @property
    def validate_every(self):
        return 500 if self.engineering else 25_000

    @property
    def checkpoint_every(self):
        return 500 if self.engineering else 50_000

    @property
    def validation(self):
        return {
            "every": self.validate_every,
            "episodes": 2 if self.engineering else 30,
            "seed": 9901800 if self.engineering else 640000,
        }

    @property
    def evaluation(self):
        return (
            {
                "id_reference": (9901810, 2),
                "ood_compound": (9901812, 2),
                "motor_fault": (9901814, 2),
            }
            if self.engineering
            else {
                "id_reference": (650000, 500),
                "ood_compound": (651000, 100),
                "motor_fault": (652000, 100),
            }
        )

    def to_dict(self):
        return {
            "campaign": "penalty_ablation_v18",
            "engineering": self.engineering,
            "conditions": CONDITIONS,
            "seeds": list(self.seeds),
            "steps": self.steps,
            "warmup": self.warmup,
            "validation": self.validation,
            "evaluation": {k: list(v) for k, v in self.evaluation.items()},
            "checkpoint_every": self.checkpoint_every,
            "validation_penalty": -500.0,
            "heldout_abort_reward": 0.0,
            "replay_action": "raw_normalized_policy_action",
            "bootstrap": {
                "type": "crossed_paired",
                "samples": 20000,
                "seeds": [180000, 180001, 180002],
                "quantile": "linear",
            },
            "decision": {
                "success_gain": 0.03,
                "positive_seed_pairs": 4,
                "unsafe_margin": 0.02,
                "abort_margin": 0.01,
            },
            "late_training_end_step_window": [int(0.75 * self.steps), self.steps],
            "diagnostic": {"draws": 4, "seed_offset": 181000, "batch": 4096},
        }


def crossed_intervals(effects: np.ndarray, *, seed: int, samples: int = 20000):
    """Resample row and shared column indices, together for all metrics.

    Input shape: training seeds, episode seeds, metrics. The bootstrap is an
    operational uncertainty estimate, not a calibrated type-I error guarantee.
    """
    matrix = np.asarray(effects, dtype=np.float64)
    if matrix.ndim != 3 or min(matrix.shape) < 1 or not np.isfinite(matrix).all():
        raise ValueError("effects must be a finite non-empty three-dimensional matrix")
    if samples <= 0 or seed < 0:
        raise ValueError("invalid bootstrap configuration")
    rng = np.random.Generator(np.random.PCG64(seed))
    nr, nc, nm = matrix.shape
    distribution = np.empty((samples, nm))
    for start in range(0, samples, 64):
        size = min(64, samples - start)
        rows = rng.integers(nr, size=(size, nr))
        cols = rng.integers(nc, size=(size, nc))
        distribution[start : start + size] = matrix[rows[:, :, None], cols[:, None, :], :].mean(
            axis=(1, 2)
        )
    quantiles = np.quantile(distribution, [0.025, 0.975, 0.95], axis=0, method="linear")
    return [
        asdict(PairedInterval(float(matrix[:, :, i].mean()), *map(float, quantiles[:, i])))
        for i in range(nm)
    ]


def classify_result(scenarios, seed_success, late_abort_delta, *, valid=True):
    if not valid:
        return "incomplete"
    if len(seed_success) != 5 or not np.isfinite(seed_success).all():
        raise ValueError("decision requires five finite paired seed effects")
    values = [late_abort_delta]
    for scenario in ("id_reference", "ood_compound", "motor_fault"):
        for metric in ("success", "unsafe_episode", "safety_infeasible"):
            values.extend(scenarios[scenario][metric].values())
    if not np.isfinite(values).all():
        raise ValueError("decision inputs must be finite")
    primary = scenarios["id_reference"]
    veto = late_abort_delta > 0.01 or any(
        row["unsafe_episode"]["difference"] > 0.02 or row["safety_infeasible"]["difference"] > 0.01
        for row in scenarios.values()
    )
    success = primary["success"]
    if success["difference"] <= 0 or veto:
        return "no_go"
    if (
        success["difference"] >= 0.03
        and success["ci95_low"] > 0
        and sum(value > 0 for value in seed_success) >= 4
        and primary["unsafe_episode"]["upper_one_sided_95"] <= 0.02
        and primary["safety_infeasible"]["upper_one_sided_95"] <= 0.01
    ):
        return "advance"
    return "inconclusive"
