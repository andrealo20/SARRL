"""Statistics and canonical records for uncertainty-gate calibration."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .planar_v15 import (
    V15_PHASE_A_BOOTSTRAP_SAMPLES,
    V15_PHASE_A_BOOTSTRAP_SEED,
    V15_PHASE_A_MIN_COMMON_EPISODES,
    V15_PHASE_A_MIN_FINITE_PAIRS,
    V15_PHASE_A_SCREENING_THRESHOLD,
)


@dataclass(frozen=True)
class PhaseAEpisode:
    policy: str
    training_seed: int
    ensemble_seed: int
    episode_seed: int
    attempted_pairs: int
    retained_pairs: int
    excluded_nonfinite_pairs: int
    terminated: bool
    truncated: bool
    qualifies: bool
    zero_variance: bool
    spearman_rho: float | None
    uncertainty_median: float | None
    error_median: float | None


def spearman_or_zero(x, y) -> tuple[float, bool]:
    """Return Spearman rho, retaining constant inputs as conservative zeroes."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.ndim != 1 or ya.ndim != 1 or xa.size != ya.size or xa.size == 0:
        raise ValueError("Spearman inputs must be aligned non-empty vectors")
    if not np.all(np.isfinite(xa)) or not np.all(np.isfinite(ya)):
        raise ValueError("Spearman inputs must be finite")
    zero_variance = bool(np.ptp(xa) == 0.0 or np.ptp(ya) == 0.0)
    if zero_variance:
        return 0.0, True
    rho = float(np.corrcoef(rankdata(xa), rankdata(ya))[0, 1])
    if not math.isfinite(rho):
        raise FloatingPointError("non-finite Spearman correlation")
    return rho, False


def summarize_episode(
    *,
    policy: str,
    training_seed: int,
    ensemble_seed: int,
    episode_seed: int,
    uncertainty,
    error,
    attempted_pairs: int,
    terminated: bool,
    truncated: bool,
) -> PhaseAEpisode:
    """Apply the frozen finite-pair and constant-variable episode rules."""
    u = np.asarray(uncertainty, dtype=np.float64)
    e = np.asarray(error, dtype=np.float64)
    if u.shape != e.shape or u.ndim != 1:
        raise ValueError("uncertainty and error must be aligned vectors")
    if attempted_pairs < u.size:
        raise ValueError("attempted pair count cannot be smaller than retained pairs")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(e)):
        raise ValueError("retained episode pairs must be finite")
    qualifies = u.size >= V15_PHASE_A_MIN_FINITE_PAIRS
    rho: float | None = None
    zero_variance = False
    uncertainty_median: float | None = None
    error_median: float | None = None
    if qualifies:
        rho, zero_variance = spearman_or_zero(u, e)
        uncertainty_median = float(np.median(u))
        error_median = float(np.median(e))
    return PhaseAEpisode(
        policy=policy,
        training_seed=training_seed,
        ensemble_seed=ensemble_seed,
        episode_seed=episode_seed,
        attempted_pairs=attempted_pairs,
        retained_pairs=int(u.size),
        excluded_nonfinite_pairs=attempted_pairs - int(u.size),
        terminated=terminated,
        truncated=truncated,
        qualifies=qualifies,
        zero_variance=zero_variance,
        spearman_rho=rho,
        uncertainty_median=uncertainty_median,
        error_median=error_median,
    )


def _cell_key(row: PhaseAEpisode) -> tuple[str, int, int]:
    return row.policy, row.training_seed, row.ensemble_seed


def analyze_phase_a(
    episodes: list[PhaseAEpisode],
    *,
    bootstrap_samples: int = V15_PHASE_A_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = V15_PHASE_A_BOOTSTRAP_SEED,
) -> dict:
    """Compute the frozen common-seed cell estimates and screening decision."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    groups: dict[tuple[str, int, int], dict[int, PhaseAEpisode]] = {}
    for row in episodes:
        cell = _cell_key(row)
        if row.episode_seed in groups.setdefault(cell, {}):
            raise ValueError(f"duplicate episode in cell {cell}: {row.episode_seed}")
        groups[cell][row.episode_seed] = row
    if len(groups) != 10:
        raise ValueError(f"Phase A requires exactly 10 cells, found {len(groups)}")

    qualifying_sets = [
        {seed for seed, row in rows.items() if row.qualifies} for rows in groups.values()
    ]
    common = sorted(set.intersection(*qualifying_sets))
    cell_rows = []
    rho_matrix = []
    for cell in sorted(groups):
        rows = groups[cell]
        selected = [rows[seed] for seed in common]
        rho = np.asarray([row.spearman_rho for row in selected], dtype=np.float64)
        if rho.size and not np.all(np.isfinite(rho)):
            raise FloatingPointError("qualifying episode has non-finite rho")
        nonconstant = np.asarray(
            [row.spearman_rho for row in selected if not row.zero_variance],
            dtype=np.float64,
        )
        between_rho = None
        between_zero_variance = False
        if selected:
            between_rho, between_zero_variance = spearman_or_zero(
                [row.uncertainty_median for row in selected],
                [row.error_median for row in selected],
            )
        cell_rows.append(
            {
                "policy": cell[0],
                "training_seed": cell[1],
                "ensemble_seed": cell[2],
                "qualifying_episodes": sum(row.qualifies for row in rows.values()),
                "common_episodes": len(common),
                "zero_variance_episodes": sum(row.zero_variance for row in selected),
                "cell_median_spearman_rho": float(np.median(rho)) if rho.size else None,
                "sensitivity_median_excluding_zero_variance": (
                    float(np.median(nonconstant)) if nonconstant.size else None
                ),
                "between_episode_medians_spearman_rho": between_rho,
                "between_episode_zero_variance": between_zero_variance,
            }
        )
        rho_matrix.append(rho)

    payload = {
        "common_episode_seeds": common,
        "common_episode_count": len(common),
        "minimum_common_episode_count": V15_PHASE_A_MIN_COMMON_EPISODES,
        "cells": cell_rows,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "screening_threshold": V15_PHASE_A_SCREENING_THRESHOLD,
        "scope": "conditional_on_five_frozen_artifact_pairs",
    }
    if len(common) < V15_PHASE_A_MIN_COMMON_EPISODES:
        payload.update(
            {
                "target_median_rho": None,
                "ci95_low": None,
                "ci95_high": None,
                "decision": "inconclusive",
                "decision_reason": "fewer_than_90_global_common_qualifying_episodes",
            }
        )
        return payload

    matrix = np.stack(rho_matrix, axis=0)
    cell_estimates = np.median(matrix, axis=1)
    target = float(np.median(cell_estimates))
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(0, len(common), size=(bootstrap_samples, len(common)))
    sampled_cell_medians = np.median(matrix[:, draws], axis=2)
    distribution = np.median(sampled_cell_medians, axis=0)
    low, high = np.quantile(distribution, [0.025, 0.975])
    if low >= V15_PHASE_A_SCREENING_THRESHOLD:
        decision = "proceed_phase_b"
    elif high < V15_PHASE_A_SCREENING_THRESHOLD:
        decision = "retire_gate"
    else:
        decision = "inconclusive"
    payload.update(
        {
            "target_median_rho": target,
            "ci95_low": float(low),
            "ci95_high": float(high),
            "decision": decision,
            "decision_reason": "preregistered_heuristic_screen",
        }
    )
    return payload


def write_dataclass_csv(path: str | Path, rows: list[PhaseAEpisode]) -> None:
    """Write episode summaries with a stable schema, including empty outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(PhaseAEpisode.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
