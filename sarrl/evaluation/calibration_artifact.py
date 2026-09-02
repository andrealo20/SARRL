"""Construction helpers for the versioned v1.5 gate-calibration artifact."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .gate_calibration import PhaseAEpisode
from .planar_v15 import (
    V15_PHASE_A_EPISODES,
    V15_PHASE_A_POLICIES,
    V15_PHASE_A_TRAINING_SEEDS,
)


def derive_reference_uncertainties(episodes: list[PhaseAEpisode]) -> list[dict]:
    """Derive per-ensemble u_ref from 200 equally weighted episode medians."""
    output = []
    for seed in V15_PHASE_A_TRAINING_SEEDS:
        medians = []
        counts = {}
        for policy in V15_PHASE_A_POLICIES:
            selected = sorted(
                (
                    row
                    for row in episodes
                    if row.policy == policy
                    and row.training_seed == seed
                    and row.ensemble_seed == seed
                ),
                key=lambda row: row.episode_seed,
            )
            if len(selected) != V15_PHASE_A_EPISODES:
                raise ValueError(f"{policy} seed {seed} does not have exactly 100 episodes")
            # With zero exclusions, scale eligibility is independent of error:
            # every attempted uncertainty was finite and is represented here.
            if any(row.excluded_nonfinite_pairs != 0 for row in selected):
                raise ValueError(
                    "cannot derive error-independent u_ref when Phase A omitted pairs"
                )
            if any(
                row.attempted_pairs < 10
                or row.uncertainty_median is None
                or not math.isfinite(row.uncertainty_median)
                for row in selected
            ):
                raise ValueError(f"{policy} seed {seed} has a scale-ineligible episode")
            counts[policy] = len(selected)
            medians.extend(float(row.uncertainty_median) for row in selected)
        reference = float(np.median(np.asarray(medians, dtype=np.float64)))
        if not math.isfinite(reference) or reference <= 0.0:
            raise ValueError(f"ensemble seed {seed} produced invalid u_ref")
        output.append(
            {
                "training_seed": seed,
                "ensemble_seed": seed,
                "a2_scale_eligible_episodes": counts["A2"],
                "a3_scale_eligible_episodes": counts["A3"],
                "pooled_episode_medians": len(medians),
                "u_ref": reference,
            }
        )
    return output


def canonical_float(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("canonical artifacts forbid non-finite floats")
    return "0" if number == 0.0 else format(number, ".17g")


def canonical_json(value) -> str:
    """Serialize the Phase-B JSON subset with its frozen number convention."""
    if value is None:
        return "null"
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return canonical_float(float(value))
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return "{" + ",".join(
            canonical_json(key) + ":" + canonical_json(value[key])
            for key in sorted(value)
        ) + "}"
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def write_canonical_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(canonical_json(payload) + "\n", encoding="utf-8", newline="\n")


def _csv_scalar(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return canonical_float(float(value))
    return str(value)


def write_canonical_csv(path: str | Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for row in rows:
            if tuple(row) != fields:
                raise ValueError("row does not match frozen canonical CSV schema")
            writer.writerow({key: _csv_scalar(value) for key, value in row.items()})
