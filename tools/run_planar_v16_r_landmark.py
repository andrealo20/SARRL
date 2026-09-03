#!/usr/bin/env python3
"""Build the required, hash-bound v1.6-R landmark evidence table.

Per PLAN.md ("Predictor: landmark window" / "Reproducibility"): no association
analysis may begin until this artifact exists. This script does exactly one
thing — derive `landmark_episodes.csv` from the local-only raw Phase-C
transition table and the retained gate-off arm — and computes no association
between predictor and endpoint. Run `run_planar_v16_r_analysis.py` after this.

Window: raw rows `step = 0..24` (zero-indexed, 25 transitions), fixed for
every episode regardless of scenario. Any episode with fewer than 25 rows in
that range is an error, not a silent truncation — the frozen feasibility check
established the shortest gate-off episode runs 27 steps, so this cannot occur
on the real arm.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from sarrl.evaluation import assert_repository_import_root, write_run_manifest

CONDITION = "A6c_gate_off_control"
POPULATION = "safety"
WINDOW_FIRST_STEP = 0
WINDOW_LAST_STEP = 24
EXPECTED_ROWS_IN_WINDOW = 25

SCHEMA = (
    "population",
    "condition",
    "training_seed",
    "ensemble_seed",
    "scenario",
    "episode_seed",
    "window_first_step",
    "window_last_step",
    "n_rows_in_window",
    "uncertainty_norm_landmark_median",
    "uncertainty_ratio_landmark_median",
    "unsafe_episode",
    "safety_infeasible",
    "operational_failure",
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_of(row: dict) -> tuple:
    return (
        row["population"],
        row["condition"],
        row["training_seed"],
        row["scenario"],
        row["episode_seed"],
    )


def load_outcomes(path: Path) -> dict:
    """Endpoint components from safety_diagnostics.csv, keyed to the arm."""
    outcomes = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["condition"] != CONDITION or row["population"] != POPULATION:
                continue
            key = (
                row["population"],
                row["condition"],
                row["training_seed"],
                row["scenario"],
                row["seed"],
            )
            outcomes[key] = {
                "unsafe_episode": row["unsafe_episode"],
                "safety_infeasible": row["safety_infeasible"],
            }
    return outcomes


def load_ensemble_seeds(path: Path) -> dict:
    """episode key -> ensemble_seed, from gate_episodes.csv."""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["condition"] != CONDITION or row["population"] != POPULATION:
                continue
            out[key_of(row)] = row["ensemble_seed"]
    return out


def build_landmark_rows(raw_path: Path, outcomes: dict, ensemble_seeds: dict) -> list:
    windows: dict = defaultdict(lambda: {"norm": [], "ratio": []})
    with gzip.open(raw_path, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["condition"] != CONDITION or row["population"] != POPULATION:
                continue
            step = int(row["step"])
            if step < WINDOW_FIRST_STEP or step > WINDOW_LAST_STEP:
                continue
            k = key_of(row)
            windows[k]["norm"].append(float(row["uncertainty_norm"]))
            windows[k]["ratio"].append(float(row["uncertainty_ratio"]))

    if set(windows) != set(outcomes):
        missing_outcome = set(windows) - set(outcomes)
        missing_window = set(outcomes) - set(windows)
        raise RuntimeError(
            f"episode key mismatch between raw transitions and safety_diagnostics: "
            f"{len(missing_outcome)} without an outcome, {len(missing_window)} "
            f"without a window"
        )

    rows = []
    for k in sorted(windows):
        norms = windows[k]["norm"]
        ratios = windows[k]["ratio"]
        if len(norms) != EXPECTED_ROWS_IN_WINDOW:
            raise RuntimeError(
                f"episode {k} has {len(norms)} rows in the window, "
                f"expected exactly {EXPECTED_ROWS_IN_WINDOW}; refusing to "
                f"silently truncate"
            )
        population, condition, training_seed, scenario, episode_seed = k
        unsafe = outcomes[k]["unsafe_episode"].strip().lower() == "true"
        infeasible = outcomes[k]["safety_infeasible"].strip().lower() == "true"
        rows.append(
            {
                "population": population,
                "condition": condition,
                "training_seed": training_seed,
                "ensemble_seed": ensemble_seeds[k],
                "scenario": scenario,
                "episode_seed": episode_seed,
                "window_first_step": WINDOW_FIRST_STEP,
                "window_last_step": WINDOW_LAST_STEP,
                "n_rows_in_window": len(norms),
                "uncertainty_norm_landmark_median": float(np.median(norms)),
                "uncertainty_ratio_landmark_median": float(np.median(ratios)),
                "unsafe_episode": unsafe,
                "safety_infeasible": infeasible,
                "operational_failure": unsafe or infeasible,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the v1.6-R landmark table")
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("results/uncertainty_gate_calibration/phase_c/transitions.csv.gz"),
    )
    parser.add_argument(
        "--gate-episodes",
        type=Path,
        default=Path("results/uncertainty_gate_calibration/phase_c/gate_episodes.csv"),
    )
    parser.add_argument(
        "--safety-diagnostics",
        type=Path,
        default=Path("results/uncertainty_gate_calibration/phase_c/safety_diagnostics.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/uncertainty_gate_calibration/phase_r")
    )
    args = parser.parse_args()

    root = assert_repository_import_root(Path(__file__).resolve().parents[1])
    if not args.raw.exists():
        raise SystemExit(
            f"missing local raw table {args.raw}: it is not committed (106 MiB, "
            f"exceeds GitHub's per-file limit) and must be present on this machine"
        )

    outcomes = load_outcomes(args.safety_diagnostics)
    ensemble_seeds = load_ensemble_seeds(args.gate_episodes)
    rows = build_landmark_rows(args.raw, outcomes, ensemble_seeds)

    expected_episodes = 1500
    if len(rows) != expected_episodes:
        raise RuntimeError(
            f"built {len(rows)} landmark rows, expected exactly {expected_episodes}"
        )

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    landmark_path = out / "landmark_episodes.csv"
    with open(landmark_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SCHEMA)
        writer.writeheader()
        writer.writerows(rows)

    raw_hash = sha256_of(args.raw)
    landmark_hash = sha256_of(landmark_path)

    n_unsafe = sum(1 for r in rows if r["unsafe_episode"])
    n_infeasible = sum(1 for r in rows if r["safety_infeasible"])
    n_composite = sum(1 for r in rows if r["operational_failure"])

    write_run_manifest(
        out / "landmark_manifest.json",
        {
            "campaign": "uncertainty_gate_calibration_phase_r_landmark",
            "condition": CONDITION,
            "population": POPULATION,
            "window_first_step": WINDOW_FIRST_STEP,
            "window_last_step": WINDOW_LAST_STEP,
            "n_rows_in_window": EXPECTED_ROWS_IN_WINDOW,
            "episodes": len(rows),
            "unsafe_episodes": n_unsafe,
            "infeasible_episodes": n_infeasible,
            "operational_failure_episodes": n_composite,
            "raw_source_path": str(args.raw),
            "raw_source_sha256": raw_hash,
            "landmark_table_sha256": landmark_hash,
            "landmark_table_path": str(landmark_path),
        },
        root=root,
    )

    print(
        f"landmark table built: {len(rows)} episodes, "
        f"{n_unsafe} unsafe, {n_infeasible} infeasible, "
        f"{n_composite} operational_failure ({100 * n_composite / len(rows):.1f}%)"
    )
    print(f"raw source sha256:      {raw_hash}")
    print(f"landmark table sha256:  {landmark_hash}")


if __name__ == "__main__":
    main()
