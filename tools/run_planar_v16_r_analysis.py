#!/usr/bin/env python3
"""Compute the preregistered SARRL v1.6-R association and decision.

Runs only after `run_planar_v16_r_landmark.py` has produced the hash-bound
landmark table; the hash is re-verified here before anything is read.

Primary estimand: the fixed-window observational association between ensemble
disagreement and operational failure in the gate-off arm, conditional on the
five frozen training artifacts. Per-scenario AUC over the five seeds pooled,
with a clustered bootstrap that resamples the 100 shared episode seeds into one
joint index applied identically across scenarios.

Decision: intersection-union test on marginal one-sided lower bounds at the
recalibrated per-scenario critical quantiles (Amendments 1-3). Positive requires
the lower bound to exceed 0.60 in BOTH primary scenarios.

Secondary analyses are computed and reported but cannot change the decision.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sarrl.evaluation import assert_repository_import_root, write_run_manifest

PRIMARY_SCENARIOS = ("id_reference", "ood_compound")
SECONDARY_SCENARIO = "motor_fault"
THRESHOLD = 0.60
BOOTSTRAP_DRAWS = 10_000
ANALYSIS_RNG_SEED = 160_000
EPISODE_SEEDS = 100

# Onset-anchored secondary window for motor_fault (PLAN.md decision rule):
# the fault activates when steps >= 20 but the predictor for that row is
# recorded beforehand, so post-onset predictor rows are 21..45.
ONSET_FIRST_STEP = 21
ONSET_LAST_STEP = 45
ONSET_ROWS = ONSET_LAST_STEP - ONSET_FIRST_STEP + 1


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC with midranks; NaN when a class is absent."""
    pos = int(y.sum())
    neg = int(y.size - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    sx = x[order]
    start = 0
    for i in range(1, x.size + 1):
        if i == x.size or sx[i] != sx[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _auc_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    b, n = x.shape
    order = np.argsort(x, axis=1, kind="stable")
    ranks = np.empty((b, n), dtype=np.float64)
    ranks[np.arange(b)[:, None], order] = np.arange(1, n + 1, dtype=np.float64)[None, :]
    pos = y.sum(axis=1)
    rank_sum = np.where(y == 1, ranks, 0.0).sum(axis=1)
    neg = n - pos
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return np.where((pos == 0) | (neg == 0), np.nan, out)


def load_landmark(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def as_matrices(rows: list, scenario: str, predictor: str, endpoint: str):
    """Shape a scenario into (artifact, seed) matrices keyed by episode seed."""
    seeds = sorted({r["episode_seed"] for r in rows if r["scenario"] == scenario})
    artifacts = sorted({r["training_seed"] for r in rows if r["scenario"] == scenario})
    x = np.full((len(artifacts), len(seeds)), np.nan)
    y = np.full((len(artifacts), len(seeds)), -1, dtype=np.int8)
    si = {s: i for i, s in enumerate(seeds)}
    ai = {a: i for i, a in enumerate(artifacts)}
    for r in rows:
        if r["scenario"] != scenario:
            continue
        i, j = ai[r["training_seed"]], si[r["episode_seed"]]
        x[i, j] = float(r[predictor])
        y[i, j] = 1 if r[endpoint].strip().lower() == "true" else 0
    if np.isnan(x).any() or (y < 0).any():
        raise RuntimeError(f"incomplete matrix for scenario {scenario}")
    return x, y, seeds, artifacts


def analyse(rows: list, predictor: str, endpoint: str, critical: dict, rng) -> dict:
    """Per-scenario AUC with one joint clustered bootstrap index."""
    data = {s: as_matrices(rows, s, predictor, endpoint) for s in PRIMARY_SCENARIOS}
    n_seeds = data[PRIMARY_SCENARIOS[0]][0].shape[1]
    if n_seeds != EPISODE_SEEDS:
        raise RuntimeError(f"expected {EPISODE_SEEDS} episode seeds, found {n_seeds}")

    idx = rng.integers(0, n_seeds, size=(BOOTSTRAP_DRAWS, n_seeds))
    out = {}
    for scenario in PRIMARY_SCENARIOS:
        x, y, _, _ = data[scenario]
        point = auc(x.ravel(), y.ravel())
        xb = np.moveaxis(x[:, idx], 1, 0).reshape(BOOTSTRAP_DRAWS, -1)
        yb = np.moveaxis(y[:, idx], 1, 0).reshape(BOOTSTRAP_DRAWS, -1)
        draws = _auc_rows(xb, yb)
        q = float(critical[scenario])
        out[scenario] = {
            "auc": point,
            "events": int(y.sum()),
            "episodes": int(y.size),
            "critical_quantile": q,
            "lower_bound": float(np.nanpercentile(draws, q)),
            "upper_bound": float(np.nanpercentile(draws, 100.0 - q)),
            "per_cell_auc": [
                auc(x[i], y[i]) for i in range(x.shape[0])
            ],
        }
    return out


def decide(primary: dict) -> str:
    if all(primary[s]["lower_bound"] > THRESHOLD for s in PRIMARY_SCENARIOS):
        return "positive"
    if all(primary[s]["upper_bound"] <= THRESHOLD for s in PRIMARY_SCENARIOS):
        return "negative"
    return "inconclusive"


def window_medians(raw: Path, first: int, last: int, scenarios: set) -> dict:
    """Median uncertainty_norm over a raw step window, per episode."""
    acc = defaultdict(list)
    with gzip.open(raw, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["condition"] != "A6c_gate_off_control" or row["population"] != "safety":
                continue
            if row["scenario"] not in scenarios:
                continue
            step = int(row["step"])
            if step < first or step > last:
                continue
            acc[(row["training_seed"], row["scenario"], row["episode_seed"])].append(
                float(row["uncertainty_norm"])
            )
    return acc


def first_unsafe_index(path: Path) -> dict:
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["condition"] != "A6c_gate_off_control" or row["population"] != "safety":
                continue
            raw = row["first_unsafe_observation"].strip()
            key = (row["training_seed"], row["scenario"], row["seed"])
            out[key] = None if raw in ("", "-1") else int(float(raw))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="v1.6-R preregistered association")
    base = Path("results/uncertainty_gate_calibration")
    p.add_argument("--landmark", type=Path, default=base / "phase_r/landmark_episodes.csv")
    p.add_argument(
        "--landmark-manifest", type=Path,
        default=base / "phase_r/landmark_manifest.json",
    )
    p.add_argument(
        "--recalibration", type=Path,
        default=base / "phase_r/recalibration_manifest.json",
    )
    p.add_argument("--raw", type=Path, default=base / "phase_c/transitions.csv.gz")
    p.add_argument("--safety", type=Path, default=base / "phase_c/safety_diagnostics.csv")
    p.add_argument("--output", type=Path, default=base / "phase_r")
    args = p.parse_args()

    root = assert_repository_import_root(Path(__file__).resolve().parents[1])

    manifest = json.loads(args.landmark_manifest.read_text())["config"]
    actual = sha256_of(args.landmark)
    if actual != manifest["landmark_table_sha256"]:
        raise SystemExit(
            f"landmark table hash mismatch: manifest says "
            f"{manifest['landmark_table_sha256']}, file is {actual}"
        )
    critical = json.loads(args.recalibration.read_text())["config"]["critical_quantiles"]
    print(f"landmark hash verified; critical quantiles {critical}")

    rows = load_landmark(args.landmark)
    rng = np.random.default_rng(ANALYSIS_RNG_SEED)

    primary = analyse(
        rows, "uncertainty_norm_landmark_median", "operational_failure", critical, rng
    )
    verdict = decide(primary)

    # --- secondary analyses; excluded from the decision -----------------------
    secondary = {}
    rng2 = np.random.default_rng(ANALYSIS_RNG_SEED)
    secondary["predictor_uncertainty_ratio"] = analyse(
        rows, "uncertainty_ratio_landmark_median", "operational_failure", critical, rng2
    )
    rng3 = np.random.default_rng(ANALYSIS_RNG_SEED)
    secondary["endpoint_unsafe_episode_only"] = analyse(
        rows, "uncertainty_norm_landmark_median", "unsafe_episode", critical, rng3
    )

    # whole-episode predictor, to quantify censoring contamination
    whole = window_medians(args.raw, 0, 10**9, set(PRIMARY_SCENARIOS))
    by_key = {(r["training_seed"], r["scenario"], r["episode_seed"]): r for r in rows}
    whole_rows = []
    for k, vals in whole.items():
        r = dict(by_key[k])
        r["uncertainty_norm_landmark_median"] = float(np.median(vals))
        whole_rows.append(r)
    rng4 = np.random.default_rng(ANALYSIS_RNG_SEED)
    secondary["predictor_whole_episode"] = analyse(
        whole_rows, "uncertainty_norm_landmark_median", "operational_failure", critical, rng4
    )

    # pre-event-truncated sensitivity: rows strictly before first unsafe obs
    first_unsafe = first_unsafe_index(args.safety)
    trunc_acc = defaultdict(list)
    with gzip.open(args.raw, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["condition"] != "A6c_gate_off_control" or row["population"] != "safety":
                continue
            if row["scenario"] not in PRIMARY_SCENARIOS:
                continue
            key = (row["training_seed"], row["scenario"], row["episode_seed"])
            cut = first_unsafe.get(key)
            step = int(row["step"])
            if step > 24:
                continue
            if cut is not None and step >= cut:
                continue
            trunc_acc[key].append(float(row["uncertainty_norm"]))
    trunc_rows, dropped = [], 0
    for k, r in by_key.items():
        if r["scenario"] not in PRIMARY_SCENARIOS:
            continue  # out of scope for the primary estimand, not a dropped episode
        vals = trunc_acc.get(k, [])
        if not vals:
            dropped += 1
            continue
        rr = dict(r)
        rr["uncertainty_norm_landmark_median"] = float(np.median(vals))
        trunc_rows.append(rr)
    rng5 = np.random.default_rng(ANALYSIS_RNG_SEED)
    try:
        secondary["predictor_pre_event_truncated"] = analyse(
            trunc_rows, "uncertainty_norm_landmark_median", "operational_failure", critical, rng5
        )
        secondary["predictor_pre_event_truncated"]["episodes_dropped_no_pre_event_rows"] = dropped
    except RuntimeError as exc:
        secondary["predictor_pre_event_truncated"] = {
            "error": str(exc),
            "episodes_dropped_no_pre_event_rows": dropped,
        }

    # motor_fault onset-anchored conditional secondary
    onset = window_medians(args.raw, ONSET_FIRST_STEP, ONSET_LAST_STEP, {SECONDARY_SCENARIO})
    fault_rows = [r for r in load_landmark(args.landmark) if r["scenario"] == SECONDARY_SCENARIO]
    fault_by_key = {(r["training_seed"], r["scenario"], r["episode_seed"]): r for r in fault_rows}
    included, excluded = [], []
    for k, r in fault_by_key.items():
        vals = onset.get(k, [])
        if len(vals) == ONSET_ROWS:
            rr = dict(r)
            rr["onset_median"] = float(np.median(vals))
            included.append(rr)
        else:
            excluded.append(r)
    fx = np.array([float(r["onset_median"]) for r in included])
    fy = np.array(
        [1 if r["operational_failure"].strip().lower() == "true" else 0 for r in included],
        dtype=np.int8,
    )
    excl_by_outcome = defaultdict(int)
    for r in excluded:
        excl_by_outcome[r["operational_failure"]] += 1
    secondary["motor_fault_onset_conditional"] = {
        "window_first_step": ONSET_FIRST_STEP,
        "window_last_step": ONSET_LAST_STEP,
        "included_episodes": len(included),
        "excluded_episodes": len(excluded),
        "excluded_by_operational_failure": dict(excl_by_outcome),
        "auc": auc(fx, fy) if len(included) else float("nan"),
        "note": (
            "conditional post-onset estimand with an outcome-dependent inclusion "
            "criterion; must not be ranked against or compared with the primary "
            "scenario AUCs, and cannot change the decision"
        ),
    }

    config = {
        "campaign": "uncertainty_gate_calibration_phase_r_analysis",
        "estimand": "fixed-window observational association, conditional on five frozen artifacts",
        "primary_scenarios": list(PRIMARY_SCENARIOS),
        "predictor": "uncertainty_norm_landmark_median (raw steps 0..24)",
        "endpoint": "operational_failure = unsafe_episode OR safety_infeasible",
        "threshold": THRESHOLD,
        "critical_quantiles": critical,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "rng_seed": ANALYSIS_RNG_SEED,
        "landmark_table_sha256": actual,
        "power_label": (
            "low-power feasibility screen: joint power 38.5% at AUC 0.70 / ICC 0.10 "
            "against an 80% target; a non-rejection is inconclusive, NOT evidence of "
            "absence; the procedure's validated worst-case size is 3.65% [2.91%, 4.56%]"
        ),
        "primary": primary,
        "decision": verdict,
        "secondary_excluded_from_decision": secondary,
    }
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_run_manifest(out / "analysis_manifest.json", config, root=root)

    print()
    print("=== PRIMARY")
    for s in PRIMARY_SCENARIOS:
        d = primary[s]
        print(
            f"  {s:<14} AUC {d['auc']:.4f}  events {d['events']}/{d['episodes']}  "
            f"one-sided lower bound (q={d['critical_quantile']}) {d['lower_bound']:.4f}  "
            f"upper {d['upper_bound']:.4f}"
        )
    print(f"  DECISION: {verdict.upper()}")


if __name__ == "__main__":
    main()
