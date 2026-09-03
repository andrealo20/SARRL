#!/usr/bin/env python3
"""Preregistered operating-characteristic simulation for the SARRL v1.6-R screen.

This runs BEFORE any association between predictor and endpoint is computed. It
uses no observed predictor-endpoint relationship: only the design constants and
the scenario-specific endpoint prevalences, which are outcome marginals.

Data-generating model, per the frozen protocol:

  1. Seed effects ``u_s ~ N(0, var_seed)`` for each of the 100 shared episode
     seeds and artifact effects ``v_a ~ N(0, var_artifact)`` for each of the 5
     training seeds. Distinct components, both shared across the two primary
     scenarios; only the idiosyncratic term is redrawn per episode-scenario.
  2. Probit liability ``L = u_s + v_a + e`` with ``e ~ N(0, 1)``; ``Y = 1`` iff
     ``L > tau``, with ``tau`` solved numerically so the marginal prevalence
     equals the scenario's observed value.
  3. Predictor drawn conditional on ``Y`` alone: ``X | Y=1 ~ N(mu, 1)`` and
     ``X | Y=0 ~ N(0, 1)`` with ``mu = sqrt(2) * Phi^-1(AUC_target)``. Because
     AUC is a functional of those two conditional distributions only, this
     imposes the target AUC exactly in the population regardless of how ``Y`` is
     generated.

Latent-scale mapping: ``ICC_seed = var_seed / (var_seed + var_artifact + 1)``
with ``var_artifact = var_seed / 2``.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import norm

from sarrl.evaluation import (
    assert_repository_import_root,
    assert_source_tree_clean,
    write_run_manifest,
)

# --- frozen protocol constants ------------------------------------------------

EPISODE_SEEDS = 100
TRAINING_SEEDS = 5
PRIMARY_SCENARIOS = ("id_reference", "ood_compound")
# Observed composite-endpoint prevalence in the gate-off arm. Outcome marginals.
PREVALENCE = {"id_reference": 24 / 500, "ood_compound": 92 / 500}

AUC_GRID = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
ICC_GRID = (0.0, 0.05, 0.10, 0.20)

THRESHOLD = 0.60
BOOTSTRAP_DRAWS = 10_000
LOWER_QUANTILE = 5.0  # one-sided 95% lower bound
RNG_SEED = 160_000

TARGET_AUC = 0.70
TARGET_ICC = 0.10
TARGET_JOINT_POWER = 0.80

VERIFY_REPLICATES = 200
VERIFY_TOL_PREVALENCE = 0.005
VERIFY_TOL_AUC = 0.005
VERIFY_TOL_LATENT_ICC = 0.02


def variance_components(icc_seed: float) -> tuple[float, float]:
    """Solve ``var_seed`` from the latent-scale ICC with var_artifact = var_seed/2."""
    if not 0.0 <= icc_seed < 2.0 / 3.0:
        raise ValueError("icc_seed must lie in [0, 2/3)")
    var_seed = icc_seed / (1.0 - 1.5 * icc_seed)
    return float(var_seed), float(var_seed / 2.0)


def solve_tau(prevalence: float, var_total: float) -> float:
    """Bisect for the liability threshold giving the target marginal prevalence."""
    if not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence must lie in (0, 1)")
    sigma = math.sqrt(var_total)
    lo, hi = -20.0 * sigma, 20.0 * sigma
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - norm.cdf(mid / sigma) > prevalence:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mu_for_auc(auc: float) -> float:
    """Binormal shift imposing the target AUC exactly."""
    if not 0.0 < auc < 1.0:
        raise ValueError("auc must lie in (0, 1)")
    return math.sqrt(2.0) * norm.ppf(auc)


def simulate_replicate(rng, auc: float, icc_seed: float) -> dict:
    """One complete 100-seed x 5-artifact x 2-scenario clustered dataset."""
    var_seed, var_artifact = variance_components(icc_seed)
    var_total = var_seed + var_artifact + 1.0
    mu = mu_for_auc(auc)

    u = (
        rng.normal(0.0, math.sqrt(var_seed), size=EPISODE_SEEDS)
        if var_seed > 0
        else np.zeros(EPISODE_SEEDS)
    )
    v = (
        rng.normal(0.0, math.sqrt(var_artifact), size=TRAINING_SEEDS)
        if var_artifact > 0
        else np.zeros(TRAINING_SEEDS)
    )
    shared = u[None, :] + v[:, None]  # (artifact, seed)

    out = {"latent": {}, "x": {}, "y": {}}
    for scenario in PRIMARY_SCENARIOS:
        tau = solve_tau(PREVALENCE[scenario], var_total)
        eps = rng.normal(0.0, 1.0, size=(TRAINING_SEEDS, EPISODE_SEEDS))
        liability = shared + eps
        y = (liability > tau).astype(np.int8)
        x = rng.normal(0.0, 1.0, size=y.shape) + mu * y
        out["latent"][scenario] = liability
        out["x"][scenario] = x
        out["y"][scenario] = y
    return out


def auc_from_arrays(x: np.ndarray, y: np.ndarray) -> float:
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
    """Vectorised AUC for each row of (B, n) predictor/label matrices."""
    b, n = x.shape
    order = np.argsort(x, axis=1, kind="stable")
    ranks = np.empty((b, n), dtype=np.float64)
    rows = np.arange(b)[:, None]
    ranks[rows, order] = np.arange(1, n + 1, dtype=np.float64)[None, :]
    pos = y.sum(axis=1)
    rank_sum = np.where(y == 1, ranks, 0.0).sum(axis=1)
    neg = n - pos
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return np.where((pos == 0) | (neg == 0), np.nan, auc)


def bootstrap_lower_bounds(rng, rep: dict, draws: int) -> dict:
    """One joint seed-resampling index applied identically to both scenarios."""
    idx = rng.integers(0, EPISODE_SEEDS, size=(draws, EPISODE_SEEDS))
    bounds = {}
    for scenario in PRIMARY_SCENARIOS:
        x = rep["x"][scenario][:, idx]          # (artifact, draws, seed)
        y = rep["y"][scenario][:, idx]
        x = np.moveaxis(x, 1, 0).reshape(draws, -1)
        y = np.moveaxis(y, 1, 0).reshape(draws, -1)
        aucs = _auc_rows(x, y)
        bounds[scenario] = float(np.nanpercentile(aucs, LOWER_QUANTILE))
    return bounds


def decide(bounds: dict) -> bool:
    """Positive iff the one-sided 95% lower bound exceeds the threshold in both."""
    return all(b > THRESHOLD for b in bounds.values())


def verify(rng, auc: float, icc_seed: float, replicates: int) -> dict:
    """Check prevalence, AUC and latent within-seed correlation against targets."""
    var_seed, var_artifact = variance_components(icc_seed)
    latent_target = var_seed / (var_seed + var_artifact + 1.0)
    prevalence: dict = {s: [] for s in PRIMARY_SCENARIOS}
    empirical_auc: dict = {s: [] for s in PRIMARY_SCENARIOS}
    latent_icc: dict = {s: [] for s in PRIMARY_SCENARIOS}
    binary_icc: dict = {s: [] for s in PRIMARY_SCENARIOS}

    for _ in range(replicates):
        rep = simulate_replicate(rng, auc, icc_seed)
        for scenario in PRIMARY_SCENARIOS:
            y = rep["y"][scenario]
            x = rep["x"][scenario]
            lat = rep["latent"][scenario]
            prevalence[scenario].append(float(y.mean()))
            empirical_auc[scenario].append(auc_from_arrays(x.ravel(), y.ravel()))
            # within-seed correlation on the latent scale: between-seed variance
            # of the column means over the total variance.
            col = lat.mean(axis=0)
            latent_icc[scenario].append(float(np.var(col, ddof=1) / np.var(lat, ddof=1)))
            yc = y.mean(axis=0)
            binary_icc[scenario].append(float(np.var(yc, ddof=1) / max(np.var(y, ddof=1), 1e-12)))

    report = {
        "auc_target": auc,
        "icc_seed_target": icc_seed,
        "latent_icc_formula_target": latent_target,
        "replicates": replicates,
        "scenarios": {},
        "passed": True,
    }
    for scenario in PRIMARY_SCENARIOS:
        p = float(np.mean(prevalence[scenario]))
        a = float(np.nanmean(empirical_auc[scenario]))
        li = float(np.mean(latent_icc[scenario]))
        ok = (
            abs(p - PREVALENCE[scenario]) <= VERIFY_TOL_PREVALENCE
            and abs(a - auc) <= VERIFY_TOL_AUC
            and abs(li - latent_target) <= VERIFY_TOL_LATENT_ICC
        )
        report["scenarios"][scenario] = {
            "prevalence_empirical": p,
            "prevalence_target": PREVALENCE[scenario],
            "auc_empirical": a,
            "auc_target": auc,
            "latent_within_seed_icc_empirical": li,
            "latent_within_seed_icc_target": latent_target,
            # Reported descriptively only; binary ICC depends on prevalence and
            # threshold and is NOT the latent probit correlation.
            "binary_icc_descriptive_no_target": float(np.mean(binary_icc[scenario])),
            "passed": bool(ok),
        }
        report["passed"] = report["passed"] and bool(ok)
    return report


def run_cell(args_tuple) -> dict:
    """Estimate one grid cell. Deterministic per cell via a SeedSequence spawn."""
    auc, icc, replicates, draws, cell_index = args_tuple
    rng = np.random.default_rng(np.random.SeedSequence([RNG_SEED, cell_index]))
    positives = 0
    marginal = {s: 0 for s in PRIMARY_SCENARIOS}
    for _ in range(replicates):
        rep = simulate_replicate(rng, auc, icc)
        bounds = bootstrap_lower_bounds(rng, rep, draws)
        for s, b in bounds.items():
            if b > THRESHOLD:
                marginal[s] += 1
        if decide(bounds):
            positives += 1
    return {
        "auc_target": auc,
        "icc_seed": icc,
        "replicates": replicates,
        "joint_positive_power": positives / replicates,
        "marginal_component_power": {s: marginal[s] / replicates for s in PRIMARY_SCENARIOS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preregistered v1.6-R operating-characteristic simulation"
    )
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--verify-replicates", type=int, default=VERIFY_REPLICATES)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/uncertainty_gate_calibration/phase_r"),
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="time a single replicate and exit")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = assert_repository_import_root(Path(__file__).resolve().parents[1])
    if not args.benchmark:
        assert_source_tree_clean(root)
    rng = np.random.default_rng(RNG_SEED)

    if args.benchmark:
        import time

        rep = simulate_replicate(rng, TARGET_AUC, TARGET_ICC)
        t0 = time.perf_counter()
        bounds = bootstrap_lower_bounds(rng, rep, args.draws)
        dt = time.perf_counter() - t0
        print(json.dumps({"seconds_per_replicate": dt, "bounds": bounds}, indent=2))
        return

    print("=== verification (must pass before power is estimated)")
    ver = verify(rng, TARGET_AUC, TARGET_ICC, args.verify_replicates)
    print(json.dumps(ver, indent=2, sort_keys=True))
    if not ver["passed"]:
        raise SystemExit("verification failed: the simulator is defective; do not estimate power")

    if args.verify_only:
        return

    print("=== power grid")
    cells = [
        (auc, icc, args.replicates, args.draws, i)
        for i, (auc, icc) in enumerate((a, c) for a in AUC_GRID for c in ICC_GRID)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        grid = list(pool.map(run_cell, cells))
    for entry in grid:
        print(json.dumps(entry))

    target = next(
        e for e in grid if e["auc_target"] == TARGET_AUC and e["icc_seed"] == TARGET_ICC
    )
    config = {
        "campaign": "uncertainty_gate_calibration_phase_r_power",
        "primary_scenarios": list(PRIMARY_SCENARIOS),
        "prevalence": PREVALENCE,
        "auc_grid": list(AUC_GRID),
        "icc_grid": list(ICC_GRID),
        "threshold": THRESHOLD,
        "bootstrap_draws": args.draws,
        "lower_quantile": LOWER_QUANTILE,
        "rng_seed": RNG_SEED,
        "replicates": args.replicates,
        "verification": ver,
        "grid": grid,
        "target": {
            "auc": TARGET_AUC,
            "icc_seed": TARGET_ICC,
            "required_joint_power": TARGET_JOINT_POWER,
            "achieved_joint_power": target["joint_positive_power"],
            "meets_target": target["joint_positive_power"] >= TARGET_JOINT_POWER,
        },
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_run_manifest(out / "power_manifest.json", config, root=root)
    print(json.dumps(config["target"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
