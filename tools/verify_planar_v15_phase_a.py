#!/usr/bin/env python3
"""Independently verify retained SARRL v1.5 Phase-A evidence."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from sarrl.dynamics import PlanarArm, PlanarArmParams
from sarrl.evaluation import PhaseAEpisode, analyze_phase_a, summarize_episode

if __package__:
    from tools.run_planar_v15_phase_a import TRANSITION_FIELDS
else:
    from run_planar_v15_phase_a import TRANSITION_FIELDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool(value: str) -> bool:
    if value not in {"True", "False", "true", "false"}:
        raise ValueError(f"invalid boolean: {value}")
    return value.lower() == "true"


def _read_episodes(path: Path) -> list[PhaseAEpisode]:
    rows = []
    float_fields = {"spearman_rho", "uncertainty_median", "error_median"}
    bool_fields = {"terminated", "truncated", "qualifies", "zero_variance"}
    with path.open(encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            parsed = {}
            for key, value in source.items():
                if key == "policy":
                    parsed[key] = value
                elif key in bool_fields:
                    parsed[key] = _bool(value)
                elif key in float_fields:
                    parsed[key] = None if value in {"", "None"} else float(value)
                else:
                    parsed[key] = int(value)
            rows.append(PhaseAEpisode(**parsed))
    return rows


def _params(row: dict[str, str]) -> PlanarArmParams:
    return PlanarArmParams(
        m1=float(row["m1"]),
        m2=float(row["m2"]),
        l1=float(row["l1"]),
        l2=float(row["l2"]),
        lc1=float(row["lc1"]),
        lc2=float(row["lc2"]),
        i1=float(row["i1"]),
        i2=float(row["i2"]),
        gravity=float(row["gravity"]),
        viscous=(float(row["viscous1"]), float(row["viscous2"])),
        coulomb=(float(row["coulomb1"]), float(row["coulomb2"])),
        friction_smoothing=float(row["friction_smoothing"]),
        payload_mass=float(row["payload_mass"]),
    )


def _close(actual, expected, label: str) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-12):
        raise ValueError(f"transition invariant failed: {label}")


def verify(root: Path, out: Path) -> dict:
    out = out.resolve()
    manifest = json.loads((out / "evaluation_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["config"]["outputs"].items():
        path = out / name
        if not path.is_file() and name == "transitions.csv":
            continue
        if _sha256(path) != expected:
            raise ValueError(f"aggregate output hash mismatch: {name}")
    verified_shard_files = 0
    for item in manifest["config"]["shard_hashes"]:
        path = root / item["path"]
        if not path.is_file():
            continue
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"shard hash mismatch: {item['path']}")
        verified_shard_files += 1
    with gzip.open(out / "transitions.csv.gz", "rb") as compressed:
        digest = hashlib.sha256()
        for block in iter(lambda: compressed.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != manifest["config"]["outputs"]["transitions.csv"]:
        raise ValueError("compressed transition payload does not match canonical CSV hash")

    expected_episodes = _read_episodes(out / "episodes.csv")
    expected_by_key = {
        (row.policy, row.training_seed, row.ensemble_seed, row.episode_seed): row
        for row in expected_episodes
    }
    if len(expected_by_key) != len(expected_episodes):
        raise ValueError("duplicate retained episode summary")

    pairs: dict[tuple[str, int, int, int], tuple[list[float], list[float]]] = {}
    prior_order = None
    arm_cache = {}
    transition_count = 0
    policy_order = {"A2": 0, "A3": 1}
    transition_path = out / "transitions.csv"
    if transition_path.is_file():
        handle = transition_path.open(encoding="utf-8", newline="")
    else:
        handle = gzip.open(out / "transitions.csv.gz", mode="rt", encoding="utf-8", newline="")
    with handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TRANSITION_FIELDS:
            raise ValueError("transition schema mismatch")
        for row in reader:
            transition_count += 1
            key = (
                row["policy"],
                int(row["training_seed"]),
                int(row["ensemble_seed"]),
                int(row["episode_seed"]),
            )
            order = (policy_order[key[0]], key[1], key[2], key[3], int(row["step"]))
            if prior_order is not None and order <= prior_order:
                raise ValueError("transitions are not in canonical order")
            prior_order = order
            numeric = [
                float(value)
                for name, value in row.items()
                if name not in {"policy", "terminated", "truncated"}
            ]
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError("non-finite value in canonical transition CSV")

            params = _params(row)
            if key not in arm_cache:
                arm_cache[key] = PlanarArm(params)
            elif arm_cache[key].params != params:
                raise ValueError("arm parameters changed within an ID-reference episode")
            arm = arm_cache[key]
            state = np.array(
                [float(row[name]) for name in ("q1", "q2", "dq1", "dq2")],
                dtype=np.float64,
            )
            commanded = np.array(
                [float(row["commanded_torque1"]), float(row["commanded_torque2"])],
                dtype=np.float64,
            )
            plant_input = np.array(
                [float(row["plant_input_torque1"]), float(row["plant_input_torque2"])],
                dtype=np.float64,
            )
            observed = np.array(
                [
                    float(row["observed_acceleration1"]),
                    float(row["observed_acceleration2"]),
                ],
                dtype=np.float64,
            )
            _close(
                observed,
                arm.forward_dynamics(state[:2], state[2:], plant_input),
                "observed acceleration",
            )
            nominal = PlanarArm()
            target = observed - nominal.forward_dynamics(state[:2], state[2:], commanded)
            logged_target = np.array(
                [float(row["residual_target1"]), float(row["residual_target2"])],
                dtype=np.float64,
            )
            _close(logged_target, target, "residual target")
            uncertainty = np.array(
                [
                    float(row["ensemble_uncertainty1"]),
                    float(row["ensemble_uncertainty2"]),
                ],
                dtype=np.float64,
            )
            uncertainty_norm = float(row["uncertainty_norm"])
            _close(uncertainty_norm, np.linalg.norm(uncertainty), "uncertainty norm")
            mean = np.array(
                [float(row["ensemble_mean1"]), float(row["ensemble_mean2"])],
                dtype=np.float64,
            )
            error_norm = float(row["prediction_error_norm"])
            _close(error_norm, np.linalg.norm(mean - target), "prediction error norm")
            u, e = pairs.setdefault(key, ([], []))
            u.append(uncertainty_norm)
            e.append(error_norm)

    recomputed = []
    for key, expected in expected_by_key.items():
        if key not in pairs:
            raise ValueError(f"episode has no retained transition pairs: {key}")
        uncertainty, error = pairs[key]
        row = summarize_episode(
            policy=key[0],
            training_seed=key[1],
            ensemble_seed=key[2],
            episode_seed=key[3],
            uncertainty=uncertainty,
            error=error,
            attempted_pairs=expected.attempted_pairs,
            terminated=expected.terminated,
            truncated=expected.truncated,
        )
        if row != expected:
            raise ValueError(f"episode summary mismatch: {key}")
        recomputed.append(row)

    decision = json.loads((out / "decision.json").read_text(encoding="utf-8"))
    recalculated = analyze_phase_a(recomputed)
    if recalculated != decision:
        raise ValueError("Phase-A decision does not reproduce from raw transitions")
    return {
        "episodes": len(recomputed),
        "transitions": transition_count,
        "excluded_nonfinite_pairs": sum(row.excluded_nonfinite_pairs for row in recomputed),
        "common_episode_count": recalculated["common_episode_count"],
        "target_median_rho": recalculated["target_median_rho"],
        "ci95_low": recalculated["ci95_low"],
        "ci95_high": recalculated["ci95_high"],
        "decision": recalculated["decision"],
        "verified_shard_files": verified_shard_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/uncertainty_gate_calibration/phase_a"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(verify(root, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
