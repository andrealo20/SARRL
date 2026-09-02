#!/usr/bin/env python3
"""Build or verify the versioned SARRL v1.5 gate-calibration artifact."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np

from sarrl.evaluation import (
    V15_PHASE_A_POLICIES,
    V15_PHASE_A_TRAINING_SEEDS,
    canonical_json,
    derive_reference_uncertainties,
    planar_id_randomization_dict,
    repository_commit,
    write_canonical_csv,
    write_canonical_json,
)

if __package__:
    from tools.run_planar_v15_phase_a import _read_episode_rows
    from tools.verify_planar_v15_phase_a import verify as verify_phase_a
else:
    from run_planar_v15_phase_a import _read_episode_rows
    from verify_planar_v15_phase_a import verify as verify_phase_a


REFERENCE_FIELDS = (
    "training_seed",
    "ensemble_seed",
    "a2_scale_eligible_episodes",
    "a3_scale_eligible_episodes",
    "pooled_episode_medians",
    "u_ref",
)
SENSITIVITY_FIELDS = (
    "scope",
    "ensemble_seed",
    "gain",
    "transitions",
    "scale_min",
    "scale_p05",
    "scale_median",
    "scale_mean",
    "scale_p95",
    "scale_max",
    "floor_fraction",
)
SENSITIVITY_GAINS = (0.5, 1.0, 2.0, 4.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _logical_path(path: str | Path) -> str:
    normalized = str(path).replace("\\", "/")
    marker = "/SARRL/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    return PurePosixPath(normalized).as_posix().lstrip("./")


def _uncertainties(path: Path) -> dict[int, np.ndarray]:
    values = {seed: [] for seed in V15_PHASE_A_TRAINING_SEEDS}
    with gzip.open(path, mode="rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values[int(row["ensemble_seed"])].append(float(row["uncertainty_norm"]))
    output = {seed: np.asarray(items, dtype=np.float64) for seed, items in values.items()}
    if any(array.size == 0 or not np.all(np.isfinite(array)) for array in output.values()):
        raise ValueError("Phase A uncertainty records are missing or non-finite")
    return output


def _scale_row(scope: str, seed: int, gain: float, scale: np.ndarray) -> dict:
    return {
        "scope": scope,
        "ensemble_seed": seed,
        "gain": gain,
        "transitions": int(scale.size),
        "scale_min": float(np.min(scale)),
        "scale_p05": float(np.quantile(scale, 0.05)),
        "scale_median": float(np.median(scale)),
        "scale_mean": float(np.mean(scale)),
        "scale_p95": float(np.quantile(scale, 0.95)),
        "scale_max": float(np.max(scale)),
        "floor_fraction": float(np.mean(scale == 0.1)),
    }


def sensitivity_rows(
    uncertainties: dict[int, np.ndarray], references: list[dict]
) -> list[dict]:
    reference_map = {row["ensemble_seed"]: row["u_ref"] for row in references}
    rows = []
    for gain in SENSITIVITY_GAINS:
        pooled = []
        for seed in V15_PHASE_A_TRAINING_SEEDS:
            ratio = uncertainties[seed] / reference_map[seed]
            scale = np.maximum(0.1, 1.0 / (1.0 + gain * ratio))
            rows.append(_scale_row("per_ensemble", seed, gain, scale))
            pooled.append(scale)
        rows.append(_scale_row("pooled_transitions", -1, gain, np.concatenate(pooled)))
    return rows


def _source_hashes(root: Path, phase_manifest: dict, phase_dir: Path) -> list[dict]:
    hashes = []
    role_fields = (
        ("a2_policy", "a2_policy_checkpoint", "a2_policy_checkpoint_sha256"),
        ("a3_policy", "policy_checkpoint", "policy_checkpoint_sha256"),
        ("context", "context_checkpoint", "context_checkpoint_sha256"),
        ("ensemble", "ensemble_checkpoint", "ensemble_checkpoint_sha256"),
    )
    for record in phase_manifest["config"]["inputs"]:
        seed = int(record["training_seed"])
        for role, path_field, hash_field in role_fields:
            actual_path = Path(record[path_field])
            actual_hash = _sha256(actual_path)
            if actual_hash != record[hash_field]:
                raise ValueError(f"source artifact hash mismatch: {role} seed {seed}")
            hashes.append(
                {
                    "logical_role": role,
                    "training_seed": seed,
                    "path": _logical_path(actual_path),
                    "sha256": actual_hash,
                }
            )
    source_manifests = (
        ("phase_a_manifest", phase_dir / "evaluation_manifest.json"),
        ("v1_3_source_manifest", root / "results/ood_fault_robustness/evaluation_manifest.json"),
    )
    for role, path in source_manifests:
        hashes.append(
            {
                "logical_role": role,
                "training_seed": -1,
                "path": _logical_path(path),
                "sha256": _sha256(path),
            }
        )
    hashes.extend(
        [
            {
                "logical_role": "phase_a_transitions_csv",
                "training_seed": -1,
                "path": _logical_path(phase_dir / "transitions.csv"),
                "sha256": phase_manifest["config"]["outputs"]["transitions.csv"],
            },
            {
                "logical_role": "phase_a_transitions_gzip",
                "training_seed": -1,
                "path": _logical_path(phase_dir / "transitions.csv.gz"),
                "sha256": _sha256(phase_dir / "transitions.csv.gz"),
            },
        ]
    )
    return sorted(hashes, key=lambda row: (row["logical_role"], row["training_seed"], row["path"]))


def build(root: Path, phase_dir: Path, output_dir: Path) -> dict:
    audit = verify_phase_a(root, phase_dir)
    if audit["decision"] != "proceed_phase_b":
        raise ValueError("Phase A does not authorize calibration")
    if audit["excluded_nonfinite_pairs"] != 0:
        raise ValueError("cannot establish error-independent scale eligibility")
    phase_manifest = json.loads(
        (phase_dir / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    decision = json.loads((phase_dir / "decision.json").read_text(encoding="utf-8"))
    episodes = _read_episode_rows(phase_dir / "episodes.csv")
    references = derive_reference_uncertainties(episodes)
    uncertainties = _uncertainties(phase_dir / "transitions.csv.gz")
    sensitivity = sensitivity_rows(uncertainties, references)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = output_dir / "u_ref.csv"
    sensitivity_path = output_dir / "sensitivity.csv"
    write_canonical_csv(reference_path, REFERENCE_FIELDS, references)
    write_canonical_csv(sensitivity_path, SENSITIVITY_FIELDS, sensitivity)
    hashes = _source_hashes(root, phase_manifest, phase_dir)
    hashes.extend(
        [
            {
                "logical_role": "u_ref_table",
                "training_seed": -1,
                "path": _logical_path(reference_path),
                "sha256": _sha256(reference_path),
            },
            {
                "logical_role": "sensitivity_table",
                "training_seed": -1,
                "path": _logical_path(sensitivity_path),
                "sha256": _sha256(sensitivity_path),
            },
        ]
    )
    hashes.sort(key=lambda row: (row["logical_role"], row["training_seed"], row["path"]))
    payload = {
        "schema_version": 1,
        "artifact_type": "sarrl_uncertainty_gate_calibration",
        "release_target": "v1.5.0",
        "generation_commit": repository_commit(root),
        "environment_randomization": planar_id_randomization_dict(),
        "seed_pairing": {
            "policies": list(V15_PHASE_A_POLICIES),
            "training_and_ensemble_seeds": list(V15_PHASE_A_TRAINING_SEEDS),
            "evaluation_seed_start": 60_000,
            "episodes_per_cell": 100,
        },
        "phase_a": {
            "decision": decision["decision"],
            "target_median_rho": decision["target_median_rho"],
            "ci95_low": decision["ci95_low"],
            "ci95_high": decision["ci95_high"],
            "screening_threshold": decision["screening_threshold"],
            "common_episode_count": decision["common_episode_count"],
            "bootstrap_seed": decision["bootstrap_seed"],
            "bootstrap_samples": decision["bootstrap_samples"],
            "scope": decision["scope"],
        },
        "gate": {
            "formula": "max(min_scale,1/(1+gain*uncertainty_norm/u_ref))",
            "gain": 1.0,
            "min_scale": 0.1,
            "reference_scale": 0.5,
            "floor_ratio": 9.0,
        },
        "u_ref": references,
        "sensitivity_gains": list(SENSITIVITY_GAINS),
        "hashes": hashes,
    }
    calibration_path = output_dir / "calibration.json"
    write_canonical_json(calibration_path, payload)
    return payload


def verify(root: Path, phase_dir: Path, output_dir: Path) -> dict:
    path = output_dir / "calibration.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported calibration schema")
    if payload.get("artifact_type") != "sarrl_uncertainty_gate_calibration":
        raise ValueError("wrong calibration artifact type")
    if path.read_text(encoding="utf-8") != canonical_json(payload) + "\n":
        raise ValueError("calibration JSON is not canonical")
    audit = verify_phase_a(root, phase_dir)
    if audit["decision"] != "proceed_phase_b" or audit["excluded_nonfinite_pairs"] != 0:
        raise ValueError("retained Phase A does not support this calibration")
    phase_manifest = json.loads(
        (phase_dir / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    episodes = _read_episode_rows(phase_dir / "episodes.csv")
    expected_references = derive_reference_uncertainties(episodes)
    if payload["u_ref"] != expected_references:
        raise ValueError("u_ref values do not reproduce from Phase A episodes")
    expected_sensitivity = sensitivity_rows(
        _uncertainties(phase_dir / "transitions.csv.gz"), expected_references
    )
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        expected_reference_path = temporary_path / "u_ref.csv"
        expected_sensitivity_path = temporary_path / "sensitivity.csv"
        write_canonical_csv(expected_reference_path, REFERENCE_FIELDS, expected_references)
        write_canonical_csv(
            expected_sensitivity_path, SENSITIVITY_FIELDS, expected_sensitivity
        )
        if expected_reference_path.read_bytes() != (output_dir / "u_ref.csv").read_bytes():
            raise ValueError("u_ref CSV does not reproduce")
        if expected_sensitivity_path.read_bytes() != (
            output_dir / "sensitivity.csv"
        ).read_bytes():
            raise ValueError("sensitivity CSV does not reproduce")
    expected_hashes = _source_hashes(root, phase_manifest, phase_dir)
    expected_hashes.extend(
        [
            {
                "logical_role": "u_ref_table",
                "training_seed": -1,
                "path": _logical_path(output_dir / "u_ref.csv"),
                "sha256": _sha256(output_dir / "u_ref.csv"),
            },
            {
                "logical_role": "sensitivity_table",
                "training_seed": -1,
                "path": _logical_path(output_dir / "sensitivity.csv"),
                "sha256": _sha256(output_dir / "sensitivity.csv"),
            },
        ]
    )
    expected_hashes.sort(
        key=lambda row: (row["logical_role"], row["training_seed"], row["path"])
    )
    if payload["hashes"] != expected_hashes:
        raise ValueError("calibration source hashes do not reproduce")
    references = {row["ensemble_seed"]: row["u_ref"] for row in payload["u_ref"]}
    if set(references) != set(V15_PHASE_A_TRAINING_SEEDS):
        raise ValueError("calibration does not contain all five ensemble seeds")
    if any(not math.isfinite(value) or value <= 0.0 for value in references.values()):
        raise ValueError("calibration contains invalid u_ref")
    return {
        "schema_version": payload["schema_version"],
        "calibration_sha256": _sha256(path),
        "u_ref": references,
        "source_hashes": len(payload["hashes"]),
        "phase_a_decision": payload["phase_a"]["decision"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--phase-a",
        type=Path,
        default=Path("results/uncertainty_gate_calibration/phase_a"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/uncertainty_gate_calibration"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    phase_dir = args.phase_a.resolve()
    output_dir = args.output.resolve()
    result = (
        verify(root, phase_dir, output_dir)
        if args.verify
        else build(root, phase_dir, output_dir)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
