#!/usr/bin/env python3
"""Independently verify retained SARRL v1.5 Phase-C evidence."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

if __package__:
    from tools.run_planar_v15_phase_c import RAW_FIELDS
else:
    from run_planar_v15_phase_c import RAW_FIELDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: str) -> bool:
    if value not in {"true", "false"}:
        raise ValueError(f"invalid canonical boolean: {value}")
    return value == "true"


def _close(actual, expected, label: str) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-12):
        raise ValueError(f"Phase-C invariant failed: {label}")


def _unique(rows: list[dict[str, str]], label: str) -> dict[tuple, dict[str, str]]:
    selected = {}
    for row in rows:
        key = (
            row["population"],
            row["condition"],
            int(row["training_seed"]),
            row["scenario"],
            int(row["seed"]),
        )
        if key in selected:
            raise ValueError(f"duplicate {label} key: {key}")
        selected[key] = row
    return selected


def verify(root: Path, out: Path) -> dict:
    out = out.resolve()
    manifest = json.loads((out / "evaluation_manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    calibration = Path(config["calibration_path"])
    if _sha256(calibration) != config["calibration_sha256"]:
        raise ValueError("calibration hash mismatch")
    references = {
        int(row["ensemble_seed"]): float(row["u_ref"])
        for row in json.loads(calibration.read_text(encoding="utf-8"))["u_ref"]
    }
    for name, expected in config["outputs"].items():
        if _sha256(out / name) != expected:
            raise ValueError(f"aggregate output hash mismatch: {name}")
    for item in config["shard_hashes"]:
        path = root / item["path"]
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"shard hash mismatch: {item['path']}")

    episodes = _read(out / "episodes.csv")
    safety = _read(out / "safety_diagnostics.csv")
    gates = _read(out / "gate_episodes.csv")
    if len(episodes) != 7_000 or len(safety) != 7_000 or len(gates) != 5_500:
        raise ValueError("Phase-C aggregate row counts do not match the frozen protocol")
    episode_map = _unique(episodes, "episode")
    safety_map = _unique(safety, "safety")
    if episode_map.keys() != safety_map.keys():
        raise ValueError("episode and safety populations differ")
    for key, episode in episode_map.items():
        diagnostic = safety_map[key]
        if (
            int(episode["steps"]) != int(diagnostic["steps"])
            or _bool(episode["success"]) != _bool(diagnostic["success"])
            or _bool(episode["fault_seen"]) != _bool(diagnostic["fault_seen"])
        ):
            raise ValueError(f"episode/safety mismatch: {key}")

    gate_map = {}
    for row in gates:
        key = (
            row["population"],
            row["condition"],
            int(row["training_seed"]),
            row["scenario"],
            int(row["episode_seed"]),
        )
        if key in gate_map:
            raise ValueError(f"duplicate gate episode: {key}")
        gate_map[key] = row
        reference = references[key[2]]
        _close(
            float(row["uncertainty_ratio_median"]),
            float(row["uncertainty_norm_median"]) / reference,
            "gate episode normalization",
        )
        scale_min = float(row["selected_scale_min"])
        scale_mean = float(row["selected_scale_mean"])
        if not (0.1 <= scale_min <= scale_mean <= 1.0):
            raise ValueError(f"invalid gate scale summary: {key}")

    transition_count = 0
    raw_values: dict[tuple, tuple[list[float], list[float], list[float]]] = {}
    with gzip.open(out / "transitions.csv.gz", mode="rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError("Phase-C transition schema mismatch")
        for row in reader:
            transition_count += 1
            numeric = [
                float(value)
                for name, value in row.items()
                if value and name not in {"population", "condition", "scenario"}
            ]
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError("non-finite Phase-C transition value")
            seed = int(row["training_seed"])
            condition = row["condition"]
            uncertainty = np.array(
                [float(row["uncertainty1"]), float(row["uncertainty2"])],
                dtype=np.float64,
            )
            norm = float(row["uncertainty_norm"])
            ratio = float(row["uncertainty_ratio"])
            scale = float(row["selected_scale"])
            raw = np.array(
                [float(row["raw_residual1"]), float(row["raw_residual2"])],
                dtype=np.float64,
            )
            gated = np.array(
                [float(row["gated_residual1"]), float(row["gated_residual2"])],
                dtype=np.float64,
            )
            baseline = np.array(
                [float(row["baseline_torque1"]), float(row["baseline_torque2"])],
                dtype=np.float64,
            )
            query = np.array(
                [
                    float(row["ensemble_query_torque1"]),
                    float(row["ensemble_query_torque2"]),
                ],
                dtype=np.float64,
            )
            candidate = np.array(
                [float(row["candidate_torque1"]), float(row["candidate_torque2"])],
                dtype=np.float64,
            )
            _close(norm, np.linalg.norm(uncertainty), "uncertainty norm")
            _close(ratio, norm / references[seed], "uncertainty normalization")
            expected_scale = (
                1.0 if condition == "A6c_gate_off_control" else max(0.1, 1.0 / (1.0 + ratio))
            )
            _close(scale, expected_scale, "gate scale")
            _close(gated, raw * scale, "gated residual")
            _close(query, np.clip(baseline + raw, -40.0, 40.0), "ensemble query")
            _close(candidate, baseline + gated, "candidate torque")
            projected = np.asarray(
                [float(row["projected_torque1"]), float(row["projected_torque2"])]
            )
            if np.max(np.abs(projected)) > 40.0 + 1e-12:
                raise ValueError("projected torque exceeds configured bound")
            executable = _bool(row["executable"])
            if executable != bool(row["plant_input_torque1"]):
                raise ValueError("plant-input presence does not match executability")
            key = (
                row["population"],
                condition,
                seed,
                row["scenario"],
                int(row["episode_seed"]),
            )
            norms, ratios, scales = raw_values.setdefault(key, ([], [], []))
            norms.append(norm)
            ratios.append(ratio)
            scales.append(scale)

    for key, (norms, ratios, scales) in raw_values.items():
        gate = gate_map[key]
        _close(float(gate["uncertainty_norm_median"]), np.median(norms), "median norm")
        _close(float(gate["uncertainty_ratio_median"]), np.median(ratios), "median ratio")
        _close(float(gate["selected_scale_mean"]), np.mean(scales), "mean scale")
        _close(float(gate["selected_scale_min"]), np.min(scales), "minimum scale")
        if int(gate["attempts"]) != len(norms):
            raise ValueError(f"gate attempt count mismatch: {key}")

    decision = json.loads((out / "decision.json").read_text(encoding="utf-8"))
    if set(decision) != {"A4c", "A6c"}:
        raise ValueError("unexpected Phase-C decision schema")
    return {
        "episodes": len(episodes),
        "gate_episodes": len(gates),
        "transitions": transition_count,
        "decision": decision,
        "verified_shard_files": len(config["shard_hashes"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/uncertainty_gate_calibration/phase_c"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(verify(root, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
