#!/usr/bin/env python3
"""Run or aggregate the frozen SARRL v1.5 calibrated-gate evaluation."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp

from sarrl.adaptation import AdaptiveContextEnv, DynamicsContextEncoder
from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm
from sarrl.envs import PlanarReachEnv
from sarrl.evaluation import (
    V15_PHASE_A_TRAINING_SEEDS,
    assert_repository_import_root,
    assert_source_tree_clean,
    canonical_json,
    evaluate_safety_episodes,
    planar_safety_config,
    v13_scenarios,
    write_run_manifest,
)
from sarrl.models import ResidualDynamicsEnsemble, UncertaintyGate
from sarrl.rl import SACAgent
from sarrl.runtime import ControlStackConfig, SARRLControlStack
from sarrl.safety import HOCBFSafetyFilter

if __package__:
    from tools.run_planar_v13 import _freeze_module
    from tools.run_planar_v14 import inputs_from_v13_manifest
else:
    from run_planar_v13 import _freeze_module
    from run_planar_v14 import inputs_from_v13_manifest


HELDOUT_SEED = 40_000
SAFETY_SEED = 50_000
EPISODES = 100
BOOTSTRAP_SEED = 150_001
BOOTSTRAP_SAMPLES = 10_000
CONDITIONS = ("A4c", "A3", "A6c", "A6c_gate_off_control")
RAW_FIELDS = (
    "population",
    "condition",
    "training_seed",
    "ensemble_seed",
    "scenario",
    "episode_seed",
    "step",
    "q1",
    "q2",
    "dq1",
    "dq2",
    "raw_residual1",
    "raw_residual2",
    "uncertainty1",
    "uncertainty2",
    "uncertainty_norm",
    "uncertainty_ratio",
    "selected_scale",
    "baseline_torque1",
    "baseline_torque2",
    "gated_residual1",
    "gated_residual2",
    "ensemble_query_torque1",
    "ensemble_query_torque2",
    "candidate_torque1",
    "candidate_torque2",
    "projected_torque1",
    "projected_torque2",
    "plant_input_torque1",
    "plant_input_torque2",
    "executable",
    "safety_certified",
    "terminated",
    "truncated",
)
GATE_EPISODE_FIELDS = (
    "population",
    "condition",
    "training_seed",
    "ensemble_seed",
    "scenario",
    "episode_seed",
    "attempts",
    "uncertainty_norm_median",
    "uncertainty_ratio_median",
    "selected_scale_mean",
    "selected_scale_min",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return ""
        return "0" if number == 0.0 else format(number, ".17g")
    return str(value)


def _write_rows(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _scalar(row.get(key)) for key in fields})


def _load_calibration(path: Path, record: dict) -> tuple[dict, float, str]:
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if raw != canonical_json(payload) + "\n":
        raise ValueError("calibration artifact is not canonical")
    if payload.get("schema_version") != 1 or payload.get("phase_a", {}).get(
        "decision"
    ) != "proceed_phase_b":
        raise ValueError("calibration artifact is not authorized by Phase A")
    if payload.get("gate") != {
        "floor_ratio": 9.0,
        "formula": "max(min_scale,1/(1+gain*uncertainty_norm/u_ref))",
        "gain": 1.0,
        "min_scale": 0.1,
        "reference_scale": 0.5,
    }:
        raise ValueError("calibration gate parameters do not match the frozen protocol")
    seed = int(record["training_seed"])
    selected = [row for row in payload["u_ref"] if row["ensemble_seed"] == seed]
    if len(selected) != 1:
        raise ValueError(f"calibration lacks unique u_ref for seed {seed}")
    reference = float(selected[0]["u_ref"])
    if not math.isfinite(reference) or reference <= 0.0:
        raise ValueError("calibration u_ref must be finite and positive")
    expected = {
        "a2_policy": record["a2_policy_checkpoint_sha256"],
        "a3_policy": record["policy_checkpoint_sha256"],
        "context": record["context_checkpoint_sha256"],
        "ensemble": record["ensemble_checkpoint_sha256"],
    }
    for role, digest in expected.items():
        matches = [
            row
            for row in payload["hashes"]
            if row["logical_role"] == role and row["training_seed"] == seed
        ]
        if len(matches) != 1 or matches[0]["sha256"] != digest:
            raise ValueError(f"calibration/source mismatch: {role} seed {seed}")
    return payload, reference, _sha256(path)


def _env(scenario) -> PlanarReachEnv:
    return PlanarReachEnv(
        mode="torque", randomization=scenario.randomization, fault=scenario.fault
    )


def _stacks(record: dict, reference: float, device: str):
    a2 = SACAgent.from_checkpoint(record["a2_policy_checkpoint"], seed=0, load_optimizers=False)
    a3 = SACAgent.from_checkpoint(record["policy_checkpoint"], seed=0, load_optimizers=False)
    encoder = DynamicsContextEncoder.load(record["context_checkpoint"], map_location="cpu")
    ensemble = ResidualDynamicsEnsemble.load(record["ensemble_checkpoint"], map_location=device)
    for module in (a2.actor, a3.actor, encoder, ensemble):
        _freeze_module(module)
    nominal = PlanarArm()
    observer = HOCBFSafetyFilter(nominal, planar_safety_config())
    gate = UncertaintyGate(gain=1.0, min_scale=0.1, reference_uncertainty=reference)
    gate_off = UncertaintyGate(gain=0.0, min_scale=0.1, reference_uncertainty=reference)
    unfiltered = ControlStackConfig(clip_ensemble_query=True)
    filtered = ControlStackConfig(require_safety=True, clip_ensemble_query=True)
    common = {"dynamics_ensemble": ensemble, "device": device}
    return {
        "A4c": (
            SARRLControlStack(
                ComputedTorqueController(nominal),
                a2,
                unfiltered,
                uncertainty_gate=gate,
                **common,
            ),
            None,
        ),
        "A3": (SARRLControlStack(ComputedTorqueController(nominal), a3), encoder),
        "A6c": (
            SARRLControlStack(
                ComputedTorqueController(nominal),
                a3,
                filtered,
                safety_filter=observer,
                uncertainty_gate=gate,
                **common,
            ),
            encoder,
        ),
        "A6c_gate_off_control": (
            SARRLControlStack(
                ComputedTorqueController(nominal),
                a3,
                filtered,
                safety_filter=observer,
                uncertainty_gate=gate_off,
                **common,
            ),
            encoder,
        ),
    }, observer


def _prefixed_rows(population, condition, training_seed, scenario, rows) -> list[dict]:
    return [
        {
            "population": population,
            "condition": condition,
            "training_seed": training_seed,
            **asdict(row),
        }
        for row in rows
    ]


def _evaluate_cell(
    *,
    population: str,
    condition: str,
    training_seed: int,
    scenario,
    seed: int,
    stack,
    encoder,
    observer,
    reference: float,
    raw_writer,
) -> tuple[list[dict], list[dict], list[dict]]:
    base_env = _env(scenario)
    env = AdaptiveContextEnv(base_env, encoder, device="cpu") if encoder is not None else base_env
    gate_values: dict[int, tuple[list[float], list[float]]] = {}
    capture_raw = population == "safety" and condition in {
        "A6c",
        "A6c_gate_off_control",
    }

    def callback(item):
        command = item["command"]
        episode_seed = int(item["episode_seed"])
        uncertainty_norm = float(np.linalg.norm(command.uncertainty))
        norms, scales = gate_values.setdefault(episode_seed, ([], []))
        norms.append(uncertainty_norm)
        scales.append(float(command.uncertainty_scale))
        if not capture_raw:
            return
        info = item["info"]
        candidate = command.baseline_torque + command.gated_residual
        plant = None if info is None else np.asarray(info["plant_input_torque"])
        row = {
            "population": population,
            "condition": condition,
            "training_seed": training_seed,
            "ensemble_seed": training_seed,
            "scenario": scenario.key,
            "episode_seed": episode_seed,
            "step": item["step"],
            "q1": item["state"][0],
            "q2": item["state"][1],
            "dq1": item["state"][2],
            "dq2": item["state"][3],
            "raw_residual1": command.raw_residual[0],
            "raw_residual2": command.raw_residual[1],
            "uncertainty1": command.uncertainty[0],
            "uncertainty2": command.uncertainty[1],
            "uncertainty_norm": uncertainty_norm,
            "uncertainty_ratio": uncertainty_norm / reference,
            "selected_scale": command.uncertainty_scale,
            "baseline_torque1": command.baseline_torque[0],
            "baseline_torque2": command.baseline_torque[1],
            "gated_residual1": command.gated_residual[0],
            "gated_residual2": command.gated_residual[1],
            "ensemble_query_torque1": command.ensemble_query_torque[0],
            "ensemble_query_torque2": command.ensemble_query_torque[1],
            "candidate_torque1": candidate[0],
            "candidate_torque2": candidate[1],
            "projected_torque1": command.torque[0],
            "projected_torque2": command.torque[1],
            "plant_input_torque1": None if plant is None else plant[0],
            "plant_input_torque2": None if plant is None else plant[1],
            "executable": command.executable,
            "safety_certified": command.safety_certified,
            "terminated": item["terminated"],
            "truncated": item["truncated"],
        }
        raw_writer.writerow({key: _scalar(row[key]) for key in RAW_FIELDS})

    outcomes, diagnostics = evaluate_safety_episodes(
        stack,
        observer,
        env,
        EPISODES,
        seed,
        scenario=scenario.key,
        controller=f"{condition}_train_seed_{training_seed}",
        context_residual_limit=8.0 if encoder is not None else None,
        transition_callback=callback if condition != "A3" else None,
    )
    gate_rows = []
    for episode_seed, (norms, scales) in sorted(gate_values.items()):
        gate_rows.append(
            {
                "population": population,
                "condition": condition,
                "training_seed": training_seed,
                "ensemble_seed": training_seed,
                "scenario": scenario.key,
                "episode_seed": episode_seed,
                "attempts": len(norms),
                "uncertainty_norm_median": float(np.median(norms)),
                "uncertainty_ratio_median": float(np.median(norms)) / reference,
                "selected_scale_mean": float(np.mean(scales)),
                "selected_scale_min": float(np.min(scales)),
            }
        )
    print(
        f"{population} {condition} seed={training_seed} scenario={scenario.key}: "
        f"success={sum(row.success for row in outcomes)}/{len(outcomes)}",
        flush=True,
    )
    return (
        _prefixed_rows(population, condition, training_seed, scenario.key, outcomes),
        _prefixed_rows(population, condition, training_seed, scenario.key, diagnostics),
        gate_rows,
    )


def run_shard(root: Path, out: Path, record: dict, calibration: Path, device: str) -> None:
    seed = int(record["training_seed"])
    shard = out / "shards" / f"training_seed_{seed}"
    complete = shard / "complete.json"
    if complete.is_file():
        payload = json.loads(complete.read_text(encoding="utf-8"))
        for name, expected in payload["sha256"].items():
            if _sha256(shard / name) != expected:
                raise ValueError(f"completed Phase-C shard hash mismatch: {name}")
        print(f"Phase-C seed {seed} already complete; verified and skipped")
        return
    _, reference, calibration_hash = _load_calibration(calibration, record)
    shard.mkdir(parents=True, exist_ok=True)
    manifest = {
        "release_target": "v1.5.0",
        "campaign": "calibrated_gate_evaluation_phase_c",
        "training_seed": seed,
        "input": record,
        "calibration_path": str(calibration.resolve()),
        "calibration_sha256": calibration_hash,
        "u_ref": reference,
        "device": device,
        "heldout": {"seed_start": HELDOUT_SEED, "episodes": EPISODES},
        "safety": {
            "seed_start": SAFETY_SEED,
            "episodes_per_scenario": EPISODES,
            "scenarios": [scenario.key for scenario in v13_scenarios()],
        },
    }
    write_run_manifest(shard / "evaluation_manifest.json", manifest, root=root)
    stacks, observer = _stacks(record, reference, device)
    outcomes = []
    diagnostics = []
    gate_rows = []
    raw_path = shard / "transitions.csv.gz"
    with raw_path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, mtime=0
        ) as binary:
            with io.TextIOWrapper(binary, encoding="utf-8", newline="") as handle:
                raw_writer = csv.DictWriter(
                    handle, fieldnames=RAW_FIELDS, lineterminator="\n"
                )
                raw_writer.writeheader()
                heldout = next(
                    scenario
                    for scenario in v13_scenarios()
                    if scenario.key == "id_reference"
                )
                for condition in ("A4c", "A6c"):
                    rows = _evaluate_cell(
                        population="heldout",
                        condition=condition,
                        training_seed=seed,
                        scenario=heldout,
                        seed=HELDOUT_SEED,
                        stack=stacks[condition][0],
                        encoder=stacks[condition][1],
                        observer=observer,
                        reference=reference,
                        raw_writer=raw_writer,
                    )
                    outcomes.extend(rows[0])
                    diagnostics.extend(rows[1])
                    gate_rows.extend(rows[2])
                for scenario in v13_scenarios():
                    for condition in CONDITIONS:
                        rows = _evaluate_cell(
                            population="safety",
                            condition=condition,
                            training_seed=seed,
                            scenario=scenario,
                            seed=SAFETY_SEED,
                            stack=stacks[condition][0],
                            encoder=stacks[condition][1],
                            observer=observer,
                            reference=reference,
                            raw_writer=raw_writer,
                        )
                        outcomes.extend(rows[0])
                        diagnostics.extend(rows[1])
                        gate_rows.extend(rows[2])
    _write_rows(shard / "episodes.csv", tuple(outcomes[0]), outcomes)
    _write_rows(shard / "safety_diagnostics.csv", tuple(diagnostics[0]), diagnostics)
    _write_rows(shard / "gate_episodes.csv", GATE_EPISODE_FIELDS, gate_rows)
    files = (
        "evaluation_manifest.json",
        "episodes.csv",
        "safety_diagnostics.csv",
        "gate_episodes.csv",
        "transitions.csv.gz",
    )
    complete.write_text(
        json.dumps(
            {
                "status": "complete",
                "training_seed": seed,
                "episodes": len(outcomes),
                "sha256": {name: _sha256(shard / name) for name in files},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: str) -> bool:
    return value.lower() == "true"


def _bootstrap_difference(matrix: np.ndarray) -> tuple[float, float, float]:
    if matrix.shape != (5, 100):
        raise ValueError(f"paired Phase-C matrix must be 5x100, got {matrix.shape}")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, 100, size=(BOOTSTRAP_SAMPLES, 100))
    distribution = matrix[:, draws].mean(axis=(0, 2))
    return (
        float(matrix.mean()),
        float(np.quantile(distribution, 0.025)),
        float(np.quantile(distribution, 0.975)),
    )


def _new_metric(rows, condition, scenario, metric) -> np.ndarray:
    selected = {
        (int(row["training_seed"]), int(row["seed"])): (
            _bool(row[metric]) if metric in {"success", "unsafe_episode"} else float(row[metric])
        )
        for row in rows
        if row["population"] == "safety"
        and row["condition"] == condition
        and row["scenario"] == scenario
    }
    return np.asarray(
        [
            [selected[(training_seed, SAFETY_SEED + offset)] for offset in range(100)]
            for training_seed in range(5)
        ],
        dtype=np.float64,
    )


def _v14_a2(root: Path, metric: str, scenario: str) -> np.ndarray:
    filename = "episodes.csv" if metric == "success" else "safety_diagnostics.csv"
    rows = _read_csv(root / "results/quantified_safety" / filename)
    pattern = re.compile(r"A2_unfiltered_train_seed_(\d+)$")
    selected = {}
    for row in rows:
        match = pattern.match(row["controller"])
        if match and row["scenario"] == scenario:
            selected[(int(match.group(1)), int(row["seed"]))] = _bool(row[metric])
    return np.asarray(
        [
            [selected[(training_seed, SAFETY_SEED + offset)] for offset in range(100)]
            for training_seed in range(5)
        ],
        dtype=np.float64,
    )


def _comparisons(root: Path, episodes: list[dict], safety: list[dict]) -> list[dict]:
    rows = []
    specs = (
        ("A4c_vs_A2", "A4c", "A2"),
        ("A6c_vs_gate_off", "A6c", "A6c_gate_off_control"),
        ("A6c_vs_A3_total_effect", "A6c", "A3"),
    )
    for label, treatment, reference in specs:
        for scenario in (item.key for item in v13_scenarios()):
            for metric, source in (("success", episodes), ("unsafe_episode", safety)):
                treatment_values = _new_metric(source, treatment, scenario, metric)
                reference_values = (
                    _v14_a2(root, metric, scenario)
                    if reference == "A2"
                    else _new_metric(source, reference, scenario, metric)
                )
                estimate, low, high = _bootstrap_difference(
                    treatment_values - reference_values
                )
                rows.append(
                    {
                        "comparison": label,
                        "scenario": scenario,
                        "metric": metric,
                        "difference": estimate,
                        "ci95_low": low,
                        "ci95_high": high,
                        "bootstrap_seed": BOOTSTRAP_SEED,
                        "bootstrap_samples": BOOTSTRAP_SAMPLES,
                    }
                )
    return rows


def _criteria(comparisons: list[dict]) -> dict:
    result = {}
    for stack, label in (("A4c", "A4c_vs_A2"), ("A6c", "A6c_vs_gate_off")):
        selected = [row for row in comparisons if row["comparison"] == label]
        noninferior = all(
            row["ci95_low"] >= -0.05
            for row in selected
            if row["metric"] == "success"
        ) and all(
            row["ci95_high"] <= 0.05
            for row in selected
            if row["metric"] == "unsafe_episode"
        )
        ood = next(
            row
            for row in selected
            if row["metric"] == "success" and row["scenario"] == "ood_compound"
        )
        strict_benefit = ood["ci95_low"] > 0.0
        result[stack] = {
            "all_noninferiority_bounds_pass": noninferior,
            "ood_success_strict_benefit_pass": strict_benefit,
            "useful_calibrated_gate": noninferior and strict_benefit,
        }
    return result


def _heldout_comparisons(root: Path, episodes: list[dict]) -> list[dict]:
    rows = []
    specs = (
        (
            "A4c_vs_A2",
            "A4c",
            root / "artifacts/planar_sac_5seed/heldout_episodes.csv",
        ),
        (
            "A6c_vs_A3",
            "A6c",
            root / "results/planar_ablations/A3_residual_sac_context/heldout_episodes.csv",
        ),
    )
    for label, condition, reference_path in specs:
        treatment = {
            (int(row["training_seed"]), int(row["seed"])): _bool(row["success"])
            for row in episodes
            if row["population"] == "heldout" and row["condition"] == condition
        }
        reference = {}
        pattern = re.compile(r"sac_train_seed_(\d+)$")
        for row in _read_csv(reference_path):
            match = pattern.match(row["controller"])
            if match:
                reference[(int(match.group(1)), int(row["seed"]))] = _bool(row["success"])
        matrix = np.asarray(
            [
                [
                    float(treatment[(seed, HELDOUT_SEED + offset)])
                    - float(reference[(seed, HELDOUT_SEED + offset)])
                    for offset in range(100)
                ]
                for seed in range(5)
            ]
        )
        estimate, low, high = _bootstrap_difference(matrix)
        rows.append(
            {
                "comparison": label,
                "metric": "success",
                "difference": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
            }
        )
    return rows


def _ks_rows(root: Path, gate_rows: list[dict], calibration: Path) -> list[dict]:
    phase_a = _read_csv(
        root / "results/uncertainty_gate_calibration/phase_a/episodes.csv"
    )
    calibration_payload = json.loads(calibration.read_text(encoding="utf-8"))
    references = {
        int(row["ensemble_seed"]): float(row["u_ref"])
        for row in calibration_payload["u_ref"]
    }
    rows = []
    for condition, calibration_policy in (
        ("A4c", "A2"),
        ("A6c", "A3"),
        ("A6c_gate_off_control", "A3"),
    ):
        for seed in V15_PHASE_A_TRAINING_SEEDS:
            calibration = [
                float(row["uncertainty_median"]) / references[seed]
                for row in phase_a
                if row["policy"] == calibration_policy
                and int(row["training_seed"]) == seed
            ]
            for population, scenario in (
                ("heldout", "id_reference"),
                *(("safety", item.key) for item in v13_scenarios()),
            ):
                if condition == "A6c_gate_off_control" and population == "heldout":
                    continue
                deployment = [
                    float(row["uncertainty_ratio_median"])
                    for row in gate_rows
                    if row["condition"] == condition
                    and row["population"] == population
                    and row["scenario"] == scenario
                    and int(row["training_seed"]) == seed
                ]
                if len(calibration) < 90 or len(deployment) < 90:
                    raise ValueError("KS diagnostic requires at least 90 episodes per sample")
                rows.append(
                    {
                        "condition": condition,
                        "training_seed": seed,
                        "population": population,
                        "scenario": scenario,
                        "calibration_policy": calibration_policy,
                        "calibration_episodes": len(calibration),
                        "deployment_episodes": len(deployment),
                        "ks_distance": float(ks_2samp(calibration, deployment).statistic),
                    }
                )
    return rows


def _summary_rows(episodes: list[dict], safety: list[dict]) -> list[dict]:
    rows = []
    keys = sorted(
        {
            (row["population"], row["condition"], row["scenario"])
            for row in episodes
        }
    )
    for population, condition, scenario in keys:
        outcome_rows = [
            row
            for row in episodes
            if (row["population"], row["condition"], row["scenario"])
            == (population, condition, scenario)
        ]
        safety_rows = [
            row
            for row in safety
            if (row["population"], row["condition"], row["scenario"])
            == (population, condition, scenario)
        ]
        rows.append(
            {
                "population": population,
                "condition": condition,
                "scenario": scenario,
                "episodes": len(outcome_rows),
                "successes": sum(_bool(row["success"]) for row in outcome_rows),
                "success_rate": float(
                    np.mean([_bool(row["success"]) for row in outcome_rows])
                ),
                "unsafe_episodes": sum(
                    _bool(row["unsafe_episode"]) for row in safety_rows
                ),
                "unsafe_episode_rate": float(
                    np.mean([_bool(row["unsafe_episode"]) for row in safety_rows])
                ),
                "safety_infeasible_episodes": sum(
                    _bool(row["safety_infeasible"]) for row in safety_rows
                ),
            }
        )
    return rows


def aggregate(root: Path, out: Path, calibration: Path) -> None:
    episodes = []
    safety = []
    gates = []
    shard_hashes = []
    for seed in V15_PHASE_A_TRAINING_SEEDS:
        shard = out / "shards" / f"training_seed_{seed}"
        complete = json.loads((shard / "complete.json").read_text(encoding="utf-8"))
        if complete["episodes"] != 1_400:
            raise ValueError(f"Phase-C seed {seed} has wrong episode count")
        for name, expected in complete["sha256"].items():
            actual = _sha256(shard / name)
            if actual != expected:
                raise ValueError(f"Phase-C seed {seed} hash mismatch: {name}")
            shard_hashes.append(
                {
                    "training_seed": seed,
                    "path": str((shard / name).resolve().relative_to(root)),
                    "sha256": actual,
                }
            )
        episodes.extend(_read_csv(shard / "episodes.csv"))
        safety.extend(_read_csv(shard / "safety_diagnostics.csv"))
        gates.extend(_read_csv(shard / "gate_episodes.csv"))
    _write_rows(out / "episodes.csv", tuple(episodes[0]), episodes)
    _write_rows(out / "safety_diagnostics.csv", tuple(safety[0]), safety)
    _write_rows(out / "gate_episodes.csv", GATE_EPISODE_FIELDS, gates)
    raw_path = out / "transitions.csv.gz"
    with raw_path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, mtime=0
        ) as binary:
            with io.TextIOWrapper(binary, encoding="utf-8", newline="") as destination:
                wrote_header = False
                for seed in V15_PHASE_A_TRAINING_SEEDS:
                    source_path = (
                        out
                        / "shards"
                        / f"training_seed_{seed}"
                        / "transitions.csv.gz"
                    )
                    with gzip.open(
                        source_path, mode="rt", encoding="utf-8", newline=""
                    ) as source:
                        header = source.readline()
                        if not wrote_header:
                            destination.write(header)
                            wrote_header = True
                        elif header.rstrip("\r\n").split(",") != list(RAW_FIELDS):
                            raise ValueError("Phase-C transition schema mismatch")
                        for line in source:
                            destination.write(line)
    comparisons = _comparisons(root, episodes, safety)
    heldout_comparisons = _heldout_comparisons(root, episodes)
    ks_rows = _ks_rows(root, gates, calibration)
    summaries = _summary_rows(episodes, safety)
    _write_rows(out / "paired_comparisons.csv", tuple(comparisons[0]), comparisons)
    _write_rows(
        out / "heldout_comparisons.csv",
        tuple(heldout_comparisons[0]),
        heldout_comparisons,
    )
    _write_rows(out / "distribution_shift.csv", tuple(ks_rows[0]), ks_rows)
    _write_rows(out / "summary.csv", tuple(summaries[0]), summaries)
    criteria = _criteria(comparisons)
    (out / "decision.json").write_text(
        json.dumps(criteria, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "release_target": "v1.5.0",
        "campaign": "calibrated_gate_evaluation_phase_c",
        "calibration_path": str(calibration.resolve()),
        "calibration_sha256": _sha256(calibration),
        "training_seeds": list(V15_PHASE_A_TRAINING_SEEDS),
        "new_episodes": len(episodes),
        "shard_hashes": shard_hashes,
        "outputs": {
            name: _sha256(out / name)
            for name in (
                "episodes.csv",
                "safety_diagnostics.csv",
                "gate_episodes.csv",
                "transitions.csv.gz",
                "paired_comparisons.csv",
                "heldout_comparisons.csv",
                "distribution_shift.csv",
                "summary.csv",
                "decision.json",
            )
        },
    }
    write_run_manifest(out / "evaluation_manifest.json", manifest, root=root)
    print(json.dumps(criteria, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seed", type=int, choices=V15_PHASE_A_TRAINING_SEEDS)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/uncertainty_gate_calibration/phase_c"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("results/uncertainty_gate_calibration/calibration.json"),
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path("results/ood_fault_robustness/evaluation_manifest.json"),
    )
    args = parser.parse_args()
    if args.aggregate == (args.training_seed is not None):
        raise SystemExit("select exactly one of --aggregate or --training-seed")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    root = Path(__file__).resolve().parents[1]
    assert_repository_import_root(root)
    assert_source_tree_clean(root)
    if args.aggregate:
        aggregate(root, args.output.resolve(), args.calibration.resolve())
    else:
        records = inputs_from_v13_manifest(root, args.input_manifest)
        record = next(row for row in records if row["training_seed"] == args.training_seed)
        run_shard(root, args.output.resolve(), record, args.calibration.resolve(), args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
