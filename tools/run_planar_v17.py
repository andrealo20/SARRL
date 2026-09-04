#!/usr/bin/env python3
"""Run the frozen v1.7 safety-aware training and evaluation campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm
from sarrl.envs import PlanarReachEnv
from sarrl.evaluation import (
    V17_BOOTSTRAP_SAMPLES,
    V17_BOOTSTRAP_SEED,
    V17_CONDITIONS,
    V17_EVALUATION,
    V17_INFEASIBLE_REWARD,
    V17_START_STEPS,
    V17_TRAINING_SEEDS,
    V17_TRAINING_STEPS,
    V17_VALIDATION_EPISODES,
    V17_VALIDATION_SEED,
    assert_repository_import_root,
    assert_source_tree_clean,
    classify_v17_result,
    evaluate_safety_episodes,
    planar_safety_config,
    repository_commit,
    two_level_paired_interval,
    v13_scenarios,
    v17_protocol_dict,
    write_episode_csv,
    write_run_manifest,
)
from sarrl.rl import SACAgent
from sarrl.runtime import ControlStackConfig, SARRLControlStack
from sarrl.safety import HOCBFSafetyFilter

TRAINING_BATCH_SIZE = 256
TRAINING_HIDDEN = (256, 256)
TRAINING_REPLAY_CAPACITY = 200_000
TRAINING_UPDATE_EVERY = 1
TRAINING_CHECKPOINT_EVERY = 50_000
TRAINING_VALIDATE_EVERY = 25_000
INTERVENTION_TOLERANCE = 1e-9
PAIRING_PATTERN = re.compile(r"^(C[01]_[a-z0-9_]+)_train_seed_(\d+)$")
V17_TRAINING_SOURCE_COMMIT = "2c149de747d6f6cbd244339a5a54a9831b459a93"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_diagnostic_csv(path: Path, rows) -> None:
    """Write a non-empty collection of same-schema diagnostic dataclasses."""
    if not rows:
        raise ValueError("diagnostic output must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _training_dir(output: Path, condition: str, training_seed: int) -> Path:
    return output / "training" / condition / f"seed_{training_seed}"


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("train_step"))
    except ValueError:
        return -1


def _validate_training_manifest(
    manifest: dict,
    *,
    condition: str,
    training_seed: int,
    expected_commit: str,
) -> None:
    if manifest["runtime"].get("git_commit") != expected_commit:
        raise ValueError("training shard source commit does not match frozen source")
    config = manifest["config"]
    if int(config["seed"]) != training_seed:
        raise ValueError("training shard seed does not match its directory")
    if int(config["requested_steps"]) != V17_TRAINING_STEPS:
        raise ValueError("training shard has a different training budget")
    if int(config["replay_capacity"]) != TRAINING_REPLAY_CAPACITY:
        raise ValueError("training shard replay capacity does not match protocol")

    expected_agent = {
        "gamma": 0.99,
        "tau": 0.005,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "alpha_lr": 3e-4,
        "init_alpha": 0.2,
        "hidden": list(TRAINING_HIDDEN),
    }
    if config["agent_config"] != expected_agent:
        raise ValueError("training shard agent configuration does not match protocol")

    expected_trainer = {
        "batch_size": TRAINING_BATCH_SIZE,
        "start_steps": V17_START_STEPS,
        "update_every": TRAINING_UPDATE_EVERY,
        "context_checkpoint": None,
        "context_checkpoint_sha256": None,
        "training_hocbf": condition == "C1_inloop_hocbf",
        "validation_hocbf": True,
        "infeasible_reward": V17_INFEASIBLE_REWARD,
    }
    if config["trainer"] != expected_trainer:
        raise ValueError("training shard trainer configuration does not match protocol")
    if config["context"] != {
        "enabled": False,
        "checkpoint": None,
        "checkpoint_sha256": None,
        "runtime_device": None,
        "latent_dim": None,
    }:
        raise ValueError("training shard context configuration does not match protocol")

    if config["validation"] != {
        "every": TRAINING_VALIDATE_EVERY,
        "episodes": V17_VALIDATION_EPISODES,
        "seed": V17_VALIDATION_SEED,
    }:
        raise ValueError("training shard validation configuration does not match protocol")

    environment = config["environment"]
    expected_safety_config = json.loads(json.dumps(asdict(planar_safety_config())))
    if condition == "C1_inloop_hocbf":
        if environment.get("environment_type") != "safety_projected":
            raise ValueError("in-loop shard did not use the projected environment")
        if float(environment.get("infeasible_reward")) != V17_INFEASIBLE_REWARD:
            raise ValueError("in-loop shard infeasible reward does not match protocol")
        if environment.get("safety_config") != expected_safety_config:
            raise ValueError("in-loop shard safety configuration does not match protocol")
        if float(environment.get("intervention_tolerance")) != INTERVENTION_TOLERANCE:
            raise ValueError("in-loop intervention tolerance does not match protocol")
        environment = environment["base_environment"]
    elif environment.get("environment_type") == "safety_projected":
        raise ValueError("post-hoc control unexpectedly used projected training")

    expected_environment = {
        "mode": "residual",
        "dt": 0.02,
        "max_steps": 250,
        "torque_limit": 40.0,
        "residual_limit": 8.0,
        "success_radius": 0.05,
        "randomization": {
            "mass_fraction": 0.15,
            "friction_fraction": 0.30,
            "motor_gain_fraction": 0.15,
            "payload_range": [0.0, 1.0],
            "sensor_noise_std": 0.0,
            "action_delay_max": 2,
        },
        "fault": None,
    }
    if environment != expected_environment:
        raise ValueError("training shard environment does not match protocol")

    expected_projected_environment = {
        "environment_type": "safety_projected",
        "base_environment": expected_environment,
        "safety_config": expected_safety_config,
        "infeasible_reward": V17_INFEASIBLE_REWARD,
        "intervention_tolerance": INTERVENTION_TOLERANCE,
    }
    safety = config["safety"]
    expected_training_environment = (
        expected_projected_environment if condition == "C1_inloop_hocbf" else None
    )
    if safety != {
        "training_hocbf": condition == "C1_inloop_hocbf",
        "validation_hocbf": True,
        "infeasible_reward": V17_INFEASIBLE_REWARD,
        "training_environment": expected_training_environment,
        "validation_environment": expected_projected_environment,
    }:
        raise ValueError("training shard safety configuration does not match protocol")


def _selected_validation_row(rows: list[dict[str, str]]) -> dict[str, str]:
    expected_steps = list(
        range(TRAINING_VALIDATE_EVERY, V17_TRAINING_STEPS + 1, TRAINING_VALIDATE_EVERY)
    )
    actual_steps = [int(row["step"]) for row in rows]
    if actual_steps != expected_steps:
        raise ValueError("training shard validation schedule does not match protocol")
    for row in rows:
        if int(row["episodes"]) != V17_VALIDATION_EPISODES:
            raise ValueError("training shard validation cell has the wrong size")
        successes = int(row["successes"])
        success_rate = float(row["success_rate"])
        metrics = (
            success_rate,
            float(row["reward_mean"]),
            float(row["reward_std"]),
            float(row["final_distance_mean"]),
        )
        if not 0 <= successes <= V17_VALIDATION_EPISODES:
            raise ValueError("training shard validation success count is invalid")
        if not all(np.isfinite(value) for value in metrics):
            raise ValueError("training shard validation metrics must be finite")
        if success_rate != successes / V17_VALIDATION_EPISODES:
            raise ValueError("training shard validation rate is inconsistent")

    best_key = max(
        (float(row["success_rate"]), float(row["reward_mean"])) for row in rows
    )
    return next(
        row
        for row in rows
        if (float(row["success_rate"]), float(row["reward_mean"])) == best_key
    )


def _validate_selection_artifacts(run_dir: Path) -> tuple[dict[str, str], str]:
    selected = run_dir / "best.pt"
    validation = run_dir / "validation.csv"
    selection_path = run_dir / "selection.json"
    validation_rows = _read_csv(validation)
    selected_row = _selected_validation_row(validation_rows)
    selection = json.loads(selection_path.read_text())
    if selection.get("checkpoint") != "best.pt":
        raise ValueError("training shard selected an unexpected checkpoint")
    if selection.get("tie_break") != "success_rate_then_reward_then_earliest_step":
        raise ValueError("training shard used an unexpected selection rule")
    if int(selection.get("best_step", -1)) != int(selected_row["step"]):
        raise ValueError("training shard selected the wrong validation step")
    selection_key = selection.get("selection_key", {})
    if float(selection_key.get("success_rate", float("nan"))) != float(
        selected_row["success_rate"]
    ):
        raise ValueError("selected checkpoint success rate is inconsistent")
    if float(selection_key.get("reward_mean", float("nan"))) != float(
        selected_row["reward_mean"]
    ):
        raise ValueError("selected checkpoint reward is inconsistent")
    selected_hash = _sha256(selected)
    if selection.get("checkpoint_sha256") != selected_hash:
        raise ValueError("selected checkpoint hash is inconsistent")
    return selected_row, selected_hash


def _validate_training_episode_artifacts(run_dir: Path, condition: str) -> None:
    episodes = _read_csv(run_dir / "episodes.csv")
    if not episodes:
        raise ValueError("training shard has no completed episodes")
    episode_numbers = [int(row["episode"]) for row in episodes]
    if episode_numbers != list(range(1, len(episodes) + 1)):
        raise ValueError("training shard episode numbering is inconsistent")
    steps = [int(row["step"]) for row in episodes]
    if steps != sorted(set(steps)) or steps[-1] > V17_TRAINING_STEPS:
        raise ValueError("training shard episode steps are inconsistent")
    for row in episodes:
        values = (float(row["reward"]), float(row["final_distance"]))
        if not all(np.isfinite(value) for value in values):
            raise ValueError("training shard episode metrics must be finite")
        if int(row["success"]) not in (0, 1):
            raise ValueError("training shard episode success flag is invalid")

    safety_path = run_dir / "training_safety.csv"
    if condition == "C0_posthoc_hocbf":
        if safety_path.is_file():
            raise ValueError("post-hoc control has unexpected training safety diagnostics")
        return

    safety_rows = _read_csv(safety_path)
    if len(safety_rows) != len(episodes):
        raise ValueError("training safety rows do not match completed episodes")
    for episode, safety in zip(episodes, safety_rows, strict=True):
        if (safety["episode"], safety["step"]) != (
            episode["episode"],
            episode["step"],
        ):
            raise ValueError("training safety rows are not aligned with episodes")
        infeasible = int(safety["safety_infeasible"])
        attempts = int(safety["command_attempts"])
        certified = int(safety["safety_certified_steps"])
        interventions = int(safety["safety_intervention_steps"])
        if infeasible not in (0, 1) or attempts <= 0:
            raise ValueError("training safety counters are invalid")
        if certified != attempts - infeasible or not 0 <= interventions <= attempts:
            raise ValueError("training safety counters are inconsistent")
        correction_sum = float(safety["safety_correction_sum"])
        correction_max = float(safety["safety_correction_max"])
        if (
            not np.isfinite(correction_sum)
            or not np.isfinite(correction_max)
            or correction_sum < 0.0
            or correction_max < 0.0
            or correction_max > correction_sum + 1e-12
        ):
            raise ValueError("training safety correction metrics are inconsistent")


def _existing_training_state(
    run_dir: Path,
    *,
    expected_commit: str,
    expected_condition: str,
    expected_training_seed: int,
) -> tuple[Path | None, bool]:
    checkpoints = [
        path for path in run_dir.glob("train_step*.pt") if _checkpoint_step(path) >= 0
    ]
    final = run_dir / "training_final.pt"
    if not final.is_file() and not checkpoints:
        return None, False

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"existing training state lacks a manifest: {run_dir}")
    manifest = json.loads(manifest_path.read_text())
    _validate_training_manifest(
        manifest,
        condition=expected_condition,
        training_seed=expected_training_seed,
        expected_commit=expected_commit,
    )

    if final.is_file():
        required = (
            run_dir / "best.pt",
            run_dir / "episodes.csv",
            run_dir / "validation.csv",
            run_dir / "selection.json",
        )
        if expected_condition == "C1_inloop_hocbf":
            required += (run_dir / "training_safety.csv",)
        if not all(path.is_file() for path in required):
            return final, False
        _validate_selection_artifacts(run_dir)
        _validate_training_episode_artifacts(run_dir, expected_condition)
        return None, True
    return max(checkpoints, key=_checkpoint_step), False


def train_shard(
    root: Path,
    output: Path,
    condition: str,
    training_seed: int,
) -> None:
    if condition not in V17_CONDITIONS:
        raise ValueError(f"unknown v1.7 condition: {condition}")
    if training_seed not in V17_TRAINING_SEEDS:
        raise ValueError(f"training seed is outside the frozen protocol: {training_seed}")

    assert_source_tree_clean(root)
    commit = repository_commit(root)
    if commit is None:
        raise RuntimeError("official v1.7 training requires a Git commit")
    run_dir = _training_dir(output, condition, training_seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    resume, complete = _existing_training_state(
        run_dir,
        expected_commit=commit,
        expected_condition=condition,
        expected_training_seed=training_seed,
    )
    if complete:
        print(f"{condition} seed={training_seed} is already complete")
        return

    command = [
        sys.executable,
        str(root / "tools" / "train_sac.py"),
        "--mode",
        "residual",
        "--steps",
        str(V17_TRAINING_STEPS),
        "--seed",
        str(training_seed),
        "--start-steps",
        str(V17_START_STEPS),
        "--batch-size",
        str(TRAINING_BATCH_SIZE),
        "--hidden",
        *(str(value) for value in TRAINING_HIDDEN),
        "--update-every",
        str(TRAINING_UPDATE_EVERY),
        "--replay-capacity",
        str(TRAINING_REPLAY_CAPACITY),
        "--randomize",
        "--validation-hocbf",
        "--infeasible-reward",
        str(V17_INFEASIBLE_REWARD),
        "--output",
        str(run_dir),
        "--checkpoint-every",
        str(TRAINING_CHECKPOINT_EVERY),
        "--validate-every",
        str(TRAINING_VALIDATE_EVERY),
        "--validation-episodes",
        str(V17_VALIDATION_EPISODES),
        "--validation-seed",
        str(V17_VALIDATION_SEED),
    ]
    if condition == "C1_inloop_hocbf":
        command.append("--training-hocbf")
    if resume is not None:
        command.extend(["--resume", str(resume)])
        print(f"resuming {condition} seed={training_seed} from {resume.name}")
    subprocess.run(command, cwd=root, check=True)


def build_checkpoint_inventory(root: Path, output: Path) -> Path:
    assert_source_tree_clean(root)
    records = []
    for condition in V17_CONDITIONS:
        for training_seed in V17_TRAINING_SEEDS:
            run_dir = _training_dir(output, condition, training_seed)
            manifest_path = run_dir / "run_manifest.json"
            final = run_dir / "training_final.pt"
            selected = run_dir / "best.pt"
            episodes = run_dir / "episodes.csv"
            validation = run_dir / "validation.csv"
            selection_path = run_dir / "selection.json"
            training_safety = run_dir / "training_safety.csv"
            required = (
                manifest_path,
                final,
                selected,
                episodes,
                validation,
                selection_path,
            )
            if condition == "C1_inloop_hocbf":
                required += (training_safety,)
            missing = [path.name for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"incomplete {condition} seed={training_seed}: {', '.join(missing)}"
                )
            manifest = json.loads(manifest_path.read_text())
            _validate_training_manifest(
                manifest,
                condition=condition,
                training_seed=training_seed,
                expected_commit=V17_TRAINING_SOURCE_COMMIT,
            )
            selected_row, selected_hash = _validate_selection_artifacts(run_dir)
            _validate_training_episode_artifacts(run_dir, condition)
            records.append(
                {
                    "condition": condition,
                    "training_seed": training_seed,
                    "best_step": int(selected_row["step"]),
                    "checkpoint": _relative(selected, root),
                    "checkpoint_sha256": selected_hash,
                    "training_checkpoint": _relative(final, root),
                    "training_checkpoint_sha256": _sha256(final),
                    "run_manifest": _relative(manifest_path, root),
                    "run_manifest_sha256": _sha256(manifest_path),
                    "episodes": _relative(episodes, root),
                    "episodes_sha256": _sha256(episodes),
                    "validation": _relative(validation, root),
                    "validation_sha256": _sha256(validation),
                    "selection": _relative(selection_path, root),
                    "selection_sha256": _sha256(selection_path),
                    "training_safety": (
                        _relative(training_safety, root)
                        if training_safety.is_file()
                        else None
                    ),
                    "training_safety_sha256": (
                        _sha256(training_safety) if training_safety.is_file() else None
                    ),
                }
            )

    path = output / "checkpoint_inventory.json"
    write_run_manifest(
        path,
        {
            "protocol": v17_protocol_dict(),
            "training_source_commit": V17_TRAINING_SOURCE_COMMIT,
            "records": records,
        },
        root=root,
    )
    print(f"retained {len(records)} selected checkpoints in {path}")
    return path


def _load_inventory(root: Path, output: Path) -> tuple[Path, dict]:
    path = output / "checkpoint_inventory.json"
    if not path.is_file():
        raise FileNotFoundError("checkpoint inventory is missing")
    payload = json.loads(path.read_text())
    commit = repository_commit(root)
    if payload["runtime"].get("git_commit") != commit:
        raise ValueError("checkpoint inventory source commit does not match current source")
    records = payload["config"].get("records", [])
    if len(records) != len(V17_CONDITIONS) * len(V17_TRAINING_SEEDS):
        raise ValueError("checkpoint inventory has the wrong number of records")
    if payload["config"].get("protocol") != v17_protocol_dict():
        raise ValueError("checkpoint inventory protocol does not match frozen protocol")
    if payload["config"].get("training_source_commit") != V17_TRAINING_SOURCE_COMMIT:
        raise ValueError("checkpoint inventory training source commit is invalid")
    expected_pairs = {
        (condition, training_seed)
        for condition in V17_CONDITIONS
        for training_seed in V17_TRAINING_SEEDS
    }
    actual_pairs = {
        (record.get("condition"), int(record.get("training_seed", -1)))
        for record in records
    }
    if actual_pairs != expected_pairs:
        raise ValueError("checkpoint inventory pairings do not match protocol")
    for record in records:
        condition = record["condition"]
        training_seed = int(record["training_seed"])
        for path_key in (
            "checkpoint",
            "run_manifest",
            "episodes",
            "validation",
            "selection",
        ):
            artifact = root / record[path_key]
            if not artifact.is_file() or _sha256(artifact) != record[f"{path_key}_sha256"]:
                raise ValueError("checkpoint inventory hash validation failed")
        manifest = json.loads((root / record["run_manifest"]).read_text())
        _validate_training_manifest(
            manifest,
            condition=condition,
            training_seed=training_seed,
            expected_commit=V17_TRAINING_SOURCE_COMMIT,
        )
        validation_rows = _read_csv(root / record["validation"])
        selected_row = _selected_validation_row(validation_rows)
        if int(record.get("best_step", -1)) != int(selected_row["step"]):
            raise ValueError("checkpoint inventory selected step is inconsistent")
        _validate_training_episode_artifacts((root / record["episodes"]).parent, condition)
        training_safety = record.get("training_safety")
        if condition == "C1_inloop_hocbf":
            if training_safety is None:
                raise ValueError("in-loop inventory lacks training safety diagnostics")
            safety_path = root / training_safety
            if (
                not safety_path.is_file()
                or _sha256(safety_path) != record.get("training_safety_sha256")
            ):
                raise ValueError("training safety diagnostics failed hash validation")
        elif training_safety is not None:
            raise ValueError("post-hoc control has unexpected training safety diagnostics")
    return path, payload


def _record_for(payload: dict, condition: str, training_seed: int) -> dict:
    matches = [
        record
        for record in payload["config"]["records"]
        if record["condition"] == condition
        and int(record["training_seed"]) == training_seed
    ]
    if len(matches) != 1:
        raise ValueError("checkpoint inventory pairing is incomplete or ambiguous")
    return matches[0]


def evaluate_shard(root: Path, output: Path, training_seed: int) -> None:
    if training_seed not in V17_TRAINING_SEEDS:
        raise ValueError(f"training seed is outside the frozen protocol: {training_seed}")
    assert_source_tree_clean(root)
    inventory_path, inventory = _load_inventory(root, output)
    scenarios = {scenario.key: scenario for scenario in v13_scenarios()}
    outcome_rows = []
    safety_rows = []

    for condition in V17_CONDITIONS:
        record = _record_for(inventory, condition, training_seed)
        checkpoint = root / record["checkpoint"]
        policy = SACAgent.from_checkpoint(checkpoint, seed=0, load_optimizers=False)
        nominal = PlanarArm()
        observer = HOCBFSafetyFilter(nominal, planar_safety_config())
        stack = SARRLControlStack(
            ComputedTorqueController(nominal),
            policy,
            ControlStackConfig(require_safety=True),
            safety_filter=observer,
        )
        controller = f"{condition}_train_seed_{training_seed}"

        for scenario_key, evaluation in V17_EVALUATION.items():
            scenario = scenarios[scenario_key]
            env = PlanarReachEnv(
                mode="torque",
                randomization=scenario.randomization,
                fault=scenario.fault,
            )
            outcomes, diagnostics = evaluate_safety_episodes(
                stack,
                observer,
                env,
                episodes=evaluation["episodes"],
                seed=evaluation["seed"],
                scenario=scenario_key,
                controller=controller,
                intervention_tolerance=INTERVENTION_TOLERANCE,
            )
            outcome_rows.extend(outcomes)
            safety_rows.extend(diagnostics)
            print(
                f"{condition} seed={training_seed} scenario={scenario_key} "
                f"success={sum(row.success for row in diagnostics)}/{len(diagnostics)} "
                f"unsafe={sum(row.unsafe_episode for row in diagnostics)}/{len(diagnostics)}"
            )

    shard = output / "evaluation_shards" / f"seed_{training_seed}"
    shard.mkdir(parents=True, exist_ok=True)
    episodes_path = shard / "episodes.csv"
    safety_path = shard / "safety_diagnostics.csv"
    write_episode_csv(episodes_path, outcome_rows)
    _write_diagnostic_csv(safety_path, safety_rows)
    write_run_manifest(
        shard / "evaluation_manifest.json",
        {
            "training_seed": training_seed,
            "protocol": v17_protocol_dict(),
            "checkpoint_inventory": _relative(inventory_path, root),
            "checkpoint_inventory_sha256": _sha256(inventory_path),
            "episodes": _relative(episodes_path, root),
            "episodes_sha256": _sha256(episodes_path),
            "safety_diagnostics": _relative(safety_path, root),
            "safety_diagnostics_sha256": _sha256(safety_path),
        },
        root=root,
    )


def _boolean(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered not in {"true", "false"}:
        raise ValueError(f"invalid boolean value: {value}")
    return lowered == "true"


def _condition_and_seed(controller: str) -> tuple[str, int]:
    match = PAIRING_PATTERN.fullmatch(controller)
    if match is None or match.group(1) not in V17_CONDITIONS:
        raise ValueError(f"invalid v1.7 controller label: {controller}")
    return match.group(1), int(match.group(2))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_dict_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _metric_value(row: dict[str, str], metric: str) -> float:
    if metric in {"success", "unsafe_episode", "safety_infeasible"}:
        return float(_boolean(row[metric]))
    return float(row[metric])


def _summary(
    condition: str,
    training_seed: int,
    scenario: str,
    outcomes: list[dict[str, str]],
    diagnostics: list[dict[str, str]],
) -> dict:
    if len(outcomes) != len(diagnostics) or not diagnostics:
        raise ValueError("summary requires matching non-empty outcome and diagnostic rows")
    attempts = sum(int(row["command_attempts"]) for row in diagnostics)
    observations = sum(int(row["state_observations"]) for row in diagnostics)
    intervention_steps = sum(int(row["safety_intervention_steps"]) for row in diagnostics)
    correction_sum = sum(
        float(row["safety_correction_mean"]) * int(row["command_attempts"])
        for row in diagnostics
    )
    return {
        "condition": condition,
        "training_seed": training_seed,
        "scenario": scenario,
        "episodes": len(diagnostics),
        "success_rate": float(np.mean([_boolean(row["success"]) for row in diagnostics])),
        "reward_mean": float(np.mean([float(row["reward"]) for row in outcomes])),
        "unsafe_episode_rate": float(
            np.mean([_boolean(row["unsafe_episode"]) for row in diagnostics])
        ),
        "unsafe_state_fraction": sum(
            int(row["unsafe_state_observations"]) for row in diagnostics
        )
        / observations,
        "normalized_violation_integral_mean": float(
            np.mean([float(row["normalized_violation_integral"]) for row in diagnostics])
        ),
        "safety_infeasible_rate": float(
            np.mean([_boolean(row["safety_infeasible"]) for row in diagnostics])
        ),
        "safety_intervention_fraction": intervention_steps / attempts,
        "safety_correction_mean": correction_sum / attempts,
    }


def aggregate_evaluation(root: Path, output: Path) -> None:
    assert_source_tree_clean(root)
    commit = repository_commit(root)
    inventory_path, _ = _load_inventory(root, output)
    all_outcomes = []
    all_diagnostics = []
    shard_records = []

    for training_seed in V17_TRAINING_SEEDS:
        shard = output / "evaluation_shards" / f"seed_{training_seed}"
        manifest_path = shard / "evaluation_manifest.json"
        episodes_path = shard / "episodes.csv"
        safety_path = shard / "safety_diagnostics.csv"
        if not all(path.is_file() for path in (manifest_path, episodes_path, safety_path)):
            raise FileNotFoundError(f"evaluation shard is incomplete: seed {training_seed}")
        manifest = json.loads(manifest_path.read_text())
        config = manifest["config"]
        if manifest.get("runtime", {}).get("git_commit") != commit:
            raise ValueError("evaluation shard source commit does not match current source")
        if int(config["training_seed"]) != training_seed:
            raise ValueError("evaluation shard training seed mismatch")
        if config.get("protocol") != v17_protocol_dict():
            raise ValueError("evaluation shard protocol does not match frozen protocol")
        if config["checkpoint_inventory_sha256"] != _sha256(inventory_path):
            raise ValueError("evaluation shard used a different checkpoint inventory")
        if config["episodes_sha256"] != _sha256(episodes_path):
            raise ValueError("evaluation episode hash mismatch")
        if config["safety_diagnostics_sha256"] != _sha256(safety_path):
            raise ValueError("evaluation diagnostic hash mismatch")
        outcomes = _read_csv(episodes_path)
        diagnostics = _read_csv(safety_path)
        all_outcomes.extend(outcomes)
        all_diagnostics.extend(diagnostics)
        shard_records.append(
            {
                "training_seed": training_seed,
                "manifest": _relative(manifest_path, root),
                "manifest_sha256": _sha256(manifest_path),
            }
        )

    expected_total = 2 * len(V17_TRAINING_SEEDS) * sum(
        config["episodes"] for config in V17_EVALUATION.values()
    )
    if len(all_outcomes) != expected_total or len(all_diagnostics) != expected_total:
        raise ValueError("v1.7 aggregate has the wrong total episode count")

    outcome_groups = defaultdict(list)
    diagnostic_groups = defaultdict(list)
    outcome_seen = set()
    diagnostic_seen = set()
    for row in all_outcomes:
        condition, training_seed = _condition_and_seed(row["controller"])
        identity = (condition, training_seed, row["scenario"], int(row["seed"]))
        if identity in outcome_seen:
            raise ValueError("duplicate v1.7 outcome identity")
        outcome_seen.add(identity)
        key = (condition, training_seed, row["scenario"])
        outcome_groups[key].append(row)
    for row in all_diagnostics:
        condition, training_seed = _condition_and_seed(row["controller"])
        identity = (condition, training_seed, row["scenario"], int(row["seed"]))
        if identity in diagnostic_seen:
            raise ValueError("duplicate v1.7 episode identity")
        diagnostic_seen.add(identity)
        key = (condition, training_seed, row["scenario"])
        diagnostic_groups[key].append(row)
    if outcome_seen != diagnostic_seen:
        raise ValueError("outcome and diagnostic episode identities do not match")

    summaries = []
    for condition in V17_CONDITIONS:
        for training_seed in V17_TRAINING_SEEDS:
            for scenario, evaluation in V17_EVALUATION.items():
                key = (condition, training_seed, scenario)
                if (
                    len(outcome_groups[key]) != evaluation["episodes"]
                    or len(diagnostic_groups[key]) != evaluation["episodes"]
                ):
                    raise ValueError(f"v1.7 cell has the wrong size: {key}")
                expected_episode_seeds = set(
                    range(evaluation["seed"], evaluation["seed"] + evaluation["episodes"])
                )
                actual_episode_seeds = {
                    int(row["seed"]) for row in diagnostic_groups[key]
                }
                if actual_episode_seeds != expected_episode_seeds:
                    raise ValueError(f"v1.7 cell has the wrong episode seeds: {key}")
                summaries.append(
                    _summary(
                        condition,
                        training_seed,
                        scenario,
                        outcome_groups[key],
                        diagnostic_groups[key],
                    )
                )

    metric_names = (
        "success",
        "unsafe_episode",
        "safety_infeasible",
        "unsafe_state_fraction",
        "normalized_violation_integral",
        "safety_intervention_fraction",
        "safety_correction_mean",
    )
    comparisons = []
    intervals = {}
    per_seed_success = {}
    for scenario in V17_EVALUATION:
        for metric in metric_names:
            values = {condition: defaultdict(dict) for condition in V17_CONDITIONS}
            for row in all_diagnostics:
                if row["scenario"] != scenario:
                    continue
                condition, training_seed = _condition_and_seed(row["controller"])
                values[condition][training_seed][int(row["seed"])] = _metric_value(
                    row, metric
                )
            interval = two_level_paired_interval(
                values["C1_inloop_hocbf"],
                values["C0_posthoc_hocbf"],
                bootstrap=V17_BOOTSTRAP_SAMPLES,
                seed=V17_BOOTSTRAP_SEED,
            )
            intervals[(scenario, metric)] = interval
            comparisons.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    **asdict(interval),
                    "bootstrap_samples": V17_BOOTSTRAP_SAMPLES,
                    "bootstrap_seed": V17_BOOTSTRAP_SEED,
                }
            )
            if scenario == "id_reference" and metric == "success":
                for training_seed in V17_TRAINING_SEEDS:
                    treatment = values["C1_inloop_hocbf"][training_seed]
                    control = values["C0_posthoc_hocbf"][training_seed]
                    per_seed_success[training_seed] = float(
                        np.mean(
                            [
                                treatment[episode_seed] - control[episode_seed]
                                for episode_seed in sorted(treatment)
                            ]
                        )
                    )

    decision = classify_v17_result(
        intervals[("id_reference", "success")],
        intervals[("id_reference", "unsafe_episode")],
        intervals[("id_reference", "safety_infeasible")],
        per_seed_success,
    )
    aggregate = {condition: {} for condition in V17_CONDITIONS}
    for condition in V17_CONDITIONS:
        for scenario in V17_EVALUATION:
            rows = [
                row
                for row in summaries
                if row["condition"] == condition and row["scenario"] == scenario
            ]
            aggregate[condition][scenario] = {
                key: float(np.mean([float(row[key]) for row in rows]))
                for key in (
                    "success_rate",
                    "unsafe_episode_rate",
                    "unsafe_state_fraction",
                    "normalized_violation_integral_mean",
                    "safety_infeasible_rate",
                    "safety_intervention_fraction",
                    "safety_correction_mean",
                )
            }

    retained = output / "evaluation"
    retained.mkdir(parents=True, exist_ok=True)
    episodes_path = retained / "episodes.csv"
    safety_path = retained / "safety_diagnostics.csv"
    summary_path = retained / "summary.csv"
    comparison_path = retained / "paired_comparisons.csv"
    aggregate_path = retained / "aggregate.json"
    decision_path = retained / "decision.json"
    _write_dict_csv(episodes_path, all_outcomes)
    _write_dict_csv(safety_path, all_diagnostics)
    _write_dict_csv(summary_path, summaries)
    _write_dict_csv(comparison_path, comparisons)
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    decision_path.write_text(
        json.dumps(
            {
                "decision": decision,
                "primary_success": asdict(intervals[("id_reference", "success")]),
                "unsafe_guardrail": asdict(
                    intervals[("id_reference", "unsafe_episode")]
                ),
                "infeasibility_guardrail": asdict(
                    intervals[("id_reference", "safety_infeasible")]
                ),
                "per_training_seed_success_difference": per_seed_success,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_run_manifest(
        retained / "evaluation_manifest.json",
        {
            "protocol": v17_protocol_dict(),
            "checkpoint_inventory": _relative(inventory_path, root),
            "checkpoint_inventory_sha256": _sha256(inventory_path),
            "shards": shard_records,
            "outputs": {
                _relative(path, root): _sha256(path)
                for path in (
                    episodes_path,
                    safety_path,
                    summary_path,
                    comparison_path,
                    aggregate_path,
                    decision_path,
                )
            },
        },
        root=root,
    )
    print(f"v1.7 decision: {decision}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/safety_aware_training")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train-shard")
    train.add_argument("--condition", choices=V17_CONDITIONS, required=True)
    train.add_argument("--training-seed", type=int, choices=V17_TRAINING_SEEDS, required=True)

    commands.add_parser("build-inventory")

    evaluate = commands.add_parser("evaluate-shard")
    evaluate.add_argument(
        "--training-seed", type=int, choices=V17_TRAINING_SEEDS, required=True
    )

    commands.add_parser("aggregate-evaluation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert_repository_import_root(root)
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output

    if args.command == "train-shard":
        train_shard(root, output, args.condition, args.training_seed)
    elif args.command == "build-inventory":
        build_checkpoint_inventory(root, output)
    elif args.command == "evaluate-shard":
        evaluate_shard(root, output, args.training_seed)
    elif args.command == "aggregate-evaluation":
        aggregate_evaluation(root, output)
    else:
        raise AssertionError(f"unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
