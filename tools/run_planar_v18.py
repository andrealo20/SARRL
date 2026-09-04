#!/usr/bin/env python3
"""Run the prospective terminal-penalty ablation in isolated, verified shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm
from sarrl.envs import DomainRandomization, PlanarReachEnv, SafetyProjectedEnv
from sarrl.evaluation import (
    assert_repository_import_root,
    assert_source_tree_clean,
    evaluate_safety_episodes,
    planar_safety_config,
    repository_commit,
    v13_scenarios,
)
from sarrl.evaluation.planar_v18 import (
    CONDITIONS,
    METRICS,
    Protocol,
    classify_result,
    crossed_intervals,
)
from sarrl.evaluation.provenance import runtime_metadata
from sarrl.rl import SACAgent
from sarrl.runtime import ControlStackConfig, SARRLControlStack
from sarrl.safety import HOCBFSafetyFilter

PLAN_HASH = "0e5456bccac220e94aabc69a1e59db73bb937a6490fb8a8ef409c12775e72dd1"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _boolean(value):
    if value not in ("True", "False"):
        raise ValueError("invalid boolean value")
    return value == "True"


def _metric_value(row, metric):
    if metric in ("success", "unsafe_episode", "safety_infeasible"):
        return float(_boolean(row[metric]))
    return float(row[metric])


def _summary(condition, seed, scenario, episodes, rows):
    attempts = sum(int(r["command_attempts"]) for r in rows)
    observations = sum(int(r["state_observations"]) for r in rows)
    return {
        "condition": condition,
        "training_seed": seed,
        "scenario": scenario,
        "episodes": len(rows),
        **{
            metric + "_mean": float(np.mean([_metric_value(r, metric) for r in rows]))
            for metric in METRICS
        },
        "reward_mean": float(np.mean([float(r["reward"]) for r in episodes])),
        "unsafe_state_fraction_pooled": sum(int(r["unsafe_state_observations"]) for r in rows)
        / observations,
        "intervention_fraction_pooled": sum(int(r["safety_intervention_steps"]) for r in rows)
        / attempts,
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write empty evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@contextmanager
def locked(path):
    """Kernel locks release on crashes; a stale file never means a live owner."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def expected_config(protocol, condition, seed):
    if condition not in CONDITIONS or seed not in protocol.seeds:
        raise ValueError("condition or seed is outside the frozen protocol")
    reward = CONDITIONS[condition]
    base = PlanarReachEnv(
        mode="residual",
        randomization=DomainRandomization(
            mass_fraction=0.15,
            friction_fraction=0.30,
            motor_gain_fraction=0.15,
            payload_range=(0.0, 1.0),
            action_delay_max=2,
        ),
    )
    training = SafetyProjectedEnv(base, infeasible_reward=reward).constructor_config()
    validation = SafetyProjectedEnv(base, infeasible_reward=-500.0).constructor_config()
    result = {
        "seed": seed,
        "requested_steps": protocol.steps,
        "replay_capacity": 200000,
        "agent_config": {
            "gamma": 0.99,
            "tau": 0.005,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "alpha_lr": 3e-4,
            "init_alpha": 0.2,
            "hidden": [256, 256],
        },
        "environment": training,
        "context": {
            "enabled": False,
            "checkpoint": None,
            "checkpoint_sha256": None,
            "runtime_device": None,
            "latent_dim": None,
        },
        "trainer": {
            "training_seed": seed,
            "batch_size": 256,
            "start_steps": protocol.warmup,
            "update_every": 1,
            "context_checkpoint": None,
            "context_checkpoint_sha256": None,
            "training_hocbf": True,
            "validation_hocbf": True,
            "infeasible_reward": reward,
            "validation_infeasible_reward": -500.0,
            "validation_config": protocol.validation,
        },
        "validation": protocol.validation,
        "safety": {
            "training_hocbf": True,
            "validation_hocbf": True,
            "infeasible_reward": reward,
            "validation_infeasible_reward": -500.0,
            "training_environment": training,
            "validation_environment": validation,
        },
    }
    return json.loads(json.dumps(result))


def validate_manifest(payload, protocol, condition, seed, commit):
    if payload["runtime"]["git_commit"] != commit:
        raise ValueError("training source commit mismatch")
    config = dict(payload["config"])
    config.pop("resume")
    if config != expected_config(protocol, condition, seed):
        raise ValueError("training configuration mismatch")


def validate_session(payload, protocol, condition, seed):
    expected = expected_config(protocol, condition, seed)
    loop, replay, agent = payload["loop_state"], payload["replay"], payload["agent"]
    if (
        loop["trainer_config"] != expected["trainer"]
        or json.loads(json.dumps(payload["environment"]["constructor_config"]))
        != expected["environment"]
        or json.loads(json.dumps(agent["config"])) != expected["agent_config"]
        or agent["obs_dim"] != 8
        or agent["action_dim"] != 2
        or replay["capacity"] != 200000
        or not 0 < loop["step"] <= protocol.steps
        or replay["size"] != loop["step"]
    ):
        raise ValueError("training session configuration mismatch")
    count = replay["size"]
    for key in ("obs", "actions", "rewards", "next_obs", "dones"):
        if not np.isfinite(replay[key][:count]).all():
            raise ValueError("non-finite replay")
    if np.any(np.abs(replay["actions"][:count]) > 1.0):
        raise ValueError("replay does not contain normalized proposed actions")
    for network in ("actor", "q1", "q2", "q1_target", "q2_target"):
        if not all(torch.isfinite(value).all() for value in agent[network].values()):
            raise ValueError("non-finite network parameters")


def training_dir(output, condition, seed):
    return output / "training" / condition / f"seed_{seed}"


def artifact_hashes(directory, names):
    return {name: _sha256(directory / name) for name in names}


def verify_hashes(directory, hashes):
    for name, digest in hashes.items():
        if Path(name).name != name or _sha256(directory / name) != digest:
            raise ValueError(f"artifact hash mismatch: {name}")


def campaign(root, output, protocol):
    if not protocol.engineering:
        assert_source_tree_clean(root)
    payload = json.loads((output / "campaign.json").read_text())
    if payload["protocol"] != protocol.to_dict():
        raise ValueError("campaign protocol mismatch")
    if payload["runtime"]["git_commit"] != repository_commit(root):
        raise ValueError("campaign source commit mismatch")
    if payload["plan_sha256"] != PLAN_HASH:
        raise ValueError("campaign plan hash mismatch")
    if payload["execution"]["threads_per_process"] != 1:
        raise ValueError("campaign thread configuration mismatch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if payload["execution"]["torch_device"] != device:
        raise ValueError("campaign compute device changed")
    return payload


def initialize(root, output, protocol, workers):
    if not 1 <= workers <= 5:
        raise ValueError("workers must be between one and five")
    with locked(output / "initialization.lock"):
        if (output / "campaign.json").exists():
            payload = campaign(root, output, protocol)
            if payload["execution"]["workers"] != workers:
                raise ValueError("execution worker count is already frozen")
            return
        if any(output.glob("training/*/seed_*/run_manifest.json")):
            raise ValueError("existing data without a campaign manifest")
        if not protocol.engineering:
            assert_source_tree_clean(root)
            if not torch.cuda.is_available():
                raise RuntimeError("official SAC training requires CUDA")
            if _sha256(root / "PLAN-v1.8-penalty-ablation.md") != PLAN_HASH:
                raise ValueError("approved local protocol changed")
        write_json(
            output / "campaign.json",
            {
                "protocol": protocol.to_dict(),
                "plan_sha256": PLAN_HASH,
                "runtime": runtime_metadata(root),
                "execution": {
                    "workers": workers,
                    "threads_per_process": 1,
                    "torch_device": "cuda" if torch.cuda.is_available() else "cpu",
                    "cpu_logical_count": os.cpu_count(),
                    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                },
            },
        )


def validate_training(directory, protocol, condition, seed, commit):
    manifest = json.loads((directory / "run_manifest.json").read_text())
    validate_manifest(manifest, protocol, condition, seed, commit)
    selection = json.loads((directory / "selection.json").read_text())
    rows = _read_csv(directory / "validation.csv")
    expected_steps = list(
        range(protocol.validate_every, protocol.steps + 1, protocol.validate_every)
    )
    if [int(row["step"]) for row in rows] != expected_steps:
        raise ValueError("validation schedule mismatch")
    for row in rows:
        if (
            int(row["episodes"]) != protocol.validation["episodes"]
            or not 0 <= int(row["successes"]) <= int(row["episodes"])
            or float(row["success_rate"]) != int(row["successes"]) / int(row["episodes"])
            or not all(np.isfinite(float(value)) for value in row.values())
        ):
            raise ValueError("invalid validation evidence")
    best = max(rows, key=lambda row: (float(row["success_rate"]), float(row["reward_mean"])))
    if selection != {
        "selection_key": {
            "success_rate": float(best["success_rate"]),
            "reward_mean": float(best["reward_mean"]),
        },
        "best_step": int(best["step"]),
        "checkpoint": "best.pt",
        "checkpoint_sha256": _sha256(directory / "best.pt"),
        "tie_break": "success_rate_then_reward_then_earliest_step",
    }:
        raise ValueError("selected checkpoint is inconsistent with validation")
    episodes = _read_csv(directory / "episodes.csv")
    safety = _read_csv(directory / "training_safety.csv")
    if not episodes or len(episodes) != len(safety):
        raise ValueError("missing or misaligned training episodes")
    previous = 0
    for index, (episode, row) in enumerate(zip(episodes, safety, strict=True), 1):
        step = int(row["step"])
        abort, attempts = int(row["safety_infeasible"]), int(row["command_attempts"])
        if (
            int(row["episode"]) != index
            or row["episode"] != episode["episode"]
            or row["step"] != episode["step"]
            or not previous < step <= protocol.steps
            or attempts != step - previous
            or abort not in (0, 1)
            or int(row["safety_certified_steps"]) != attempts - abort
            or not 0 <= int(row["safety_intervention_steps"]) <= attempts
            or int(episode["success"]) not in (0, 1)
            or (abort and int(episode["success"]))
        ):
            raise ValueError("invalid training safety counters")
        if not all(np.isfinite(float(value)) for value in [*episode.values(), *row.values()]):
            raise ValueError("non-finite training evidence")
        previous = step
    final = torch.load(directory / "training_final.pt", map_location="cpu", weights_only=False)
    validate_session(final, protocol, condition, seed)
    loop = final["loop_state"]
    replay = final["replay"]
    count = replay["size"]
    abort_mask = (replay["dones"][:count, 0] == 1) & (
        replay["rewards"][:count, 0] == CONDITIONS[condition]
    )
    if not np.array_equal(
        replay["obs"][:count][abort_mask], replay["next_obs"][:count][abort_mask]
    ) or int(abort_mask.sum()) != sum(int(row["safety_infeasible"]) for row in safety):
        raise ValueError("training abort replay and episode evidence mismatch")
    for stored, written in (
        (loop["rows"], episodes),
        (loop["safety_rows"], safety),
        (loop["validation_rows"], rows),
    ):
        if not np.array_equal(
            np.asarray(stored, dtype=float),
            np.asarray([list(row.values()) for row in written], dtype=float),
        ):
            raise ValueError("training checkpoint and CSV rows mismatch")
    if (
        loop["step"] != protocol.steps
        or loop["episode"] != len(episodes)
        or loop["trainer_config"] != manifest["config"]["trainer"]
        or loop["best_step"] != int(best["step"])
        or hashlib.sha256(loop["selected_checkpoint_bytes"]).hexdigest()
        != selection["checkpoint_sha256"]
        or final["replay"]["size"] != protocol.steps
    ):
        raise ValueError("final training session is inconsistent")
    late = [row for row in safety if 0.75 * protocol.steps < int(row["step"]) <= protocol.steps]
    if not late:
        raise ValueError("missing late training episodes")
    names = [
        "best.pt",
        "final.pt",
        "training_final.pt",
        "run_manifest.json",
        "episodes.csv",
        "training_safety.csv",
        "validation.csv",
        "selection.json",
    ]
    names += [
        f"train_step{step}.pt"
        for step in range(protocol.checkpoint_every, protocol.steps + 1, protocol.checkpoint_every)
    ]
    return {
        "condition": condition,
        "training_seed": seed,
        "best_step": int(best["step"]),
        "hashes": artifact_hashes(directory, names),
        "late_training_abort_rate": sum(int(row["safety_infeasible"]) for row in late) / len(late),
        "late_training_completed_episodes": len(late),
        "trailing_partial_episode_decisions": protocol.steps - previous,
    }


def trainer_command(root, directory, protocol, condition, seed):
    return [
        sys.executable,
        str(root / "tools/train_sac.py"),
        "--mode",
        "residual",
        "--steps",
        str(protocol.steps),
        "--seed",
        str(seed),
        "--start-steps",
        str(protocol.warmup),
        "--batch-size",
        "256",
        "--hidden",
        "256",
        "256",
        "--update-every",
        "1",
        "--replay-capacity",
        "200000",
        "--randomize",
        "--training-hocbf",
        "--validation-hocbf",
        "--infeasible-reward",
        str(CONDITIONS[condition]),
        "--validation-infeasible-reward",
        "-500",
        "--output",
        str(directory),
        "--checkpoint-every",
        str(protocol.checkpoint_every),
        "--validate-every",
        str(protocol.validate_every),
        "--validation-episodes",
        str(protocol.validation["episodes"]),
        "--validation-seed",
        str(protocol.validation["seed"]),
    ]


def train(root, output, protocol, condition, seed):
    cfg = campaign(root, output, protocol)
    directory = training_dir(output, condition, seed)
    with locked(directory / "training.lock"):
        expected_config(protocol, condition, seed)
        marker = directory / "complete.json"
        if marker.exists():
            record = json.loads(marker.read_text())
            verify_hashes(directory, record["hashes"])
            if record != validate_training(
                directory, protocol, condition, seed, cfg["runtime"]["git_commit"]
            ):
                raise ValueError("training completion record mismatch")
            print(f"training already complete: {condition} seed={seed}", flush=True)
            return
        command = trainer_command(root, directory, protocol, condition, seed)
        checkpoints = list(directory.glob("train_step*.pt"))
        final = directory / "training_final.pt"
        if checkpoints or final.exists():
            validate_manifest(
                json.loads((directory / "run_manifest.json").read_text()),
                protocol,
                condition,
                seed,
                cfg["runtime"]["git_commit"],
            )
            resume = final if final.exists() else max(checkpoints, key=lambda p: int(p.stem[10:]))
            payload = torch.load(resume, map_location="cpu", weights_only=False)
            validate_session(payload, protocol, condition, seed)
            if (
                payload["loop_state"]["trainer_config"]
                != expected_config(protocol, condition, seed)["trainer"]
            ):
                raise ValueError("resume checkpoint configuration mismatch")
            del payload
            command += ["--resume", str(resume)]
        elif (directory / "run_manifest.json").exists():
            raise ValueError(
                "interrupted shard has no recoverable checkpoint; manual audit required"
            )
        environment = dict(
            os.environ,
            PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
        )
        with (directory / "training.log").open("a") as log:
            subprocess.run(
                command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        record = validate_training(
            directory, protocol, condition, seed, cfg["runtime"]["git_commit"]
        )
        write_json(marker, record)
        print(f"training complete: {condition} seed={seed}", flush=True)


def inventory(root, output, protocol):
    cfg = campaign(root, output, protocol)
    records = []
    for condition in CONDITIONS:
        for seed in protocol.seeds:
            directory = training_dir(output, condition, seed)
            record = json.loads((directory / "complete.json").read_text())
            verify_hashes(directory, record["hashes"])
            if record != validate_training(
                directory, protocol, condition, seed, cfg["runtime"]["git_commit"]
            ):
                raise ValueError("training inventory audit failed")
            records.append(record)
    path = output / "checkpoint_inventory.json"
    payload = {"campaign_sha256": _sha256(output / "campaign.json"), "records": records}
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError("frozen inventory differs from current artifacts")
    else:
        write_json(path, payload)
    return payload


def load_inventory(root, output, protocol):
    campaign(root, output, protocol)
    payload = json.loads((output / "checkpoint_inventory.json").read_text())
    if payload["campaign_sha256"] != _sha256(output / "campaign.json"):
        raise ValueError("inventory campaign mismatch")
    expected = {(condition, seed) for condition in CONDITIONS for seed in protocol.seeds}
    records = payload["records"]
    if (
        len(records) != len(expected)
        or {(r["condition"], r["training_seed"]) for r in records} != expected
    ):
        raise ValueError("inventory pairings incomplete")
    for record in records:
        directory = training_dir(output, record["condition"], record["training_seed"])
        verify_hashes(directory, record["hashes"])
    return payload


def validate_evaluation(directory, protocol, seed):
    outcomes = _read_csv(directory / "episodes.csv")
    diagnostics = _read_csv(directory / "safety_diagnostics.csv")
    expected = {
        (f"{condition}_train_seed_{seed}", scenario, s)
        for condition in CONDITIONS
        for scenario, (start, count) in protocol.evaluation.items()
        for s in range(start, start + count)
    }

    def keyed(rows):
        mapping = {(r["controller"], r["scenario"], int(r["seed"])): r for r in rows}
        if len(mapping) != len(rows) or mapping.keys() != expected:
            raise ValueError("evaluation keys, counts or pairing mismatch")
        return mapping

    outcome_map, diagnostic_map = keyed(outcomes), keyed(diagnostics)
    for key, row in diagnostic_map.items():
        episode = outcome_map[key]
        steps, attempts = int(row["steps"]), int(row["command_attempts"])
        unsafe, observations = int(row["unsafe_state_observations"]), int(row["state_observations"])
        abort = _boolean(row["safety_infeasible"])
        if (
            int(episode["steps"]) != steps
            or not 0 <= steps <= 250
            or observations != steps + 1
            or not 0 <= unsafe <= observations
            or _boolean(row["unsafe_episode"]) != (unsafe > 0)
            or not np.isclose(float(row["unsafe_state_fraction"]), unsafe / observations)
            or attempts != steps + int(abort)
            or not _boolean(row["safety_enabled"])
            or int(row["safety_certified_steps"]) != steps
            or not 0 <= int(row["safety_intervention_steps"]) <= attempts
            or row["success"] != episode["success"]
            or (abort and _boolean(row["success"]))
            or (steps == 0 and float(episode["reward"]) != 0.0)
        ):
            raise ValueError("evaluation physical invariant failed")
        values = [float(episode["reward"]), *[_metric_value(row, metric) for metric in METRICS]]
        if not np.isfinite(values).all():
            raise ValueError("non-finite evaluation metrics")
    return outcomes, diagnostics


def evaluation_record(output, directory, protocol, seed):
    validate_evaluation(directory, protocol, seed)
    return {
        "training_seed": seed,
        "inventory_sha256": _sha256(output / "checkpoint_inventory.json"),
        "campaign_sha256": _sha256(output / "campaign.json"),
        "hashes": artifact_hashes(directory, ["episodes.csv", "safety_diagnostics.csv"]),
    }


def evaluate(root, output, protocol, seed):
    if seed not in protocol.seeds:
        raise ValueError("unknown evaluation training seed")
    payload = load_inventory(root, output, protocol)
    directory = output / "evaluation_shards" / f"seed_{seed}"
    with locked(directory / "evaluation.lock"):
        marker = directory / "complete.json"
        if marker.exists():
            record = json.loads(marker.read_text())
            if record != evaluation_record(output, directory, protocol, seed):
                raise ValueError("evaluation completion record mismatch")
            print(f"evaluation already complete: seed={seed}", flush=True)
            return
        outcomes, diagnostics = [], []
        scenarios = {scenario.key: scenario for scenario in v13_scenarios()}
        for condition in CONDITIONS:
            record = next(
                r
                for r in payload["records"]
                if (r["condition"], r["training_seed"]) == (condition, seed)
            )
            checkpoint = training_dir(output, condition, seed) / "best.pt"
            if _sha256(checkpoint) != record["hashes"]["best.pt"]:
                raise ValueError("selected checkpoint changed")
            policy = SACAgent.from_checkpoint(checkpoint, seed=0, load_optimizers=False)
            nominal = PlanarArm()
            observer = HOCBFSafetyFilter(nominal, planar_safety_config())
            stack = SARRLControlStack(
                ComputedTorqueController(nominal),
                policy,
                ControlStackConfig(require_safety=True),
                safety_filter=observer,
            )
            for scenario, (start, count) in protocol.evaluation.items():
                specification = scenarios[scenario]
                env = PlanarReachEnv(
                    mode="torque",
                    randomization=specification.randomization,
                    fault=specification.fault,
                )
                result, safety = evaluate_safety_episodes(
                    stack,
                    observer,
                    env,
                    episodes=count,
                    seed=start,
                    scenario=scenario,
                    controller=f"{condition}_train_seed_{seed}",
                    intervention_tolerance=1e-9,
                )
                outcomes.extend(asdict(row) for row in result)
                diagnostics.extend(asdict(row) for row in safety)
                print(
                    f"evaluated {condition} seed={seed} scenario={scenario} episodes={count}",
                    flush=True,
                )
        write_csv(directory / "episodes.csv", outcomes)
        write_csv(directory / "safety_diagnostics.csv", diagnostics)
        write_json(marker, evaluation_record(output, directory, protocol, seed))


def aggregate(root, output, protocol):
    payload = load_inventory(root, output, protocol)
    outcomes, diagnostics, shards = [], [], []
    for seed in protocol.seeds:
        directory = output / "evaluation_shards" / f"seed_{seed}"
        record = json.loads((directory / "complete.json").read_text())
        if record != evaluation_record(output, directory, protocol, seed):
            raise ValueError("evaluation aggregate provenance mismatch")
        left, right = validate_evaluation(directory, protocol, seed)
        outcomes.extend(left)
        diagnostics.extend(right)
        shards.append({"seed": seed, "completion_sha256": _sha256(directory / "complete.json")})
    groups = {}
    summaries = []
    scenarios = {}
    per_seed = {}
    for scenario_index, (scenario, (start, count)) in enumerate(protocol.evaluation.items()):
        for condition in CONDITIONS:
            for seed in protocol.seeds:
                controller = f"{condition}_train_seed_{seed}"
                rows = [
                    r
                    for r in diagnostics
                    if r["controller"] == controller and r["scenario"] == scenario
                ]
                episode_rows = [
                    r
                    for r in outcomes
                    if r["controller"] == controller and r["scenario"] == scenario
                ]
                groups[condition, seed, scenario] = {int(r["seed"]): r for r in rows}
                summaries.append(_summary(condition, seed, scenario, episode_rows, rows))
        arms = []
        for condition in CONDITIONS:
            arms.append(
                np.array(
                    [
                        [
                            [
                                _metric_value(groups[condition, seed, scenario][s], m)
                                for m in METRICS
                            ]
                            for s in range(start, start + count)
                        ]
                        for seed in protocol.seeds
                    ]
                )
            )
        effects = arms[1] - arms[0]
        scenarios[scenario] = dict(
            zip(METRICS, crossed_intervals(effects, seed=180000 + scenario_index), strict=True)
        )
        per_seed[scenario] = {
            str(seed): float(value)
            for seed, value in zip(protocol.seeds, effects[:, :, 0].mean(axis=1), strict=True)
        }
    rates = {
        (r["condition"], r["training_seed"]): r["late_training_abort_rate"]
        for r in payload["records"]
    }
    control, treatment = CONDITIONS
    late_delta = float(np.mean([rates[treatment, s] - rates[control, s] for s in protocol.seeds]))
    seed_effects = list(per_seed["id_reference"].values())
    decision = (
        "engineering_only"
        if protocol.engineering
        else classify_result(scenarios, seed_effects, late_delta)
    )
    result = {
        "engineering": protocol.engineering,
        "decision": decision,
        "scientific_decision": None if protocol.engineering else decision,
        "episodes": len(outcomes),
        "scenarios": scenarios,
        "per_seed_success": per_seed,
        "late_training_abort_delta": late_delta,
        "late_training": [
            {k: v for k, v in r.items() if k != "hashes"} for r in payload["records"]
        ],
    }
    with locked(output / "aggregation.lock"):
        write_csv(output / "episodes.csv", outcomes)
        write_csv(output / "safety_diagnostics.csv", diagnostics)
        write_csv(output / "summary.csv", summaries)
        write_json(output / "aggregate.json", result)
        write_json(
            output / "complete.json",
            {
                "campaign_sha256": _sha256(output / "campaign.json"),
                "inventory_sha256": _sha256(output / "checkpoint_inventory.json"),
                "shards": shards,
                "hashes": artifact_hashes(
                    output,
                    ["episodes.csv", "safety_diagnostics.csv", "summary.csv", "aggregate.json"],
                ),
            },
        )
    print(
        f"aggregation complete: {len(outcomes)} episodes; engineering={protocol.engineering}",
        flush=True,
    )


@torch.inference_mode()
def diagnose(root, output, protocol):
    """Describe final-checkpoint Bellman residuals on each arm's own replay."""
    payload = load_inventory(root, output, protocol)
    records = []
    for record in payload["records"]:
        condition, seed = record["condition"], record["training_seed"]
        directory = training_dir(output, condition, seed)
        final = torch.load(directory / "training_final.pt", map_location="cpu", weights_only=False)
        validate_session(final, protocol, condition, seed)
        agent = SACAgent.from_state_dict(final["agent"], seed=181000 + seed, load_optimizers=False)
        replay, groups = final["replay"], {}
        count = replay["size"]
        done = replay["dones"][:count, 0] == 1
        abort = done & (replay["rewards"][:count, 0] == CONDITIONS[condition])
        if not np.array_equal(replay["obs"][:count][abort], replay["next_obs"][:count][abort]):
            raise ValueError("abort transitions advanced the plant")
        safety = _read_csv(directory / "training_safety.csv")
        if int(abort.sum()) != sum(int(r["safety_infeasible"]) for r in safety):
            raise ValueError("replay abort count differs from completed episode diagnostics")
        # An abort always ends an episode immediately, so a trailing partial
        # episode cannot contain a completed abort hidden from the CSV.
        for start in range(0, count, 4096):
            stop = min(start + 4096, count)
            batch = {
                key: torch.as_tensor(replay[key][start:stop], device=agent.device)
                for key in ("obs", "actions", "rewards", "next_obs", "dones")
            }
            q1, q2 = (
                agent.q1(batch["obs"], batch["actions"]),
                agent.q2(batch["obs"], batch["actions"]),
            )
            targets = []
            for _ in range(4):
                action, logp, _ = agent.actor.sample(batch["next_obs"])
                qnext = torch.minimum(
                    agent.q1_target(batch["next_obs"], action),
                    agent.q2_target(batch["next_obs"], action),
                )
                targets.append(
                    agent.compute_bellman_target(batch["rewards"], batch["dones"], qnext, logp)
                )
            target = torch.stack(targets)
            mean_target = target.mean(0)
            values = {
                "sse": (((q1 - target).square() + (q2 - target).square()) / 2).mean(0),
                "mean_target_sse": ((q1 - mean_target).square() + (q2 - mean_target).square()) / 2,
                "critic_gap_sum": (q1 - q2).abs(),
            }
            arrays = {key: value.cpu().numpy().reshape(-1) for key, value in values.items()}
            if not all(np.isfinite(value).all() for value in arrays.values()):
                raise ValueError("non-finite critic diagnostics")
            masks = {
                "all": np.ones(stop - start, dtype=bool),
                "abort": abort[start:stop],
                "success_terminal": done[start:stop] & ~abort[start:stop],
                "nonterminal": ~done[start:stop],
            }
            for group, mask in masks.items():
                stat = groups.setdefault(group, {"n": 0, **{key: 0.0 for key in arrays}})
                stat["n"] += int(mask.sum())
                for key, value in arrays.items():
                    stat[key] += float(value[mask].astype(np.float64).sum())
        for stat in groups.values():
            stat["rmse"] = (stat["sse"] / stat["n"]) ** 0.5 if stat["n"] else None
            stat["mean_target_rmse"] = (
                (stat["mean_target_sse"] / stat["n"]) ** 0.5 if stat["n"] else None
            )
        groups["abort"]["sse_share"] = (
            groups["abort"]["sse"] / groups["all"]["sse"] if groups["all"]["sse"] else None
        )
        records.append(
            {
                "condition": condition,
                "training_seed": seed,
                "alpha": agent.alpha.item(),
                "groups": groups,
            }
        )
    write_json(
        output / "critic_diagnostics.json",
        {
            "inventory_sha256": _sha256(output / "checkpoint_inventory.json"),
            "records": records,
            "interpretation": "descriptive_on_own_training_replay_not_causal_mediation",
        },
    )


def run_all(root, output, protocol):
    cfg = campaign(root, output, protocol)
    workers = cfg["execution"]["workers"]
    with locked(output / "campaign.lock"):
        # Every Future is resolved. One failed child prevents heldout evaluation.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tasks = [
                pool.submit(train, root, output, protocol, condition, seed)
                for condition in CONDITIONS
                for seed in protocol.seeds
            ]
            failures = []
            for task in tasks:
                try:
                    task.result()
                except Exception as exc:
                    failures.append(str(exc))
            if failures:
                raise RuntimeError("training failed: " + "; ".join(failures))
        inventory(root, output, protocol)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            commands = [
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output",
                    str(output),
                    *(["--engineering"] if protocol.engineering else []),
                    "evaluate",
                    "--seed",
                    str(seed),
                ]
                for seed in protocol.seeds
            ]
            tasks = [
                pool.submit(subprocess.run, command, cwd=root, check=True) for command in commands
            ]
            for task in tasks:
                task.result()
        aggregate(root, output, protocol)
        diagnose(root, output, protocol)
        write_json(
            output / "workflow_complete.json",
            {
                "evaluation_completion_sha256": _sha256(output / "complete.json"),
                "critic_diagnostics_sha256": _sha256(output / "critic_diagnostics.json"),
            },
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--engineering", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("initialize")
    init.add_argument("--workers", type=int, default=5)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--condition", choices=CONDITIONS, required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--seed", type=int, required=True)
    for name in ("inventory", "aggregate", "diagnose", "run-all"):
        sub.add_parser(name)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert_repository_import_root(root)
    torch.set_num_threads(1)
    output = Path(args.output).resolve()
    if not output.is_relative_to(root / "results"):
        raise ValueError("campaign output must be inside repository results")
    protocol = Protocol(args.engineering)
    if args.command == "initialize":
        initialize(root, output, protocol, args.workers)
    elif args.command == "train":
        train(root, output, protocol, args.condition, args.seed)
    elif args.command == "evaluate":
        evaluate(root, output, protocol, args.seed)
    else:
        {"inventory": inventory, "aggregate": aggregate, "diagnose": diagnose, "run-all": run_all}[
            args.command
        ](root, output, protocol)


if __name__ == "__main__":
    main()
