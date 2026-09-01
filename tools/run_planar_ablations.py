#!/usr/bin/env python3
"""Run the reproducible SARRL v1.2 planar ablation campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm
from sarrl.envs import DomainRandomization, PlanarReachEnv
from sarrl.evaluation import (
    V12_CONTEXT_DATA_SEED_BASE,
    V12_CONTEXT_DATA_SEED_STRIDE,
    V12_CONTEXT_HISTORY,
    V12_CONTEXT_SAMPLES,
    V12_CONTEXT_TRAINING_STEPS,
    V12_ENSEMBLE_BATCH_SIZE,
    V12_ENSEMBLE_DATA_SEED_BASE,
    V12_ENSEMBLE_DATA_SEED_STRIDE,
    V12_ENSEMBLE_SAMPLES,
    V12_ENSEMBLE_TRAINING_STEPS,
    EpisodeResult,
    aggregate,
    assert_repository_import_root,
    assert_source_tree_clean,
    evaluate_gated_policy_episodes,
    evaluate_policy_episodes,
    paired_success_difference,
    planar_ensemble_randomization_dict,
    planar_id_randomization,
    planar_id_randomization_dict,
    repository_commit,
    seed_ranges_overlap,
    validate_context_data_range,
    validate_ensemble_data_range,
    write_episode_csv,
    write_run_manifest,
    write_summary_json,
)
from sarrl.models import ResidualDynamicsConfig, ResidualDynamicsEnsemble, UncertaintyGate
from sarrl.rl import SACAgent
from sarrl.runtime import ControlStackConfig, SARRLControlStack


@dataclass(frozen=True)
class AblationCondition:
    key: str
    label: str
    policy_source: str
    context: bool
    uncertainty_gate: bool
    hocbf: bool
    status: str


CONDITIONS = (
    AblationCondition("A0", "Computed torque", "none", False, False, False, "ready"),
    AblationCondition("A1", "Direct SAC", "train-direct-sac", False, False, False, "ready"),
    AblationCondition("A2", "Residual SAC", "retained-v1.1.0", False, False, False, "ready"),
    AblationCondition(
        "A3",
        "Residual SAC + context",
        "train-context-conditioned-residual-sac",
        True,
        False,
        False,
        "ready",
    ),
    AblationCondition(
        "A4",
        "Residual SAC + uncertainty gate",
        "retained-v1.1.0",
        False,
        True,
        False,
        "ready",
    ),
    AblationCondition(
        "A5",
        "Residual SAC + HOCBF",
        "retained-v1.1.0",
        False,
        False,
        True,
        "needs-safety-evaluation-runner",
    ),
    AblationCondition(
        "A6",
        "Full adaptive stack",
        "context-conditioned-residual-sac",
        True,
        True,
        True,
        "needs-full-stack-wiring",
    ),
)


class _ZeroResidualPolicy:
    def act(self, obs, deterministic=False):
        return np.zeros(2, dtype=np.float32)


def _randomization() -> DomainRandomization:
    return planar_id_randomization()


def build_protocol(
    seeds: list[int],
    steps: int,
    validation_seed: int,
    validation_episodes: int,
    heldout_seed: int,
    heldout_episodes: int,
) -> dict:
    if len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be unique")
    if any(seed < 0 for seed in seeds):
        raise ValueError("training seeds must be non-negative")
    if steps <= 0:
        raise ValueError("training steps must be positive")
    if validation_episodes <= 0 or heldout_episodes <= 0:
        raise ValueError("evaluation episode counts must be positive")
    if validation_seed < 0 or heldout_seed < 0:
        raise ValueError("evaluation seeds must be non-negative")

    if seed_ranges_overlap(
        validation_seed,
        validation_episodes,
        heldout_seed,
        heldout_episodes,
    ):
        raise ValueError("validation and held-out seed ranges must not overlap")

    return {
        "release_target": "v1.2.0",
        "campaign": "planar_ablations",
        "training": {
            "seeds": seeds,
            "steps_per_seed": steps,
            "start_steps": 5_000,
            "batch_size": 256,
            "hidden": [256, 256],
            "update_every": 1,
            "replay_capacity": 200_000,
        },
        "validation": {
            "seed_start": validation_seed,
            "episodes": validation_episodes,
            "every_steps": 25_000,
            "checkpoint_selection": "success_rate_then_reward",
        },
        "heldout": {
            "seed_start": heldout_seed,
            "episodes_per_policy": heldout_episodes,
        },
        "domain_randomization": planar_id_randomization_dict(),
        "context_pretraining": {
            "per_training_seed": True,
            "samples_per_seed": V12_CONTEXT_SAMPLES,
            "history": V12_CONTEXT_HISTORY,
            "optimization_steps": V12_CONTEXT_TRAINING_STEPS,
            "data_seed_base": V12_CONTEXT_DATA_SEED_BASE,
            "data_seed_stride": V12_CONTEXT_DATA_SEED_STRIDE,
            "device": "cpu",
            "encoder": {
                "context_dim": 8,
                "latent_dim": 16,
                "hidden_dim": 64,
            },
            "excitation_action_range": [-0.7, 0.7],
            "supervision_target": "raw_dynamics_context",
            "target_normalization": "none",
            "runtime_ground_truth_access": False,
        },
        "uncertainty_gate": {
            "gain": 4.0,
            "min_scale": 0.1,
            "safety_certificate": False,
            "policy_source": "A2_retained_residual_sac",
            "ensemble_pairing": "one_per_training_seed",
        },
        "ensemble_pretraining": {
            "per_training_seed": True,
            "samples_per_seed": V12_ENSEMBLE_SAMPLES,
            "optimization_steps": V12_ENSEMBLE_TRAINING_STEPS,
            "batch_size": V12_ENSEMBLE_BATCH_SIZE,
            "data_seed_base": V12_ENSEMBLE_DATA_SEED_BASE,
            "data_seed_stride": V12_ENSEMBLE_DATA_SEED_STRIDE,
            "device": "cpu",
            "ensemble_config": {
                "state_dim": 4,
                "action_dim": 2,
                "output_dim": 2,
                "hidden": [128, 128],
                "ensemble_size": 5,
                "learning_rate": 1e-3,
                "weight_decay": 1e-6,
            },
            "domain_randomization": planar_ensemble_randomization_dict(),
            "excitation": {
                "distribution": "uniform",
                "low": -30.0,
                "high": 30.0,
                "space": "commanded_torque_nm",
            },
            "supervision": {
                "target": "observed_minus_nominal_acceleration",
                "torque_input": "commanded_torque",
            },
        },
        "statistics": {
            "multi_seed_spread": "sample_sd_ddof_1",
            "episode_success_interval": "wilson_95",
            "paired_comparison": "paired_bootstrap_95",
        },
        "conditions": [asdict(condition) for condition in CONDITIONS],
    }


def run_a0(out: Path, heldout_seed: int, heldout_episodes: int) -> None:
    """Evaluate the computed-torque baseline on the fixed held-out seeds."""
    env = PlanarReachEnv(mode="residual", randomization=_randomization())

    rows = evaluate_policy_episodes(
        _ZeroResidualPolicy(),
        env,
        heldout_episodes,
        heldout_seed,
        scenario="id_randomized",
        controller="A0_computed_torque",
    )

    condition_out = out / "A0_computed_torque"
    write_episode_csv(condition_out / "heldout_episodes.csv", rows)

    metrics = aggregate(rows)
    write_summary_json(
        condition_out / "summary.json",
        {"A0_computed_torque": metrics},
        metadata={
            "condition": "A0",
            "heldout_seed_start": heldout_seed,
            "heldout_episodes": heldout_episodes,
        },
    )

    print(
        f"A0 computed torque: {metrics.successes}/{metrics.n} = {100.0 * metrics.success_rate:.1f}%"
    )


def run_a1(
    root: Path,
    out: Path,
    seeds: list[int],
    steps: int,
    validation_seed: int,
    validation_episodes: int,
    heldout_seed: int,
    heldout_episodes: int,
    confirm_training: bool,
) -> None:
    """Run the Direct SAC multi-seed campaign."""
    cmd = [
        sys.executable,
        str(root / "tools" / "run_sac_sweep.py"),
        "--seeds",
        *[str(seed) for seed in seeds],
        "--mode",
        "torque",
        "--steps",
        str(steps),
        "--randomize",
        "--validation-seed",
        str(validation_seed),
        "--validation-episodes",
        str(validation_episodes),
        "--heldout-seed",
        str(heldout_seed),
        "--heldout-episodes",
        str(heldout_episodes),
        "--resume-existing",
        "--output",
        str(out / "A1_direct_sac"),
    ]

    print("A1 command:")
    print(" ".join(cmd))

    if not confirm_training:
        print("A1 training NOT started. Pass --confirm-training to execute it.")
        return

    subprocess.run(cmd, cwd=root, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_a3_context_artifact(
    context_dir: Path,
    training_seed: int,
    expected_commit: str | None = None,
) -> bool:
    """Validate one retained A3 context-pretraining artifact.

    Returns False only when no artifact exists yet. Partial or incompatible
    artifacts are rejected rather than silently overwritten or reused.
    """
    checkpoint = context_dir / "context.pt"
    dataset = context_dir / "context.npz"
    manifest = context_dir / "context_manifest.json"

    existing = [
        checkpoint.exists(),
        dataset.exists(),
        manifest.exists(),
    ]

    if not any(existing):
        return False

    if not all(existing):
        raise ValueError(
            f"partial A3 context artifact exists for training seed {training_seed}: {context_dir}"
        )

    payload = json.loads(manifest.read_text())
    config = payload.get("config", {})
    runtime = payload.get("runtime", {})
    extra = payload.get("extra", {})

    if expected_commit is not None and runtime.get("git_commit") != expected_commit:
        raise ValueError(
            "A3 context artifact git commit mismatch for "
            f"training seed {training_seed}: "
            f"{runtime.get('git_commit')} != {expected_commit}"
        )

    data_seed_start, data_seed_end = validate_context_data_range(
        training_seed,
        V12_CONTEXT_SAMPLES,
    )

    expected = {
        "purpose": "A3 learned dynamics context pretraining",
        "training_seed": training_seed,
        "data_seed_start": data_seed_start,
        "data_seed_end": data_seed_end,
        "samples": V12_CONTEXT_SAMPLES,
        "history": V12_CONTEXT_HISTORY,
        "optimization_steps": V12_CONTEXT_TRAINING_STEPS,
        "device": "cpu",
        "domain_randomization": planar_id_randomization_dict(),
        "excitation": {
            "distribution": "uniform",
            "low": -0.7,
            "high": 0.7,
            "space": "normalized_residual_action",
        },
        "supervision": {
            "target": "raw_dynamics_context",
            "normalization": "none",
            "runtime_ground_truth_access": False,
        },
    }

    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"A3 context artifact protocol mismatch for training seed {training_seed}: {key}"
            )

    context_cfg = config.get("context_config", {})

    expected_context_cfg = {
        "obs_dim": 8,
        "action_dim": 2,
        "context_dim": 8,
        "latent_dim": 16,
        "hidden_dim": 64,
        "history": V12_CONTEXT_HISTORY,
        "learning_rate": 1e-3,
    }

    if context_cfg != expected_context_cfg:
        raise ValueError(
            f"A3 context encoder configuration mismatch for training seed {training_seed}"
        )

    if extra.get("dataset_file") != "context.npz":
        raise ValueError(
            f"A3 context dataset provenance mismatch for training seed {training_seed}"
        )

    recorded_sha256 = extra.get("checkpoint_sha256")
    actual_sha256 = _sha256(checkpoint)

    if recorded_sha256 != actual_sha256:
        raise ValueError(f"A3 context checkpoint SHA256 mismatch for training seed {training_seed}")

    return True


def prepare_a3_contexts(
    root: Path,
    out: Path,
    seeds: list[int],
) -> Path:
    """Prepare or safely reuse one frozen context encoder per A3 seed."""
    condition_out = out / "A3_residual_sac_context"
    context_root = condition_out / "contexts"
    context_root.mkdir(parents=True, exist_ok=True)

    current_commit = repository_commit(root)

    if current_commit is None:
        raise RuntimeError("A3 context preparation requires a Git commit")

    for training_seed in seeds:
        context_dir = context_root / f"context_seed_{training_seed}"

        if _validate_a3_context_artifact(
            context_dir,
            training_seed,
            current_commit,
        ):
            print(f"A3 context seed={training_seed} already complete; reusing validated artifact")
            continue

        checkpoint = context_dir / "context.pt"

        cmd = [
            sys.executable,
            str(root / "tools" / "train_context.py"),
            "--samples",
            str(V12_CONTEXT_SAMPLES),
            "--history",
            str(V12_CONTEXT_HISTORY),
            "--steps",
            str(V12_CONTEXT_TRAINING_STEPS),
            "--seed",
            str(training_seed),
            "--device",
            "cpu",
            "--output",
            str(checkpoint),
        ]

        print(f"A3 context seed={training_seed}: " + " ".join(cmd))

        subprocess.run(
            cmd,
            cwd=root,
            check=True,
        )

        if not _validate_a3_context_artifact(
            context_dir,
            training_seed,
            current_commit,
        ):
            raise RuntimeError(
                "A3 context training completed without producing "
                f"a valid artifact for seed {training_seed}"
            )

    return context_root


def run_a3(
    root: Path,
    out: Path,
    seeds: list[int],
    steps: int,
    validation_seed: int,
    validation_episodes: int,
    heldout_seed: int,
    heldout_episodes: int,
    confirm_training: bool,
) -> None:
    """Run Residual SAC + causal context using independent context encoders."""
    condition_out = out / "A3_residual_sac_context"
    context_root = condition_out / "contexts"

    cmd = [
        sys.executable,
        str(root / "tools" / "run_sac_sweep.py"),
        "--seeds",
        *[str(seed) for seed in seeds],
        "--mode",
        "residual",
        "--steps",
        str(steps),
        "--randomize",
        "--start-steps",
        "5000",
        "--batch-size",
        "256",
        "--hidden",
        "256",
        "256",
        "--update-every",
        "1",
        "--replay-capacity",
        "200000",
        "--validate-every",
        "25000",
        "--validation-seed",
        str(validation_seed),
        "--validation-episodes",
        str(validation_episodes),
        "--heldout-seed",
        str(heldout_seed),
        "--heldout-episodes",
        str(heldout_episodes),
        "--context-root",
        str(context_root),
        "--resume-existing",
        "--output",
        str(condition_out),
    ]

    print("A3 command:")
    print(" ".join(cmd))

    if not confirm_training:
        print(
            "A3 training NOT started. Pass --confirm-training to pretrain contexts and execute it."
        )
        return

    prepared_root = prepare_a3_contexts(
        root,
        out,
        seeds,
    )

    if prepared_root.resolve() != context_root.resolve():
        raise RuntimeError("unexpected A3 context-root resolution")

    subprocess.run(
        cmd,
        cwd=root,
        check=True,
    )


def register_a2(root: Path, out: Path) -> None:
    """Register the retained v1.1 residual-SAC evidence as condition A2."""
    source = root / "artifacts" / "planar_sac_5seed"

    required = [
        source / "summary.csv",
        source / "heldout_episodes.csv",
        source / "aggregate.json",
        source / "paired_comparison.csv",
        source / "result.json",
    ]

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("A2 retained evidence is incomplete: " + ", ".join(missing))

    condition_out = out / "A2_residual_sac"
    condition_out.mkdir(parents=True, exist_ok=True)

    payload = {
        "condition": "A2",
        "label": "Residual SAC",
        "source": str(source),
        "source_release": "v1.1.0",
        "training_commit": "9f832614ce8b51c207873ff4861986ab72903115",
        "reused_without_retraining": True,
        "files": [str(path.relative_to(root)) for path in required],
    }

    (condition_out / "retained_source.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    print(f"A2 retained evidence verified: {source}")


def _retained_a2_checkpoint_hashes(root: Path) -> dict[int, str]:
    """Return the frozen A2 policy hashes keyed by training seed."""
    record = root / "artifacts" / "planar_sac_5seed" / "checkpoint_sha256.txt"

    if not record.is_file():
        raise FileNotFoundError(f"missing retained A2 checkpoint hashes: {record}")

    hashes: dict[int, str] = {}

    for line in record.read_text().splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError("invalid retained A2 checkpoint hash record")
        digest, checkpoint = fields
        seed_dir = Path(checkpoint).parent.name

        try:
            seed = int(seed_dir.removeprefix("seed_"))
        except ValueError as exc:
            raise ValueError("invalid retained A2 checkpoint seed") from exc

        if seed in hashes or len(digest) != 64:
            raise ValueError("invalid retained A2 checkpoint hash record")
        hashes[seed] = digest

    return hashes


def _write_gate_diagnostics(path: Path, rows) -> None:
    if not rows:
        raise ValueError("cannot write empty A4 gate diagnostics")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _read_episode_csv(path: Path) -> list[EpisodeResult]:
    if not path.is_file():
        raise FileNotFoundError(f"missing retained episode evidence: {path}")

    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                EpisodeResult(
                    scenario=row["scenario"],
                    controller=row["controller"],
                    seed=int(row["seed"]),
                    reward=float(row["reward"]),
                    steps=int(row["steps"]),
                    success=row["success"].lower() == "true",
                    final_distance=float(row["final_distance"]),
                    max_speed=float(row["max_speed"]),
                    max_command_torque=float(row["max_command_torque"]),
                    fault_seen=row["fault_seen"].lower() == "true",
                )
            )
    return rows


def _validate_a4_ensemble_artifact(
    checkpoint: Path,
    training_seed: int,
    expected_commit: str | None = None,
) -> bool:
    """Validate one retained A4 ensemble and its training provenance."""
    checkpoint = Path(checkpoint)
    dataset = checkpoint.with_suffix(".npz")
    manifest = checkpoint.parent / "ensemble_manifest.json"
    existing = [checkpoint.exists(), dataset.exists(), manifest.exists()]

    if not any(existing):
        return False
    if not all(existing):
        raise ValueError(
            f"partial A4 ensemble artifact exists for training seed {training_seed}: "
            f"{checkpoint.parent}"
        )

    payload = json.loads(manifest.read_text())
    config = payload.get("config", {})
    runtime = payload.get("runtime", {})
    extra = payload.get("extra", {})

    if expected_commit is not None and runtime.get("git_commit") != expected_commit:
        raise ValueError(
            "A4 ensemble artifact git commit mismatch for "
            f"training seed {training_seed}: "
            f"{runtime.get('git_commit')} != {expected_commit}"
        )

    data_seed_start, data_seed_end = validate_ensemble_data_range(
        training_seed,
        V12_ENSEMBLE_SAMPLES,
    )
    expected = {
        "purpose": "A4 residual-dynamics ensemble pretraining",
        "training_seed": training_seed,
        "data_seed_start": data_seed_start,
        "data_seed_end": data_seed_end,
        "samples": V12_ENSEMBLE_SAMPLES,
        "optimization_steps": V12_ENSEMBLE_TRAINING_STEPS,
        "batch_size": V12_ENSEMBLE_BATCH_SIZE,
        "device": "cpu",
        "domain_randomization": planar_ensemble_randomization_dict(),
        "excitation": {
            "distribution": "uniform",
            "low": -30.0,
            "high": 30.0,
            "space": "commanded_torque_nm",
        },
        "supervision": {
            "target": "observed_minus_nominal_acceleration",
            "torque_input": "commanded_torque",
        },
    }

    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"A4 ensemble artifact protocol mismatch for training seed "
                f"{training_seed}: {key}"
            )

    ensemble_config = config.get("ensemble_config", {})
    expected_ensemble_config = asdict(ResidualDynamicsConfig())
    expected_ensemble_config["hidden"] = list(expected_ensemble_config["hidden"])

    if ensemble_config != expected_ensemble_config:
        raise ValueError(
            f"A4 ensemble configuration mismatch for training seed {training_seed}"
        )
    if extra.get("dataset_file") != dataset.name:
        raise ValueError(
            f"A4 ensemble dataset provenance mismatch for training seed {training_seed}"
        )
    if extra.get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError(
            f"A4 ensemble checkpoint SHA256 mismatch for training seed {training_seed}"
        )

    return True


def prepare_a4_ensembles(
    root: Path,
    out: Path,
    seeds: list[int],
) -> list[Path]:
    """Prepare or safely reuse one frozen residual-dynamics ensemble per A4 seed."""
    ensemble_root = out / "A4_residual_sac_uncertainty_gate" / "ensembles"
    ensemble_root.mkdir(parents=True, exist_ok=True)
    current_commit = repository_commit(root)

    if current_commit is None:
        raise RuntimeError("A4 ensemble preparation requires a Git commit")

    checkpoints = []

    for training_seed in seeds:
        checkpoint = ensemble_root / f"ensemble_seed_{training_seed}" / "ensemble.pt"

        if _validate_a4_ensemble_artifact(
            checkpoint,
            training_seed,
            current_commit,
        ):
            print(
                f"A4 ensemble seed={training_seed} already complete; "
                "reusing validated artifact"
            )
            checkpoints.append(checkpoint.resolve())
            continue

        cmd = [
            sys.executable,
            str(root / "tools" / "train_residual_dynamics.py"),
            "--samples",
            str(V12_ENSEMBLE_SAMPLES),
            "--steps",
            str(V12_ENSEMBLE_TRAINING_STEPS),
            "--batch-size",
            str(V12_ENSEMBLE_BATCH_SIZE),
            "--seed",
            str(training_seed),
            "--device",
            "cpu",
            "--output",
            str(checkpoint),
        ]
        print(f"A4 ensemble seed={training_seed}: " + " ".join(cmd))
        subprocess.run(cmd, cwd=root, check=True)

        if not _validate_a4_ensemble_artifact(
            checkpoint,
            training_seed,
            current_commit,
        ):
            raise RuntimeError(
                f"A4 ensemble training did not produce a valid artifact for seed {training_seed}"
            )
        checkpoints.append(checkpoint.resolve())

    return checkpoints


def run_a4(
    root: Path,
    out: Path,
    seeds: list[int],
    heldout_seed: int,
    heldout_episodes: int,
    policy_checkpoints: list[Path],
    ensemble_checkpoints: list[Path],
    gate_gain: float = 4.0,
    gate_min_scale: float = 0.1,
) -> None:
    """Evaluate retained A2 policies with one uncertainty ensemble per seed."""
    if gate_gain != 4.0 or gate_min_scale != 0.1:
        raise ValueError("A4 uncertainty-gate parameters are frozen at gain=4.0, min_scale=0.1")
    if len(policy_checkpoints) != len(seeds):
        raise ValueError("A4 requires one policy checkpoint per training seed")
    if len(ensemble_checkpoints) != len(seeds):
        raise ValueError("A4 requires one ensemble checkpoint per training seed")

    gate = UncertaintyGate(gain=gate_gain, min_scale=gate_min_scale)
    retained_hashes = _retained_a2_checkpoint_hashes(root)
    current_commit = repository_commit(root)
    if current_commit is None:
        raise RuntimeError("A4 evaluation requires a Git commit")
    inputs = []

    for training_seed, policy_path, ensemble_path in zip(
        seeds,
        policy_checkpoints,
        ensemble_checkpoints,
        strict=True,
    ):
        policy_path = Path(policy_path).resolve()
        ensemble_path = Path(ensemble_path).resolve()

        if not policy_path.is_file():
            raise FileNotFoundError(f"missing A4 policy checkpoint: {policy_path}")
        if not ensemble_path.is_file():
            raise FileNotFoundError(f"missing A4 ensemble checkpoint: {ensemble_path}")
        if training_seed not in retained_hashes:
            raise ValueError(f"no retained A2 checkpoint hash for training seed {training_seed}")
        if not _validate_a4_ensemble_artifact(
            ensemble_path,
            training_seed,
            current_commit,
        ):
            raise ValueError(
                f"missing A4 ensemble artifact for training seed {training_seed}"
            )

        policy_sha256 = _sha256(policy_path)
        if policy_sha256 != retained_hashes[training_seed]:
            raise ValueError(
                "A4 policy checkpoint does not match retained A2 evidence for "
                f"training seed {training_seed}"
            )

        inputs.append(
            {
                "training_seed": training_seed,
                "policy_checkpoint": str(policy_path),
                "policy_checkpoint_sha256": policy_sha256,
                "ensemble_checkpoint": str(ensemble_path),
                "ensemble_checkpoint_sha256": _sha256(ensemble_path),
            }
        )

    condition_out = out / "A4_residual_sac_uncertainty_gate"
    condition_out.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        condition_out / "evaluation_manifest.json",
        {
            "condition": "A4",
            "label": "Residual SAC + uncertainty gate",
            "training_seeds": seeds,
            "policy_source": "retained_A2_residual_SAC",
            "heldout": {
                "seed_start": heldout_seed,
                "episodes_per_policy": heldout_episodes,
            },
            "domain_randomization": planar_id_randomization_dict(),
            "uncertainty_gate": {
                "gain": gate.gain,
                "min_scale": gate.min_scale,
                "safety_certificate": False,
            },
            "inputs": inputs,
        },
        root=root,
    )

    summary_rows = []
    paired_rows = []
    all_episode_rows = []
    all_gate_rows = []
    retained_a2_rows = _read_episode_csv(
        root / "artifacts" / "planar_sac_5seed" / "heldout_episodes.csv"
    )

    for record in inputs:
        training_seed = record["training_seed"]
        agent = SACAgent.from_checkpoint(
            record["policy_checkpoint"],
            seed=0,
            load_optimizers=False,
        )
        if (
            agent.obs_dim != 8
            or agent.action_dim != 2
            or agent.config.hidden != (256, 256)
        ):
            raise ValueError(
                f"A4 policy architecture mismatch for training seed {training_seed}"
            )

        ensemble = ResidualDynamicsEnsemble.load(
            record["ensemble_checkpoint"],
            map_location="cpu",
        )
        if (
            ensemble.config.state_dim != 4
            or ensemble.config.action_dim != 2
            or ensemble.config.output_dim != 2
        ):
            raise ValueError(
                f"A4 ensemble dimensions mismatch for training seed {training_seed}"
            )
        ensemble.eval()
        for parameter in ensemble.parameters():
            parameter.requires_grad_(False)

        nominal = PlanarArm()
        stack = SARRLControlStack(
            ComputedTorqueController(nominal),
            agent,
            ControlStackConfig(require_safety=False),
            dynamics_ensemble=ensemble,
            uncertainty_gate=gate,
            device="cpu",
        )
        env = PlanarReachEnv(
            mode="torque",
            randomization=_randomization(),
        )
        controller = f"A4_train_seed_{training_seed}"
        rows, gate_rows = evaluate_gated_policy_episodes(
            stack,
            env,
            heldout_episodes,
            heldout_seed,
            scenario="id_randomized_heldout",
            controller=controller,
        )
        metrics = aggregate(rows)
        paired_a2 = [
            row
            for row in retained_a2_rows
            if row.controller == f"sac_train_seed_{training_seed}"
            and heldout_seed <= row.seed < heldout_seed + heldout_episodes
        ]
        if len(paired_a2) != heldout_episodes:
            raise ValueError(
                f"A2 paired evidence coverage mismatch for training seed {training_seed}"
            )
        paired_diff, paired_low, paired_high = paired_success_difference(
            rows,
            paired_a2,
            bootstrap=10_000,
            seed=training_seed,
        )
        all_episode_rows.extend(rows)
        all_gate_rows.extend(gate_rows)
        summary_rows.append(
            {
                "training_seed": training_seed,
                "policy_checkpoint": record["policy_checkpoint"],
                "policy_checkpoint_sha256": record["policy_checkpoint_sha256"],
                "ensemble_checkpoint": record["ensemble_checkpoint"],
                "ensemble_checkpoint_sha256": record["ensemble_checkpoint_sha256"],
                "successes": metrics.successes,
                "episodes": metrics.n,
                "success_rate": metrics.success_rate,
                "success_ci95_low": metrics.success_ci95_low,
                "success_ci95_high": metrics.success_ci95_high,
                "reward_mean": metrics.reward_mean,
                "reward_std": metrics.reward_std,
                "final_distance_mean": metrics.final_distance_mean,
                "gate_scale_mean": float(
                    np.mean([row.uncertainty_scale_mean for row in gate_rows])
                ),
                "gate_scale_min": float(
                    np.min([row.uncertainty_scale_min for row in gate_rows])
                ),
                "uncertainty_norm_mean": float(
                    np.mean([row.uncertainty_norm_mean for row in gate_rows])
                ),
                "paired_vs_a2_success_difference": paired_diff,
                "paired_vs_a2_ci95_low": paired_low,
                "paired_vs_a2_ci95_high": paired_high,
            }
        )
        paired_rows.append(
            {
                "training_seed": training_seed,
                "episodes": heldout_episodes,
                "a4_successes": metrics.successes,
                "a2_successes": sum(int(row.success) for row in paired_a2),
                "success_difference": paired_diff,
                "paired_bootstrap_ci95_low": paired_low,
                "paired_bootstrap_ci95_high": paired_high,
                "bootstrap_samples": 10_000,
                "bootstrap_seed": training_seed,
            }
        )
        print(
            f"A4 seed={training_seed} heldout={metrics.successes}/{metrics.n} "
            f"({100.0 * metrics.success_rate:.1f}%)"
        )

    write_episode_csv(condition_out / "heldout_episodes.csv", all_episode_rows)
    _write_gate_diagnostics(condition_out / "gate_diagnostics.csv", all_gate_rows)

    with (condition_out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (condition_out / "paired_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    rates = np.asarray([row["success_rate"] for row in summary_rows], dtype=np.float64)
    rewards = np.asarray([row["reward_mean"] for row in summary_rows], dtype=np.float64)
    aggregate_payload = {
        "condition": "A4",
        "training_seeds": seeds,
        "models": len(summary_rows),
        "heldout_episodes_per_model": heldout_episodes,
        "success_rate_mean": float(rates.mean()),
        "success_rate_std": (float(rates.std(ddof=1)) if rates.size > 1 else None),
        "success_rate_min": float(rates.min()),
        "success_rate_max": float(rates.max()),
        "reward_mean_across_models": float(rewards.mean()),
        "reward_std_across_models": (
            float(rewards.std(ddof=1)) if rewards.size > 1 else None
        ),
        "gate_scale_mean_across_models": float(
            np.mean([row["gate_scale_mean"] for row in summary_rows])
        ),
        "gate_scale_min": float(
            np.min([row["gate_scale_min"] for row in summary_rows])
        ),
        "paired_vs_a2_success_difference_mean": float(
            np.mean([row["success_difference"] for row in paired_rows])
        ),
        "paired_vs_a2_success_difference_std": (
            float(
                np.std(
                    [row["success_difference"] for row in paired_rows],
                    ddof=1,
                )
            )
            if len(paired_rows) > 1
            else None
        ),
    }
    (condition_out / "aggregate.json").write_text(
        json.dumps(aggregate_payload, indent=2, sort_keys=True) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--validation-seed", type=int, default=20_000)
    parser.add_argument("--validation-episodes", type=int, default=30)
    parser.add_argument("--heldout-seed", type=int, default=40_000)
    parser.add_argument("--heldout-episodes", type=int, default=100)
    parser.add_argument("--output", default="results/planar_ablations")
    parser.add_argument(
        "--execute",
        nargs="*",
        choices=["A0", "A1", "A2", "A3", "A4"],
        default=[],
        help="Ready conditions to execute/register.",
    )
    parser.add_argument(
        "--a4-policy-checkpoints",
        type=Path,
        nargs="+",
        default=None,
        help="One retained A2 best.pt checkpoint per --seeds entry.",
    )
    parser.add_argument(
        "--a4-ensemble-checkpoints",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One residual-dynamics ensemble checkpoint per --seeds entry. "
            "Omit with --confirm-training to prepare canonical per-seed ensembles."
        ),
    )
    parser.add_argument(
        "--confirm-training",
        action="store_true",
        help="Required before expensive A1, A3 or A4 auxiliary training is launched.",
    )
    args = parser.parse_args()

    try:
        protocol = build_protocol(
            args.seeds,
            args.steps,
            args.validation_seed,
            args.validation_episodes,
            args.heldout_seed,
            args.heldout_episodes,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    root = Path(__file__).resolve().parents[1]
    assert_repository_import_root(root)

    if args.confirm_training and any(condition in {"A1", "A3"} for condition in args.execute):
        assert_source_tree_clean(root)
    if "A4" in args.execute:
        assert_source_tree_clean(root)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    write_run_manifest(
        out / "experiment_manifest.json",
        protocol,
        root=root,
    )

    print("SARRL v1.2 planar ablation protocol")
    print("------------------------------------")
    for condition in CONDITIONS:
        print(f"{condition.key}: {condition.label:<35} [{condition.status}]")

    print()
    print(f"training seeds: {args.seeds}")
    print(f"steps/seed:     {args.steps}")
    print(
        f"validation:     {args.validation_seed}"
        f"..{args.validation_seed + args.validation_episodes - 1}"
    )
    print(f"held-out:       {args.heldout_seed}..{args.heldout_seed + args.heldout_episodes - 1}")
    print(f"manifest:       {out / 'experiment_manifest.json'}")

    if "A0" in args.execute:
        run_a0(out, args.heldout_seed, args.heldout_episodes)

    if "A1" in args.execute:
        run_a1(
            root,
            out,
            args.seeds,
            args.steps,
            args.validation_seed,
            args.validation_episodes,
            args.heldout_seed,
            args.heldout_episodes,
            args.confirm_training,
        )

    if "A2" in args.execute:
        register_a2(root, out)

    if "A3" in args.execute:
        run_a3(
            root,
            out,
            args.seeds,
            args.steps,
            args.validation_seed,
            args.validation_episodes,
            args.heldout_seed,
            args.heldout_episodes,
            args.confirm_training,
        )

    if "A4" in args.execute:
        if args.a4_policy_checkpoints is None:
            raise SystemExit(
                "A4 requires --a4-policy-checkpoints"
            )
        ensemble_checkpoints = args.a4_ensemble_checkpoints
        if ensemble_checkpoints is None:
            if not args.confirm_training:
                raise SystemExit(
                    "A4 requires --a4-ensemble-checkpoints or --confirm-training"
                )
            ensemble_checkpoints = prepare_a4_ensembles(root, out, args.seeds)
        try:
            run_a4(
                root,
                out,
                args.seeds,
                args.heldout_seed,
                args.heldout_episodes,
                args.a4_policy_checkpoints,
                ensemble_checkpoints,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
