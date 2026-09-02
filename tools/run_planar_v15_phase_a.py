#!/usr/bin/env python3
"""Run or aggregate the frozen SARRL v1.5 uncertainty-gate Phase A."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch

from sarrl.adaptation import AdaptiveContextEnv, DynamicsContextEncoder
from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm, PlanarArmParams
from sarrl.envs import PlanarReachEnv
from sarrl.evaluation import (
    V15_PHASE_A_EPISODES,
    V15_PHASE_A_EVALUATION_SEED,
    V15_PHASE_A_POLICIES,
    V15_PHASE_A_TRAINING_SEEDS,
    PhaseAEpisode,
    analyze_phase_a,
    assert_repository_import_root,
    assert_source_tree_clean,
    planar_id_randomization,
    summarize_episode,
    v15_phase_a_protocol_dict,
    write_dataclass_csv,
    write_run_manifest,
)
from sarrl.models import ResidualDynamicsEnsemble, residual_acceleration_target
from sarrl.rl import SACAgent

if __package__:
    from tools.run_planar_v13 import _freeze_module
    from tools.run_planar_v14 import inputs_from_v13_manifest
else:
    from run_planar_v13 import _freeze_module
    from run_planar_v14 import inputs_from_v13_manifest


# The Phase-A shards were collected before private, non-scientific planning
# files were removed from public history. This surviving commit has the exact
# same Phase-A implementation and is the portable source reference.
COLLECTION_SOURCE_EQUIVALENT_COMMIT = "e71fa2683eb8df565de38d234932f4875c56fb7a"


TRANSITION_FIELDS = (
    "policy",
    "training_seed",
    "ensemble_seed",
    "episode_seed",
    "step",
    "q1",
    "q2",
    "dq1",
    "dq2",
    "m1",
    "m2",
    "l1",
    "l2",
    "lc1",
    "lc2",
    "i1",
    "i2",
    "gravity",
    "viscous1",
    "viscous2",
    "coulomb1",
    "coulomb2",
    "friction_smoothing",
    "payload_mass",
    "baseline_torque1",
    "baseline_torque2",
    "raw_residual1",
    "raw_residual2",
    "commanded_torque1",
    "commanded_torque2",
    "delayed_torque1",
    "delayed_torque2",
    "actuator_scaled_torque1",
    "actuator_scaled_torque2",
    "plant_input_torque1",
    "plant_input_torque2",
    "observed_acceleration1",
    "observed_acceleration2",
    "ensemble_mean1",
    "ensemble_mean2",
    "ensemble_uncertainty1",
    "ensemble_uncertainty2",
    "uncertainty_norm",
    "residual_target1",
    "residual_target2",
    "prediction_error_norm",
    "terminated",
    "truncated",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_scalar(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("canonical transition CSV forbids non-finite numbers")
        if number == 0.0:
            return "0"
        return format(number, ".17g")
    return str(value)


def _write_transition(writer: csv.DictWriter, row: dict) -> None:
    if tuple(row) != TRANSITION_FIELDS:
        raise ValueError("transition row does not match frozen schema")
    writer.writerow({key: _canonical_scalar(value) for key, value in row.items()})


def _arm_fields(params: PlanarArmParams) -> dict:
    return {
        "m1": params.m1,
        "m2": params.m2,
        "l1": params.l1,
        "l2": params.l2,
        "lc1": params.lc1,
        "lc2": params.lc2,
        "i1": params.i1,
        "i2": params.i2,
        "gravity": params.gravity,
        "viscous1": params.viscous[0],
        "viscous2": params.viscous[1],
        "coulomb1": params.coulomb[0],
        "coulomb2": params.coulomb[1],
        "friction_smoothing": params.friction_smoothing,
        "payload_mass": params.payload_mass,
    }


def _episode_rows(
    *,
    policy_name: str,
    training_seed: int,
    policy,
    encoder,
    ensemble: ResidualDynamicsEnsemble,
    device: str,
    output: Path,
) -> list[PhaseAEpisode]:
    base_env = PlanarReachEnv(mode="torque", randomization=planar_id_randomization())
    # The context encoder deliberately retains its exact-resume CPU inference
    # path; policy and ensemble inference use the requested accelerator.
    env = AdaptiveContextEnv(base_env, encoder, device="cpu") if encoder is not None else base_env
    nominal = PlanarArm()
    controller = ComputedTorqueController(nominal)
    summaries = []
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=TRANSITION_FIELDS,
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for episode in range(V15_PHASE_A_EPISODES):
            episode_seed = V15_PHASE_A_EVALUATION_SEED + episode
            obs, _ = env.reset(seed=episode_seed)
            uncertainties: list[float] = []
            errors: list[float] = []
            attempted = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                state = np.asarray(env.state, dtype=np.float64).copy()
                baseline = controller.command(state[:2], state[2:], env.q_des)
                normalized_action = np.clip(
                    np.asarray(policy.act(obs, deterministic=True), dtype=np.float64),
                    -1.0,
                    1.0,
                )
                raw_residual = normalized_action * 8.0
                commanded = np.clip(baseline + raw_residual, -40.0, 40.0)
                ensemble_mean, uncertainty = ensemble.predict(
                    state.astype(np.float32),
                    commanded.astype(np.float32),
                    device=device,
                )
                params = base_env.arm.params
                if encoder is None:
                    obs, _, terminated, truncated, info = env.step_torque(
                        commanded, baseline=baseline
                    )
                else:
                    obs, _, terminated, truncated, info = env.step_torque(
                        commanded,
                        baseline=baseline,
                        context_action=normalized_action,
                    )
                if not np.array_equal(info["commanded_torque_exact"], commanded):
                    raise AssertionError("environment changed the clipped commanded torque")
                observed = np.asarray(info["pre_step_acceleration"], dtype=np.float64)
                independent_arm = PlanarArm(params)
                recomputed = independent_arm.forward_dynamics(
                    state[:2], state[2:], info["plant_input_torque"]
                )
                if not np.allclose(observed, recomputed, rtol=0.0, atol=1e-12):
                    raise AssertionError("pre-step acceleration invariant failed")
                target = residual_acceleration_target(
                    nominal,
                    state,
                    commanded,
                    observed,
                    dtype=np.float64,
                )
                mean64 = np.asarray(ensemble_mean, dtype=np.float64)
                uncertainty64 = np.asarray(uncertainty, dtype=np.float64)
                uncertainty_norm = float(np.linalg.norm(uncertainty64))
                error_norm = float(np.linalg.norm(mean64 - target))
                attempted += 1
                if math.isfinite(uncertainty_norm) and math.isfinite(error_norm):
                    uncertainties.append(uncertainty_norm)
                    errors.append(error_norm)
                    delayed = np.asarray(info["delayed_torque_exact"], dtype=np.float64)
                    actuator = np.asarray(info["actuator_scaled_torque"], dtype=np.float64)
                    plant_input = np.asarray(info["plant_input_torque"], dtype=np.float64)
                    row = {
                        "policy": policy_name,
                        "training_seed": training_seed,
                        "ensemble_seed": training_seed,
                        "episode_seed": episode_seed,
                        "step": attempted - 1,
                        "q1": state[0],
                        "q2": state[1],
                        "dq1": state[2],
                        "dq2": state[3],
                        **_arm_fields(params),
                        "baseline_torque1": baseline[0],
                        "baseline_torque2": baseline[1],
                        "raw_residual1": raw_residual[0],
                        "raw_residual2": raw_residual[1],
                        "commanded_torque1": commanded[0],
                        "commanded_torque2": commanded[1],
                        "delayed_torque1": delayed[0],
                        "delayed_torque2": delayed[1],
                        "actuator_scaled_torque1": actuator[0],
                        "actuator_scaled_torque2": actuator[1],
                        "plant_input_torque1": plant_input[0],
                        "plant_input_torque2": plant_input[1],
                        "observed_acceleration1": observed[0],
                        "observed_acceleration2": observed[1],
                        "ensemble_mean1": mean64[0],
                        "ensemble_mean2": mean64[1],
                        "ensemble_uncertainty1": uncertainty64[0],
                        "ensemble_uncertainty2": uncertainty64[1],
                        "uncertainty_norm": uncertainty_norm,
                        "residual_target1": target[0],
                        "residual_target2": target[1],
                        "prediction_error_norm": error_norm,
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                    _write_transition(writer, row)
            summaries.append(
                summarize_episode(
                    policy=policy_name,
                    training_seed=training_seed,
                    ensemble_seed=training_seed,
                    episode_seed=episode_seed,
                    uncertainty=uncertainties,
                    error=errors,
                    attempted_pairs=attempted,
                    terminated=terminated,
                    truncated=truncated,
                )
            )
            print(
                f"{policy_name} seed={training_seed} episode={episode + 1}/"
                f"{V15_PHASE_A_EPISODES} steps={attempted}",
                flush=True,
            )
    return summaries


def run_shard(root: Path, out: Path, record: dict, device: str) -> None:
    training_seed = int(record["training_seed"])
    shard = out / "shards" / f"training_seed_{training_seed}"
    complete_path = shard / "complete.json"
    if complete_path.is_file():
        completed = json.loads(complete_path.read_text(encoding="utf-8"))
        for name, expected in completed["sha256"].items():
            if _sha256(shard / name) != expected:
                raise ValueError(f"completed shard hash mismatch: {name}")
        print(f"training seed {training_seed} already complete; verified and skipped")
        return
    shard.mkdir(parents=True, exist_ok=True)
    manifest = v15_phase_a_protocol_dict()
    manifest["selected_training_seed"] = training_seed
    manifest["input"] = record
    manifest["device"] = device
    write_run_manifest(shard / "evaluation_manifest.json", manifest, root=root)

    ensemble = ResidualDynamicsEnsemble.load(record["ensemble_checkpoint"], map_location=device)
    _freeze_module(ensemble)
    all_episodes = []
    for policy_name in V15_PHASE_A_POLICIES:
        checkpoint = record["a2_policy_checkpoint" if policy_name == "A2" else "policy_checkpoint"]
        policy = SACAgent.from_checkpoint(checkpoint, seed=0, load_optimizers=False)
        _freeze_module(policy.actor)
        encoder = None
        if policy_name == "A3":
            encoder = DynamicsContextEncoder.load(record["context_checkpoint"], map_location="cpu")
            _freeze_module(encoder)
        rows = _episode_rows(
            policy_name=policy_name,
            training_seed=training_seed,
            policy=policy,
            encoder=encoder,
            ensemble=ensemble,
            device=device,
            output=shard / f"{policy_name}_transitions.csv",
        )
        all_episodes.extend(rows)
    write_dataclass_csv(shard / "episodes.csv", all_episodes)
    files = [
        "evaluation_manifest.json",
        "A2_transitions.csv",
        "A3_transitions.csv",
        "episodes.csv",
    ]
    complete_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "training_seed": training_seed,
                "episode_count": len(all_episodes),
                "sha256": {name: _sha256(shard / name) for name in files},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_episode_rows(path: Path) -> list[PhaseAEpisode]:
    rows = []
    optional_floats = {
        "spearman_rho",
        "uncertainty_median",
        "error_median",
    }
    with path.open(encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            parsed = {}
            for key, value in source.items():
                if key in {"policy"}:
                    parsed[key] = value
                elif key in {"terminated", "truncated", "qualifies", "zero_variance"}:
                    parsed[key] = value.lower() == "true"
                elif key in optional_floats:
                    parsed[key] = None if value in {"", "None"} else float(value)
                else:
                    parsed[key] = int(value)
            rows.append(PhaseAEpisode(**parsed))
    return rows


def aggregate_shards(root: Path, out: Path) -> None:
    out = out.resolve()
    episodes = []
    shard_hashes = []
    inputs = []
    collection_runtimes = []
    for seed in V15_PHASE_A_TRAINING_SEEDS:
        shard = out / "shards" / f"training_seed_{seed}"
        complete_path = shard / "complete.json"
        if not complete_path.is_file():
            raise FileNotFoundError(f"missing completed shard for training seed {seed}")
        completed = json.loads(complete_path.read_text(encoding="utf-8"))
        if completed.get("episode_count") != 2 * V15_PHASE_A_EPISODES:
            raise ValueError(f"wrong episode count in shard {seed}")
        for name, expected in completed["sha256"].items():
            actual = _sha256(shard / name)
            if actual != expected:
                raise ValueError(f"shard {seed} hash mismatch: {name}")
            shard_hashes.append(
                {
                    "training_seed": seed,
                    "path": str((shard / name).relative_to(root)),
                    "sha256": actual,
                }
            )
        shard_manifest = json.loads(
            (shard / "evaluation_manifest.json").read_text(encoding="utf-8")
        )
        inputs.append(shard_manifest["config"]["input"])
        collection_runtime = dict(shard_manifest["runtime"])
        collection_runtime["git_commit"] = COLLECTION_SOURCE_EQUIVALENT_COMMIT
        collection_runtime["provenance_note"] = (
            "source-equivalent public commit after removal of private "
            "non-scientific planning files"
        )
        collection_runtimes.append(collection_runtime)
        episodes.extend(_read_episode_rows(shard / "episodes.csv"))

    transition_path = out / "transitions.csv"
    with transition_path.open("w", encoding="utf-8", newline="") as destination:
        wrote_header = False
        for policy in V15_PHASE_A_POLICIES:
            for seed in V15_PHASE_A_TRAINING_SEEDS:
                source_path = out / "shards" / f"training_seed_{seed}" / f"{policy}_transitions.csv"
                with source_path.open(encoding="utf-8", newline="") as source:
                    header = source.readline()
                    if not wrote_header:
                        destination.write(header)
                        wrote_header = True
                    elif header.rstrip("\r\n").split(",") != list(TRANSITION_FIELDS):
                        raise ValueError(f"transition schema mismatch: {source_path}")
                    for line in source:
                        destination.write(line)
    compressed_transition_path = out / "transitions.csv.gz"
    with transition_path.open("rb") as source, compressed_transition_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as destination:
            shutil.copyfileobj(source, destination)

    episodes.sort(key=lambda row: (_cell_key_for_sort(row), row.episode_seed))
    write_dataclass_csv(out / "episodes.csv", episodes)
    analysis = analyze_phase_a(episodes)
    with (out / "cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(analysis["cells"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(analysis["cells"])
    (out / "decision.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = v15_phase_a_protocol_dict()
    manifest["inputs"] = inputs
    manifest["collection_runtimes"] = collection_runtimes
    manifest["shard_hashes"] = shard_hashes
    manifest["outputs"] = {
        name: _sha256(out / name)
        for name in (
            "transitions.csv",
            "transitions.csv.gz",
            "episodes.csv",
            "cells.csv",
            "decision.json",
        )
    }
    write_run_manifest(out / "evaluation_manifest.json", manifest, root=root)
    print(
        f"Phase A: rho={analysis['target_median_rho']} "
        f"CI=[{analysis['ci95_low']}, {analysis['ci95_high']}] "
        f"decision={analysis['decision']}"
    )


def _cell_key_for_sort(row: PhaseAEpisode) -> tuple[int, int, int]:
    return V15_PHASE_A_POLICIES.index(row.policy), row.training_seed, row.ensemble_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seed", type=int, choices=V15_PHASE_A_TRAINING_SEEDS)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/uncertainty_gate_calibration/phase_a"),
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
        raise SystemExit("CUDA was requested but is not available")
    root = Path(__file__).resolve().parents[1]
    assert_repository_import_root(root)
    assert_source_tree_clean(root)
    if args.aggregate:
        aggregate_shards(root, args.output)
    else:
        records = inputs_from_v13_manifest(root, args.input_manifest)
        record = next(row for row in records if row["training_seed"] == args.training_seed)
        run_shard(root, args.output, record, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
