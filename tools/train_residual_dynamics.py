#!/usr/bin/env python3
"""Generate randomized plant data and fit an ensemble residual dynamics model."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from sarrl.dynamics import PlanarArm
from sarrl.envs import PlanarReachEnv
from sarrl.evaluation import (
    V12_ENSEMBLE_BATCH_SIZE,
    V12_ENSEMBLE_SAMPLES,
    V12_ENSEMBLE_TRAINING_STEPS,
    assert_repository_import_root,
    ensemble_data_seed,
    planar_ensemble_randomization,
    planar_ensemble_randomization_dict,
    validate_ensemble_data_range,
    write_run_manifest,
)
from sarrl.models import (
    ResidualDynamicsEnsemble,
    residual_acceleration_target,
    train_residual_ensemble,
)
from sarrl.utils import seed_everything


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(samples: int, data_seed: int):
    env = PlanarReachEnv(
        mode="torque",
        randomization=planar_ensemble_randomization(),
    )
    nominal = PlanarArm()
    rng = np.random.default_rng(data_seed)
    states, torques, targets = [], [], []
    env.reset(seed=data_seed)
    for i in range(samples):
        if i % 64 == 0:
            env.reset(seed=data_seed + i)
        state = env.state.copy()
        commanded = rng.uniform(-30.0, 30.0, size=2)
        applied = commanded * env.motor_gain
        actual_qdd = env.arm.forward_dynamics(state[:2], state[2:], applied)
        # The learned model receives the commanded torque at runtime. Keep
        # motor-gain error inside the residual target rather than hiding it by
        # replacing the command with the already-degraded applied torque.
        states.append(state.astype(np.float32))
        torques.append(commanded.astype(np.float32))
        targets.append(residual_acceleration_target(nominal, state, commanded, actual_qdd))
        env.state = env.arm.step_rk4(state, applied, env.dt)
    return np.asarray(states), np.asarray(torques), np.asarray(targets)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=V12_ENSEMBLE_SAMPLES)
    p.add_argument("--steps", type=int, default=V12_ENSEMBLE_TRAINING_STEPS)
    p.add_argument("--batch-size", type=int, default=V12_ENSEMBLE_BATCH_SIZE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="First A4 identification-data seed; defaults to the per-seed namespace.",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Official A4 ensemble training uses CPU.",
    )
    p.add_argument("--output", default="results/residual_dynamics/ensemble.pt")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    assert_repository_import_root(root)

    if args.seed < 0 or args.samples <= 0 or args.steps <= 0 or args.batch_size <= 0:
        raise SystemExit("seed must be non-negative and samples/steps/batch-size positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA ensemble training requested but CUDA is unavailable")

    data_seed = ensemble_data_seed(args.seed) if args.data_seed is None else args.data_seed
    if data_seed < 0:
        raise SystemExit("ensemble data seed must be non-negative")
    if args.data_seed is None:
        data_seed_start, data_seed_end = validate_ensemble_data_range(
            args.seed,
            args.samples,
        )
    else:
        data_seed_start = data_seed
        data_seed_end = data_seed + args.samples - 1

    seed_everything(args.seed)
    states, actions, targets = collect(args.samples, data_seed)
    model = ResidualDynamicsEnsemble(seed=args.seed)
    stats = train_residual_ensemble(
        model,
        states,
        actions,
        targets,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)

    dataset_path = path.with_suffix(".npz")
    np.savez_compressed(
        dataset_path,
        states=states,
        actions=actions,
        targets=targets,
    )
    checkpoint_sha256 = _sha256(path)

    write_run_manifest(
        path.parent / "ensemble_manifest.json",
        {
            "purpose": "A4 residual-dynamics ensemble pretraining",
            "training_seed": args.seed,
            "data_seed_start": data_seed_start,
            "data_seed_end": data_seed_end,
            "samples": args.samples,
            "optimization_steps": args.steps,
            "batch_size": args.batch_size,
            "device": args.device,
            "ensemble_config": asdict(model.config),
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
        root=root,
        extra={
            "initial_loss": stats.initial_loss,
            "final_loss": stats.final_loss,
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_file": dataset_path.name,
        },
    )

    print(f"residual dynamics loss: {stats.initial_loss:.6f} -> {stats.final_loss:.6f}")
    print(f"data seeds: {data_seed_start}..{data_seed_end}")
    print(f"checkpoint sha256: {checkpoint_sha256}")
    print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
