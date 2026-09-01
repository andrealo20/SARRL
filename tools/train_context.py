#!/usr/bin/env python3
"""Collect v1.2 planar transitions and train a causal dynamics context encoder."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from sarrl.adaptation import (
    ContextConfig,
    DynamicsContextEncoder,
    train_context_encoder,
)
from sarrl.envs import PlanarReachEnv
from sarrl.evaluation import (
    V12_CONTEXT_HISTORY,
    V12_CONTEXT_SAMPLES,
    V12_CONTEXT_TRAINING_STEPS,
    assert_repository_import_root,
    context_data_seed,
    planar_id_randomization,
    planar_id_randomization_dict,
    validate_context_data_range,
    write_run_manifest,
)
from sarrl.utils import seed_everything


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(
    samples: int,
    history: int,
    data_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect a deterministic supervised context dataset.

    Ground-truth dynamics parameters are used only as supervised labels.
    Runtime context inference remains causal and does not query these labels.
    """
    if samples <= 0 or history <= 0 or data_seed < 0:
        raise ValueError("samples/history must be positive and data_seed non-negative")

    env = PlanarReachEnv(
        mode="residual",
        randomization=planar_id_randomization(),
        max_steps=history + 5,
    )

    # Dataset generation is independent from neural-network initialization.
    rng = np.random.default_rng(data_seed)

    sequences = []
    targets = []

    transition_dim = 18

    for episode in range(samples):
        obs, _ = env.reset(seed=data_seed + episode)
        rows = []

        for _ in range(history):
            action = rng.uniform(
                -0.7,
                0.7,
                size=2,
            ).astype(np.float32)

            next_obs, _, terminated, truncated, _ = env.step(action)

            rows.append(
                DynamicsContextEncoder.transition_feature(
                    obs,
                    action,
                    next_obs,
                )
            )
            obs = next_obs

            if terminated or truncated:
                break

        while len(rows) < history:
            rows.insert(
                0,
                np.zeros(transition_dim, dtype=np.float32),
            )

        sequences.append(np.asarray(rows[-history:], dtype=np.float32))
        targets.append(env.dynamics_context())

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=V12_CONTEXT_SAMPLES)
    p.add_argument("--history", type=int, default=V12_CONTEXT_HISTORY)
    p.add_argument("--steps", type=int, default=V12_CONTEXT_TRAINING_STEPS)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Neural-network initialization and optimization seed.",
    )
    p.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help=(
            "First environment seed used for context-data collection. "
            "Defaults to the v1.2 per-training-seed namespace."
        ),
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Context pretraining device. Official v1.2 runs use CPU.",
    )
    p.add_argument(
        "--output",
        default="results/context/context.pt",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    assert_repository_import_root(root)

    if args.seed < 0:
        raise SystemExit("context training seed must be non-negative")

    if args.samples <= 0 or args.history <= 0 or args.steps <= 0:
        raise SystemExit("samples/history/steps must be positive")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA context training requested but CUDA is unavailable")

    data_seed = context_data_seed(args.seed) if args.data_seed is None else args.data_seed

    if data_seed < 0:
        raise SystemExit("context data seed must be non-negative")

    if args.data_seed is None:
        data_seed_start, data_seed_end = validate_context_data_range(
            args.seed,
            args.samples,
        )
    else:
        data_seed_start = data_seed
        data_seed_end = data_seed + args.samples - 1

    # Fix all relevant RNGs before constructing the neural network.
    seed_everything(args.seed)

    cfg = ContextConfig(history=args.history)

    x, y = collect(
        samples=args.samples,
        history=args.history,
        data_seed=data_seed,
    )

    model = DynamicsContextEncoder(cfg)

    stats = train_context_encoder(
        model,
        x,
        y,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
    )

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.save(path)

    dataset_path = path.with_suffix(".npz")
    np.savez_compressed(
        dataset_path,
        sequences=x,
        targets=y,
    )

    checkpoint_sha256 = _sha256(path)

    root = Path(__file__).resolve().parents[1]

    write_run_manifest(
        path.parent / "context_manifest.json",
        {
            "purpose": "A3 learned dynamics context pretraining",
            "training_seed": args.seed,
            "data_seed_start": data_seed_start,
            "data_seed_end": data_seed_end,
            "samples": args.samples,
            "history": args.history,
            "optimization_steps": args.steps,
            "device": args.device,
            "context_config": {
                "obs_dim": cfg.obs_dim,
                "action_dim": cfg.action_dim,
                "context_dim": cfg.context_dim,
                "latent_dim": cfg.latent_dim,
                "hidden_dim": cfg.hidden_dim,
                "history": cfg.history,
                "learning_rate": cfg.learning_rate,
            },
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
        },
        root=root,
        extra={
            "initial_loss": stats.initial_loss,
            "final_loss": stats.final_loss,
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_file": dataset_path.name,
        },
    )

    print(f"context loss: {stats.initial_loss:.6f} -> {stats.final_loss:.6f}")
    print(f"data seeds: {data_seed_start}..{data_seed_end}")
    print(f"checkpoint sha256: {checkpoint_sha256}")
    print(f"saved: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
