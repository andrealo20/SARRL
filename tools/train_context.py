#!/usr/bin/env python3
"""Collect randomized planar-arm transitions and train the online context encoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sarrl.adaptation import ContextConfig, DynamicsContextEncoder, train_context_encoder
from sarrl.envs import DomainRandomization, PlanarReachEnv


def collect(samples: int, history: int, seed: int):
    randomization = DomainRandomization(
        mass_fraction=0.25,
        friction_fraction=0.4,
        motor_gain_fraction=0.25,
        payload_range=(0.0, 1.5),
        action_delay_max=3,
    )
    env = PlanarReachEnv(mode="residual", randomization=randomization, max_steps=history + 5)
    rng = np.random.default_rng(seed)
    sequences = []
    targets = []
    for episode in range(samples):
        obs, _ = env.reset(seed=seed + episode)
        rows = []
        for _ in range(history):
            # Bounded broadband excitation around the stabilising baseline.
            action = rng.uniform(-0.7, 0.7, size=2).astype(np.float32)
            next_obs, _, terminated, truncated, _ = env.step(action)
            rows.append(DynamicsContextEncoder.transition_feature(obs, action, next_obs))
            obs = next_obs
            if terminated or truncated:
                break
        while len(rows) < history:
            rows.insert(0, np.zeros(18, dtype=np.float32))
        sequences.append(np.asarray(rows[-history:], dtype=np.float32))
        targets.append(env.dynamics_context())
    return np.asarray(sequences), np.asarray(targets)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--history", type=int, default=16)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="results/context/context.pt")
    args = p.parse_args()
    cfg = ContextConfig(history=args.history)
    x, y = collect(args.samples, args.history, args.seed)
    model = DynamicsContextEncoder(cfg)
    stats = train_context_encoder(model, x, y, steps=args.steps, seed=args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    np.savez_compressed(path.with_suffix(".npz"), sequences=x, targets=y)
    print(f"context loss: {stats.initial_loss:.6f} -> {stats.final_loss:.6f}")
    print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
