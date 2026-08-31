#!/usr/bin/env python3
"""Evaluate the zero-residual computed-torque baseline on fixed seeds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from sarrl.envs import PlanarReachEnv


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--output", default="results/v0_1_nominal.csv")
    args = p.parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")

    env = PlanarReachEnv(mode="residual")
    rows = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        _, info0 = env.reset(seed=seed)
        total = 0.0
        for step in range(1, env.max_steps + 1):
            _, reward, terminated, truncated, info = env.step(np.zeros(2))
            total += reward
            if terminated or truncated:
                rows.append(
                    [
                        seed,
                        float(info0["target"][0]),
                        float(info0["target"][1]),
                        int(info["success"]),
                        step,
                        total,
                        float(info["distance"]),
                    ]
                )
                break

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "target_x", "target_y", "success", "steps", "reward", "distance"])
        w.writerows(rows)

    successes = sum(r[3] for r in rows)
    mean_steps = np.mean([r[4] for r in rows])
    mean_distance = np.mean([r[6] for r in rows])
    print(f"success: {successes}/{len(rows)} = {100 * successes / len(rows):.1f}%")
    print(f"mean steps: {mean_steps:.2f}")
    print(f"mean terminal distance: {mean_distance:.5f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
