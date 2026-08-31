#!/usr/bin/env python3
"""Train from-scratch SAC on the analytical SARRL reaching environment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from sarrl.envs.planar_reach import DomainRandomization, PlanarReachEnv
from sarrl.rl import ReplayBuffer, SACAgent
from sarrl.utils import seed_everything


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["torque", "residual"], default="residual")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-steps", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--output", default="results/run_seed0")
    args = p.parse_args()

    seed_everything(args.seed)
    dr = DomainRandomization(0.15, 0.30, 0.15) if args.randomize else DomainRandomization()
    env = PlanarReachEnv(mode=args.mode, randomization=dr)
    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], seed=args.seed)
    replay = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], 500_000, args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset(seed=args.seed)
    ep_reward = 0.0
    episode = 0
    rows = []
    for step in range(1, args.steps + 1):
        action = env.action_space.sample(env._rng) if step <= args.start_steps else agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        replay.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        ep_reward += reward

        if len(replay) >= args.batch_size and step > args.start_steps:
            metrics = agent.update(replay.sample(args.batch_size))
        else:
            metrics = {}

        if terminated or truncated:
            episode += 1
            rows.append((episode, step, ep_reward, int(info["success"]), info["distance"]))
            if episode % 20 == 0:
                alpha = metrics.get("alpha", float("nan"))
                print(
                    f"episode={episode:5d} step={step:8d} reward={ep_reward:9.2f} "
                    f"success={int(info['success'])} alpha={alpha:.3f}"
                )
            obs, _ = env.reset()
            ep_reward = 0.0

    agent.save(out / "final.pt")
    with (out / "episodes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "step", "reward", "success", "final_distance"])
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
