#!/usr/bin/env python3
"""Deterministic evaluation of a SARRL SAC checkpoint."""

from __future__ import annotations

import argparse

import numpy as np

from sarrl.envs import PlanarReachEnv
from sarrl.rl import SACAgent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--mode", choices=["torque", "residual"], default="residual")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10_000)
    args = p.parse_args()

    env = PlanarReachEnv(mode=args.mode)
    agent = SACAgent.from_checkpoint(args.checkpoint, seed=0, load_optimizers=False)
    if agent.obs_dim != env.observation_space.shape[0] or agent.action_dim != env.action_space.shape[0]:
        raise SystemExit("checkpoint dimensions do not match evaluation environment")
    success = 0
    rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total = 0.0
        while True:
            obs, reward, terminated, truncated, info = env.step(agent.act(obs, deterministic=True))
            total += reward
            if terminated or truncated:
                success += int(info["success"])
                rewards.append(total)
                break
    print(f"success: {success}/{args.episodes} = {100.0 * success / args.episodes:.1f}%")
    print(f"reward:  {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
