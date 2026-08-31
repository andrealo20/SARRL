#!/usr/bin/env python3
"""Deterministic evaluation of a SARRL SAC checkpoint."""

from __future__ import annotations

import argparse

from sarrl.envs import PlanarReachEnv
from sarrl.evaluation import evaluate_policy
from sarrl.rl import SACAgent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--mode", choices=["torque", "residual"], default="residual")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10_000)
    args = p.parse_args()
    if args.episodes <= 0 or args.seed < 0:
        raise SystemExit("episodes must be positive and seed non-negative")

    env = PlanarReachEnv(mode=args.mode)
    agent = SACAgent.from_checkpoint(args.checkpoint, seed=0, load_optimizers=False)
    if (
        agent.obs_dim != env.observation_space.shape[0]
        or agent.action_dim != env.action_space.shape[0]
    ):
        raise SystemExit("checkpoint dimensions do not match evaluation environment")
    result = evaluate_policy(agent, env, args.episodes, args.seed)
    print(
        f"success: {result.successes}/{result.episodes} = "
        f"{100.0 * result.success_rate:.1f}%"
    )
    print(f"reward:  {result.reward_mean:.2f} +/- {result.reward_std:.2f}")
    print(f"final distance: {result.final_distance_mean:.4f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
