#!/usr/bin/env python3
"""Train from-scratch SAC on the analytical SARRL reaching environment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from sarrl.envs.planar_reach import DomainRandomization, PlanarReachEnv
from sarrl.rl import (
    ReplayBuffer,
    SACAgent,
    load_training_checkpoint,
    save_training_checkpoint,
)
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
    p.add_argument("--resume", default=None)
    p.add_argument("--checkpoint-every", type=int, default=50_000)
    args = p.parse_args()
    if args.steps <= 0 or args.start_steps < 0 or args.batch_size <= 0:
        raise SystemExit("steps/batch-size must be positive and start-steps non-negative")
    if args.checkpoint_every < 0:
        raise SystemExit("checkpoint-every must be non-negative")

    seed_everything(args.seed)
    dr = (
        DomainRandomization(
            mass_fraction=0.15,
            friction_fraction=0.30,
            motor_gain_fraction=0.15,
            payload_range=(0.0, 1.0),
            action_delay_max=2,
        )
        if args.randomize
        else DomainRandomization()
    )
    env = PlanarReachEnv(mode=args.mode, randomization=dr)
    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], seed=args.seed)
    replay = ReplayBuffer(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        500_000,
        args.seed,
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.resume is not None:
        loop = load_training_checkpoint(args.resume, agent, replay, env)
        step0 = int(loop["step"])
        episode = int(loop["episode"])
        ep_reward = float(loop["ep_reward"])
        obs = np.asarray(loop["obs"], dtype=np.float32)
        rows = list(loop.get("rows", []))
        if step0 >= args.steps:
            raise SystemExit("resume checkpoint has already reached requested --steps")
    else:
        obs, _ = env.reset(seed=args.seed)
        ep_reward = 0.0
        episode = 0
        step0 = 0
        rows = []

    for step in range(step0 + 1, args.steps + 1):
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

        if args.checkpoint_every and step % args.checkpoint_every == 0:
            save_training_checkpoint(
                out / f"train_step{step}.pt",
                agent,
                replay,
                env,
                {
                    "step": step,
                    "episode": episode,
                    "ep_reward": ep_reward,
                    "obs": obs,
                    "rows": rows,
                },
            )

    agent.save(out / "final.pt")
    save_training_checkpoint(
        out / "training_final.pt",
        agent,
        replay,
        env,
        {
            "step": args.steps,
            "episode": episode,
            "ep_reward": ep_reward,
            "obs": obs,
            "rows": rows,
        },
    )
    with (out / "episodes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "step", "reward", "success", "final_distance"])
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
