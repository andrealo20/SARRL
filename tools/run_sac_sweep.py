#!/usr/bin/env python3
"""Run a reproducible multi-seed SAC campaign and held-out evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from sarrl.envs import DomainRandomization, PlanarReachEnv
from sarrl.evaluation import (
    aggregate,
    evaluate_policy_episodes,
    seed_ranges_overlap,
    write_episode_csv,
    write_run_manifest,
)
from sarrl.rl import SACAgent


def _randomization(enabled: bool) -> DomainRandomization:
    if not enabled:
        return DomainRandomization()
    return DomainRandomization(
        mass_fraction=0.15,
        friction_fraction=0.30,
        motor_gain_fraction=0.15,
        payload_range=(0.0, 1.0),
        action_delay_max=2,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--mode", choices=["torque", "residual"], default="residual")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--start-steps", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, nargs=2, default=(256, 256), metavar=("H1", "H2"))
    p.add_argument("--update-every", type=int, default=1)
    p.add_argument("--replay-capacity", type=int, default=200_000)
    p.add_argument("--validate-every", type=int, default=25_000)
    p.add_argument("--validation-episodes", type=int, default=30)
    p.add_argument("--validation-seed", type=int, default=20_000)
    p.add_argument("--heldout-episodes", type=int, default=100)
    p.add_argument("--heldout-seed", type=int, default=40_000)
    p.add_argument("--output", default="results/sac_sweep")
    args = p.parse_args()

    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise SystemExit("training seeds must be unique and non-negative")
    if args.steps <= 0 or args.heldout_episodes <= 0 or args.validation_episodes <= 0:
        raise SystemExit("steps and evaluation episode counts must be positive")
    if seed_ranges_overlap(
        args.validation_seed,
        args.validation_episodes,
        args.heldout_seed,
        args.heldout_episodes,
    ):
        raise SystemExit("validation and held-out seed ranges must not overlap")

    root = Path(__file__).resolve().parents[1]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        out / "sweep_manifest.json",
        {
            "seeds": args.seeds,
            "mode": args.mode,
            "steps": args.steps,
            "randomize": args.randomize,
            "start_steps": args.start_steps,
            "batch_size": args.batch_size,
            "hidden": list(args.hidden),
            "update_every": args.update_every,
            "replay_capacity": args.replay_capacity,
            "validation_episodes": args.validation_episodes,
            "validation_seed": args.validation_seed,
            "heldout_episodes": args.heldout_episodes,
            "heldout_seed": args.heldout_seed,
        },
        root=root,
    )

    summary_rows = []
    all_episode_rows = []
    dr = _randomization(args.randomize)
    for training_seed in args.seeds:
        run_dir = out / f"seed_{training_seed}"
        cmd = [
            sys.executable,
            str(root / "tools" / "train_sac.py"),
            "--mode",
            args.mode,
            "--steps",
            str(args.steps),
            "--seed",
            str(training_seed),
            "--start-steps",
            str(args.start_steps),
            "--batch-size",
            str(args.batch_size),
            "--hidden",
            str(args.hidden[0]),
            str(args.hidden[1]),
            "--update-every",
            str(args.update_every),
            "--replay-capacity",
            str(args.replay_capacity),
            "--validate-every",
            str(args.validate_every),
            "--validation-episodes",
            str(args.validation_episodes),
            "--validation-seed",
            str(args.validation_seed),
            "--output",
            str(run_dir),
        ]
        if args.randomize:
            cmd.append("--randomize")
        subprocess.run(cmd, cwd=root, check=True)

        checkpoint = run_dir / "best.pt"
        if not checkpoint.exists():
            checkpoint = run_dir / "final.pt"
        agent = SACAgent.from_checkpoint(checkpoint, seed=0, load_optimizers=False)
        env = PlanarReachEnv(mode=args.mode, randomization=dr)
        rows = evaluate_policy_episodes(
            agent,
            env,
            args.heldout_episodes,
            args.heldout_seed,
            scenario="heldout",
            controller=f"sac_train_seed_{training_seed}",
        )
        metrics = aggregate(rows)
        all_episode_rows.extend(rows)
        summary_rows.append(
            {
                "training_seed": training_seed,
                "checkpoint": str(checkpoint),
                "successes": metrics.successes,
                "episodes": metrics.n,
                "success_rate": metrics.success_rate,
                "success_ci95_low": metrics.success_ci95_low,
                "success_ci95_high": metrics.success_ci95_high,
                "reward_mean": metrics.reward_mean,
                "reward_std": metrics.reward_std,
                "final_distance_mean": metrics.final_distance_mean,
            }
        )
        print(
            f"seed={training_seed} heldout={metrics.successes}/{metrics.n} "
            f"({100.0 * metrics.success_rate:.1f}%) reward={metrics.reward_mean:.2f}"
        )

    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    write_episode_csv(out / "heldout_episodes.csv", all_episode_rows)

    rates = np.asarray([row["success_rate"] for row in summary_rows], dtype=np.float64)
    rewards = np.asarray([row["reward_mean"] for row in summary_rows], dtype=np.float64)
    aggregate_payload = {
        "training_seeds": args.seeds,
        "models": len(summary_rows),
        "heldout_episodes_per_model": args.heldout_episodes,
        "success_rate_mean": float(rates.mean()),
        "success_rate_std": float(rates.std()),
        "success_rate_min": float(rates.min()),
        "success_rate_max": float(rates.max()),
        "reward_mean_across_models": float(rewards.mean()),
        "reward_std_across_models": float(rewards.std()),
    }
    (out / "aggregate.json").write_text(
        json.dumps(aggregate_payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        "multi-seed success: "
        f"{100.0 * rates.mean():.1f}% +/- {100.0 * rates.std():.1f}% "
        f"across {len(rates)} training seeds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
