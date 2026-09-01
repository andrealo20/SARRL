#!/usr/bin/env python3
"""Run a reproducible multi-seed SAC campaign and held-out evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from sarrl.adaptation import AdaptiveContextEnv, DynamicsContextEncoder
from sarrl.envs import DomainRandomization, PlanarReachEnv
from sarrl.evaluation import (
    aggregate,
    evaluate_policy_episodes,
    repository_commit,
    seed_ranges_overlap,
    write_episode_csv,
    write_run_manifest,
)
from sarrl.rl import SACAgent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _context_checkpoint_for_seed(
    context_root: Path,
    training_seed: int,
) -> Path:
    checkpoint = context_root / f"context_seed_{training_seed}" / "context.pt"

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"missing context checkpoint for training seed {training_seed}: {checkpoint}"
        )

    return checkpoint.resolve()


def _sample_std(values: np.ndarray) -> float | None:
    """Sample standard deviation across independent training runs."""
    if values.size < 2:
        return None
    return float(values.std(ddof=1))


def _checkpoint_step(path: Path) -> int:
    """Return the training step encoded by train_stepN.pt."""
    try:
        return int(path.stem.removeprefix("train_step"))
    except ValueError:
        return -1


def _resume_plan(
    run_dir: Path,
    requested_steps: int,
    current_commit: str | None,
    expected_context_sha256: str | None = None,
) -> tuple[Path | None, bool]:
    """Return (resume_checkpoint, already_complete) for an existing run."""
    final_training = run_dir / "training_final.pt"
    periodic = [path for path in run_dir.glob("train_step*.pt") if _checkpoint_step(path) >= 0]

    if not final_training.exists() and not periodic:
        return None, False

    manifest = run_dir / "run_manifest.json"
    if not manifest.exists():
        raise ValueError(f"existing checkpoints in {run_dir} have no run_manifest.json")

    payload = json.loads(manifest.read_text())
    previous_commit = payload.get("runtime", {}).get("git_commit")

    if (
        previous_commit is not None
        and current_commit is not None
        and previous_commit != current_commit
    ):
        raise ValueError(
            "refusing to resume training across different git commits: "
            f"{previous_commit} != {current_commit}"
        )

    previous_context = payload.get("config", {}).get("context", {}).get("checkpoint_sha256")

    if previous_context != expected_context_sha256:
        raise ValueError(
            "refusing to resume or reuse training with a different "
            "context checkpoint: "
            f"{previous_context} != {expected_context_sha256}"
        )

    previous_requested = int(payload["config"]["requested_steps"])

    if final_training.exists():
        if previous_requested == requested_steps:
            return None, True
        if previous_requested < requested_steps:
            return final_training, False
        raise ValueError("existing completed run requested more steps than the new campaign")

    latest = max(periodic, key=_checkpoint_step)
    latest_step = _checkpoint_step(latest)

    if latest_step > requested_steps:
        raise ValueError(
            f"latest checkpoint step {latest_step} exceeds requested {requested_steps}"
        )

    return latest, False


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
    p.add_argument(
        "--context-root",
        default=None,
        help=(
            "Directory containing per-training-seed context checkpoints "
            "as context_seed_<seed>/context.pt."
        ),
    )
    p.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume interrupted seed runs and reuse completed runs.",
    )
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

    if args.context_root is not None and args.mode != "residual":
        raise SystemExit("--context-root requires --mode residual")

    context_records = {}

    if args.context_root is not None:
        context_root = Path(args.context_root).resolve()

        for training_seed in args.seeds:
            try:
                checkpoint = _context_checkpoint_for_seed(
                    context_root,
                    training_seed,
                )
            except FileNotFoundError as exc:
                raise SystemExit(str(exc)) from exc

            context_records[training_seed] = {
                "checkpoint": checkpoint,
                "sha256": _sha256(checkpoint),
            }

    root = Path(__file__).resolve().parents[1]
    current_commit = repository_commit(root)
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
            "resume_existing": args.resume_existing,
            "context": {
                "enabled": args.context_root is not None,
                "root": (
                    None if args.context_root is None else str(Path(args.context_root).resolve())
                ),
                "per_training_seed": {
                    str(seed): {
                        "checkpoint": str(record["checkpoint"]),
                        "checkpoint_sha256": record["sha256"],
                    }
                    for seed, record in context_records.items()
                },
            },
        },
        root=root,
    )

    summary_rows = []
    all_episode_rows = []
    dr = _randomization(args.randomize)
    for training_seed in args.seeds:
        run_dir = out / f"seed_{training_seed}"

        context_record = context_records.get(training_seed)
        expected_context_sha256 = None if context_record is None else context_record["sha256"]

        resume_checkpoint = None
        training_complete = False
        if args.resume_existing:
            try:
                resume_checkpoint, training_complete = _resume_plan(
                    run_dir,
                    args.steps,
                    current_commit,
                    expected_context_sha256,
                )
            except ValueError as exc:
                raise SystemExit(f"seed {training_seed}: {exc}") from exc

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

        if context_record is not None:
            cmd.extend(
                [
                    "--context-checkpoint",
                    str(context_record["checkpoint"]),
                ]
            )

        if training_complete:
            print(f"seed={training_seed} training already complete; reusing retained checkpoints")
        else:
            if resume_checkpoint is not None:
                cmd.extend(["--resume", str(resume_checkpoint)])
                print(f"seed={training_seed} resuming from {resume_checkpoint}")
            subprocess.run(cmd, cwd=root, check=True)

        run_manifest_path = run_dir / "run_manifest.json"

        if not run_manifest_path.is_file():
            raise SystemExit(f"seed {training_seed}: missing run_manifest.json")

        run_manifest = json.loads(run_manifest_path.read_text())

        recorded_context_sha256 = (
            run_manifest.get("config", {}).get("context", {}).get("checkpoint_sha256")
        )

        if recorded_context_sha256 != expected_context_sha256:
            raise SystemExit(
                f"seed {training_seed}: training manifest context SHA "
                "does not match the requested context checkpoint"
            )

        checkpoint = run_dir / "best.pt"
        if not checkpoint.exists():
            checkpoint = run_dir / "final.pt"

        agent = SACAgent.from_checkpoint(
            checkpoint,
            seed=0,
            load_optimizers=False,
        )

        base_env = PlanarReachEnv(
            mode=args.mode,
            randomization=dr,
        )

        if context_record is None:
            env = base_env
        else:
            encoder = DynamicsContextEncoder.load(
                context_record["checkpoint"],
                map_location="cpu",
            )

            encoder.eval()
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)

            env = AdaptiveContextEnv(
                base_env,
                encoder,
                device="cpu",
            )
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
                "context_checkpoint_sha256": expected_context_sha256,
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
    rates_std = _sample_std(rates)
    rewards_std = _sample_std(rewards)

    aggregate_payload = {
        "training_seeds": args.seeds,
        "models": len(summary_rows),
        "heldout_episodes_per_model": args.heldout_episodes,
        "success_rate_mean": float(rates.mean()),
        "success_rate_std": rates_std,
        "success_rate_min": float(rates.min()),
        "success_rate_max": float(rates.max()),
        "reward_mean_across_models": float(rewards.mean()),
        "reward_std_across_models": rewards_std,
    }
    (out / "aggregate.json").write_text(
        json.dumps(aggregate_payload, indent=2, sort_keys=True) + "\n"
    )
    spread = "n/a" if rates_std is None else f"{100.0 * rates_std:.1f}%"
    print(
        "multi-seed success: "
        f"{100.0 * rates.mean():.1f}% +/- {spread} "
        f"across {len(rates)} training seeds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
