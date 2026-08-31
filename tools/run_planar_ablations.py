#!/usr/bin/env python3
"""Run the reproducible SARRL v1.2 planar ablation campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sarrl.envs import DomainRandomization, PlanarReachEnv
from sarrl.evaluation import (
    aggregate,
    evaluate_policy_episodes,
    seed_ranges_overlap,
    write_episode_csv,
    write_run_manifest,
    write_summary_json,
)


@dataclass(frozen=True)
class AblationCondition:
    key: str
    label: str
    policy_source: str
    context: bool
    uncertainty_gate: bool
    hocbf: bool
    status: str


CONDITIONS = (
    AblationCondition("A0", "Computed torque", "none", False, False, False, "ready"),
    AblationCondition(
        "A1", "Direct SAC", "train-direct-sac", False, False, False, "ready"
    ),
    AblationCondition(
        "A2", "Residual SAC", "retained-v1.1.0", False, False, False, "ready"
    ),
    AblationCondition(
        "A3",
        "Residual SAC + context",
        "train-context-conditioned-residual-sac",
        True,
        False,
        False,
        "needs-context-policy-wiring",
    ),
    AblationCondition(
        "A4",
        "Residual SAC + uncertainty gate",
        "retained-v1.1.0",
        False,
        True,
        False,
        "needs-ensemble-evaluation-wiring",
    ),
    AblationCondition(
        "A5",
        "Residual SAC + HOCBF",
        "retained-v1.1.0",
        False,
        False,
        True,
        "needs-safety-evaluation-runner",
    ),
    AblationCondition(
        "A6",
        "Full adaptive stack",
        "context-conditioned-residual-sac",
        True,
        True,
        True,
        "needs-full-stack-wiring",
    ),
)


class _ZeroResidualPolicy:
    def act(self, obs, deterministic=False):
        return np.zeros(2, dtype=np.float32)


def _randomization() -> DomainRandomization:
    return DomainRandomization(
        mass_fraction=0.15,
        friction_fraction=0.30,
        motor_gain_fraction=0.15,
        payload_range=(0.0, 1.0),
        action_delay_max=2,
    )


def build_protocol(
    seeds: list[int],
    steps: int,
    validation_seed: int,
    validation_episodes: int,
    heldout_seed: int,
    heldout_episodes: int,
) -> dict:
    if len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be unique")
    if any(seed < 0 for seed in seeds):
        raise ValueError("training seeds must be non-negative")
    if steps <= 0:
        raise ValueError("training steps must be positive")
    if validation_episodes <= 0 or heldout_episodes <= 0:
        raise ValueError("evaluation episode counts must be positive")
    if validation_seed < 0 or heldout_seed < 0:
        raise ValueError("evaluation seeds must be non-negative")

    if seed_ranges_overlap(
        validation_seed,
        validation_episodes,
        heldout_seed,
        heldout_episodes,
    ):
        raise ValueError("validation and held-out seed ranges must not overlap")

    return {
        "release_target": "v1.2.0",
        "campaign": "planar_ablations",
        "training": {
            "seeds": seeds,
            "steps_per_seed": steps,
            "start_steps": 5_000,
            "batch_size": 256,
            "hidden": [256, 256],
            "update_every": 1,
            "replay_capacity": 200_000,
        },
        "validation": {
            "seed_start": validation_seed,
            "episodes": validation_episodes,
            "every_steps": 25_000,
            "checkpoint_selection": "success_rate_then_reward",
        },
        "heldout": {
            "seed_start": heldout_seed,
            "episodes_per_policy": heldout_episodes,
        },
        "domain_randomization": {
            "mass_fraction": 0.15,
            "friction_fraction": 0.30,
            "motor_gain_fraction": 0.15,
            "payload_range": [0.0, 1.0],
            "action_delay_max": 2,
        },
        "statistics": {
            "multi_seed_spread": "sample_sd_ddof_1",
            "episode_success_interval": "wilson_95",
            "paired_comparison": "paired_bootstrap_95",
        },
        "conditions": [asdict(condition) for condition in CONDITIONS],
    }


def run_a0(out: Path, heldout_seed: int, heldout_episodes: int) -> None:
    """Evaluate the computed-torque baseline on the fixed held-out seeds."""
    env = PlanarReachEnv(mode="residual", randomization=_randomization())

    rows = evaluate_policy_episodes(
        _ZeroResidualPolicy(),
        env,
        heldout_episodes,
        heldout_seed,
        scenario="id_randomized",
        controller="A0_computed_torque",
    )

    condition_out = out / "A0_computed_torque"
    write_episode_csv(condition_out / "heldout_episodes.csv", rows)

    metrics = aggregate(rows)
    write_summary_json(
        condition_out / "summary.json",
        {"A0_computed_torque": metrics},
        metadata={
            "condition": "A0",
            "heldout_seed_start": heldout_seed,
            "heldout_episodes": heldout_episodes,
        },
    )

    print(
        "A0 computed torque: "
        f"{metrics.successes}/{metrics.n} "
        f"= {100.0 * metrics.success_rate:.1f}%"
    )


def run_a1(
    root: Path,
    out: Path,
    seeds: list[int],
    steps: int,
    validation_seed: int,
    validation_episodes: int,
    heldout_seed: int,
    heldout_episodes: int,
    confirm_training: bool,
) -> None:
    """Run the Direct SAC multi-seed campaign."""
    cmd = [
        sys.executable,
        str(root / "tools" / "run_sac_sweep.py"),
        "--seeds",
        *[str(seed) for seed in seeds],
        "--mode",
        "torque",
        "--steps",
        str(steps),
        "--randomize",
        "--validation-seed",
        str(validation_seed),
        "--validation-episodes",
        str(validation_episodes),
        "--heldout-seed",
        str(heldout_seed),
        "--heldout-episodes",
        str(heldout_episodes),
        "--output",
        str(out / "A1_direct_sac"),
    ]

    print("A1 command:")
    print(" ".join(cmd))

    if not confirm_training:
        print("A1 training NOT started. Pass --confirm-training to execute it.")
        return

    subprocess.run(cmd, cwd=root, check=True)


def register_a2(root: Path, out: Path) -> None:
    """Register the retained v1.1 residual-SAC evidence as condition A2."""
    source = root / "artifacts" / "planar_sac_5seed"

    required = [
        source / "summary.csv",
        source / "heldout_episodes.csv",
        source / "aggregate.json",
        source / "paired_comparison.csv",
        source / "result.json",
    ]

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "A2 retained evidence is incomplete: " + ", ".join(missing)
        )

    condition_out = out / "A2_residual_sac"
    condition_out.mkdir(parents=True, exist_ok=True)

    payload = {
        "condition": "A2",
        "label": "Residual SAC",
        "source": str(source),
        "source_release": "v1.1.0",
        "training_commit": "9f832614ce8b51c207873ff4861986ab72903115",
        "reused_without_retraining": True,
        "files": [str(path.relative_to(root)) for path in required],
    }

    (condition_out / "retained_source.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    print(f"A2 retained evidence verified: {source}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--validation-seed", type=int, default=20_000)
    parser.add_argument("--validation-episodes", type=int, default=30)
    parser.add_argument("--heldout-seed", type=int, default=40_000)
    parser.add_argument("--heldout-episodes", type=int, default=100)
    parser.add_argument("--output", default="results/planar_ablations")
    parser.add_argument(
        "--execute",
        nargs="*",
        choices=["A0", "A1", "A2"],
        default=[],
        help="Ready conditions to execute/register.",
    )
    parser.add_argument(
        "--confirm-training",
        action="store_true",
        help="Required before the expensive A1 Direct SAC training is launched.",
    )
    args = parser.parse_args()

    try:
        protocol = build_protocol(
            args.seeds,
            args.steps,
            args.validation_seed,
            args.validation_episodes,
            args.heldout_seed,
            args.heldout_episodes,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    root = Path(__file__).resolve().parents[1]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    write_run_manifest(
        out / "experiment_manifest.json",
        protocol,
        root=root,
    )

    print("SARRL v1.2 planar ablation protocol")
    print("------------------------------------")
    for condition in CONDITIONS:
        print(f"{condition.key}: {condition.label:<35} [{condition.status}]")

    print()
    print(f"training seeds: {args.seeds}")
    print(f"steps/seed:     {args.steps}")
    print(
        f"validation:     {args.validation_seed}"
        f"..{args.validation_seed + args.validation_episodes - 1}"
    )
    print(
        f"held-out:       {args.heldout_seed}"
        f"..{args.heldout_seed + args.heldout_episodes - 1}"
    )
    print(f"manifest:       {out / 'experiment_manifest.json'}")

    if "A0" in args.execute:
        run_a0(out, args.heldout_seed, args.heldout_episodes)

    if "A1" in args.execute:
        run_a1(
            root,
            out,
            args.seeds,
            args.steps,
            args.validation_seed,
            args.validation_episodes,
            args.heldout_seed,
            args.heldout_episodes,
            args.confirm_training,
        )

    if "A2" in args.execute:
        register_a2(root, out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
