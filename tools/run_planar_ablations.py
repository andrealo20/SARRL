#!/usr/bin/env python3
"""Plan the reproducible SARRL v1.2 planar ablation campaign."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from sarrl.evaluation import seed_ranges_overlap, write_run_manifest


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
    AblationCondition(
        "A0",
        "Computed torque",
        "none",
        False,
        False,
        False,
        "ready",
    ),
    AblationCondition(
        "A1",
        "Direct SAC",
        "train-direct-sac",
        False,
        False,
        False,
        "ready",
    ),
    AblationCondition(
        "A2",
        "Residual SAC",
        "retained-v1.1.0",
        False,
        False,
        False,
        "ready",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--validation-seed", type=int, default=20_000)
    parser.add_argument("--validation-episodes", type=int, default=30)
    parser.add_argument("--heldout-seed", type=int, default=40_000)
    parser.add_argument("--heldout-episodes", type=int, default=100)
    parser.add_argument("--output", default="results/planar_ablations")
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

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parents[1]
    write_run_manifest(
        out / "experiment_manifest.json",
        protocol,
        root=root,
    )

    print("SARRL v1.2 planar ablation protocol")
    print("------------------------------------")
    for condition in CONDITIONS:
        print(
            f"{condition.key}: {condition.label:<35} "
            f"[{condition.status}]"
        )

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
