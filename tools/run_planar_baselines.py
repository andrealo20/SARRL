#!/usr/bin/env python3
"""Run reproducible non-learned SARRL baselines across nominal/ID/OOD/fault scenarios."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sarrl.envs import DomainRandomization, FaultSpec, PlanarReachEnv
from sarrl.evaluation import EpisodeResult, aggregate, repository_commit, write_episode_csv, write_summary_json


@dataclass(frozen=True)
class Scenario:
    name: str
    randomization: DomainRandomization
    fault: FaultSpec | None = None


def scenarios():
    return [
        Scenario("nominal", DomainRandomization()),
        Scenario(
            "id_randomized",
            DomainRandomization(
                mass_fraction=0.15,
                friction_fraction=0.30,
                motor_gain_fraction=0.15,
                payload_range=(0.0, 1.0),
                action_delay_max=2,
            ),
        ),
        Scenario(
            "ood_dynamics",
            DomainRandomization(
                mass_fraction=0.30,
                friction_fraction=0.50,
                motor_gain_fraction=0.25,
                payload_range=(1.25, 1.75),
                action_delay_max=3,
            ),
        ),
        Scenario(
            "motor_fault",
            DomainRandomization(payload_range=(0.4, 0.8)),
            FaultSpec(start_step=20, motor_gain_multiplier=(1.0, 0.55)),
        ),
    ]


def run_scenario(scenario: Scenario, seeds: range):
    env = PlanarReachEnv(
        mode="residual",
        randomization=scenario.randomization,
        fault=scenario.fault,
    )
    out = []
    for seed in seeds:
        _, _ = env.reset(seed=seed)
        total = 0.0
        max_speed = 0.0
        max_torque = 0.0
        info = None
        while True:
            _, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
            total += reward
            max_speed = max(max_speed, float(np.linalg.norm(env.state[2:])))
            max_torque = max(max_torque, float(np.max(np.abs(info["commanded_torque"]))))
            if terminated or truncated:
                break
        out.append(
            EpisodeResult(
                scenario=scenario.name,
                controller="computed_torque_zero_residual",
                seed=seed,
                reward=total,
                steps=env.steps,
                success=bool(info["success"]),
                final_distance=float(info["distance"]),
                max_speed=max_speed,
                max_command_torque=max_torque,
                fault_seen=bool(info["fault_active"]),
            )
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--output", default="results/v0_9_baselines")
    p.add_argument("--scenario", default="all", choices=["all", "nominal", "id_randomized", "ood_dynamics", "motor_fault"])
    args = p.parse_args()
    if args.episodes <= 0:
        raise SystemExit("episodes must be positive")
    seeds = range(args.seed, args.seed + args.episodes)
    all_results = []
    summaries = {}
    selected = scenarios() if args.scenario == "all" else [s for s in scenarios() if s.name == args.scenario]
    for scenario in selected:
        rows = run_scenario(scenario, seeds)
        all_results.extend(rows)
        summary = aggregate(rows)
        summaries[scenario.name] = summary
        print(
            f"{scenario.name:14s}: {summary.successes:3d}/{summary.n} "
            f"({100*summary.success_rate:5.1f}%, "
            f"95% CI {100*summary.success_ci95_low:4.1f}-{100*summary.success_ci95_high:4.1f}%)"
        )
    out = Path(args.output)
    write_episode_csv(out.with_suffix(".csv"), all_results)
    write_summary_json(
        out.with_suffix(".json"),
        summaries,
        metadata={"seed_start": args.seed, "episodes_per_scenario": args.episodes, "git_commit": repository_commit()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
