#!/usr/bin/env python3
"""Run the exploratory adaptive-nominal pilot and write its summary.

The pilot is descriptive. It compares the frozen nominal with the online
command-space estimate, each with and without the HOCBF filter, on the six
historical diagnostic cases and on fresh seeds outside every official range.
It trains nothing, and its counts are not population estimates.
"""

# Numerical thread limits precede all numerical imports.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

from sarrl.controllers import AdaptiveNominalConfig
from sarrl.envs.planar_reach import ObstacleSpec
from sarrl.evaluation import assert_repository_import_root
from sarrl.evaluation.adaptive_pilot import (
    ARMS,
    RESIDUAL_ARMS,
    PlantOptions,
    episodes_to_records,
    pilot_cases,
    run_case,
    summarize,
)
from sarrl.evaluation.provenance import runtime_metadata
from sarrl.rl import SACAgent

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "sarrl/controllers/adaptive_nominal.py",
    "sarrl/controllers/computed_torque.py",
    "sarrl/dynamics/planar_arm.py",
    "sarrl/envs/planar_reach.py",
    "sarrl/evaluation/adaptive_pilot.py",
    "sarrl/evaluation/safety_audit.py",
    "sarrl/evaluation/planar_v13.py",
    "sarrl/safety/filter.py",
    "tools/run_adaptive_pilot.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown_summary(table: dict, config: AdaptiveNominalConfig, records: list[dict]) -> str:
    lines = [
        "# Adaptive nominal pilot",
        "",
        "Exploratory comparison on shared initial conditions. Counts are not",
        "population estimates. Estimator settings: "
        + ", ".join(f"{k}={v}" for k, v in asdict(config).items() if k not in ("kp", "kd")),
        "",
    ]
    for origin in ("historical", "fresh"):
        lines += [f"## {origin.capitalize()} cases", ""]
        lines.append(
            "| Scenario | Arm | Episodes | Success | Timeout | Abort | Unsafe "
            "| Median final distance, m | Max normalized violation | Lag correct "
            "| Obstacle violations | Contacts (tip / link) |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for key, row in table.items():
            key_origin, scenario, arm = key.split("/")
            if key_origin != origin:
                continue
            lag = "" if row["lag_correct"] is None else str(row["lag_correct"])
            lines.append(
                f"| {scenario} | {arm} | {row['episodes']} | {row['success']} | "
                f"{row['timeout']} | {row['abort']} | {row['unsafe_episodes']} | "
                f"{row['median_final_distance_m']:.3f} | "
                f"{row['max_normalized_violation']:.4f} | {lag} | "
                f"{row['obstacle_violation_episodes']} | {row['obstacle_contact_episodes']} "
                f"({row['obstacle_tip_contact_episodes']} / "
                f"{row['obstacle_link_contact_episodes']}) |"
            )
        lines.append("")
    lines += ["## Historical cases, one row per episode", ""]
    lines.append(
        "| Scenario | Seed | Arm | Outcome | Steps | Final distance, m | Unsafe "
        "| Max normalized violation | True delay | Selected lag | Prediction RMS, N m |"
    )
    lines.append("| --- | ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |")
    for record in records:
        if record["origin"] != "historical":
            continue
        rms = record["prediction_error_rms"]
        rms_text = "" if rms is None else f"{rms[0]:.3f} / {rms[1]:.3f}"
        lag = "" if record["selected_lag"] is None else str(record["selected_lag"])
        lines.append(
            f"| {record['scenario']} | {record['seed']} | {record['arm']} | "
            f"{record['outcome']} | {record['steps']} | {record['final_distance']:.3f} | "
            f"{'yes' if record['unsafe_episode'] else 'no'} | "
            f"{record['normalized_violation_max']:.4f} | {record['true_delay']} | "
            f"{lag} | {rms_text} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fresh", type=int, default=20, help="fresh seeds per scenario")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS + RESIDUAL_ARMS)
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="selected SAC checkpoint for the residual arms; evaluated deterministically",
    )
    parser.add_argument("--process-noise", type=float, default=None)
    parser.add_argument("--forgetting", type=float, default=None)
    parser.add_argument("--max-lag", type=int, default=None)
    parser.add_argument("--payload-prior", type=float, default=None)
    parser.add_argument("--filter-gate", action="store_true")
    parser.add_argument("--gate-threshold", type=float, default=None)
    parser.add_argument("--historical", action="store_true", help="include the historical cases")
    parser.add_argument("--delay-compensation", action="store_true")
    parser.add_argument("--plant", choices=("analytical", "mujoco"), default="analytical")
    parser.add_argument("--sensor-noise", type=float, default=0.0)
    parser.add_argument(
        "--actuator-grid",
        type=float,
        nargs="+",
        default=None,
        help="candidate actuator time constants for the estimator, in seconds",
    )
    parser.add_argument("--armature", type=float, default=0.0)
    parser.add_argument("--actuator-tau", type=float, default=0.0)
    parser.add_argument("--armature-range", type=float, nargs=2, default=None)
    parser.add_argument("--actuator-tau-range", type=float, nargs=2, default=None)
    parser.add_argument(
        "--obstacle", action="store_true", help="one circular obstacle per episode across the path"
    )
    parser.add_argument("--obstacle-radius", type=float, default=None)
    parser.add_argument("--obstacle-margin", type=float, default=None)
    args = parser.parse_args()
    assert_repository_import_root(ROOT)
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "results"):
        raise ValueError("pilot output must be inside repository results")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; use a new directory")
    residual_arms = [arm for arm in args.arms if arm in RESIDUAL_ARMS]
    if bool(residual_arms) != (args.policy is not None):
        raise ValueError("residual arms need --policy, and --policy needs a residual arm")
    policy = None
    if args.policy is not None:
        policy = SACAgent.from_checkpoint(args.policy, seed=0, load_optimizers=False)
    output.mkdir(parents=True, exist_ok=True)

    overrides = {
        key: value
        for key, value in (
            ("process_noise", args.process_noise),
            ("forgetting", args.forgetting),
            ("max_lag", args.max_lag),
            ("payload_prior", args.payload_prior),
            ("gate_threshold", args.gate_threshold),
            ("filter_gate", True if args.filter_gate else None),
            (
                "actuator_time_constants",
                tuple(args.actuator_grid) if args.actuator_grid is not None else None,
            ),
        )
        if value is not None
    }
    config = AdaptiveNominalConfig(**overrides)
    obstacle = None
    if args.obstacle:
        obstacle = ObstacleSpec(
            **{
                key: value
                for key, value in (
                    ("radius", args.obstacle_radius),
                    ("margin", args.obstacle_margin),
                )
                if value is not None
            }
        )
    options = PlantOptions(
        obstacle=obstacle,
        sensor_noise_std=args.sensor_noise,
        armature=args.armature,
        actuator_time_constant=args.actuator_tau,
        armature_range=tuple(args.armature_range) if args.armature_range else None,
        actuator_time_constant_range=(
            tuple(args.actuator_tau_range) if args.actuator_tau_range else None
        ),
    )
    cases = pilot_cases(args.fresh, historical=args.historical)
    started = time.time()
    episodes = []
    for index, (scenario, seed, origin) in enumerate(cases, start=1):
        for arm in args.arms:
            episodes.append(
                run_case(
                    arm,
                    scenario,
                    seed,
                    origin,
                    config,
                    args.delay_compensation,
                    args.plant,
                    options,
                    policy if arm in RESIDUAL_ARMS else None,
                )
            )
        print(f"[{index}/{len(cases)}] {scenario} {seed} done", flush=True)
    records = episodes_to_records(episodes)
    table = summarize(episodes)
    (output / "episodes.json").write_text(json.dumps(records, indent=1) + "\n")
    (output / "summary.json").write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    (output / "summary.md").write_text(markdown_summary(table, config, records))
    manifest = {
        "experiment": "adaptive_nominal_pilot",
        "exploratory": True,
        "training": False,
        "arms": list(args.arms),
        "cases": [list(case) for case in cases],
        "estimator": asdict(config),
        "delay_compensation": bool(args.delay_compensation),
        "plant": args.plant,
        "plant_options": asdict(options),
        "policy": None if args.policy is None else str(args.policy),
        "policy_sha256": None if args.policy is None else sha(args.policy),
        "source_hashes": {name: sha(ROOT / name) for name in SOURCES},
        "runtime": runtime_metadata(ROOT),
        "elapsed_seconds": time.time() - started,
        "episodes": len(episodes),
        "physical_steps": sum(e.steps for e in episodes),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
