#!/usr/bin/env python3
"""Run the frozen SARRL v1.4 paired quantified-safety campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sarrl.adaptation import AdaptiveContextEnv, DynamicsContextEncoder
from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm
from sarrl.evaluation import (
    V14_EPISODES,
    V14_EVALUATION_SEED,
    V14_PAIRINGS,
    V14_TRAINING_SEEDS,
    aggregate,
    assert_repository_import_root,
    assert_source_tree_clean,
    evaluate_safety_episodes,
    paired_diagnostic_difference,
    planar_safety_config,
    v13_scenarios,
    v14_protocol_dict,
    wilson_interval,
    write_episode_csv,
    write_run_manifest,
)
from sarrl.models import ResidualDynamicsEnsemble, UncertaintyGate
from sarrl.rl import SACAgent
from sarrl.runtime import ControlStackConfig, SARRLControlStack
from sarrl.safety import HOCBFSafetyFilter

if __package__:
    from tools.run_planar_v13 import _env, _freeze_module, validate_inputs
else:
    from run_planar_v13 import _env, _freeze_module, validate_inputs


def _write_dataclass_csv(path: Path, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _write_dict_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite_min(values) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return min(finite) if finite else None


def _summarize(condition: str, training_seed: int, scenario: str, outcomes, rows) -> dict:
    metrics = aggregate(outcomes)
    state_observations = sum(row.state_observations for row in rows)
    unsafe_observations = sum(row.unsafe_state_observations for row in rows)
    command_attempts = sum(row.command_attempts for row in rows)
    candidate_violations = sum(row.candidate_constraint_violation_steps for row in rows)
    interventions = sum(row.safety_intervention_steps for row in rows)
    unsafe_episodes = sum(int(row.unsafe_episode) for row in rows)
    infeasible_episodes = sum(int(row.safety_infeasible) for row in rows)
    unsafe_low, unsafe_high = wilson_interval(unsafe_episodes, len(rows))
    infeasible_low, infeasible_high = wilson_interval(infeasible_episodes, len(rows))
    return {
        "condition": condition,
        "training_seed": training_seed,
        "scenario": scenario,
        "episodes": len(rows),
        "successes": metrics.successes,
        "success_rate": metrics.success_rate,
        "success_ci95_low": metrics.success_ci95_low,
        "success_ci95_high": metrics.success_ci95_high,
        "unsafe_episodes": unsafe_episodes,
        "unsafe_episode_rate": unsafe_episodes / len(rows),
        "unsafe_episode_ci95_low": unsafe_low,
        "unsafe_episode_ci95_high": unsafe_high,
        "state_observations": state_observations,
        "unsafe_state_observations": unsafe_observations,
        "unsafe_state_fraction": unsafe_observations / state_observations,
        "unsafe_entry_count": sum(row.unsafe_entry_count for row in rows),
        "joint_position_violation_max_rad": max(
            row.joint_position_violation_max_rad for row in rows
        ),
        "joint_velocity_violation_max_rad_s": max(
            row.joint_velocity_violation_max_rad_s for row in rows
        ),
        "normalized_violation_integral_mean": float(
            np.mean([row.normalized_violation_integral for row in rows])
        ),
        "normalized_violation_integral_max": max(
            row.normalized_violation_integral for row in rows
        ),
        "command_attempts": command_attempts,
        "candidate_constraint_violation_steps": candidate_violations,
        "candidate_constraint_violation_fraction": candidate_violations / command_attempts,
        "candidate_constraint_margin_min": min(
            row.candidate_constraint_margin_min for row in rows
        ),
        "executed_constraint_margin_min": _finite_min(
            row.executed_constraint_margin_min for row in rows
        ),
        "safety_enabled": rows[0].safety_enabled,
        "safety_infeasible_episodes": infeasible_episodes,
        "safety_infeasible_rate": infeasible_episodes / len(rows),
        "safety_infeasible_ci95_low": infeasible_low,
        "safety_infeasible_ci95_high": infeasible_high,
        "safety_intervention_steps": interventions,
        "safety_intervention_fraction": interventions / command_attempts,
        "safety_correction_mean": sum(
            row.safety_correction_mean * row.command_attempts for row in rows
        )
        / command_attempts,
        "safety_correction_max": max(row.safety_correction_max for row in rows),
        "fault_exposure_rate": sum(int(row.fault_seen) for row in rows) / len(rows),
    }


def _paired_rows(groups: dict, scenario_keys: list[str]) -> list[dict]:
    metric_extractors = {
        "success_rate": lambda row: float(row.success),
        "unsafe_episode_rate": lambda row: float(row.unsafe_episode),
        "unsafe_state_fraction": lambda row: row.unsafe_state_fraction,
        "normalized_violation_integral": lambda row: row.normalized_violation_integral,
        "joint_position_violation_max_rad": (
            lambda row: row.joint_position_violation_max_rad
        ),
        "joint_velocity_violation_max_rad_s": (
            lambda row: row.joint_velocity_violation_max_rad_s
        ),
    }
    comparisons = []
    for filtered, reference, label in V14_PAIRINGS:
        for training_seed in V14_TRAINING_SEEDS:
            for scenario_index, scenario in enumerate(scenario_keys):
                filtered_rows = groups[(filtered, training_seed, scenario)]
                reference_rows = groups[(reference, training_seed, scenario)]
                for metric_index, (metric_name, extractor) in enumerate(
                    metric_extractors.items()
                ):
                    difference, low, high = paired_diagnostic_difference(
                        filtered_rows,
                        reference_rows,
                        extractor,
                        bootstrap=10_000,
                        seed=10_000 * metric_index + 100 * scenario_index + training_seed,
                    )
                    comparisons.append(
                        {
                            "pairing": label,
                            "filtered": filtered,
                            "reference": reference,
                            "training_seed": training_seed,
                            "scenario": scenario,
                            "metric": metric_name,
                            "difference": difference,
                            "paired_bootstrap_ci95_low": low,
                            "paired_bootstrap_ci95_high": high,
                            "bootstrap_samples": 10_000,
                        }
                    )
    return comparisons


def _aggregate_payload(summaries: list[dict], comparisons: list[dict]) -> dict:
    payload: dict = {"conditions": {}, "paired_effects": {}}
    for condition in sorted({row["condition"] for row in summaries}):
        payload["conditions"][condition] = {}
        for scenario in sorted(
            {row["scenario"] for row in summaries if row["condition"] == condition}
        ):
            selected = [
                row
                for row in summaries
                if row["condition"] == condition and row["scenario"] == scenario
            ]
            values = {}
            for metric in (
                "success_rate",
                "unsafe_episode_rate",
                "unsafe_state_fraction",
                "normalized_violation_integral_mean",
                "safety_infeasible_rate",
                "safety_intervention_fraction",
            ):
                metric_values = np.asarray([row[metric] for row in selected], dtype=np.float64)
                values[metric + "_mean"] = float(np.mean(metric_values))
                values[metric + "_std"] = float(np.std(metric_values, ddof=1))
            values["models"] = len(selected)
            payload["conditions"][condition][scenario] = values

    for pairing in sorted({row["pairing"] for row in comparisons}):
        payload["paired_effects"][pairing] = {}
        for scenario in sorted(
            {row["scenario"] for row in comparisons if row["pairing"] == pairing}
        ):
            payload["paired_effects"][pairing][scenario] = {}
            for metric in sorted(
                {
                    row["metric"]
                    for row in comparisons
                    if row["pairing"] == pairing and row["scenario"] == scenario
                }
            ):
                differences = np.asarray(
                    [
                        row["difference"]
                        for row in comparisons
                        if row["pairing"] == pairing
                        and row["scenario"] == scenario
                        and row["metric"] == metric
                    ],
                    dtype=np.float64,
                )
                payload["paired_effects"][pairing][scenario][metric] = {
                    "models": len(differences),
                    "difference_mean": float(np.mean(differences)),
                    "difference_std": float(np.std(differences, ddof=1)),
                }
    return payload


def _a2_stacks(record: dict):
    policy = SACAgent.from_checkpoint(
        record["a2_policy_checkpoint"], seed=0, load_optimizers=False
    )
    nominal = PlanarArm()
    observer = HOCBFSafetyFilter(nominal, planar_safety_config())
    unfiltered = SARRLControlStack(ComputedTorqueController(nominal), policy)
    filtered = SARRLControlStack(
        ComputedTorqueController(nominal),
        policy,
        ControlStackConfig(require_safety=True),
        safety_filter=observer,
    )
    return unfiltered, filtered, observer


def _a6_stacks(record: dict):
    policy = SACAgent.from_checkpoint(
        record["policy_checkpoint"], seed=0, load_optimizers=False
    )
    encoder = DynamicsContextEncoder.load(record["context_checkpoint"], map_location="cpu")
    ensemble = ResidualDynamicsEnsemble.load(
        record["ensemble_checkpoint"], map_location="cpu"
    )
    _freeze_module(encoder)
    _freeze_module(ensemble)
    nominal = PlanarArm()
    observer = HOCBFSafetyFilter(nominal, planar_safety_config())
    common = {
        "dynamics_ensemble": ensemble,
        "uncertainty_gate": UncertaintyGate(gain=4.0, min_scale=0.1),
        "device": "cpu",
    }
    prefilter = SARRLControlStack(
        ComputedTorqueController(nominal), policy, **common
    )
    filtered = SARRLControlStack(
        ComputedTorqueController(nominal),
        policy,
        ControlStackConfig(require_safety=True),
        safety_filter=observer,
        **common,
    )
    return prefilter, filtered, observer, encoder


def run_campaign(
    root: Path,
    out: Path,
    inputs: list[dict],
    scenario_keys: list[str],
    evaluation_seed: int,
    episodes: int,
) -> None:
    """Run all paired v1.4 filter comparisons and retain raw diagnostics."""
    scenario_map = {scenario.key: scenario for scenario in v13_scenarios()}
    scenarios = [scenario_map[key] for key in scenario_keys]
    out.mkdir(parents=True, exist_ok=True)
    manifest = v14_protocol_dict()
    manifest["selected_scenarios"] = scenario_keys
    manifest["inputs"] = inputs
    write_run_manifest(out / "evaluation_manifest.json", manifest, root=root)

    outcome_groups = {}
    diagnostic_groups = {}
    all_outcomes = []
    all_diagnostics = []

    def retain(condition: str, training_seed: int, scenario: str, outcomes, rows) -> None:
        outcome_groups[(condition, training_seed, scenario)] = outcomes
        diagnostic_groups[(condition, training_seed, scenario)] = rows
        all_outcomes.extend(outcomes)
        all_diagnostics.extend(rows)
        unsafe = sum(int(row.unsafe_episode) for row in rows)
        successes = sum(int(row.success) for row in rows)
        print(
            f"{condition} seed={training_seed} scenario={scenario}: "
            f"success={successes}/{len(rows)} unsafe={unsafe}/{len(rows)}"
        )

    for record in inputs:
        training_seed = record["training_seed"]
        a2_unfiltered, a5_hocbf, a2_observer = _a2_stacks(record)
        a6_prefilter, a6_hocbf, a6_observer, encoder = _a6_stacks(record)
        for scenario in scenarios:
            for condition, stack, observer, uses_context in (
                ("A2_unfiltered", a2_unfiltered, a2_observer, False),
                ("A5_hocbf", a5_hocbf, a2_observer, False),
                ("A6_prefilter", a6_prefilter, a6_observer, True),
                ("A6_hocbf", a6_hocbf, a6_observer, True),
            ):
                base_env = _env(scenario, "torque")
                env = (
                    AdaptiveContextEnv(base_env, encoder, device="cpu")
                    if uses_context
                    else base_env
                )
                outcomes, rows = evaluate_safety_episodes(
                    stack,
                    observer,
                    env,
                    episodes,
                    evaluation_seed,
                    scenario=scenario.key,
                    controller=f"{condition}_train_seed_{training_seed}",
                    context_residual_limit=8.0 if uses_context else None,
                )
                retain(condition, training_seed, scenario.key, outcomes, rows)

    write_episode_csv(out / "episodes.csv", all_outcomes)
    _write_dataclass_csv(out / "safety_diagnostics.csv", all_diagnostics)
    summaries = [
        _summarize(condition, training_seed, scenario, outcome_groups[key], rows)
        for key, rows in diagnostic_groups.items()
        for condition, training_seed, scenario in [key]
    ]
    _write_dict_csv(out / "summary.csv", summaries)
    comparisons = _paired_rows(diagnostic_groups, scenario_keys)
    _write_dict_csv(out / "paired_comparisons.csv", comparisons)
    (out / "aggregate.json").write_text(
        json.dumps(
            _aggregate_payload(summaries, comparisons), indent=2, sort_keys=True
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=[scenario.key for scenario in v13_scenarios()],
        default=[scenario.key for scenario in v13_scenarios()],
    )
    parser.add_argument("--evaluation-seed", type=int, default=V14_EVALUATION_SEED)
    parser.add_argument("--episodes", type=int, default=V14_EPISODES)
    parser.add_argument("--output", type=Path, default=Path("results/quantified_safety"))
    parser.add_argument("--a2-policy-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--a3-policy-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--context-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--ensemble-checkpoints", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    if args.episodes <= 0 or args.evaluation_seed < 0:
        raise SystemExit("episodes must be positive and evaluation seed non-negative")
    root = Path(__file__).resolve().parents[1]
    assert_repository_import_root(root)
    assert_source_tree_clean(root)
    try:
        inputs = validate_inputs(
            root,
            list(V14_TRAINING_SEEDS),
            args.a2_policy_checkpoints,
            args.a3_policy_checkpoints,
            args.context_checkpoints,
            args.ensemble_checkpoints,
        )
        run_campaign(
            root,
            args.output,
            inputs,
            list(args.scenarios),
            args.evaluation_seed,
            args.episodes,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
