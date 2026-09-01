#!/usr/bin/env python3
"""Run the frozen SARRL v1.3 planar OOD and motor-fault campaign."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sarrl.adaptation import AdaptiveContextEnv, DynamicsContextEncoder
from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm
from sarrl.envs import PlanarReachEnv
from sarrl.evaluation import (
    V13_CONDITIONS,
    V13_EPISODES,
    V13_EVALUATION_SEED,
    V13_TRAINING_SEEDS,
    aggregate,
    assert_repository_import_root,
    assert_source_tree_clean,
    evaluate_gated_policy_episodes,
    evaluate_policy_episodes,
    evaluate_stack_episodes,
    paired_success_difference,
    planar_safety_config,
    v13_protocol_dict,
    v13_scenarios,
    write_episode_csv,
    write_run_manifest,
)
from sarrl.models import ResidualDynamicsEnsemble, UncertaintyGate
from sarrl.rl import SACAgent
from sarrl.runtime import ControlStackConfig, SARRLControlStack
from sarrl.safety import HOCBFSafetyFilter
from tools.run_planar_ablations import (
    _retained_a2_checkpoint_hashes,
    _retained_a3_checkpoint_hashes,
    _sha256,
    _validate_a6_input,
)


class _ZeroResidualPolicy:
    def act(self, obs, deterministic=True):
        del obs, deterministic
        return np.zeros(2, dtype=np.float32)


def _require_count(name: str, paths: list[Path], count: int) -> None:
    if len(paths) != count:
        raise ValueError(f"{name} requires exactly {count} checkpoints")


def validate_inputs(
    root: Path,
    training_seeds: list[int],
    a2_policies: list[Path],
    a3_policies: list[Path],
    contexts: list[Path],
    ensembles: list[Path],
) -> list[dict]:
    """Validate all v1.2 artifacts reused by the v1.3 campaign."""
    count = len(training_seeds)
    _require_count("A2", a2_policies, count)
    _require_count("A3", a3_policies, count)
    _require_count("context", contexts, count)
    _require_count("ensemble", ensembles, count)
    a2_hashes = _retained_a2_checkpoint_hashes(root)
    a3_hashes = _retained_a3_checkpoint_hashes(root)
    records = []

    for seed, a2_path, a3_path, context_path, ensemble_path in zip(
        training_seeds,
        a2_policies,
        a3_policies,
        contexts,
        ensembles,
        strict=True,
    ):
        a2_path = Path(a2_path).resolve()
        if not a2_path.is_file():
            raise FileNotFoundError(f"missing A2 policy checkpoint: {a2_path}")
        a2_sha = _sha256(a2_path)
        if a2_sha != a2_hashes.get(seed):
            raise ValueError(f"A2 policy hash mismatch for training seed {seed}")
        adaptive = _validate_a6_input(
            root,
            seed,
            Path(a3_path).resolve(),
            Path(context_path).resolve(),
            Path(ensemble_path).resolve(),
            a3_hashes,
        )
        records.append(
            {
                "training_seed": seed,
                "a2_policy_checkpoint": str(a2_path),
                "a2_policy_checkpoint_sha256": a2_sha,
                **adaptive,
            }
        )
    return records


def _env(scenario, mode: str) -> PlanarReachEnv:
    return PlanarReachEnv(
        mode=mode,
        randomization=scenario.randomization,
        fault=scenario.fault,
    )


def _freeze_module(module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _write_dataclass_csv(path: Path, rows) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def run_campaign(
    root: Path,
    out: Path,
    inputs: list[dict],
    conditions: list[str],
    scenario_keys: list[str],
    evaluation_seed: int,
    episodes: int,
) -> None:
    scenario_map = {scenario.key: scenario for scenario in v13_scenarios()}
    scenarios = [scenario_map[key] for key in scenario_keys]
    out.mkdir(parents=True, exist_ok=True)
    manifest = v13_protocol_dict()
    manifest["selected_conditions"] = conditions
    manifest["selected_scenarios"] = scenario_keys
    manifest["inputs"] = inputs
    write_run_manifest(out / "evaluation_manifest.json", manifest, root=root)

    groups: dict[tuple[str, int, str], list] = {}
    all_rows = []
    gate_diagnostics = []
    stack_diagnostics = []

    def retain(condition: str, training_seed: int, scenario_key: str, rows) -> None:
        groups[(condition, training_seed, scenario_key)] = rows
        all_rows.extend(rows)
        metrics = aggregate(rows)
        print(
            f"{condition} seed={training_seed} scenario={scenario_key}: "
            f"{metrics.successes}/{metrics.n} ({100.0 * metrics.success_rate:.1f}%)"
        )

    if "A0" in conditions:
        for scenario in scenarios:
            rows = evaluate_policy_episodes(
                _ZeroResidualPolicy(),
                _env(scenario, "residual"),
                episodes,
                evaluation_seed,
                scenario=scenario.key,
                controller="A0_computed_torque",
            )
            retain("A0", -1, scenario.key, rows)

    for record in inputs:
        training_seed = record["training_seed"]

        if "A2" in conditions:
            agent = SACAgent.from_checkpoint(
                record["a2_policy_checkpoint"], seed=0, load_optimizers=False
            )
            for scenario in scenarios:
                rows = evaluate_policy_episodes(
                    agent,
                    _env(scenario, "residual"),
                    episodes,
                    evaluation_seed,
                    scenario=scenario.key,
                    controller=f"A2_train_seed_{training_seed}",
                )
                retain("A2", training_seed, scenario.key, rows)

        if "A3" in conditions:
            agent = SACAgent.from_checkpoint(
                record["policy_checkpoint"], seed=0, load_optimizers=False
            )
            encoder = DynamicsContextEncoder.load(
                record["context_checkpoint"], map_location="cpu"
            )
            _freeze_module(encoder)
            for scenario in scenarios:
                env = AdaptiveContextEnv(
                    _env(scenario, "residual"),
                    encoder,
                    device="cpu",
                )
                rows = evaluate_policy_episodes(
                    agent,
                    env,
                    episodes,
                    evaluation_seed,
                    scenario=scenario.key,
                    controller=f"A3_train_seed_{training_seed}",
                )
                retain("A3", training_seed, scenario.key, rows)

        if "A4" in conditions:
            agent = SACAgent.from_checkpoint(
                record["a2_policy_checkpoint"], seed=0, load_optimizers=False
            )
            ensemble = ResidualDynamicsEnsemble.load(
                record["ensemble_checkpoint"], map_location="cpu"
            )
            _freeze_module(ensemble)
            nominal = PlanarArm()
            stack = SARRLControlStack(
                ComputedTorqueController(nominal),
                agent,
                dynamics_ensemble=ensemble,
                uncertainty_gate=UncertaintyGate(gain=4.0, min_scale=0.1),
                device="cpu",
            )
            for scenario in scenarios:
                rows, diagnostics = evaluate_gated_policy_episodes(
                    stack,
                    _env(scenario, "torque"),
                    episodes,
                    evaluation_seed,
                    scenario=scenario.key,
                    controller=f"A4_train_seed_{training_seed}",
                )
                gate_diagnostics.extend(diagnostics)
                retain("A4", training_seed, scenario.key, rows)

        if "A5" in conditions:
            agent = SACAgent.from_checkpoint(
                record["a2_policy_checkpoint"], seed=0, load_optimizers=False
            )
            nominal = PlanarArm()
            stack = SARRLControlStack(
                ComputedTorqueController(nominal),
                agent,
                ControlStackConfig(require_safety=True),
                safety_filter=HOCBFSafetyFilter(nominal, planar_safety_config()),
            )
            for scenario in scenarios:
                rows, diagnostics = evaluate_stack_episodes(
                    stack,
                    _env(scenario, "torque"),
                    episodes,
                    evaluation_seed,
                    scenario=scenario.key,
                    controller=f"A5_train_seed_{training_seed}",
                )
                stack_diagnostics.extend(diagnostics)
                retain("A5", training_seed, scenario.key, rows)

        if "A6" in conditions:
            agent = SACAgent.from_checkpoint(
                record["policy_checkpoint"], seed=0, load_optimizers=False
            )
            encoder = DynamicsContextEncoder.load(
                record["context_checkpoint"], map_location="cpu"
            )
            ensemble = ResidualDynamicsEnsemble.load(
                record["ensemble_checkpoint"], map_location="cpu"
            )
            _freeze_module(encoder)
            _freeze_module(ensemble)
            nominal = PlanarArm()
            stack = SARRLControlStack(
                ComputedTorqueController(nominal),
                agent,
                ControlStackConfig(require_safety=True),
                safety_filter=HOCBFSafetyFilter(nominal, planar_safety_config()),
                dynamics_ensemble=ensemble,
                uncertainty_gate=UncertaintyGate(gain=4.0, min_scale=0.1),
                device="cpu",
            )
            for scenario in scenarios:
                env = AdaptiveContextEnv(
                    _env(scenario, "torque"),
                    encoder,
                    device="cpu",
                )
                rows, diagnostics = evaluate_stack_episodes(
                    stack,
                    env,
                    episodes,
                    evaluation_seed,
                    scenario=scenario.key,
                    controller=f"A6_train_seed_{training_seed}",
                    context_residual_limit=8.0,
                )
                stack_diagnostics.extend(diagnostics)
                retain("A6", training_seed, scenario.key, rows)

    write_episode_csv(out / "heldout_episodes.csv", all_rows)
    _write_dataclass_csv(out / "gate_diagnostics.csv", gate_diagnostics)
    _write_dataclass_csv(out / "stack_diagnostics.csv", stack_diagnostics)

    summaries = []
    for (condition, training_seed, scenario_key), rows in groups.items():
        metrics = aggregate(rows)
        summaries.append(
            {
                "condition": condition,
                "training_seed": training_seed,
                "scenario": scenario_key,
                "successes": metrics.successes,
                "episodes": metrics.n,
                "success_rate": metrics.success_rate,
                "success_ci95_low": metrics.success_ci95_low,
                "success_ci95_high": metrics.success_ci95_high,
                "reward_mean": metrics.reward_mean,
                "reward_std": metrics.reward_std,
                "final_distance_mean": metrics.final_distance_mean,
                "fault_exposure_rate": sum(int(row.fault_seen) for row in rows) / metrics.n,
            }
        )
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    deltas = []
    if "id_reference" in scenario_keys:
        for (condition, training_seed, scenario_key), rows in groups.items():
            if scenario_key == "id_reference":
                continue
            reference = groups[(condition, training_seed, "id_reference")]
            difference, low, high = paired_success_difference(
                rows,
                reference,
                bootstrap=10_000,
                seed=max(training_seed, 0) + 100 * scenario_keys.index(scenario_key),
            )
            deltas.append(
                {
                    "condition": condition,
                    "training_seed": training_seed,
                    "scenario": scenario_key,
                    "reference": "id_reference",
                    "success_difference": difference,
                    "paired_bootstrap_ci95_low": low,
                    "paired_bootstrap_ci95_high": high,
                    "bootstrap_samples": 10_000,
                }
            )
    if deltas:
        with (out / "robustness_deltas.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(deltas[0]))
            writer.writeheader()
            writer.writerows(deltas)

    aggregate_payload = {"conditions": {}, "evaluation_seed": evaluation_seed}
    for condition in conditions:
        aggregate_payload["conditions"][condition] = {}
        for scenario in scenarios:
            rates = [
                row["success_rate"]
                for row in summaries
                if row["condition"] == condition and row["scenario"] == scenario.key
            ]
            if not rates:
                continue
            aggregate_payload["conditions"][condition][scenario.key] = {
                "models": len(rates),
                "success_rate_mean": float(np.mean(rates)),
                "success_rate_std": (
                    float(np.std(rates, ddof=1)) if len(rates) > 1 else None
                ),
                "success_rate_min": float(np.min(rates)),
                "success_rate_max": float(np.max(rates)),
            }
    (out / "aggregate.json").write_text(
        json.dumps(aggregate_payload, indent=2, sort_keys=True) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=V13_CONDITIONS, default=V13_CONDITIONS)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=[scenario.key for scenario in v13_scenarios()],
        default=[scenario.key for scenario in v13_scenarios()],
    )
    parser.add_argument("--evaluation-seed", type=int, default=V13_EVALUATION_SEED)
    parser.add_argument("--episodes", type=int, default=V13_EPISODES)
    parser.add_argument("--output", type=Path, default=Path("results/ood_fault_robustness"))
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
            list(V13_TRAINING_SEEDS),
            args.a2_policy_checkpoints,
            args.a3_policy_checkpoints,
            args.context_checkpoints,
            args.ensemble_checkpoints,
        )
        run_campaign(
            root,
            args.output,
            inputs,
            list(args.conditions),
            list(args.scenarios),
            args.evaluation_seed,
            args.episodes,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
