"""Preregistered v1.9 campaign: adaptive against fixed nominal control behind the HOCBF.

Everything that defines the decision lives in this module and is frozen at the
source commit that runs the campaign: arms, seeds, endpoints, bootstrap,
thresholds and the order in which the rule is applied. The runner only
orchestrates episodes and writes the files; the analysis reloads them.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from sarrl.controllers import AdaptiveNominalConfig
from sarrl.evaluation.adaptive_pilot import PilotEpisode

V19_ARMS = ("fixed", "fixed_hocbf", "adaptive", "adaptive_hocbf")
V19_PRIMARY_ARMS = ("fixed_hocbf", "adaptive_hocbf")
V19_SCENARIOS = ("id_reference", "ood_compound", "motor_fault")
V19_PRIMARY_SCENARIOS = ("id_reference", "motor_fault")
# Decision seeds: never used as episode seeds by any retained artifact.
V19_PRIMARY_SEED_START = 50100
V19_PRIMARY_EPISODES = 1900
# Reproduction seeds: the v1.3/v1.4 evaluation seeds, run by every arm and
# reported apart; the unfiltered fixed arm must reproduce the retained A0 rows.
V19_REPRODUCTION_SEED_START = 50000
V19_REPRODUCTION_EPISODES = 100
V19_REPRODUCTION_REFERENCE = "results/ood_fault_robustness/heldout_episodes.csv"
V19_REPRODUCTION_CONTROLLER = "A0_computed_torque"
V19_BOOTSTRAP_DRAWS = 10_000
V19_BOOTSTRAP_SEED = 190_000
V19_SUCCESS_GAIN = 0.20
V19_UNSAFE_MARGIN = 0.03
V19_COMPENSATE_DELAY = True
# Paths compared against the sealed commit. The seal itself lives outside them.
V19_FROZEN_PATHS = ("sarrl", "tools", "tests", "docs/experiments.md", "pyproject.toml")
V19_SEAL_FILE = "docs/v19_seal.json"


def v19_config() -> AdaptiveNominalConfig:
    """The frozen estimator configuration: library defaults, nothing tuned after the pilot."""
    return AdaptiveNominalConfig()


def v19_protocol_dict() -> dict:
    """JSON-friendly record of the frozen protocol, written into the manifest."""
    config = v19_config()
    return {
        "release_target": "v1.9.0",
        "campaign": "adaptive_nominal_v19",
        "training": False,
        "arms": list(V19_ARMS),
        "primary_contrast": {
            "treatment": "adaptive_hocbf",
            "reference": "fixed_hocbf",
            "scenarios": list(V19_PRIMARY_SCENARIOS),
            "metric": "success",
        },
        "seeds": {
            "decision": {
                "arms": list(V19_PRIMARY_ARMS),
                "start": V19_PRIMARY_SEED_START,
                "episodes_per_scenario": V19_PRIMARY_EPISODES,
            },
            "reproduction": {
                "arms": list(V19_ARMS),
                "start": V19_REPRODUCTION_SEED_START,
                "episodes_per_scenario": V19_REPRODUCTION_EPISODES,
                "reference": V19_REPRODUCTION_REFERENCE,
                "reference_controller": V19_REPRODUCTION_CONTROLLER,
                "decision_weight": False,
            },
            "scenarios": list(V19_SCENARIOS),
            "unopened_range_scanned": [
                V19_PRIMARY_SEED_START,
                V19_PRIMARY_SEED_START + V19_PRIMARY_EPISODES - 1,
            ],
        },
        "bootstrap": {
            "type": "paired_over_episode_seeds",
            "draws": V19_BOOTSTRAP_DRAWS,
            "seed": V19_BOOTSTRAP_SEED,
        },
        "decision": {
            "order": [
                "safety_veto",
                "primary_success",
                "non_inferiority",
                "otherwise_inconclusive",
            ],
            "safety_veto": {
                "metric": "unsafe_episode",
                "scenarios": list(V19_SCENARIOS),
                "veto_if": "lower_95_above_zero_or_difference_above_margin",
                "margin": V19_UNSAFE_MARGIN,
                "outcome": "no_go_safety",
            },
            "primary_success": {
                "minimum_gain": V19_SUCCESS_GAIN,
                "lower_95_above": 0.0,
                "scenarios": list(V19_PRIMARY_SCENARIOS),
            },
            "non_inferiority": {
                "metric": "unsafe_episode",
                "scenarios": list(V19_SCENARIOS),
                "upper_95_at_or_below": V19_UNSAFE_MARGIN,
                "decision_weight": True,
            },
            "go_requires": ["no_safety_veto", "primary_success", "non_inferiority_all_scenarios"],
        },
        "seal_file": V19_SEAL_FILE,
        "frozen_paths": list(V19_FROZEN_PATHS),
        "state_interface": "full_state_feedback_noise_free_same_for_every_arm",
        "estimator": {
            "process_noise": config.process_noise,
            "forgetting": config.forgetting,
            "max_lag": config.max_lag,
            "lag_min_updates": config.lag_min_updates,
            "selection_memory": config.selection_memory,
            "payload_prior": config.payload_prior,
            "filter_gate": config.filter_gate,
            "prior_std": list(config.prior_std),
            "lower": list(config.lower),
            "upper": list(config.upper),
            "compensate_delay": V19_COMPENSATE_DELAY,
        },
    }


def v19_seeds(block: str) -> range:
    if block == "reproduction":
        start = V19_REPRODUCTION_SEED_START
        return range(start, start + V19_REPRODUCTION_EPISODES)
    if block == "decision":
        return range(V19_PRIMARY_SEED_START, V19_PRIMARY_SEED_START + V19_PRIMARY_EPISODES)
    raise ValueError(f"unknown seed block {block}")


def v19_cells():
    """Every (arm, scenario, seed) in the fixed order the runner executes."""
    for scenario in V19_SCENARIOS:
        for seed in v19_seeds("reproduction"):
            for arm in V19_ARMS:
                yield arm, scenario, seed
        for seed in v19_seeds("decision"):
            for arm in V19_PRIMARY_ARMS:
                yield arm, scenario, seed


def paired_difference(rows_treatment, rows_reference, metric, rng):
    """Mean difference and percentile interval over seed-paired bootstrap draws."""
    treatment = {r.seed: float(metric(r)) for r in rows_treatment}
    reference = {r.seed: float(metric(r)) for r in rows_reference}
    if treatment.keys() != reference.keys() or not treatment:
        raise ValueError("paired contrast needs the same non-empty seed set")
    seeds = sorted(treatment)
    differences = np.asarray([treatment[s] - reference[s] for s in seeds])
    draws = rng.integers(0, len(seeds), size=(V19_BOOTSTRAP_DRAWS, len(seeds)))
    distribution = differences[draws].mean(axis=1)
    return {
        "difference": float(differences.mean()),
        "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)),
        "treatment_rate": float(np.mean(list(treatment.values()))),
        "reference_rate": float(np.mean(list(reference.values()))),
        "pairs": len(seeds),
    }


def load_episodes(path: Path) -> list[PilotEpisode]:
    """Reload the serialised episode log into validated records for the analysis."""
    episodes = []
    names = set(PilotEpisode.__dataclass_fields__)
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if set(record) != names:
            raise ValueError(f"episode record {line_number} has unexpected fields")
        if record["prediction_error_rms"] is not None:
            record["prediction_error_rms"] = tuple(record["prediction_error_rms"])
        episodes.append(PilotEpisode(**record))
    return episodes


def cell_summary(rows: list[PilotEpisode]) -> dict:
    lags = [r for r in rows if r.selected_lag is not None]
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([r.success for r in rows])),
        "unsafe_episode_rate": float(np.mean([r.unsafe_episode for r in rows])),
        "abort_rate": float(np.mean([r.outcome == "abort" for r in rows])),
        "timeout_rate": float(np.mean([r.outcome == "timeout" for r in rows])),
        "median_final_distance_m": float(np.median([r.final_distance for r in rows])),
        "normalized_violation_max": float(max(r.normalized_violation_max for r in rows)),
        "intervention_fraction_mean": float(
            np.mean([r.safety_intervention_fraction for r in rows])
        ),
        "lag_identified_fraction": (
            float(np.mean([r.selected_lag == r.true_delay for r in lags])) if lags else None
        ),
    }


def reproduction_check(rows: list[PilotEpisode], reference_path: Path) -> dict:
    """Compare the unfiltered fixed arm on the v1.3 seeds with the retained A0 rows."""
    reference = {}
    with reference_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["controller"] == V19_REPRODUCTION_CONTROLLER:
                reference[(row["scenario"], int(row["seed"]))] = (
                    row["success"] == "True",
                    float(row["final_distance"]),
                )
    compared = matched = 0
    mismatches = []
    for episode in rows:
        key = (episode.scenario, episode.seed)
        if key not in reference:
            continue
        compared += 1
        success, distance = reference[key]
        if episode.success == success and abs(episode.final_distance - distance) <= 1e-9:
            matched += 1
        else:
            mismatches.append([episode.scenario, episode.seed])
    return {
        "reference": reference_path.name,
        "reference_controller": V19_REPRODUCTION_CONTROLLER,
        "compared": compared,
        "matched": matched,
        "mismatches": mismatches[:20],
    }


def _block(episodes, arms, block):
    seeds = set(v19_seeds(block))
    return {
        (arm, scenario): [
            e for e in episodes if e.arm == arm and e.scenario == scenario and e.seed in seeds
        ]
        for arm in arms
        for scenario in V19_SCENARIOS
    }


def analyze(episodes: list[PilotEpisode], reference_path: Path | None = None) -> dict:
    """Apply the frozen rule. Vetoes are checked before the primary endpoint."""
    if [(e.arm, e.scenario, e.seed) for e in episodes] != list(v19_cells()):
        raise ValueError("episodes do not match the planned cells exactly, in order")
    decision_block = _block(episodes, V19_PRIMARY_ARMS, "decision")
    reproduction_block = _block(episodes, V19_ARMS, "reproduction")

    rng = np.random.default_rng(V19_BOOTSTRAP_SEED)
    contrasts = {}
    for scenario in V19_SCENARIOS:
        treatment = decision_block[("adaptive_hocbf", scenario)]
        reference = decision_block[("fixed_hocbf", scenario)]
        contrasts[scenario] = {
            "success": paired_difference(treatment, reference, lambda r: r.success, rng),
            "unsafe_episode": paired_difference(
                treatment, reference, lambda r: r.unsafe_episode, rng
            ),
            "abort": paired_difference(
                treatment, reference, lambda r: r.outcome == "abort", rng
            ),
        }

    vetoes = [
        scenario
        for scenario in V19_SCENARIOS
        if contrasts[scenario]["unsafe_episode"]["ci95_low"] > 0.0
        or contrasts[scenario]["unsafe_episode"]["difference"] > V19_UNSAFE_MARGIN
    ]
    non_inferior = {
        scenario: bool(contrasts[scenario]["unsafe_episode"]["ci95_high"] <= V19_UNSAFE_MARGIN)
        for scenario in V19_SCENARIOS
    }
    primary_met = all(
        contrasts[s]["success"]["difference"] >= V19_SUCCESS_GAIN
        and contrasts[s]["success"]["ci95_low"] > 0.0
        for s in V19_PRIMARY_SCENARIOS
    )
    reasons = []
    if vetoes:
        decision = "no_go_safety"
    else:
        if not primary_met:
            reasons.append("success gain not established in every primary scenario")
        not_shown = [s for s in V19_SCENARIOS if not non_inferior[s]]
        if not_shown:
            reasons.append("unsafe-episode non-inferiority not shown in " + ", ".join(not_shown))
        decision = "go" if not reasons else "inconclusive"

    report = {
        "decision": decision,
        "inconclusive_reasons": reasons,
        "safety_vetoes": vetoes,
        "unsafe_non_inferior_at_margin": non_inferior,
        "primary_met": primary_met,
        "contrasts": contrasts,
        "decision_summary": {
            f"{arm}/{scenario}": cell_summary(rows)
            for (arm, scenario), rows in decision_block.items()
        },
        "reproduction_summary": {
            f"{arm}/{scenario}": cell_summary(rows)
            for (arm, scenario), rows in reproduction_block.items()
        },
        "protocol": v19_protocol_dict(),
    }
    if reference_path is not None and reference_path.exists():
        fixed_rows = [
            e for (arm, _), rows in reproduction_block.items() if arm == "fixed" for e in rows
        ]
        report["reproduction_check"] = reproduction_check(fixed_rows, reference_path)
    return report
