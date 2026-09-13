"""Preregistered v2.0 campaign: adaptive against fixed nominal control on a MuJoCo plant.

The design mirrors v1.9 (`adaptive_campaign`): two filtered arms on fresh
paired seeds, a seed-paired percentile bootstrap, a safety veto checked
first, a success gain and a non-inferiority condition both required for
`go`. What changes is the plant: MuJoCo with per-episode reflected inertia
and a first-order actuator lag, neither in the controller's parametrisation,
and a measured state with sensor noise handed to both arms. The adaptive arm
runs the time-constant hypothesis bank. A transfer block runs the unfiltered
fixed nominal on the v1.3 seeds on both plants: the analytical rows must
reproduce the retained A0 rows, and the MuJoCo rows quantify how much the
plant change alone moves the baseline.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sarrl.controllers import AdaptiveNominalConfig
from sarrl.evaluation.adaptive_campaign import (
    V19_BOOTSTRAP_DRAWS,
    V19_REPRODUCTION_CONTROLLER,
    V19_REPRODUCTION_REFERENCE,
    cell_summary,
    load_episodes,
    paired_difference,
)
from sarrl.evaluation.adaptive_pilot import PilotEpisode, PlantOptions

V20_ARMS = ("fixed", "fixed_hocbf", "adaptive", "adaptive_hocbf")
V20_PRIMARY_ARMS = ("fixed_hocbf", "adaptive_hocbf")
V20_SCENARIOS = ("id_reference", "ood_compound", "motor_fault")
V20_PRIMARY_SCENARIOS = ("id_reference", "motor_fault")
V20_PLANT = "mujoco"
V20_PLANT_OPTIONS = PlantOptions(
    sensor_noise_std=1e-3,
    armature_range=(0.02, 0.08),
    actuator_time_constant_range=(0.01, 0.05),
)
V20_ACTUATOR_GRID = (0.0, 0.01, 0.03, 0.06)
V20_COMPENSATE_DELAY = True
# Decision seeds: never used as episode seeds by any retained artifact; the
# v1.9 block ends at 52099 and 52100..52199 stay unused as a guard band.
V20_PRIMARY_SEED_START = 52200
V20_PRIMARY_EPISODES = 1900
V20_DESCRIPTIVE_EPISODES = 100
# Transfer block: the v1.3 evaluation seeds, unfiltered fixed nominal on both plants.
V20_TRANSFER_SEED_START = 50000
V20_TRANSFER_EPISODES = 100
V20_BOOTSTRAP_SEED = 200_000
V20_SUCCESS_GAIN = 0.20
V20_UNSAFE_MARGIN = 0.03
# Estimator validity: an adaptive decision episode spending more than this
# fraction of its control steps on the nominal model is guarded, and the
# campaign cannot reach go if more than this fraction of decision episodes are.
V20_GUARDED_STEP_FRACTION = 0.10
V20_GUARDED_EPISODE_FRACTION = 0.10
V20_FROZEN_PATHS = (
    "sarrl",
    "tools",
    "tests",
    "docs/experiments.md",
    "pyproject.toml",
    V19_REPRODUCTION_REFERENCE,
)
V20_SEAL_FILE = "docs/v20_seal.json"
V20_OUTPUT = "results/adaptive_mujoco_v20"


def v20_config() -> AdaptiveNominalConfig:
    """Frozen estimator configuration: v1.9 defaults plus the time-constant grid."""
    return AdaptiveNominalConfig(actuator_time_constants=V20_ACTUATOR_GRID)


def v20_protocol_dict() -> dict:
    config = v20_config()
    return {
        "release_target": "v2.0.0",
        "campaign": "adaptive_mujoco_v20",
        "training": False,
        "plant": V20_PLANT,
        "plant_options": asdict(V20_PLANT_OPTIONS),
        "arms": list(V20_ARMS),
        "primary_contrast": {
            "treatment": "adaptive_hocbf",
            "reference": "fixed_hocbf",
            "scenarios": list(V20_PRIMARY_SCENARIOS),
            "metric": "success",
        },
        "seeds": {
            "decision": {
                "arms": list(V20_PRIMARY_ARMS),
                "start": V20_PRIMARY_SEED_START,
                "episodes_per_scenario": V20_PRIMARY_EPISODES,
                "plant": V20_PLANT,
            },
            "descriptive": {
                "arms": ["fixed", "adaptive"],
                "start": V20_PRIMARY_SEED_START,
                "episodes_per_scenario": V20_DESCRIPTIVE_EPISODES,
                "plant": V20_PLANT,
                "decision_weight": False,
            },
            "transfer": {
                "arms": ["fixed"],
                "plants": ["analytical", V20_PLANT],
                "start": V20_TRANSFER_SEED_START,
                "episodes_per_scenario": V20_TRANSFER_EPISODES,
                "reference": V19_REPRODUCTION_REFERENCE,
                "reference_controller": V19_REPRODUCTION_CONTROLLER,
                "analytical_rows_must_match_reference": "all retained fields, exactly",
                "decision_weight": False,
            },
            "scenarios": list(V20_SCENARIOS),
            "unopened_range_scanned": [
                V20_PRIMARY_SEED_START,
                V20_PRIMARY_SEED_START + V20_PRIMARY_EPISODES - 1,
            ],
        },
        "bootstrap": {
            "type": "paired_over_episode_seeds",
            "draws": V19_BOOTSTRAP_DRAWS,
            "seed": V20_BOOTSTRAP_SEED,
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
                "scenarios": list(V20_SCENARIOS),
                "veto_if": "lower_95_above_zero_or_difference_above_margin",
                "margin": V20_UNSAFE_MARGIN,
                "outcome": "no_go_safety",
            },
            "primary_success": {
                "minimum_gain": V20_SUCCESS_GAIN,
                "lower_95_above": 0.0,
                "scenarios": list(V20_PRIMARY_SCENARIOS),
            },
            "non_inferiority": {
                "metric": "unsafe_episode",
                "scenarios": list(V20_SCENARIOS),
                "upper_95_at_or_below": V20_UNSAFE_MARGIN,
                "decision_weight": True,
            },
            "estimator_validity": {
                "guarded_step_fraction_per_episode": V20_GUARDED_STEP_FRACTION,
                "max_guarded_episode_fraction": V20_GUARDED_EPISODE_FRACTION,
            },
            "go_requires": [
                "no_safety_veto",
                "primary_success",
                "non_inferiority_all_scenarios",
                "estimator_validity",
            ],
        },
        "seal_file": V20_SEAL_FILE,
        "frozen_paths": list(V20_FROZEN_PATHS),
        "state_interface": "measured_state_with_sensor_noise_same_for_every_arm",
        "estimator": {
            "process_noise": config.process_noise,
            "forgetting": config.forgetting,
            "max_lag": config.max_lag,
            "lag_min_updates": config.lag_min_updates,
            "selection_memory": config.selection_memory,
            "actuator_time_constants": list(config.actuator_time_constants),
            "actuator_substeps": config.actuator_substeps,
            "payload_prior": config.payload_prior,
            "filter_gate": config.filter_gate,
            "prediction_limit": config.prediction_limit,
            "conditioning_floor": config.conditioning_floor,
            "prior_std": list(config.prior_std),
            "lower": list(config.lower),
            "upper": list(config.upper),
            "compensate_delay": V20_COMPENSATE_DELAY,
        },
    }


def v20_seeds(block: str) -> range:
    if block == "decision":
        return range(V20_PRIMARY_SEED_START, V20_PRIMARY_SEED_START + V20_PRIMARY_EPISODES)
    if block == "descriptive":
        return range(V20_PRIMARY_SEED_START, V20_PRIMARY_SEED_START + V20_DESCRIPTIVE_EPISODES)
    if block == "transfer":
        return range(V20_TRANSFER_SEED_START, V20_TRANSFER_SEED_START + V20_TRANSFER_EPISODES)
    raise ValueError(f"unknown seed block {block}")


def v20_cells():
    """Every (arm, scenario, seed, plant) in the fixed order the runner executes."""
    for scenario in V20_SCENARIOS:
        for seed in v20_seeds("transfer"):
            for plant in ("analytical", V20_PLANT):
                yield "fixed", scenario, seed, plant
        for seed in v20_seeds("decision"):
            for arm in V20_ARMS:
                if arm in V20_PRIMARY_ARMS or seed in v20_seeds("descriptive"):
                    yield arm, scenario, seed, V20_PLANT


def cell_key(episode: PilotEpisode) -> tuple:
    return (episode.arm, episode.scenario, episode.seed, episode.plant)


def _select(episodes, arm, scenario, seeds, plant):
    wanted = set(seeds)
    return [
        e
        for e in episodes
        if e.arm == arm and e.scenario == scenario and e.seed in wanted and e.plant == plant
    ]


REFERENCE_FIELDS = (
    ("reward", float),
    ("steps", int),
    ("success", bool),
    ("final_distance", float),
    ("max_speed", float),
    ("max_command_torque", float),
    ("fault_seen", bool),
)


REPRODUCTION_RELATIVE_TOLERANCE = 1e-9
REPRODUCTION_ZERO_TOLERANCE = 1e-12


def _float_matches(ours: float, retained: float) -> bool:
    """Relative tolerance against the retained value; absolute only when it is zero."""
    if retained == 0.0:
        return abs(ours) <= REPRODUCTION_ZERO_TOLERANCE
    return abs(ours - retained) <= REPRODUCTION_RELATIVE_TOLERANCE * abs(retained)


def strict_reproduction_check(rows: list[PilotEpisode], reference_path: Path) -> dict:
    """Every retained A0 field must match the analytical transfer rows exactly."""
    if not reference_path.exists():
        raise FileNotFoundError(f"reproduction reference missing: {reference_path}")
    reference = {}
    with reference_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["controller"] != V19_REPRODUCTION_CONTROLLER:
                continue
            key = (row["scenario"], int(row["seed"]))
            if key in reference:
                raise ValueError(f"duplicate reference row {key}")
            reference[key] = {
                name: (row[name] == "True" if kind is bool else kind(row[name]))
                for name, kind in REFERENCE_FIELDS
            }
    expected = {(s, seed) for s in V20_SCENARIOS for seed in v20_seeds("transfer")}
    if set(reference) != expected:
        raise ValueError("reproduction reference does not hold exactly the 300 expected rows")
    if {(e.scenario, e.seed) for e in rows} != expected:
        raise ValueError("analytical transfer rows do not cover the 300 reference keys")
    mismatches = []
    for episode in rows:
        retained = reference[(episode.scenario, episode.seed)]
        for name, kind in REFERENCE_FIELDS:
            ours = getattr(episode, name)
            if kind is float:
                same = _float_matches(ours, retained[name])
            else:
                same = ours == retained[name]
            if not same:
                mismatches.append([episode.scenario, episode.seed, name, ours, retained[name]])
    return {
        "reference": reference_path.name,
        "reference_controller": V19_REPRODUCTION_CONTROLLER,
        "fields": [name for name, _ in REFERENCE_FIELDS],
        "relative_tolerance": REPRODUCTION_RELATIVE_TOLERANCE,
        "zero_tolerance": REPRODUCTION_ZERO_TOLERANCE,
        "compared": len(rows),
        "matched": len(rows) - len({(m[0], m[1]) for m in mismatches}),
        "mismatches": mismatches[:20],
    }


def transfer_check(episodes: list[PilotEpisode], reference_path: Path) -> dict:
    """Analytical fixed rows must reproduce the retained A0 rows; MuJoCo rows are compared."""
    analytical = [
        e
        for scenario in V20_SCENARIOS
        for e in _select(episodes, "fixed", scenario, v20_seeds("transfer"), "analytical")
    ]
    engine = {
        (e.scenario, e.seed): e
        for scenario in V20_SCENARIOS
        for e in _select(episodes, "fixed", scenario, v20_seeds("transfer"), V20_PLANT)
    }
    reproduction = strict_reproduction_check(analytical, reference_path)
    if reproduction["mismatches"]:
        raise ValueError(
            "analytical transfer rows do not reproduce the retained A0 rows: "
            f"{reproduction['mismatches'][:3]}"
        )
    per_scenario = {}
    for scenario in V20_SCENARIOS:
        pairs = [
            (e, engine[(scenario, e.seed)])
            for e in analytical
            if e.scenario == scenario and (scenario, e.seed) in engine
        ]
        per_scenario[scenario] = {
            "pairs": len(pairs),
            "success_analytical": float(np.mean([a.success for a, _ in pairs])),
            "success_mujoco": float(np.mean([m.success for _, m in pairs])),
            "same_outcome_fraction": float(np.mean([a.outcome == m.outcome for a, m in pairs])),
            "final_distance_abs_difference_median_m": float(
                np.median([abs(a.final_distance - m.final_distance) for a, m in pairs])
            ),
            "unsafe_analytical": float(np.mean([a.unsafe_episode for a, _ in pairs])),
            "unsafe_mujoco": float(np.mean([m.unsafe_episode for _, m in pairs])),
        }
    return {"reproduction_of_retained_a0": reproduction, "plant_difference": per_scenario}


def analyze(episodes: list[PilotEpisode], reference_path: Path) -> dict:
    """Apply the frozen rule. Vetoes are checked before the primary endpoint."""
    reference_path = Path(reference_path)
    if [cell_key(e) for e in episodes] != list(v20_cells()):
        raise ValueError("episodes do not match the planned cells exactly, in order")
    decision_block = {
        (arm, scenario): _select(episodes, arm, scenario, v20_seeds("decision"), V20_PLANT)
        for arm in V20_PRIMARY_ARMS
        for scenario in V20_SCENARIOS
    }
    descriptive_block = {
        (arm, scenario): _select(episodes, arm, scenario, v20_seeds("descriptive"), V20_PLANT)
        for arm in ("fixed", "adaptive")
        for scenario in V20_SCENARIOS
    }

    rng = np.random.default_rng(V20_BOOTSTRAP_SEED)
    contrasts = {}
    for scenario in V20_SCENARIOS:
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
        for scenario in V20_SCENARIOS
        if contrasts[scenario]["unsafe_episode"]["ci95_low"] > 0.0
        or contrasts[scenario]["unsafe_episode"]["difference"] > V20_UNSAFE_MARGIN
    ]
    non_inferior = {
        scenario: bool(contrasts[scenario]["unsafe_episode"]["ci95_high"] <= V20_UNSAFE_MARGIN)
        for scenario in V20_SCENARIOS
    }
    primary_met = all(
        contrasts[s]["success"]["difference"] >= V20_SUCCESS_GAIN
        and contrasts[s]["success"]["ci95_low"] > 0.0
        for s in V20_PRIMARY_SCENARIOS
    )
    adaptive_rows = [
        e for (arm, _), rows in decision_block.items() if arm == "adaptive_hocbf" for e in rows
    ]
    guarded = [
        e
        for e in adaptive_rows
        if e.control_steps > 0
        and e.model_fallback_steps / e.control_steps > V20_GUARDED_STEP_FRACTION
    ]
    guarded_fraction = len(guarded) / len(adaptive_rows)
    estimator_valid = guarded_fraction <= V20_GUARDED_EPISODE_FRACTION

    reasons = []
    if vetoes:
        decision = "no_go_safety"
    else:
        if not primary_met:
            reasons.append("success gain not established in every primary scenario")
        not_shown = [s for s in V20_SCENARIOS if not non_inferior[s]]
        if not_shown:
            reasons.append("unsafe-episode non-inferiority not shown in " + ", ".join(not_shown))
        if not estimator_valid:
            reasons.append(
                f"estimate guarded in more than {V20_GUARDED_STEP_FRACTION:.0%} of steps in "
                f"{guarded_fraction:.1%} of adaptive decision episodes"
            )
        decision = "go" if not reasons else "inconclusive"

    estimator = {
        "guarded_episode_fraction": guarded_fraction,
        "guarded_step_fraction_mean": float(
            np.mean([e.model_fallback_steps / max(e.control_steps, 1) for e in adaptive_rows])
        ),
        "estimator_valid": estimator_valid,
        "prediction_error_oracle": "analytical_parameters_in_command_coordinates",
        "time_constant_abs_error_mean_s": float(
            np.mean([abs(e.selected_time_constant - e.true_time_constant) for e in adaptive_rows])
        ),
        "lag_identified_fraction": float(
            np.mean([e.selected_lag == e.true_delay for e in adaptive_rows])
        ),
        "episodes_with_model_fallback": int(sum(e.model_fallbacks > 0 for e in adaptive_rows)),
        "episodes_with_prediction_fallback": int(
            sum(e.prediction_fallbacks > 0 for e in adaptive_rows)
        ),
    }
    return {
        "decision": decision,
        "inconclusive_reasons": reasons,
        "safety_vetoes": vetoes,
        "unsafe_non_inferior_at_margin": non_inferior,
        "primary_met": primary_met,
        "estimator_valid": estimator_valid,
        "contrasts": contrasts,
        "estimator": estimator,
        "decision_summary": {
            f"{arm}/{scenario}": cell_summary(rows)
            for (arm, scenario), rows in decision_block.items()
        },
        "descriptive_summary": {
            f"{arm}/{scenario}": cell_summary(rows)
            for (arm, scenario), rows in descriptive_block.items()
        },
        "transfer_check": transfer_check(episodes, reference_path),
        "protocol": v20_protocol_dict(),
    }


__all__ = ["analyze", "load_episodes", "v20_cells", "v20_config", "v20_protocol_dict"]
