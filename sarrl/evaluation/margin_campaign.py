"""Preregistered v2.2 campaign: an uncertainty margin on the identified certificate.

v2.1 measured that certifying the identified controller stack with its own
estimate raises success everywhere and, in distribution, the unsafe-episode
rate. v2.2 tightens the certificate's model rows by the command error the
estimator's covariance allows (`adaptive_hocbf_margin`) and compares it on
the v2.0 benchmark (MuJoCo plant, actuator options, measured state) and
fresh paired seeds with the stack without a margin (`adaptive_hocbf`) and
with a constant command margin on the same steps
(`adaptive_hocbf_constmargin`), the control for the covariance's shape.
Both margin values were chosen in the exploratory pilot at the same
in-distribution success. A reproduction block reruns `adaptive_hocbf` on
the first hundred v2.1 decision seeds per scenario before any decision seed
is opened: every retained field of the v2.1 rows must match.

The decision rule tests three preregistered statements with seed-paired
percentile bootstraps: the margin lowers the unsafe-episode rate, at no
material cost in success, and the covariance margin lowers it further than
the constant one. Every other endpoint is reported with an interval and no
threshold.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sarrl.controllers import AdaptiveNominalConfig
from sarrl.evaluation.adaptive_campaign import (
    V19_BOOTSTRAP_DRAWS,
    cell_summary,
    load_episodes,
    paired_difference,
)
from sarrl.evaluation.adaptive_pilot import PilotEpisode
from sarrl.evaluation.factorial_campaign import (
    V21_DECISION_SEED_START,
    V21_OUTPUT,
    V21_POSITION_TOLERANCE_RAD,
    V21_SEED_REGISTRY,
    V21_VELOCITY_TOLERANCE_RAD_S,
    _values_match,
    conditional_mean,
)
from sarrl.evaluation.mujoco_campaign import (
    V20_ACTUATOR_GRID,
    V20_COMPENSATE_DELAY,
    V20_GUARDED_EPISODE_FRACTION,
    V20_GUARDED_STEP_FRACTION,
    V20_PLANT,
    V20_PLANT_OPTIONS,
)

# Arm labels are the pilot's arm names; the margin is fixed per arm.
V22_NONE = "adaptive_hocbf"
V22_COVARIANCE = "adaptive_hocbf_margin"
V22_CONSTANT = "adaptive_hocbf_constmargin"
V22_ARM_LABELS = (V22_NONE, V22_COVARIANCE, V22_CONSTANT)
# Standard deviations of the predicted command for the covariance margin,
# N m per joint for the constant margin.
V22_MARGINS = {V22_NONE: 0.0, V22_COVARIANCE: 0.25, V22_CONSTANT: 2.0}
V22_SCENARIOS = ("id_reference", "ood_compound", "motor_fault")
V22_STATEMENT_SCENARIOS = ("id_reference", "motor_fault")
V22_SHAPE_SCENARIOS = ("id_reference",)
V22_PLANT = V20_PLANT
V22_PLANT_OPTIONS = V20_PLANT_OPTIONS
V22_COMPENSATE_DELAY = V20_COMPENSATE_DELAY
# Decision seeds: fresh block after the v2.1 block (54200..56099) and a guard band.
V22_DECISION_SEED_START = 56200
V22_DECISION_EPISODES = 1900
# Reproduction block: the first hundred v2.1 decision seeds per scenario, no-margin arm only.
V22_REPRODUCTION_SEED_START = V21_DECISION_SEED_START
V22_REPRODUCTION_EPISODES = 100
V22_REPRODUCTION_ARMS = (V22_NONE,)
V22_REPRODUCTION_REFERENCE = f"{V21_OUTPUT}/episodes.jsonl"
V22_BOOTSTRAP_SEED = 220_000
# Statements, each on a 95% bound of a seed-paired difference.
V22_UNSAFE_REDUCTION = 0.05  # covariance minus none, unsafe rate: upper bound below -5 pp
V22_SUCCESS_MARGIN = 0.03  # covariance minus none, success: lower bound above -3 pp
V22_SHAPE_BOUND = 0.0  # covariance minus constant, unsafe rate: upper bound below 0
V22_SEED_REGISTRY = V21_SEED_REGISTRY
V22_FROZEN_PATHS = (
    "sarrl",
    "tools",
    "tests",
    "docs/experiments.md",
    "pyproject.toml",
    ".gitignore",
    V22_REPRODUCTION_REFERENCE,
    V22_SEED_REGISTRY,
)
V22_SEAL_FILE = "docs/v22_seal.json"
V22_OUTPUT = "results/uncertainty_margin_v22"
V22_CONTRASTS = {
    "covariance_minus_none": (V22_COVARIANCE, V22_NONE),
    "constant_minus_none": (V22_CONSTANT, V22_NONE),
    "covariance_minus_constant": (V22_COVARIANCE, V22_CONSTANT),
}


def v22_config() -> AdaptiveNominalConfig:
    """The v2.0 estimator configuration, unchanged."""
    return AdaptiveNominalConfig(actuator_time_constants=V20_ACTUATOR_GRID)


def v22_protocol_dict() -> dict:
    config = v22_config()
    return {
        "release_target": "v2.2.0",
        "campaign": "uncertainty_margin_v22",
        "training": False,
        "plant": V22_PLANT,
        "plant_options": asdict(V22_PLANT_OPTIONS),
        "arms": {
            V22_NONE: "identified controller stack and certificate, no margin (v2.1 "
            "identified/identified)",
            V22_COVARIANCE: "same, model rows tightened by kappa * sum_j |A_rj| sigma_j, "
            "sigma from the selected hypothesis' covariance",
            V22_CONSTANT: "same, sigma = (1, 1) on the same steps: a constant command margin",
        },
        "margins": dict(V22_MARGINS),
        "margin_units": {
            V22_COVARIANCE: "standard deviations of the predicted command",
            V22_CONSTANT: "N m per joint",
        },
        "margin_rules": {
            "sigma_evaluated_at": "the larger of the candidate and the projection without margin",
            "infeasible_margin": "halved until feasible, dropped after 8 halvings",
            "no_margin_when": "the step uses the nominal model, or the rows without margin "
            "are infeasible",
            "rows": "every row except the four torque-box rows",
        },
        "margin_values_chosen": "in the exploratory pilot (seeds 9803600..9803899) at the "
        "same in-distribution success, 90 of 100",
        "seeds": {
            "reproduction": {
                "arms": list(V22_REPRODUCTION_ARMS),
                "start": V22_REPRODUCTION_SEED_START,
                "episodes_per_scenario": V22_REPRODUCTION_EPISODES,
                "reference": V22_REPRODUCTION_REFERENCE,
                "rows_must_match_reference": "every retained field",
                "runs_and_is_checked_before_any_decision_seed": True,
                "decision_weight": False,
            },
            "decision": {
                "arms": list(V22_ARM_LABELS),
                "start": V22_DECISION_SEED_START,
                "episodes_per_scenario": V22_DECISION_EPISODES,
                "plant": V22_PLANT,
            },
            "scenarios": list(V22_SCENARIOS),
            "unopened_range_checked_against": [
                "committed CSV/JSON/JSONL blobs of every commit reachable from any ref",
                V22_SEED_REGISTRY,
            ],
            "unopened_range": [
                V22_DECISION_SEED_START,
                V22_DECISION_SEED_START + V22_DECISION_EPISODES - 1,
            ],
        },
        "bootstrap": {
            "type": "paired_over_episode_seeds",
            "draws": V19_BOOTSTRAP_DRAWS,
            "seed": V22_BOOTSTRAP_SEED,
        },
        "statements": {
            "margin_removes_violations": {
                "contrast": "covariance_minus_none",
                "metric": "unsafe_episode",
                "scenarios": list(V22_STATEMENT_SCENARIOS),
                "upper_95_below": -V22_UNSAFE_REDUCTION,
            },
            "no_material_success_cost": {
                "contrast": "covariance_minus_none",
                "metric": "success",
                "scenarios": list(V22_STATEMENT_SCENARIOS),
                "lower_95_above": -V22_SUCCESS_MARGIN,
            },
            "covariance_shape_matters": {
                "contrast": "covariance_minus_constant",
                "metric": "unsafe_episode",
                "scenarios": list(V22_SHAPE_SCENARIOS),
                "upper_95_below": V22_SHAPE_BOUND,
            },
        },
        "reported_without_threshold": {
            "contrasts": sorted(V22_CONTRASTS),
            "scenarios": "all three, ood_compound included",
            "paired": [
                "success",
                "abort",
                "timeout",
                "unsafe_episode",
                "unsafe_beyond_tolerance (position excess above "
                f"{V21_POSITION_TOLERANCE_RAD} rad or velocity excess above "
                f"{V21_VELOCITY_TOLERANCE_RAD_S} rad/s)",
                "normalized_violation_max",
                "joint_position_violation_max_rad",
                "joint_velocity_violation_max_rad_s",
                "safety_intervention_fraction",
                "steps (observed prefix length; aborts end the prefix)",
            ],
            "conditional": "mean position and velocity excess over the unsafe episodes of "
            "each arm, unpaired percentile bootstrap",
            "margin_diagnostics": "mean enforced margin per joint, fractions of margin steps "
            "halved and dropped, episodes with a dropped margin",
            "expected_from_pilot": {
                "abort": "more frequent with the covariance margin, about +3 to +4 pp in "
                "id_reference and motor_fault",
                "ood_compound_success": "lower with the covariance margin",
            },
        },
        "decision": {
            "order": ["estimator_validity", "statements"],
            "estimator_validity": {
                "arms_with_estimator": list(V22_ARM_LABELS),
                "applied_per": "arm and scenario",
                "guarded_step_fraction_per_episode": V20_GUARDED_STEP_FRACTION,
                "max_guarded_episode_fraction": V20_GUARDED_EPISODE_FRACTION,
                "failure_outcome": "inconclusive",
            },
            "outcomes": {
                "confirmed": "all three statements hold",
                "margin_confirmed_shape_not": "the first two hold and the third does not",
                "not_confirmed": "the first or the second fails; each failed statement is named",
                "inconclusive": "estimator validity fails in any cell",
            },
        },
        "seal_file": V22_SEAL_FILE,
        "frozen_paths": list(V22_FROZEN_PATHS),
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
            "compensate_delay": V22_COMPENSATE_DELAY,
        },
    }


def v22_seeds(block: str) -> range:
    if block == "decision":
        return range(V22_DECISION_SEED_START, V22_DECISION_SEED_START + V22_DECISION_EPISODES)
    if block == "reproduction":
        return range(
            V22_REPRODUCTION_SEED_START,
            V22_REPRODUCTION_SEED_START + V22_REPRODUCTION_EPISODES,
        )
    raise ValueError(f"unknown seed block {block}")


def v22_reproduction_cells():
    """The reproduction block, executed and checked before any decision seed."""
    for scenario in V22_SCENARIOS:
        for seed in v22_seeds("reproduction"):
            for arm in V22_REPRODUCTION_ARMS:
                yield arm, scenario, seed, V22_PLANT


def v22_decision_cells():
    """The decision block, executed after the reproduction check."""
    for scenario in V22_SCENARIOS:
        for seed in v22_seeds("decision"):
            for arm in V22_ARM_LABELS:
                yield arm, scenario, seed, V22_PLANT


def v22_cells():
    """Every (arm, scenario, seed, plant) in the fixed order the runner executes."""
    yield from v22_reproduction_cells()
    yield from v22_decision_cells()


def cell_key(episode: PilotEpisode) -> tuple:
    return (episode.arm, episode.scenario, episode.seed, episode.plant)


def _select(episodes, arm, scenario, seeds):
    wanted = set(seeds)
    return [e for e in episodes if e.arm == arm and e.scenario == scenario and e.seed in wanted]


def load_reference(reference_path: Path) -> dict:
    """Retained v2.1 rows of the no-margin arm on the reproduction seeds, keyed by cell."""
    if not reference_path.exists():
        raise FileNotFoundError(f"reproduction reference missing: {reference_path}")
    wanted = set(v22_seeds("reproduction"))
    reference = {}
    for line in reference_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (record["arm"], record["scenario"], record["seed"])
        if record["arm"] in V22_REPRODUCTION_ARMS and record["seed"] in wanted:
            if key in reference:
                raise ValueError(f"duplicate reference row {key}")
            reference[key] = record
    expected = {
        (arm, scenario, seed)
        for arm in V22_REPRODUCTION_ARMS
        for scenario in V22_SCENARIOS
        for seed in wanted
    }
    if set(reference) != expected:
        raise ValueError("reproduction reference does not hold exactly the 300 expected rows")
    return reference


def reproduction_check(episodes: list[PilotEpisode], reference_path: Path) -> dict:
    """Every field of every retained row must match on the reproduction seeds."""
    reference = load_reference(Path(reference_path))
    rows = [
        e
        for arm in V22_REPRODUCTION_ARMS
        for scenario in V22_SCENARIOS
        for e in _select(episodes, arm, scenario, v22_seeds("reproduction"))
    ]
    if {(e.arm, e.scenario, e.seed) for e in rows} != set(reference):
        raise ValueError("reproduction rows do not cover the reference keys")
    fields = sorted(next(iter(reference.values())))
    mismatches = []
    for episode in rows:
        retained = reference[(episode.arm, episode.scenario, episode.seed)]
        ours = asdict(episode)
        for name in fields:
            if not _values_match(ours[name], retained[name]):
                mismatches.append(
                    [episode.arm, episode.scenario, episode.seed, name, ours[name], retained[name]]
                )
    return {
        "reference": str(reference_path),
        "fields": fields,
        "compared": len(rows),
        "matched": len(rows) - len({(m[0], m[1], m[2]) for m in mismatches}),
        "mismatches": mismatches[:20],
    }


def beyond_tolerance(row: PilotEpisode) -> bool:
    return (
        row.joint_position_violation_max_rad > V21_POSITION_TOLERANCE_RAD
        or row.joint_velocity_violation_max_rad_s > V21_VELOCITY_TOLERANCE_RAD_S
    )


def estimator_validity(cells: dict) -> dict:
    """The v2.0 guard condition, applied to every arm in every scenario."""
    per_cell = {}
    for (label, scenario), rows in cells.items():
        guarded = [
            e
            for e in rows
            if e.control_steps > 0
            and e.model_fallback_steps / e.control_steps > V20_GUARDED_STEP_FRACTION
        ]
        fraction = len(guarded) / len(rows)
        per_cell[f"{label}/{scenario}"] = {
            "guarded_episode_fraction": fraction,
            "valid": fraction <= V20_GUARDED_EPISODE_FRACTION,
            "lag_identified_fraction": float(
                np.mean([e.selected_lag == e.true_delay for e in rows])
            ),
            "time_constant_abs_error_mean_s": float(
                np.mean([abs(e.selected_time_constant - e.true_time_constant) for e in rows])
            ),
            "episodes_with_prediction_fallback": int(sum(e.prediction_fallbacks > 0 for e in rows)),
        }
    return {
        "estimator_valid": all(cell["valid"] for cell in per_cell.values()),
        "invalid_cells": [name for name, cell in per_cell.items() if not cell["valid"]],
        "cells": per_cell,
    }


def margin_diagnostics(rows: list[PilotEpisode]) -> dict:
    """Enforced margin and how often it was halved or dropped, over one margin cell."""
    steps = sum(r.margin_steps for r in rows)
    weighted = [
        (np.asarray(r.margin_mean_command, dtype=np.float64), r.margin_steps)
        for r in rows
        if r.margin_mean_command is not None and r.margin_steps > 0
    ]
    mean = (
        [float(v) for v in sum(m * w for m, w in weighted) / sum(w for _, w in weighted)]
        if weighted
        else None
    )
    return {
        "margin_steps": int(steps),
        "mean_enforced_margin_per_joint": mean,
        "halved_step_fraction": (
            float(sum(r.margin_reduced_steps for r in rows) / steps) if steps else None
        ),
        "dropped_step_fraction": (
            float(sum(r.margin_dropped_steps for r in rows) / steps) if steps else None
        ),
        "episodes_with_dropped_margin": int(sum(r.margin_dropped_steps > 0 for r in rows)),
    }


PAIRED_METRICS = {
    "success": lambda r: r.success,
    "abort": lambda r: r.outcome == "abort",
    "timeout": lambda r: r.outcome == "timeout",
    "unsafe_episode": lambda r: r.unsafe_episode,
    "unsafe_beyond_tolerance": beyond_tolerance,
    "normalized_violation_max": lambda r: r.normalized_violation_max,
    "joint_position_violation_max_rad": lambda r: r.joint_position_violation_max_rad,
    "joint_velocity_violation_max_rad_s": lambda r: r.joint_velocity_violation_max_rad_s,
    "safety_intervention_fraction": lambda r: r.safety_intervention_fraction,
    "steps": lambda r: r.steps,
}
CONDITIONAL_METRICS = {
    "joint_position_violation_max_rad": lambda r: r.joint_position_violation_max_rad,
    "joint_velocity_violation_max_rad_s": lambda r: r.joint_velocity_violation_max_rad_s,
}


def analyze(episodes: list[PilotEpisode], reference_path: Path) -> dict:
    """Apply the frozen rule: reproduction, validity, then the three statements."""
    if [cell_key(e) for e in episodes] != list(v22_cells()):
        raise ValueError("episodes do not match the planned cells exactly, in order")
    for episode in episodes:
        if episode.certificate_margin != V22_MARGINS[episode.arm]:
            raise ValueError(f"episode {cell_key(episode)} ran with the wrong margin")
    reproduction = reproduction_check(episodes, Path(reference_path))
    if reproduction["mismatches"]:
        raise ValueError(
            f"the no-margin arm does not reproduce the retained v2.1 rows: "
            f"{reproduction['mismatches'][:3]}"
        )
    cells = {
        (label, scenario): _select(episodes, label, scenario, v22_seeds("decision"))
        for label in V22_ARM_LABELS
        for scenario in V22_SCENARIOS
    }
    rng = np.random.default_rng(V22_BOOTSTRAP_SEED)
    contrasts = {
        scenario: {
            name: {
                metric_name: paired_difference(
                    cells[(treatment, scenario)], cells[(reference, scenario)], metric, rng
                )
                for metric_name, metric in PAIRED_METRICS.items()
            }
            for name, (treatment, reference) in V22_CONTRASTS.items()
        }
        for scenario in V22_SCENARIOS
    }
    conditional = {
        scenario: {
            label: {
                name: conditional_mean(cells[(label, scenario)], metric, rng)
                for name, metric in CONDITIONAL_METRICS.items()
            }
            for label in V22_ARM_LABELS
        }
        for scenario in V22_SCENARIOS
    }
    margins = {
        f"{label}/{scenario}": margin_diagnostics(cells[(label, scenario)])
        for label in (V22_COVARIANCE, V22_CONSTANT)
        for scenario in V22_SCENARIOS
    }

    def bound(scenario, contrast, metric, side):
        return contrasts[scenario][contrast][metric][side]

    statements = {
        "margin_removes_violations": all(
            bound(s, "covariance_minus_none", "unsafe_episode", "ci95_high") < -V22_UNSAFE_REDUCTION
            for s in V22_STATEMENT_SCENARIOS
        ),
        "no_material_success_cost": all(
            bound(s, "covariance_minus_none", "success", "ci95_low") > -V22_SUCCESS_MARGIN
            for s in V22_STATEMENT_SCENARIOS
        ),
        "covariance_shape_matters": all(
            bound(s, "covariance_minus_constant", "unsafe_episode", "ci95_high") < V22_SHAPE_BOUND
            for s in V22_SHAPE_SCENARIOS
        ),
    }
    validity = estimator_validity(cells)
    if not validity["estimator_valid"]:
        decision = "inconclusive"
    elif all(statements.values()):
        decision = "confirmed"
    elif statements["margin_removes_violations"] and statements["no_material_success_cost"]:
        decision = "margin_confirmed_shape_not"
    else:
        decision = "not_confirmed"
    return {
        "decision": decision,
        "statements": statements,
        "failed_statements": [name for name, held in statements.items() if not held],
        "estimator_valid": validity["estimator_valid"],
        "invalid_estimator_cells": validity["invalid_cells"],
        "contrasts": contrasts,
        "severity_conditional": conditional,
        "margin_diagnostics": margins,
        "estimator": validity["cells"],
        "cell_summary": {
            f"{label}/{scenario}": cell_summary(rows) for (label, scenario), rows in cells.items()
        },
        "reproduction_check": reproduction,
        "protocol": v22_protocol_dict(),
    }


__all__ = [
    "analyze",
    "load_episodes",
    "v22_cells",
    "v22_config",
    "v22_decision_cells",
    "v22_protocol_dict",
    "v22_reproduction_cells",
    "v22_seeds",
]
