"""Preregistered v2.1 campaign: which model matters, the controller's or the certificate's.

v2.0 compared two stacks that differed in both places at once: the fixed
nominal certified with the fixed model against the identified controller
stack (estimate-driven computed torque with delay prediction) certified
with its own estimate. This campaign crosses the two factors on the v2.0
benchmark (MuJoCo plant, actuator options, measured state) and fresh paired
seeds: controller stack in {fixed, identified} by certificate model in
{fixed, identified}, four filtered arms on every seed. The controller
factor is the whole v2.0 controller stack, model and delay prediction
together; the certificate factor is the model behind the HOCBF rows alone.
The corner arms use the v2.0 configuration, and a reproduction block reruns
them on the first hundred v2.0 decision seeds per scenario before any
decision seed is opened: every retained field of the v2.0 rows must match.

The decision rule tests three preregistered statements about task success
with seed-paired percentile bootstraps, each requiring the lower 95% bound
to clear its margin. Safety endpoints (operational outcomes, violation
rates at zero and at a physical tolerance, violation severity paired and
conditional on violating) are preregistered with their expected directions
but carry no decision weight: the pilot showed the two certificates trading
rate against severity, and the campaign estimates that trade.
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
from sarrl.evaluation.mujoco_campaign import (
    V20_ACTUATOR_GRID,
    V20_COMPENSATE_DELAY,
    V20_GUARDED_EPISODE_FRACTION,
    V20_GUARDED_STEP_FRACTION,
    V20_OUTPUT,
    V20_PLANT,
    V20_PLANT_OPTIONS,
    V20_PRIMARY_SEED_START,
    _float_matches,
)

# Arms named controller stack / certificate model; the values are the pilot's arm names.
V21_ARMS = {
    "fixed/fixed": "fixed_hocbf",
    "fixed/identified": "fixed_hocbf_adaptivemodel",
    "identified/fixed": "adaptive_hocbf_fixedmodel",
    "identified/identified": "adaptive_hocbf",
}
V21_ARM_LABELS = tuple(V21_ARMS.values())
V21_ESTIMATOR_ARMS = tuple(label for label in V21_ARM_LABELS if label != "fixed_hocbf")
# Descriptive arm: the identified stack without delay prediction; its paired
# contrast with the full stack is the prediction's effect within that stack.
V21_NO_COMPENSATION_SUFFIX = "_nocompensation"
V21_DESCRIPTIVE_ARM = "adaptive_hocbf" + V21_NO_COMPENSATION_SUFFIX
V21_SCENARIOS = ("id_reference", "ood_compound", "motor_fault")
V21_MISMATCH_SCENARIOS = ("motor_fault", "ood_compound")
V21_PLANT = V20_PLANT
V21_PLANT_OPTIONS = V20_PLANT_OPTIONS
V21_COMPENSATE_DELAY = V20_COMPENSATE_DELAY
# Decision seeds: fresh block after the v2.0 block (52200..54099) and a guard band.
V21_DECISION_SEED_START = 54200
V21_DECISION_EPISODES = 1900
V21_DESCRIPTIVE_EPISODES = 100
# Reproduction block: the first hundred v2.0 decision seeds per scenario, corner arms only.
V21_REPRODUCTION_SEED_START = V20_PRIMARY_SEED_START
V21_REPRODUCTION_EPISODES = 100
V21_REPRODUCTION_ARMS = ("fixed_hocbf", "adaptive_hocbf")
V21_REPRODUCTION_REFERENCE = f"{V20_OUTPUT}/episodes.jsonl"
V21_BOOTSTRAP_SEED = 210_000
# Statements about success: the lower 95% bound of the paired difference must clear the margin.
V21_CONTROLLER_GAIN = 0.20
V21_CERTIFICATE_GAIN = 0.10
V21_EQUIVALENCE_MARGIN = 0.03
# Physical tolerances for the second violation rate: excesses at or below
# these are counted as within tolerance.
V21_POSITION_TOLERANCE_RAD = 0.05
V21_VELOCITY_TOLERANCE_RAD_S = 0.5
V21_SEED_REGISTRY = "docs/seed_registry.json"
V21_FROZEN_PATHS = (
    "sarrl",
    "tools",
    "tests",
    "docs/experiments.md",
    "pyproject.toml",
    ".gitignore",
    V21_REPRODUCTION_REFERENCE,
    V21_SEED_REGISTRY,
)
V21_SEAL_FILE = "docs/v21_seal.json"
V21_OUTPUT = "results/certificate_factorial_v21"


def v21_config() -> AdaptiveNominalConfig:
    """The v2.0 estimator configuration, unchanged."""
    return AdaptiveNominalConfig(actuator_time_constants=V20_ACTUATOR_GRID)


def v21_protocol_dict() -> dict:
    config = v21_config()
    return {
        "release_target": "v2.1.0",
        "campaign": "certificate_factorial_v21",
        "training": False,
        "plant": V21_PLANT,
        "plant_options": asdict(V21_PLANT_OPTIONS),
        "factors": {
            "controller_stack": {
                "levels": ["fixed", "identified"],
                "identified_means": "estimate-driven computed torque with delay prediction, "
                "the v2.0 controller stack; the two are not separated",
            },
            "certificate_model": {"levels": ["fixed", "identified"]},
        },
        "arms": dict(V21_ARMS),
        "descriptive_arm": {
            "label": V21_DESCRIPTIVE_ARM,
            "stack": "identified/identified without delay prediction",
            "episodes_per_scenario": V21_DESCRIPTIVE_EPISODES,
            "seeds": "first decision seeds",
            "contrast": "identified/identified minus the same stack without prediction, "
            "seed-paired on those seeds: the prediction's effect within the identified "
            "stack, not a share of the controller effect",
            "decision_weight": False,
        },
        "seeds": {
            "reproduction": {
                "arms": list(V21_REPRODUCTION_ARMS),
                "start": V21_REPRODUCTION_SEED_START,
                "episodes_per_scenario": V21_REPRODUCTION_EPISODES,
                "reference": V21_REPRODUCTION_REFERENCE,
                "rows_must_match_reference": "every retained field",
                "runs_and_is_checked_before_any_decision_seed": True,
                "decision_weight": False,
            },
            "decision": {
                "arms": list(V21_ARM_LABELS),
                "start": V21_DECISION_SEED_START,
                "episodes_per_scenario": V21_DECISION_EPISODES,
                "plant": V21_PLANT,
            },
            "scenarios": list(V21_SCENARIOS),
            "unopened_range_checked_against": [
                "committed CSV/JSON/JSONL blobs of every commit reachable from any ref",
                V21_SEED_REGISTRY,
            ],
            "unopened_range": [
                V21_DECISION_SEED_START,
                V21_DECISION_SEED_START + V21_DECISION_EPISODES - 1,
            ],
        },
        "bootstrap": {
            "type": "paired_over_episode_seeds",
            "draws": V19_BOOTSTRAP_DRAWS,
            "seed": V21_BOOTSTRAP_SEED,
        },
        "success_statements": {
            "controller_gain_at_fixed_certificate": {
                "contrast": "identified/fixed minus fixed/fixed",
                "scenarios": list(V21_SCENARIOS),
                "lower_95_above": V21_CONTROLLER_GAIN,
            },
            "certificate_gain_under_mismatch": {
                "contrast": "identified/identified minus identified/fixed",
                "scenarios": list(V21_MISMATCH_SCENARIOS),
                "lower_95_above": V21_CERTIFICATE_GAIN,
            },
            "certificate_equivalence_in_distribution": {
                "contrast": "identified/identified minus identified/fixed",
                "scenarios": ["id_reference"],
                "interval_within": [-V21_EQUIVALENCE_MARGIN, V21_EQUIVALENCE_MARGIN],
            },
            "reported_without_threshold": [
                "interaction: (identified/identified minus identified/fixed) minus "
                "(fixed/identified minus fixed/fixed)",
                "controller at identified certificate",
                "certificate at fixed controller",
            ],
        },
        "safety_endpoints": {
            "contrast": "identified/identified minus identified/fixed, per scenario, "
            "seed-paired unless stated",
            "families": {
                "operational": ["abort", "timeout"],
                "violation_rate": [
                    "unsafe_episode (any positive excess, the v2.0 definition)",
                    "unsafe_beyond_tolerance (position excess above "
                    f"{V21_POSITION_TOLERANCE_RAD} rad or velocity excess above "
                    f"{V21_VELOCITY_TOLERANCE_RAD_S} rad/s)",
                ],
                "severity_paired": [
                    "normalized_violation_max",
                    "joint_position_violation_max_rad",
                    "joint_velocity_violation_max_rad_s",
                    "safety_intervention_fraction",
                    "steps (observed prefix length; aborts end the prefix)",
                ],
                "severity_conditional": [
                    "mean of joint_position_violation_max_rad over unsafe episodes, per arm",
                    "mean of joint_velocity_violation_max_rad_s over unsafe episodes, per arm",
                    "with unpaired percentile bootstrap intervals per arm",
                ],
            },
            "censoring": "maxima are taken over the observed prefix of each episode, which "
            "an abort shortens; the prefix length is reported alongside",
            "expected_from_pilot": {
                "unsafe_episode": "more frequent with the identified certificate in "
                "id_reference and ood_compound, less frequent under motor_fault",
                "severity_conditional": "smaller position and velocity excesses with the "
                "identified certificate in every scenario",
                "safety_intervention_fraction": "lower with the identified certificate",
                "timeout": "fewer with the identified certificate under mismatch",
            },
            "decision_weight": False,
        },
        "decision": {
            "order": ["estimator_validity", "success_statements", "otherwise"],
            "estimator_validity": {
                "arms_with_estimator": list(V21_ESTIMATOR_ARMS),
                "applied_per": "arm and scenario",
                "guarded_step_fraction_per_episode": V20_GUARDED_STEP_FRACTION,
                "max_guarded_episode_fraction": V20_GUARDED_EPISODE_FRACTION,
                "failure_outcome": "inconclusive",
            },
            "outcomes": {
                "confirmed": "all three success statements hold",
                "partial": "at least one success statement fails; each is reported",
                "inconclusive": "estimator validity fails in any estimator cell",
            },
        },
        "seal_file": V21_SEAL_FILE,
        "frozen_paths": list(V21_FROZEN_PATHS),
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
            "compensate_delay": V21_COMPENSATE_DELAY,
            "compensation_applies_to": "identified controller stack only, as in v2.0",
        },
    }


def v21_seeds(block: str) -> range:
    if block == "decision":
        return range(V21_DECISION_SEED_START, V21_DECISION_SEED_START + V21_DECISION_EPISODES)
    if block == "descriptive":
        return range(V21_DECISION_SEED_START, V21_DECISION_SEED_START + V21_DESCRIPTIVE_EPISODES)
    if block == "reproduction":
        return range(
            V21_REPRODUCTION_SEED_START,
            V21_REPRODUCTION_SEED_START + V21_REPRODUCTION_EPISODES,
        )
    raise ValueError(f"unknown seed block {block}")


def v21_reproduction_cells():
    """The reproduction block, executed and checked before any decision seed."""
    for scenario in V21_SCENARIOS:
        for seed in v21_seeds("reproduction"):
            for arm in V21_REPRODUCTION_ARMS:
                yield arm, scenario, seed, V21_PLANT


def v21_decision_cells():
    """The decision block and the descriptive arm, executed after the reproduction check."""
    for scenario in V21_SCENARIOS:
        for seed in v21_seeds("decision"):
            for arm in V21_ARM_LABELS:
                yield arm, scenario, seed, V21_PLANT
            if seed in v21_seeds("descriptive"):
                yield V21_DESCRIPTIVE_ARM, scenario, seed, V21_PLANT


def v21_cells():
    """Every (arm, scenario, seed, plant) in the fixed order the runner executes."""
    yield from v21_reproduction_cells()
    yield from v21_decision_cells()


def arm_stack_and_compensation(arm: str) -> tuple[str, bool]:
    """Map a campaign arm label to the pilot stack name and its delay-compensation flag."""
    if arm.endswith(V21_NO_COMPENSATION_SUFFIX):
        return arm[: -len(V21_NO_COMPENSATION_SUFFIX)], False
    return arm, V21_COMPENSATE_DELAY


def cell_key(episode: PilotEpisode) -> tuple:
    return (episode.arm, episode.scenario, episode.seed, episode.plant)


def _select(episodes, arm, scenario, seeds):
    wanted = set(seeds)
    return [e for e in episodes if e.arm == arm and e.scenario == scenario and e.seed in wanted]


def load_reference(reference_path: Path) -> dict:
    """Retained v2.0 rows of the corner arms on the reproduction seeds, keyed by cell."""
    if not reference_path.exists():
        raise FileNotFoundError(f"reproduction reference missing: {reference_path}")
    wanted = set(v21_seeds("reproduction"))
    reference = {}
    for line in reference_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (record["arm"], record["scenario"], record["seed"])
        if record["arm"] in V21_REPRODUCTION_ARMS and record["seed"] in wanted:
            if key in reference:
                raise ValueError(f"duplicate reference row {key}")
            reference[key] = record
    expected = {
        (arm, scenario, seed)
        for arm in V21_REPRODUCTION_ARMS
        for scenario in V21_SCENARIOS
        for seed in wanted
    }
    if set(reference) != expected:
        raise ValueError("reproduction reference does not hold exactly the 600 expected rows")
    return reference


def _values_match(ours, theirs) -> bool:
    """Recursive equality with the reproduction tolerance on floats."""
    if isinstance(theirs, bool) or isinstance(ours, bool):
        return ours == theirs
    if theirs is None or ours is None:
        return ours is None and theirs is None
    if isinstance(theirs, (int, float)) and isinstance(ours, (int, float)):
        if isinstance(theirs, int) and isinstance(ours, int):
            return ours == theirs
        return _float_matches(float(ours), float(theirs))
    if isinstance(theirs, (list, tuple)) and isinstance(ours, (list, tuple)):
        return len(ours) == len(theirs) and all(
            _values_match(a, b) for a, b in zip(ours, theirs, strict=True)
        )
    return ours == theirs


def reproduction_check(episodes: list[PilotEpisode], reference_path: Path) -> dict:
    """Every field of every retained corner row must match on the reproduction seeds."""
    reference = load_reference(Path(reference_path))
    rows = [
        e
        for arm in V21_REPRODUCTION_ARMS
        for scenario in V21_SCENARIOS
        for e in _select(episodes, arm, scenario, v21_seeds("reproduction"))
    ]
    if {(e.arm, e.scenario, e.seed) for e in rows} != set(reference):
        raise ValueError("reproduction rows do not cover the reference keys")
    fields = sorted(next(iter(reference.values())))
    mismatches = []
    for episode in rows:
        retained = reference[(episode.arm, episode.scenario, episode.seed)]
        ours_record = asdict(episode)
        for name in fields:
            if not _values_match(ours_record[name], retained[name]):
                mismatches.append(
                    [
                        episode.arm,
                        episode.scenario,
                        episode.seed,
                        name,
                        ours_record[name],
                        retained[name],
                    ]
                )
    return {
        "reference": str(reference_path),
        "fields": fields,
        "compared": len(rows),
        "matched": len(rows) - len({(m[0], m[1], m[2]) for m in mismatches}),
        "mismatches": mismatches[:20],
    }


def paired_interaction(cells: dict, scenario: str, rng) -> dict:
    """Seed-paired bootstrap of the difference of certificate effects between controllers."""
    values = {}
    for label in V21_ARM_LABELS:
        values[label] = {e.seed: float(e.success) for e in cells[(label, scenario)]}
    seeds = sorted(values["adaptive_hocbf"])
    if any(sorted(v) != seeds for v in values.values()):
        raise ValueError("interaction needs the four arms on the same seeds")
    per_seed = np.asarray(
        [
            (values["adaptive_hocbf"][s] - values["adaptive_hocbf_fixedmodel"][s])
            - (values["fixed_hocbf_adaptivemodel"][s] - values["fixed_hocbf"][s])
            for s in seeds
        ]
    )
    draws = rng.integers(0, len(seeds), size=(V19_BOOTSTRAP_DRAWS, len(seeds)))
    distribution = per_seed[draws].mean(axis=1)
    return {
        "difference": float(per_seed.mean()),
        "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)),
        "pairs": len(seeds),
    }


def conditional_mean(rows, metric, rng) -> dict:
    """Mean of a metric over unsafe episodes with an unpaired percentile bootstrap."""
    values = np.asarray([float(metric(r)) for r in rows if r.unsafe_episode])
    if values.size == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "episodes": 0}
    draws = rng.integers(0, values.size, size=(V19_BOOTSTRAP_DRAWS, values.size))
    distribution = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)),
        "episodes": int(values.size),
    }


def beyond_tolerance(row: PilotEpisode) -> bool:
    return (
        row.joint_position_violation_max_rad > V21_POSITION_TOLERANCE_RAD
        or row.joint_velocity_violation_max_rad_s > V21_VELOCITY_TOLERANCE_RAD_S
    )


def estimator_validity(cells: dict) -> dict:
    """The v2.0 guard condition, applied to every estimator arm in every scenario."""
    per_cell = {}
    for (label, scenario), rows in cells.items():
        if label not in V21_ESTIMATOR_ARMS:
            continue
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


def analyze(episodes: list[PilotEpisode], reference_path: Path) -> dict:
    """Apply the frozen rule: reproduction, validity, then the three success statements."""
    if [cell_key(e) for e in episodes] != list(v21_cells()):
        raise ValueError("episodes do not match the planned cells exactly, in order")
    reproduction = reproduction_check(episodes, Path(reference_path))
    if reproduction["mismatches"]:
        raise ValueError(
            f"corner arms do not reproduce the retained v2.0 rows: {reproduction['mismatches'][:3]}"
        )
    cells = {
        (label, scenario): _select(episodes, label, scenario, v21_seeds("decision"))
        for label in V21_ARM_LABELS
        for scenario in V21_SCENARIOS
    }
    descriptive = {
        scenario: _select(episodes, V21_DESCRIPTIVE_ARM, scenario, v21_seeds("descriptive"))
        for scenario in V21_SCENARIOS
    }
    rng = np.random.default_rng(V21_BOOTSTRAP_SEED)
    prediction_effect = {
        scenario: {
            name: paired_difference(
                _select(episodes, "adaptive_hocbf", scenario, v21_seeds("descriptive")),
                descriptive[scenario],
                metric,
                rng,
            )
            for name, metric in (
                ("success", lambda r: r.success),
                ("unsafe_episode", lambda r: r.unsafe_episode),
                ("abort", lambda r: r.outcome == "abort"),
            )
        }
        for scenario in V21_SCENARIOS
    }

    def contrast(treatment, reference, scenario, metric):
        return paired_difference(
            cells[(treatment, scenario)], cells[(reference, scenario)], metric, rng
        )

    success = {}
    for scenario in V21_SCENARIOS:
        success[scenario] = {
            "controller_at_fixed_certificate": contrast(
                "adaptive_hocbf_fixedmodel", "fixed_hocbf", scenario, lambda r: r.success
            ),
            "controller_at_identified_certificate": contrast(
                "adaptive_hocbf", "fixed_hocbf_adaptivemodel", scenario, lambda r: r.success
            ),
            "certificate_at_identified_controller": contrast(
                "adaptive_hocbf", "adaptive_hocbf_fixedmodel", scenario, lambda r: r.success
            ),
            "certificate_at_fixed_controller": contrast(
                "fixed_hocbf_adaptivemodel", "fixed_hocbf", scenario, lambda r: r.success
            ),
            "interaction": paired_interaction(cells, scenario, rng),
        }

    paired_metrics = {
        "operational": {
            "abort": lambda r: r.outcome == "abort",
            "timeout": lambda r: r.outcome == "timeout",
        },
        "violation_rate": {
            "unsafe_episode": lambda r: r.unsafe_episode,
            "unsafe_beyond_tolerance": beyond_tolerance,
        },
        "severity_paired": {
            "normalized_violation_max": lambda r: r.normalized_violation_max,
            "joint_position_violation_max_rad": lambda r: r.joint_position_violation_max_rad,
            "joint_velocity_violation_max_rad_s": lambda r: r.joint_velocity_violation_max_rad_s,
            "safety_intervention_fraction": lambda r: r.safety_intervention_fraction,
            "steps": lambda r: r.steps,
        },
    }
    conditional_metrics = {
        "joint_position_violation_max_rad": lambda r: r.joint_position_violation_max_rad,
        "joint_velocity_violation_max_rad_s": lambda r: r.joint_velocity_violation_max_rad_s,
    }
    safety = {}
    for scenario in V21_SCENARIOS:
        families = {
            family: {
                name: contrast("adaptive_hocbf", "adaptive_hocbf_fixedmodel", scenario, metric)
                for name, metric in metrics.items()
            }
            for family, metrics in paired_metrics.items()
        }
        families["severity_conditional"] = {
            label: {
                name: conditional_mean(cells[(label, scenario)], metric, rng)
                for name, metric in conditional_metrics.items()
            }
            for label in ("adaptive_hocbf", "adaptive_hocbf_fixedmodel")
        }
        safety[scenario] = families

    statements = {
        "controller_gain_at_fixed_certificate": all(
            success[s]["controller_at_fixed_certificate"]["ci95_low"] > V21_CONTROLLER_GAIN
            for s in V21_SCENARIOS
        ),
        "certificate_gain_under_mismatch": all(
            success[s]["certificate_at_identified_controller"]["ci95_low"] > V21_CERTIFICATE_GAIN
            for s in V21_MISMATCH_SCENARIOS
        ),
        "certificate_equivalence_in_distribution": (
            success["id_reference"]["certificate_at_identified_controller"]["ci95_low"]
            >= -V21_EQUIVALENCE_MARGIN
            and success["id_reference"]["certificate_at_identified_controller"]["ci95_high"]
            <= V21_EQUIVALENCE_MARGIN
        ),
    }
    validity = estimator_validity(cells)
    if not validity["estimator_valid"]:
        decision = "inconclusive"
    elif all(statements.values()):
        decision = "confirmed"
    else:
        decision = "partial"
    return {
        "decision": decision,
        "statements": statements,
        "failed_statements": [name for name, held in statements.items() if not held],
        "estimator_valid": validity["estimator_valid"],
        "invalid_estimator_cells": validity["invalid_cells"],
        "success": success,
        "safety": safety,
        "estimator": validity["cells"],
        "cell_summary": {
            f"{label}/{scenario}": cell_summary(rows) for (label, scenario), rows in cells.items()
        },
        "descriptive_summary": {
            f"{V21_DESCRIPTIVE_ARM}/{scenario}": cell_summary(rows)
            for scenario, rows in descriptive.items()
        },
        "prediction_effect_within_identified_stack": prediction_effect,
        "reproduction_check": reproduction,
        "protocol": v21_protocol_dict(),
    }


__all__ = [
    "analyze",
    "arm_stack_and_compensation",
    "load_episodes",
    "v21_cells",
    "v21_config",
    "v21_decision_cells",
    "v21_protocol_dict",
    "v21_reproduction_cells",
    "v21_seeds",
]
