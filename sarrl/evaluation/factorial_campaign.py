"""Preregistered v2.1 campaign: which model matters, the controller's or the certificate's.

v2.0 compared two stacks that differed in both places at once: the fixed
nominal certified with the fixed model against the identified nominal
certified with its own estimate. This campaign crosses the two factors on
the v2.0 benchmark (MuJoCo plant, actuator options, measured state) and
fresh paired seeds: controller model in {fixed, identified} by certificate
model in {fixed, identified}, four filtered arms on every seed. The two
corner arms are the v2.0 arms unchanged, and a reproduction block reruns
them on the first hundred v2.0 decision seeds per scenario, where every
retained field must match.

The decision rule tests three preregistered statements about task success
with seed-paired percentile bootstraps: the controller's model carries a
large gain at a fixed certificate in every scenario; the certificate's
model adds a gain under motor fault and compound OOD given the identified
controller; and in distribution the certificate's model makes no difference
beyond an equivalence margin. Safety endpoints (rate and severity of
envelope violations, aborts, intervention) are preregistered with their
expected directions but carry no decision weight: the pilot showed the two
certificates trading rate against severity, and the campaign estimates that
trade rather than adjudicating it.
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

# Arms named controller/certificate; the labels are the pilot's arm names.
V21_ARMS = {
    "fixed/fixed": "fixed_hocbf",
    "fixed/identified": "fixed_hocbf_adaptivemodel",
    "identified/fixed": "adaptive_hocbf_fixedmodel",
    "identified/identified": "adaptive_hocbf",
}
V21_ARM_LABELS = tuple(V21_ARMS.values())
V21_SCENARIOS = ("id_reference", "ood_compound", "motor_fault")
V21_MISMATCH_SCENARIOS = ("motor_fault", "ood_compound")
V21_PLANT = V20_PLANT
V21_PLANT_OPTIONS = V20_PLANT_OPTIONS
V21_COMPENSATE_DELAY = V20_COMPENSATE_DELAY
# Decision seeds: fresh block after the v2.0 block (52200..54099) and a guard band.
V21_DECISION_SEED_START = 54200
V21_DECISION_EPISODES = 1900
# Reproduction block: the first hundred v2.0 decision seeds per scenario, corner arms only.
V21_REPRODUCTION_SEED_START = V20_PRIMARY_SEED_START
V21_REPRODUCTION_EPISODES = 100
V21_REPRODUCTION_ARMS = ("fixed_hocbf", "adaptive_hocbf")
V21_REPRODUCTION_REFERENCE = f"{V20_OUTPUT}/episodes.jsonl"
V21_BOOTSTRAP_SEED = 210_000
# Statements about success, in percentage points of paired difference.
V21_CONTROLLER_GAIN = 0.20
V21_CERTIFICATE_GAIN = 0.10
V21_EQUIVALENCE_MARGIN = 0.03
V21_FROZEN_PATHS = (
    "sarrl",
    "tools",
    "tests",
    "docs/experiments.md",
    "pyproject.toml",
    V21_REPRODUCTION_REFERENCE,
)
V21_SEAL_FILE = "docs/v21_seal.json"
V21_OUTPUT = "results/certificate_factorial_v21"

# Fields of a retained v2.0 row that the reproduction must match; fields added
# to the record after v2.0 are not in the reference and are not compared.
REPRODUCTION_FIELDS = (
    ("outcome", str),
    ("steps", int),
    ("final_distance", float),
    ("reward", float),
    ("max_speed", float),
    ("max_command_torque", float),
    ("fault_seen", bool),
    ("success", bool),
    ("unsafe_episode", bool),
    ("normalized_violation_max", float),
    ("safety_infeasible", bool),
    ("safety_intervention_fraction", float),
    ("true_delay", int),
    ("selected_lag", int),
    ("selected_time_constant", float),
    ("model_fallback_steps", int),
    ("control_steps", int),
)


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
            "controller_model": ["fixed", "identified"],
            "certificate_model": ["fixed", "identified"],
        },
        "arms": dict(V21_ARMS),
        "seeds": {
            "decision": {
                "arms": list(V21_ARM_LABELS),
                "start": V21_DECISION_SEED_START,
                "episodes_per_scenario": V21_DECISION_EPISODES,
                "plant": V21_PLANT,
            },
            "reproduction": {
                "arms": list(V21_REPRODUCTION_ARMS),
                "start": V21_REPRODUCTION_SEED_START,
                "episodes_per_scenario": V21_REPRODUCTION_EPISODES,
                "reference": V21_REPRODUCTION_REFERENCE,
                "rows_must_match_reference": [name for name, _ in REPRODUCTION_FIELDS],
                "decision_weight": False,
            },
            "scenarios": list(V21_SCENARIOS),
            "unopened_range_scanned": [
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
                "minimum_difference": V21_CONTROLLER_GAIN,
                "lower_95_above": 0.0,
            },
            "certificate_gain_under_mismatch": {
                "contrast": "identified/identified minus identified/fixed",
                "scenarios": list(V21_MISMATCH_SCENARIOS),
                "minimum_difference": V21_CERTIFICATE_GAIN,
                "lower_95_above": 0.0,
            },
            "certificate_equivalence_in_distribution": {
                "contrast": "identified/identified minus identified/fixed",
                "scenarios": ["id_reference"],
                "interval_within": [-V21_EQUIVALENCE_MARGIN, V21_EQUIVALENCE_MARGIN],
            },
            "interaction": {
                "contrast": "(identified/identified minus identified/fixed) minus "
                "(fixed/identified minus fixed/fixed)",
                "scenarios": list(V21_SCENARIOS),
                "reported": "estimate and interval, no threshold",
            },
        },
        "safety_endpoints": {
            "contrast": "identified/identified minus identified/fixed, per scenario",
            "metrics": [
                "unsafe_episode",
                "abort",
                "timeout",
                "normalized_violation_max",
                "joint_position_violation_max_rad",
                "joint_velocity_violation_max_rad_s",
                "safety_intervention_fraction",
            ],
            "expected_from_pilot": {
                "unsafe_episode": "more frequent with the identified certificate in "
                "id_reference and ood_compound, less frequent under motor_fault",
                "severity": "smaller position and velocity excesses with the identified "
                "certificate in every scenario",
                "safety_intervention_fraction": "lower with the identified certificate",
            },
            "decision_weight": False,
        },
        "decision": {
            "order": ["estimator_validity", "success_statements", "otherwise"],
            "estimator_validity": {
                "arms_with_estimator": [
                    "fixed_hocbf_adaptivemodel",
                    "adaptive_hocbf_fixedmodel",
                    "adaptive_hocbf",
                ],
                "guarded_step_fraction_per_episode": V20_GUARDED_STEP_FRACTION,
                "max_guarded_episode_fraction": V20_GUARDED_EPISODE_FRACTION,
                "failure_outcome": "inconclusive",
            },
            "outcomes": {
                "confirmed": "all three success statements hold",
                "partial": "at least one success statement fails; each is reported",
                "inconclusive": "estimator validity fails",
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
            "compensation_applies_to": "identified controller only, as in v2.0",
        },
    }


def v21_seeds(block: str) -> range:
    if block == "decision":
        return range(V21_DECISION_SEED_START, V21_DECISION_SEED_START + V21_DECISION_EPISODES)
    if block == "reproduction":
        return range(
            V21_REPRODUCTION_SEED_START,
            V21_REPRODUCTION_SEED_START + V21_REPRODUCTION_EPISODES,
        )
    raise ValueError(f"unknown seed block {block}")


def v21_cells():
    """Every (arm, scenario, seed, plant) in the fixed order the runner executes."""
    for scenario in V21_SCENARIOS:
        for seed in v21_seeds("reproduction"):
            for arm in V21_REPRODUCTION_ARMS:
                yield arm, scenario, seed, V21_PLANT
        for seed in v21_seeds("decision"):
            for arm in V21_ARM_LABELS:
                yield arm, scenario, seed, V21_PLANT


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


def reproduction_check(episodes: list[PilotEpisode], reference_path: Path) -> dict:
    """Every retained field of the corner arms must match on the reproduction seeds."""
    reference = load_reference(Path(reference_path))
    rows = [
        e
        for arm in V21_REPRODUCTION_ARMS
        for scenario in V21_SCENARIOS
        for e in _select(episodes, arm, scenario, v21_seeds("reproduction"))
    ]
    if {(e.arm, e.scenario, e.seed) for e in rows} != set(reference):
        raise ValueError("reproduction rows do not cover the reference keys")
    mismatches = []
    for episode in rows:
        retained = reference[(episode.arm, episode.scenario, episode.seed)]
        for name, kind in REPRODUCTION_FIELDS:
            ours = getattr(episode, name)
            theirs = retained[name]
            if ours is None or theirs is None:
                same = ours is theirs
            elif kind is float:
                same = _float_matches(float(ours), float(theirs))
            else:
                same = ours == theirs
            if not same:
                mismatches.append([episode.arm, episode.scenario, episode.seed, name, ours, theirs])
    return {
        "reference": str(reference_path),
        "fields": [name for name, _ in REPRODUCTION_FIELDS],
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


def estimator_validity(cells: dict) -> dict:
    rows = [e for (label, _), block in cells.items() if label != "fixed_hocbf" for e in block]
    guarded = [
        e
        for e in rows
        if e.control_steps > 0
        and e.model_fallback_steps / e.control_steps > V20_GUARDED_STEP_FRACTION
    ]
    fraction = len(guarded) / len(rows)
    identified = [e for e in rows if e.arm == "adaptive_hocbf"]
    return {
        "guarded_episode_fraction": fraction,
        "estimator_valid": fraction <= V20_GUARDED_EPISODE_FRACTION,
        "episodes_with_estimator": len(rows),
        "lag_identified_fraction_identified_arm": float(
            np.mean([e.selected_lag == e.true_delay for e in identified])
        ),
        "time_constant_abs_error_mean_s_identified_arm": float(
            np.mean([abs(e.selected_time_constant - e.true_time_constant) for e in identified])
        ),
    }


def analyze(episodes: list[PilotEpisode], reference_path: Path) -> dict:
    """Apply the frozen rule: validity first, then the three success statements."""
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
    rng = np.random.default_rng(V21_BOOTSTRAP_SEED)

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

    safety_metrics = {
        "unsafe_episode": lambda r: r.unsafe_episode,
        "abort": lambda r: r.outcome == "abort",
        "timeout": lambda r: r.outcome == "timeout",
        "normalized_violation_max": lambda r: r.normalized_violation_max,
        "joint_position_violation_max_rad": lambda r: r.joint_position_violation_max_rad,
        "joint_velocity_violation_max_rad_s": lambda r: r.joint_velocity_violation_max_rad_s,
        "safety_intervention_fraction": lambda r: r.safety_intervention_fraction,
    }
    safety = {
        scenario: {
            name: contrast("adaptive_hocbf", "adaptive_hocbf_fixedmodel", scenario, metric)
            for name, metric in safety_metrics.items()
        }
        for scenario in V21_SCENARIOS
    }

    statements = {
        "controller_gain_at_fixed_certificate": all(
            success[s]["controller_at_fixed_certificate"]["difference"] >= V21_CONTROLLER_GAIN
            and success[s]["controller_at_fixed_certificate"]["ci95_low"] > 0.0
            for s in V21_SCENARIOS
        ),
        "certificate_gain_under_mismatch": all(
            success[s]["certificate_at_identified_controller"]["difference"] >= V21_CERTIFICATE_GAIN
            and success[s]["certificate_at_identified_controller"]["ci95_low"] > 0.0
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
        "success": success,
        "safety": safety,
        "estimator": validity,
        "cell_summary": {
            f"{label}/{scenario}": cell_summary(rows) for (label, scenario), rows in cells.items()
        },
        "reproduction_check": reproduction,
        "protocol": v21_protocol_dict(),
    }


__all__ = [
    "analyze",
    "load_episodes",
    "v21_cells",
    "v21_config",
    "v21_protocol_dict",
    "v21_seeds",
]
