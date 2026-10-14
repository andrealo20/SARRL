"""Isolate shared physical parameters in causal Panda delay scoring."""

from __future__ import annotations

from dataclasses import asdict

from sarrl.controllers.panda_shared_scoring import (
    PandaSharedScoreConfig,
    PandaSharedScoreController,
)
from sarrl.evaluation.panda_estimator_memory import evaluate_gate as memory_gate
from sarrl.evaluation.panda_score_memory import run_episode as score_episode
from sarrl.evaluation.panda_variable_delay import CONDITIONS, variable_delay_protocol
from sarrl.evaluation.panda_variable_delay import run_episode as variable_episode
from sarrl.evaluation.panda_variable_delay import summarize as variable_summary

ARMS = ("fixed", "adaptive", "bank_align", "known_align", "bank_exp", "bank_shared_score")
CONFIG = PandaSharedScoreConfig()


def shared_scoring_protocol():
    protocol = variable_delay_protocol(20, 9805600)
    protocol.update(
        kind="exploratory_panda_shared_parameter_scoring",
        arms=list(ARMS),
        score_memory={
            "bank_align": "cumulative",
            "bank_exp": "exponential size 50",
            "bank_shared_score": "exponential size 50, unchanged across acquisition",
        },
        shared_scoring=asdict(CONFIG),
        acquisition=(
            "selected age unchanged for 25 observations, at least 50 updates, information rank 8"
        ),
        physical_scoring_parameters=(
            "copy selected causal estimate once; never refit or reset scores"
        ),
        candidate_control_parameters=(
            "original cumulative bounded RLS, updated in all five candidates"
        ),
        privileges="no applied age, true parameter, delivered command or switch input to banks",
        frozen_controller_reference="results/panda_reference_validation/manifest.json",
        reproduction={
            **protocol["reproduction"],
            "episodes": 4000,
            "reference": "results/panda_reference_validation/episodes.json",
        },
        prerequisite="completed reference-validation gate must pass before new seeds",
        gate={
            "changing_age_gain_over_score_only": 0.20,
            "each_clean_decreasing_cell_age_gain": 0.20,
            "constant_correct_age": "no fewer than either reference per cell",
            "success": "no fewer than either reference per cell",
            "unsafe_episodes": 0,
            "tracking_tolerance_rad": 0.0001,
            "tracking": "each joint paired mean RMS after 1 s and mean final max joint error",
            "tracking_references": ["bank_align", "bank_exp"],
            "meaning": "same v2.4 exploratory screen; fresh confirmation precedes defaults",
        },
    )
    return protocol


def run_episode(seed, scenario, arm, *, retain_trace=False):
    if arm not in ARMS:
        raise ValueError(f"unknown shared-scoring arm: {arm}")
    if arm != "bank_shared_score":
        return score_episode(seed, scenario, arm, retain_trace=retain_trace)
    controller = PandaSharedScoreController(CONFIG)
    row, trace = variable_episode(
        seed, scenario, "bank_align", retain_trace=retain_trace, bank_factory=lambda: controller
    )
    row["arm"] = arm
    if retain_trace:
        for sample, innovations, parameters in zip(
            trace[:-1],
            controller.innovation_history,
            controller.scoring_parameter_history,
            strict=True,
        ):
            sample["score_innovation_squared"] = innovations
            sample["shared_scoring_parameters"] = parameters
    return row, trace


def evaluate_gate(records):
    if any(row["arm"] not in ARMS for row in records):
        raise ValueError("unexpected arm in shared-scoring comparison")
    mapped = [
        {**row, "arm": "bank_joint_memory" if row["arm"] == "bank_shared_score" else row["arm"]}
        for row in records
    ]
    return {**memory_gate(mapped), "candidate_arm": "bank_shared_score"}


def summarize(records):
    summary = variable_summary(records, arms=ARMS)
    for row in records:
        if row["arm"] in ("bank_exp", "bank_shared_score"):
            d = row["delay_estimator"]
            if (
                d["score_memory"] != {"rule": "exponential", "memory_steps": 50}
                or d["score_updates"] != 300
                or d["score_decay"] != 49 / 51
            ):
                raise ValueError("shared comparison must retain score memory")
            if row["arm"] == "bank_shared_score" and d["shared_score_config"] != asdict(CONFIG):
                raise ValueError("shared acquisition differs from the frozen rule")
    for condition in CONDITIONS:
        selected = [
            r for r in records if r["scenario"] == condition and r["arm"] == "bank_shared_score"
        ]
        summary["cells"][f"{condition}/bank_shared_score"]["calibrated_episodes"] = sum(
            r["delay_estimator"]["shared_scoring_active"] for r in selected
        )
        summary["cells"][f"{condition}/bank_shared_score"]["calibration_updates"] = [
            r["delay_estimator"]["calibrated_at_update"] for r in selected
        ]
    summary["exploratory_gate"] = evaluate_gate(records)
    return summary
