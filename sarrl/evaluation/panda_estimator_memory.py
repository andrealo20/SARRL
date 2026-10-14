"""Paired Panda comparison with fixed score memory and different parameter memory."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from sarrl.controllers.panda_estimator_memory import (
    PandaEstimatorMemoryConfig,
    PandaJointMemoryController,
)
from sarrl.evaluation.panda_score_memory import run_episode as score_episode
from sarrl.evaluation.panda_variable_delay import ARMS as REFERENCE_ARMS
from sarrl.evaluation.panda_variable_delay import CONDITIONS, variable_delay_protocol
from sarrl.evaluation.panda_variable_delay import run_episode as variable_episode
from sarrl.evaluation.panda_variable_delay import summarize as variable_summary

ARMS = (*REFERENCE_ARMS, "bank_exp", "bank_joint_memory")
CONFIG = PandaEstimatorMemoryConfig(50)
TRACKING_TOLERANCE_RAD = 0.0001
MINIMUM_AGE_GAIN = 0.20


def estimator_memory_protocol(episodes=20, seed_start=9805400):
    protocol = variable_delay_protocol(episodes, seed_start)
    protocol.update(
        kind="exploratory_panda_estimator_memory_pilot",
        arms=list(ARMS),
        score_memory={
            "bank_align": "cumulative",
            "bank_exp": {"rule": "exponential", "memory_steps": 50},
            "bank_joint_memory": {"rule": "exponential", "memory_steps": 50},
        },
        candidate_parameter_memory={
            "bank_align": "cumulative",
            "bank_exp": "cumulative",
            "bank_joint_memory": asdict(CONFIG),
        },
        candidate_parameter_decay=CONFIG.decay,
        forgetting_clock="once per observed control interval, before all seven RLS rows",
        covariance="inverse information; no calibrated uncertainty or uniform upper bound",
        estimator="same bounded RLS priors, row updates and parameter clipping",
        selection="same pre-update innovation; no applied age, switch or true parameter input",
        frozen_controller_reference="results/panda_score_memory_pilot/manifest.json",
        reproduction={
            **protocol["reproduction"],
            "episodes": 1920,
            "reference": "results/panda_score_memory_pilot/episodes.json",
        },
        exploratory_gate={
            "changing_final_age_gain_over_score_only": MINIMUM_AGE_GAIN,
            "each_clean_decreasing_cell_age_gain_over_score_only": MINIMUM_AGE_GAIN,
            "constant_correct_age": "no fewer correct final ages than either reference per cell",
            "success": "no fewer successes than either reference per cell",
            "unsafe_episodes": 0,
            "tracking_tolerance_rad": TRACKING_TOLERANCE_RAD,
            "tracking": "each joint mean RMS after 1 s and mean final max joint error per cell",
            "tracking_comparisons": ["bank_align", "bank_exp"],
            "meaning": (
                "all criteria must pass to advance to separate confirmation; no default change"
            ),
        },
    )
    return protocol


def run_episode(seed, scenario, arm, *, retain_trace=False):
    if arm not in ARMS:
        raise ValueError(f"unknown estimator-memory arm: {arm}")
    if arm != "bank_joint_memory":
        return score_episode(seed, scenario, arm, retain_trace=retain_trace)
    controller = PandaJointMemoryController(CONFIG)
    row, trace = variable_episode(
        seed, scenario, "bank_align", retain_trace=retain_trace, bank_factory=lambda: controller
    )
    row["arm"] = arm
    if retain_trace:
        for sample, innovation in zip(trace[:-1], controller.innovation_history, strict=True):
            sample["score_innovation_squared"] = innovation
    return row, trace


def evaluate_gate(records):
    for row in records:
        rms = np.asarray(row["tracking_phase_rms_rad"]["after"], dtype=float)
        final = row["joint_error_max_rad"]
        if (
            rms.shape != (7,)
            or not np.all(np.isfinite(rms))
            or np.any(rms < 0)
            or not np.isfinite(final)
            or final < 0
        ):
            raise ValueError("gate requires finite nonnegative tracking metrics")
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in records}
    seeds = sorted({r["seed"] for r in records})
    expected = {(c, seed, arm) for c in CONDITIONS for seed in seeds for arm in ARMS}
    if not seeds or len(lookup) != len(records) or set(lookup) != expected:
        raise ValueError("gate requires a complete paired factorial without duplicate cells")
    failures = []
    for condition in CONDITIONS:
        target = [lookup[condition, seed, "bank_joint_memory"] for seed in seeds]
        if any(row["unsafe_episode"] for row in target):
            failures.append({"condition": condition, "criterion": "unsafe_episode"})
        for reference in ("bank_align", "bank_exp"):
            base = [lookup[condition, seed, reference] for seed in seeds]
            if sum(row["success"] for row in target) < sum(row["success"] for row in base):
                failures.append(
                    {"condition": condition, "reference": reference, "criterion": "success_count"}
                )
            if condition.endswith(("fixed10", "fixed40")):
                if sum(row["delay_estimator"]["final_delay_correct"] for row in target) < sum(
                    row["delay_estimator"]["final_delay_correct"] for row in base
                ):
                    failures.append(
                        {
                            "condition": condition,
                            "reference": reference,
                            "criterion": "constant_final_age_count",
                        }
                    )
            for field in ("after_tracking_rms_rad", "joint_error_max_rad"):

                def metric(row, field=field):
                    return (
                        np.asarray(row["tracking_phase_rms_rad"]["after"])
                        if field == "after_tracking_rms_rad"
                        else row[field]
                    )

                differences = np.atleast_1d(
                    np.mean(
                        [metric(a) - metric(b) for a, b in zip(target, base, strict=True)], axis=0
                    )
                )
                for index, difference in enumerate(differences):
                    if difference > TRACKING_TOLERANCE_RAD:
                        failure = {
                            "condition": condition,
                            "reference": reference,
                            "criterion": field,
                            "difference_rad": float(difference),
                        }
                        if field == "after_tracking_rms_rad":
                            failure["joint"] = index + 1
                        failures.append(failure)
    gains = {}
    for label, conditions in (
        ("all_changing", [c for c in CONDITIONS if c.endswith(("_up", "_down"))]),
        ("cold_clean_down", ["cold_clean_down"]),
        ("warm_clean_down", ["warm_clean_down"]),
    ):
        pairs = [
            (lookup[c, seed, "bank_joint_memory"], lookup[c, seed, "bank_exp"])
            for c in conditions
            for seed in seeds
        ]
        gain = sum(
            a["delay_estimator"]["final_delay_correct"]
            - b["delay_estimator"]["final_delay_correct"]
            for a, b in pairs
        ) / len(pairs)
        gains[label] = gain
        if gain < MINIMUM_AGE_GAIN:
            failures.append({"condition": label, "criterion": "final_age_gain", "gain": gain})
    return {
        "candidate_advances": not failures,
        "reference_controller": "bank_align",
        "final_age_gains_over_score_only": gains,
        "failures": failures,
        "interpretation": "exploratory screen; independent confirmation precedes default changes",
    }


def summarize(records):
    for row in records:
        if row["arm"] in ("bank_exp", "bank_joint_memory"):
            diagnostic = row["delay_estimator"]
            if (
                diagnostic["score_memory"] != {"rule": "exponential", "memory_steps": 50}
                or diagnostic["score_decay"] != CONFIG.decay
                or diagnostic["score_updates"] != 300
            ):
                raise ValueError("score memory must remain fixed across the comparison")
            if row["arm"] == "bank_joint_memory" and (
                diagnostic["parameter_memory"] != asdict(CONFIG)
                or diagnostic["parameter_decay"] != CONFIG.decay
            ):
                raise ValueError("candidate parameter memory differs from the frozen configuration")
    summary = variable_summary(records, arms=ARMS)
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in records}
    seeds = sorted({r["seed"] for r in records})
    for condition in CONDITIONS:
        for reference in ("bank_align", "bank_exp"):
            pairs = [
                (lookup[condition, seed, "bank_joint_memory"], lookup[condition, seed, reference])
                for seed in seeds
            ]
            summary["paired_contrasts"][f"{condition}/bank_joint_memory-{reference}"] = {
                "success_count_difference": sum(a["success"] - b["success"] for a, b in pairs),
                "final_age_count_difference": sum(
                    a["delay_estimator"]["final_delay_correct"]
                    - b["delay_estimator"]["final_delay_correct"]
                    for a, b in pairs
                ),
                "mean_after_tracking_rms_difference_rad": float(
                    np.mean(
                        [
                            np.mean(a["tracking_phase_rms_rad"]["after"])
                            - np.mean(b["tracking_phase_rms_rad"]["after"])
                            for a, b in pairs
                        ]
                    )
                ),
            }
    summary["exploratory_gate"] = evaluate_gate(records)
    return summary
