"""Paired comparison of cumulative, exponential and windowed delay scores."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from sarrl.controllers.panda_score_memory import PandaScoreMemoryConfig, PandaScoreMemoryController
from sarrl.evaluation.panda_variable_delay import ARMS as REFERENCE_ARMS
from sarrl.evaluation.panda_variable_delay import CONDITIONS, variable_delay_protocol
from sarrl.evaluation.panda_variable_delay import run_episode as variable_episode
from sarrl.evaluation.panda_variable_delay import summarize as variable_summary

MEMORY_CONFIGS = {
    "bank_exp": PandaScoreMemoryConfig("exponential", 50),
    "bank_window": PandaScoreMemoryConfig("window", 50),
}
ARMS = (*REFERENCE_ARMS, *MEMORY_CONFIGS)


def score_memory_protocol(episodes=20, seed_start=9805300):
    protocol = variable_delay_protocol(episodes, seed_start)
    protocol.update(
        kind="exploratory_panda_score_memory_pilot",
        arms=list(ARMS),
        score_memory={a: asdict(c) for a, c in MEMORY_CONFIGS.items()},
        score_inputs="pre-update squared innovation; no age, switch or arrival metadata",
        score_exponential_decay=49 / 51,
        score_window="sum of last 50 transitions, including the current transition",
        score_pairing="50-interval window and exponential effective sample size 50",
        estimator="original cumulative candidate RLS unchanged; no parameter forgetting",
        frozen_controller_reference="results/panda_variable_delay_pilot/manifest.json",
        reproduction={
            **protocol["reproduction"],
            "episodes": 1280,
            "reference": "results/panda_variable_delay_pilot/episodes.json",
        },
    )
    return protocol


def run_episode(seed, scenario, arm, *, retain_trace=False):
    if arm not in ARMS:
        raise ValueError(f"unknown score-memory arm: {arm}")
    if arm not in MEMORY_CONFIGS:
        return variable_episode(seed, scenario, arm, retain_trace=retain_trace)
    controller = PandaScoreMemoryController(MEMORY_CONFIGS[arm])
    row, trace = variable_episode(
        seed,
        scenario,
        "bank_align",
        retain_trace=retain_trace,
        bank_factory=lambda: controller,
    )
    row["arm"] = arm
    if retain_trace:
        for sample, innovation in zip(trace[:-1], controller.innovation_history, strict=True):
            sample["score_innovation_squared"] = innovation
    return row, trace


def summarize(records):
    summary = variable_summary(records, arms=ARMS)
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in records}
    seeds = sorted({r["seed"] for r in records})
    for row in records:
        arm = row["arm"]
        if arm in MEMORY_CONFIGS:
            diag = row["delay_estimator"]
            cfg = MEMORY_CONFIGS[arm]
            expected_decay = cfg.decay if cfg.rule == "exponential" else None
            if (
                diag["score_memory"] != asdict(cfg)
                or diag["score_updates"] != 300
                or diag["score_decay"] != expected_decay
            ):
                raise ValueError("incorrect score-memory configuration or update count")
    for condition in CONDITIONS:
        for arm in ("bank_align", *MEMORY_CONFIGS):
            rows = [lookup[condition, seed, arm] for seed in seeds]
            cell = summary["cells"][f"{condition}/{arm}"]
            cell["mean_selection_switches"] = float(
                np.mean([r["delay_estimator"]["selection_switches"] for r in rows])
            )
            cell["mean_after_selection_switches"] = float(
                np.mean(
                    [
                        np.count_nonzero(np.diff(r["delay_estimator"]["selected_history"][99:]))
                        for r in rows
                    ]
                )
            )
        for treatment in MEMORY_CONFIGS:
            pairs = [
                (lookup[condition, seed, treatment], lookup[condition, seed, "bank_align"])
                for seed in seeds
            ]
            summary["paired_contrasts"][f"{condition}/{treatment}-bank_align"] = {
                "success_count_difference": sum(a["success"] - b["success"] for a, b in pairs),
                "final_delay_correct_count_difference": sum(
                    a["delay_estimator"]["final_delay_correct"]
                    - b["delay_estimator"]["final_delay_correct"]
                    for a, b in pairs
                ),
                "mean_paired_final_joint_error_difference_rad": float(
                    np.mean([a["joint_error_max_rad"] - b["joint_error_max_rad"] for a, b in pairs])
                ),
                "mean_paired_after_tracking_rms_difference_rad": float(
                    np.mean(
                        [
                            np.mean(a["tracking_phase_rms_rad"]["after"])
                            - np.mean(b["tracking_phase_rms_rad"]["after"])
                            for a, b in pairs
                        ]
                    )
                ),
            }
    return summary
