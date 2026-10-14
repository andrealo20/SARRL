"""Fresh paired validation of the frozen constant-delay alignment reference."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from sarrl.evaluation.panda_adaptation import summarize as base_summary
from sarrl.evaluation.panda_posture_delay import (
    CONDITION_PARTS,
    CONDITIONS,
    condition_config,
    posture_delay_protocol,
)
from sarrl.evaluation.panda_posture_delay import run_episode as grid_episode

ARMS = ("fixed", "adaptive", "bank_align", "known_align")
SEEDS = tuple(range(9805500, 9805550))
TRACKING_TOLERANCE_RAD = 0.0001


def reference_protocol():
    protocol = posture_delay_protocol(len(SEEDS), SEEDS[0])
    protocol.update(
        kind="panda_constant_delay_reference_validation",
        arms=list(ARMS),
        prediction="disabled in all arms",
        privileged_delay_arms=["known_align"],
        oracle="known_align receives constant age; no arm receives true parameters",
        reproduction={
            **protocol["reproduction"],
            "episodes": 1920,
            "reference": "results/panda_estimator_memory_pilot/episodes.json",
        },
        frozen_controller_reference="results/panda_estimator_memory_pilot/manifest.json",
        gate={
            "aggregate_success_fraction": 0.98,
            "aggregate_final_age_fraction": 0.98,
            "each_cell_success_fraction": 0.95,
            "each_cell_final_age_fraction": 0.95,
            "unsafe_episodes": 0,
            "tracking_tolerance_rad": TRACKING_TOLERANCE_RAD,
            "tracking_reference": "known_align",
            "tracking": "each joint mean full-episode RMS and mean final max joint error",
            "meaning": "engineering acceptance in the declared grid; no safety certificate",
        },
        uncertainty={
            "unit": "paired episode seed across all twenty conditions",
            "draws": 5000,
            "analysis_rng_seed": 2305,
            "interval": "two-sided 95% percentile bootstrap of mean paired differences",
            "scope": (
                "conditional on sampled tasks; no success-probability bound from zero failures"
            ),
        },
    )
    return protocol


def run_episode(seed, scenario, arm, *, retain_trace=False):
    if arm not in ARMS:
        raise ValueError(f"unknown reference-validation arm: {arm}")
    return grid_episode(seed, scenario, arm, retain_trace=retain_trace)


def evaluate_gate(records):
    seeds = sorted({r["seed"] for r in records})
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in records}
    expected = {(c, seed, arm) for c in CONDITIONS for seed in seeds for arm in ARMS}
    if not seeds or len(lookup) != len(records) or set(lookup) != expected:
        raise ValueError("validation needs a complete paired factorial without duplicate cells")
    failures = []
    targets = [r for r in records if r["arm"] == "bank_align"]
    for field, threshold in (("success", 0.98), ("final_delay_correct", 0.98)):
        count = sum(
            r[field] if field == "success" else r["delay_estimator"][field] for r in targets
        )
        if count / len(targets) < threshold:
            failures.append({"condition": "all", "criterion": field, "count": count})
    for condition in CONDITIONS:
        target = [lookup[condition, seed, "bank_align"] for seed in seeds]
        base = [lookup[condition, seed, "known_align"] for seed in seeds]
        if any(r["unsafe_episode"] for r in target):
            failures.append({"condition": condition, "criterion": "unsafe_episode"})
        for field in ("success", "final_delay_correct"):
            count = sum(
                r[field] if field == "success" else r["delay_estimator"][field] for r in target
            )
            if count / len(seeds) < 0.95:
                failures.append({"condition": condition, "criterion": field, "count": count})
        for field in ("tracking_rms_rad", "joint_error_max_rad"):
            values = np.array([r[field] for r in target], dtype=float)
            baseline = np.array([r[field] for r in base], dtype=float)
            if (
                values.shape != baseline.shape
                or values.shape
                != ((len(seeds), 7) if field == "tracking_rms_rad" else (len(seeds),))
                or not np.all(np.isfinite([values, baseline]))
                or np.any(values < 0)
                or np.any(baseline < 0)
            ):
                raise ValueError("validation requires finite nonnegative tracking metrics")
            difference = np.atleast_1d(np.mean(values - baseline, axis=0))
            for joint, increase in enumerate(difference):
                if increase > TRACKING_TOLERANCE_RAD:
                    failure = {
                        "condition": condition,
                        "criterion": field,
                        "difference_rad": float(increase),
                    }
                    if field == "tracking_rms_rad":
                        failure["joint"] = joint + 1
                    failures.append(failure)
    return {"reference_passes": not failures, "failures": failures, "controller": "bank_align"}


def paired_bootstrap(records):
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in records}
    seeds = sorted({r["seed"] for r in records})
    draws = np.random.default_rng(2305).integers(0, len(seeds), (5000, len(seeds)))
    contrasts = {}
    for reference in ("adaptive", "known_align"):
        for field in ("tracking_rms_rad", "joint_error_max_rad"):
            per_seed = np.array(
                [
                    np.mean(
                        [
                            np.mean(lookup[c, seed, "bank_align"][field])
                            - np.mean(lookup[c, seed, reference][field])
                            for c in CONDITIONS
                        ]
                    )
                    for seed in seeds
                ]
            )
            means = np.mean(per_seed[draws], axis=1)
            contrasts[f"bank_align-{reference}/{field}"] = {
                "mean_difference_rad": float(np.mean(per_seed)),
                "percentile_95_interval_rad": np.quantile(means, [0.025, 0.975]).tolist(),
                "paired_seed_count": len(seeds),
            }
    return contrasts


def summarize(records):
    gate = evaluate_gate(records)
    summary = base_summary(records, scenarios=CONDITIONS, arms=ARMS)
    for seed in sorted({r["seed"] for r in records}):
        paired = [r for r in records if r["seed"] == seed]
        if len({str(r["true_parameters"]) for r in paired}) != 1:
            raise ValueError("unpaired physical parameter draws")
        for noise in ("clean", "noise"):
            if (
                len(
                    {r["measurement_noise_sha256"] for r in paired if f"_{noise}_" in r["scenario"]}
                )
                != 1
            ):
                raise ValueError("unpaired exogenous noise")
        for posture in ("narrow", "wide"):
            pose_rows = [r for r in paired if r["posture"] == posture]
            for field in ("start_q", "goal_q", "target_xyz_m", "task_sampling"):
                if any(r[field] != pose_rows[0][field] for r in pose_rows):
                    raise ValueError(f"unpaired {field}")
    for row in records:
        if (
            row["interface_config"] != asdict(condition_config(row["scenario"]))
            or row["posture"] != CONDITION_PARTS[row["scenario"]][0]
            or row["privileged_delay"] != (row["arm"] == "known_align")
            or row["privileged_parameters"]
        ):
            raise ValueError("invalid interface or privilege label")
    for condition in CONDITIONS:
        for arm in ("bank_align", "known_align"):
            group = [r for r in records if r["scenario"] == condition and r["arm"] == arm]
            summary["cells"][f"{condition}/{arm}"]["final_delay_correct_episodes"] = sum(
                r["delay_estimator"]["final_delay_correct"] for r in group
            )
    summary["reference_gate"] = gate
    summary["paired_seed_bootstrap"] = paired_bootstrap(records)
    return summary
