"""Separate command alignment, delay identification and prediction on paired Panda tasks."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from sarrl.controllers.panda_delay import PandaDelayConfig, PandaDelayController
from sarrl.evaluation.panda_adaptation import run_episode as adaptation_episode
from sarrl.evaluation.panda_adaptation import summarize as adaptation_summary
from sarrl.evaluation.panda_robustness import CONDITIONS, condition_config, robustness_protocol

ARMS = ("fixed", "adaptive", "known_align", "bank_align", "known_predict", "bank_predict", "oracle")
CONTRASTS = (
    ("known_align", "adaptive"),
    ("bank_align", "adaptive"),
    ("bank_align", "known_align"),
    ("known_predict", "known_align"),
    ("bank_predict", "bank_align"),
    ("bank_predict", "known_predict"),
)


def delay_protocol(episodes=20, seed_start=9804700):
    protocol = robustness_protocol(episodes, seed_start)
    protocol.update(
        kind="exploratory_panda_delay_identification_pilot",
        arms=list(ARMS),
        delay_model=asdict(PandaDelayConfig()),
        estimator_inputs=(
            "sent-command history and cached measured pre/post q, qd; "
            "lag-indexed targets for the delay arms"
        ),
        selection=(
            "minimum accumulated squared seven-joint pre-update innovation; "
            "ties use the lowest delay"
        ),
        prediction=(
            "RK4 on the selected estimated dynamics, through queued sent commands; "
            "reference advanced by the same horizon"
        ),
        privileged_delay_arms=["known_align", "known_predict"],
        oracle="physical parameters known, no delay knowledge or compensation",
        reproduction={
            **protocol["reproduction"],
            "reference": "results/panda_robustness_pilot/episodes.json",
        },
    )
    return protocol


def run_episode(seed, scenario, arm, *, retain_trace=False):
    if arm not in ARMS:
        raise ValueError(f"unknown Panda delay arm: {arm}")
    config = condition_config(scenario)
    privileged_delay = arm in ("known_align", "known_predict")
    controller = None
    base_arm = arm
    if arm in ("known_align", "bank_align", "known_predict", "bank_predict"):
        controller = PandaDelayController(
            known_delay_steps=config.command_delay_steps if privileged_delay else None,
            predict=arm.endswith("predict"),
        )
        base_arm = "adaptive"
    record, trace = adaptation_episode(
        seed,
        "payload_friction",
        base_arm,
        retain_trace=retain_trace,
        interface_config=config,
        controller_override=controller,
    )
    record.update(
        scenario=scenario,
        arm=arm,
        plant_scenario="payload_friction",
        privileged_delay=privileged_delay,
    )
    if controller is not None:
        selected = controller.selected_history
        truth = config.command_delay_steps
        incorrect = [i for i, lag in enumerate(selected) if lag != truth]
        record["delay_estimator"].update(
            final_delay_correct=controller.selected_delay_steps == truth,
            correct_selection_fraction=sum(lag == truth for lag in selected) / len(selected),
            final_correct_run_start_s=(
                (incorrect[-1] + 2) * controller.delay_config.control_dt
                if incorrect
                else controller.delay_config.control_dt
            )
            if selected[-1] == truth
            else None,
        )
    return record, trace


def summarize(records):
    summary = adaptation_summary(records, scenarios=CONDITIONS, arms=ARMS)
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in records}
    seeds = sorted({r["seed"] for r in records})
    for seed in seeds:
        reference = lookup["ideal", seed, "fixed"]
        for condition in CONDITIONS:
            noise_hashes = []
            for arm in ARMS:
                row = lookup[condition, seed, arm]
                for field in ("start_q", "goal_q", "target_xyz_m", "true_parameters"):
                    if row[field] != reference[field]:
                        raise ValueError(f"unpaired {field} across delay comparisons")
                if row["interface_config"] != asdict(condition_config(condition)):
                    raise ValueError("incorrect interface condition")
                if row["privileged_delay"] != (arm in ("known_align", "known_predict")):
                    raise ValueError("incorrect privileged-delay label")
                noise_hashes.append(row["measurement_noise_sha256"])
            if len(set(noise_hashes)) != 1:
                raise ValueError("unpaired sensor noise across delay arms")
        if (
            lookup["noise", seed, "fixed"]["measurement_noise_sha256"]
            != lookup["noise_delay", seed, "fixed"]["measurement_noise_sha256"]
        ):
            raise ValueError("unpaired sensor noise across conditions")
    summary["delay_contrasts"] = {}
    for condition in CONDITIONS:
        for arm in ARMS:
            rows = [lookup[condition, seed, arm] for seed in seeds]
            cell = summary["cells"][f"{condition}/{arm}"]
            if rows[0].get("delay_estimator") is not None:
                cell.update(
                    final_delay_correct_episodes=sum(
                        r["delay_estimator"]["final_delay_correct"] for r in rows
                    ),
                    mean_correct_selection_fraction=float(
                        np.mean([r["delay_estimator"]["correct_selection_fraction"] for r in rows])
                    ),
                    prediction_fallbacks=sum(
                        r["delay_estimator"]["prediction_fallbacks"] for r in rows
                    ),
                    prediction_virtual_substeps=sum(
                        r["delay_estimator"]["prediction_virtual_substeps"] for r in rows
                    ),
                )
            if rows[0]["estimator"] is not None:
                cell["mean_bound_hit_fraction"] = float(
                    np.mean(
                        [r["estimator"]["bound_hits"] / r["estimator"]["updates"] for r in rows]
                    )
                )
        for treatment, baseline in CONTRASTS:
            treatment_rows = [lookup[condition, seed, treatment] for seed in seeds]
            baseline_rows = [lookup[condition, seed, baseline] for seed in seeds]
            summary["delay_contrasts"][f"{condition}/{treatment}-{baseline}"] = {
                "success_count_difference": sum(r["success"] for r in treatment_rows)
                - sum(r["success"] for r in baseline_rows),
                "mean_paired_final_joint_error_difference_rad": float(
                    np.mean(
                        [
                            a["joint_error_max_rad"] - b["joint_error_max_rad"]
                            for a, b in zip(treatment_rows, baseline_rows, strict=True)
                        ]
                    )
                ),
                "mean_paired_tracking_rms_difference_rad": float(
                    np.mean(
                        [
                            np.mean(a["tracking_rms_rad"]) - np.mean(b["tracking_rms_rad"])
                            for a, b in zip(treatment_rows, baseline_rows, strict=True)
                        ]
                    )
                ),
            }
    summary["prediction_virtual_substeps"] = sum(
        r.get("delay_estimator", {}).get("prediction_virtual_substeps", 0) for r in records
    )
    return summary
