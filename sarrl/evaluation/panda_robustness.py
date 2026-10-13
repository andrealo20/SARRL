"""Descriptive noise-by-command-delay comparison with unchanged Panda controllers."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from sarrl.envs.panda_interface import NOISE_SEED_MASK, PandaInterfaceConfig
from sarrl.evaluation.panda_adaptation import ARMS, pilot_protocol
from sarrl.evaluation.panda_adaptation import run_episode as adaptation_episode
from sarrl.evaluation.panda_adaptation import summarize as adaptation_summary

CONDITIONS = ("ideal", "noise", "delay", "noise_delay")


def condition_config(condition: str) -> PandaInterfaceConfig:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown Panda interface condition: {condition}")
    noise = condition in ("noise", "noise_delay")
    return PandaInterfaceConfig(
        position_noise_std_rad=0.001 if noise else 0.0,
        velocity_noise_std_rad_s=0.001 if noise else 0.0,
        command_delay_steps=2 if condition in ("delay", "noise_delay") else 0,
    )


def robustness_protocol(episodes=20, seed_start=9804500):
    protocol = pilot_protocol(episodes, seed_start)
    protocol.update(
        kind="exploratory_panda_noise_delay_pilot",
        scenarios=list(CONDITIONS),
        plant_scenario="payload_friction",
        interfaces={c: asdict(condition_config(c)) for c in CONDITIONS},
        state_interface="one cached q/qd sample per control boundary, shared by control and RLS",
        noise_rng=(
            f"default_rng(episode_seed ^ {NOISE_SEED_MASK}), "
            "fourteen independent normals per boundary"
        ),
        actuation="sent command clipped, then pure FIFO delay; queue starts with zeros",
        estimator_inputs="current sent command and cached measured pre/post q, qd; no delay model",
        oracle="true inertial/friction parameters only; same measured state and delayed actuation",
        reproduction={
            **protocol["reproduction"],
            "episodes": 240,
            "reference": "results/panda_adaptation_pilot/episodes.json",
        },
    )
    return protocol


def run_episode(seed, scenario, arm, *, retain_trace=False):
    record, trace = adaptation_episode(
        seed,
        "payload_friction",
        arm,
        retain_trace=retain_trace,
        interface_config=condition_config(scenario),
    )
    record.update(scenario=scenario, plant_scenario="payload_friction")
    return record, trace


def summarize(records):
    summary = adaptation_summary(records, scenarios=CONDITIONS)
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in records}
    seeds = sorted({r["seed"] for r in records})
    for seed in seeds:
        reference = lookup["ideal", seed, "fixed"]
        for condition in CONDITIONS:
            for arm in ARMS:
                row = lookup[condition, seed, arm]
                for field in ("start_q", "goal_q", "target_xyz_m", "true_parameters"):
                    if row[field] != reference[field]:
                        raise ValueError(f"unpaired {field} across interface conditions")
                if row["interface_config"] != asdict(condition_config(condition)):
                    raise ValueError(
                        "recorded interface configuration does not match its condition"
                    )
            hashes = [lookup[condition, seed, arm]["measurement_noise_sha256"] for arm in ARMS]
            if len(set(hashes)) != 1:
                raise ValueError("controllers did not receive matched exogenous sensor noise")
        if (
            lookup["noise", seed, "fixed"]["measurement_noise_sha256"]
            != lookup["noise_delay", seed, "fixed"]["measurement_noise_sha256"]
        ):
            raise ValueError("noisy conditions did not receive matched exogenous sensor noise")
    summary["interface_minus_ideal"] = {}
    for condition in CONDITIONS:
        for arm in ARMS:
            rows = [lookup[condition, seed, arm] for seed in seeds]
            reference = [lookup["ideal", seed, arm] for seed in seeds]
            cell = summary["cells"][f"{condition}/{arm}"]
            cell["mean_sent_delivered_rms_nm"] = float(
                np.mean([r["sent_delivered_rms_nm"] for r in rows])
            )
            cell["mean_actuation_mismatch_steps"] = float(
                np.mean([r["actuation_mismatch_steps"] for r in rows])
            )
            if arm == "adaptive":
                cell["mean_estimator_bound_hit_fraction"] = float(
                    np.mean(
                        [r["estimator"]["bound_hits"] / r["estimator"]["updates"] for r in rows]
                    )
                )
            summary["interface_minus_ideal"][f"{condition}/{arm}"] = {
                "success_count_difference": sum(r["success"] for r in rows)
                - sum(r["success"] for r in reference),
                "mean_paired_final_joint_error_difference_rad": float(
                    np.mean(
                        [
                            r["joint_error_max_rad"] - b["joint_error_max_rad"]
                            for r, b in zip(rows, reference, strict=True)
                        ]
                    )
                ),
            }
    return summary
