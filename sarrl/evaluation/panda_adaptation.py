"""Paired exploratory Panda payload/friction pilot, with an oracle diagnostic arm."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from sarrl.controllers.panda_adaptive import PandaAdaptiveController, PandaEstimatorConfig
from sarrl.controllers.panda_nominal import PandaComputedTorqueController
from sarrl.envs.panda_reach import PandaReachConfig, PandaReachEnv
from sarrl.robots.panda import PandaDynamics, PandaModelParameters

ARMS = ("fixed", "adaptive", "oracle")
SCENARIOS = ("nominal", "payload", "friction", "payload_friction")
REPRODUCTION_RTOL = 1e-9
REPRODUCTION_ATOL = 1e-10


def _reproduction_value_matches(actual, retained) -> bool:
    if isinstance(retained, bool) or isinstance(retained, int):
        return actual == retained
    if isinstance(retained, float):
        return bool(np.isclose(actual, retained, rtol=REPRODUCTION_RTOL, atol=REPRODUCTION_ATOL))
    if isinstance(retained, list):
        return (
            isinstance(actual, (list, tuple))
            and len(actual) == len(retained)
            and all(
                _reproduction_value_matches(a, b) for a, b in zip(actual, retained, strict=True)
            )
        )
    return actual == retained


def reproduction_mismatches(actual: dict, retained: dict) -> list[str]:
    """Check all fields, with an absolute floor for the nominal's near-zero errors."""
    return sorted(
        name
        for name in set(actual) | set(retained)
        if name not in actual
        or name not in retained
        or not _reproduction_value_matches(actual[name], retained[name])
    )


def sample_parameters(seed: int, scenario: str) -> PandaModelParameters:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown Panda scenario: {scenario}")
    # Independent of the task RNG; the same draws form the four factorial cells.
    rng = np.random.default_rng(seed ^ 0x7AD0F)
    mass = float(rng.uniform(0.5, 2.5))
    damping = tuple(float(v) for v in rng.uniform(0.5, 3.0, 7))
    return PandaModelParameters(
        payload_mass=mass if scenario in ("payload", "payload_friction") else 0.0,
        viscous_extra=damping if scenario in ("friction", "payload_friction") else (0.0,) * 7,
    )


def pilot_protocol(episodes: int = 20, seed_start: int = 9804300) -> dict:
    return {
        "kind": "exploratory_panda_adaptation_pilot",
        "arms": list(ARMS),
        "scenarios": list(SCENARIOS),
        "seeds": list(range(seed_start, seed_start + episodes)),
        "task": asdict(PandaReachConfig()),
        "estimator": asdict(PandaEstimatorConfig()),
        "controller": {"kp": 100.0, "kd": 20.0},
        "reproduction": {
            "episodes": 20,
            "relative_tolerance": REPRODUCTION_RTOL,
            "absolute_tolerance": REPRODUCTION_ATOL,
        },
        "plant_parameters": {
            "payload_mass_range_kg": [0.5, 2.5],
            "extra_viscous_range_nm_s_per_rad": [0.5, 3.0],
            "nominal_parameters": "zero added payload and zero extra viscous friction",
            "sample_rng": "default_rng(episode_seed ^ 0x7AD0F), mass then seven dampings",
        },
        "state_interface": "noise-free q and qd sampled at the control period",
        "estimator_inputs": "sent clipped command, pre/post q and qd, control interval",
        "oracle": "privileged true-parameter diagnostic; excluded from adaptive/fixed comparison",
        "decision": "descriptive only; no preregistered acceptance rule or population inference",
    }


def run_episode(seed: int, scenario: str, arm: str, *, retain_trace=False):
    if arm not in ARMS:
        raise ValueError(f"unknown Panda controller arm: {arm}")
    parameters = sample_parameters(seed, scenario)
    env = PandaReachEnv(parameters=parameters)
    _, reset_info = env.reset(seed=seed)
    if arm == "adaptive":
        controller = PandaAdaptiveController()
    elif arm == "oracle":
        controller = PandaComputedTorqueController(PandaDynamics(parameters=parameters))
    else:
        controller = PandaComputedTorqueController()
    start, goal = env.start.copy(), env.goal.copy()
    tracking = np.zeros(7)
    error_squares = np.zeros(7)
    saturated = 0
    trace = []
    for _ in range(env.horizon):
        reference = env.reference()
        q, qd = env.data.qpos.copy(), env.data.qvel.copy()
        error = q - reference[0]
        tracking = np.maximum(tracking, np.abs(error))
        error_squares += error**2
        torque = controller.command(q, qd, *reference)
        saturated += int(np.any(np.abs(torque - controller.last_raw_torque) > 1e-12))
        if retain_trace:
            trace.append(
                {
                    "time_s": env.steps * env.config.control_dt,
                    "q": q.tolist(),
                    "qd": qd.tolist(),
                    "reference": reference[0].tolist(),
                    "torque_nm": torque.tolist(),
                    "estimated_parameters": (
                        controller.estimator.parameters.tolist() if arm == "adaptive" else None
                    ),
                }
            )
        env.step_torque(torque)
        if arm == "adaptive":
            controller.observe_transition(
                q, qd, env.data.qpos.copy(), env.data.qvel.copy(), torque, env.config.control_dt
            )
    if retain_trace:
        trace.append(
            {
                "time_s": env.steps * env.config.control_dt,
                "q": env.data.qpos.tolist(),
                "qd": env.data.qvel.tolist(),
                "reference": env.goal.tolist(),
                "torque_nm": None,
                "estimated_parameters": (
                    controller.estimator.parameters.tolist() if arm == "adaptive" else None
                ),
            }
        )
    estimator = controller.estimator.diagnostics() if arm == "adaptive" else None
    if estimator is not None:
        estimator["parameter_error"] = (
            controller.estimator.parameters - parameters.vector()
        ).tolist()
    return {
        "seed": seed,
        "scenario": scenario,
        "arm": arm,
        "privileged_parameters": arm == "oracle",
        "true_parameters": asdict(parameters),
        "start_q": start.tolist(),
        "goal_q": goal.tolist(),
        "target_xyz_m": env.target.tolist(),
        "final_q": env.data.qpos.tolist(),
        "sampling_attempts": reset_info["sampling_attempts"],
        "tracking_error_max_rad": tracking.tolist(),
        "tracking_rms_rad": np.sqrt(error_squares / env.horizon).tolist(),
        "controller_saturated_steps": saturated,
        "physical_steps": env.steps * env.substeps,
        "estimator": estimator,
        **env.diagnostics(),
    }, trace


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("the Panda adaptation pilot needs at least one episode")
    seeds = sorted({r["seed"] for r in records})
    expected = {(scenario, seed, arm) for scenario in SCENARIOS for seed in seeds for arm in ARMS}
    keys = [(r["scenario"], r["seed"], r["arm"]) for r in records]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("pilot records must contain exactly one of every paired cell")
    cells = {}
    contrasts = {}
    for scenario in SCENARIOS:
        by_arm = {}
        for arm in ARMS:
            rows = sorted(
                (r for r in records if r["scenario"] == scenario and r["arm"] == arm),
                key=lambda r: r["seed"],
            )
            by_arm[arm] = rows
            cells[f"{scenario}/{arm}"] = {
                "episodes": len(rows),
                "successes": sum(r["success"] for r in rows),
                "unsafe_episodes": sum(r["unsafe_episode"] for r in rows),
                "contact_episodes": sum(r["contact_substeps"] > 0 for r in rows),
                "saturated_episodes": sum(r["controller_saturated_steps"] > 0 for r in rows),
                "mean_final_tip_error_m": float(np.mean([r["tip_error_m"] for r in rows])),
                "mean_final_joint_error_rad": float(
                    np.mean([r["joint_error_max_rad"] for r in rows])
                ),
                "mean_tracking_rms_rad": float(
                    np.mean([np.mean(r["tracking_rms_rad"]) for r in rows])
                ),
                "max_position_excess_rad": max(r["joint_position_excess_rad"] for r in rows),
                "max_velocity_excess_rad_s": max(r["joint_velocity_excess_rad_s"] for r in rows),
            }
            if arm == "adaptive":
                cells[f"{scenario}/{arm}"].update(
                    {
                        "mean_payload_absolute_error_kg": float(
                            np.mean([abs(r["estimator"]["parameter_error"][0]) for r in rows])
                        ),
                        "mean_viscous_absolute_error_nm_s_per_rad": float(
                            np.mean([np.abs(r["estimator"]["parameter_error"][1:]) for r in rows])
                        ),
                        "full_rank_episodes": sum(
                            r["estimator"]["information_rank"] == 8 for r in rows
                        ),
                    }
                )
        fixed, adaptive = by_arm["fixed"], by_arm["adaptive"]
        for index, seed in enumerate(seeds):
            matched = [by_arm[arm][index] for arm in ARMS]
            for field in ("seed", "true_parameters", "start_q", "goal_q", "target_xyz_m"):
                if any(row[field] != matched[0][field] for row in matched[1:]):
                    raise ValueError(f"unpaired {field} in {scenario}, seed {seed}")
        contrasts[scenario] = {
            "success_count_difference": sum(r["success"] for r in adaptive)
            - sum(r["success"] for r in fixed),
            "mean_paired_final_tip_error_difference_m": float(
                np.mean(
                    [
                        a["tip_error_m"] - f["tip_error_m"]
                        for a, f in zip(adaptive, fixed, strict=True)
                    ]
                )
            ),
            "adaptive_lower_final_joint_error_pairs": sum(
                a["joint_error_max_rad"] < f["joint_error_max_rad"]
                for a, f in zip(adaptive, fixed, strict=True)
            ),
        }
    return {
        "scope": "descriptive paired engineering pilot, no safety certificate",
        "episodes": len(records),
        "physical_steps": sum(r["physical_steps"] for r in records),
        "cells": cells,
        "adaptive_minus_fixed": contrasts,
    }
