"""Paired exploratory Panda payload/friction pilot, with an oracle diagnostic arm."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from sarrl.controllers.panda_adaptive import PandaAdaptiveController, PandaEstimatorConfig
from sarrl.controllers.panda_nominal import PandaComputedTorqueController
from sarrl.envs.panda_interface import PandaControlInterface, PandaInterfaceConfig
from sarrl.envs.panda_reach import PandaReachConfig, PandaReachEnv, quintic_reference
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
    if isinstance(retained, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(retained)
            and all(_reproduction_value_matches(actual[k], v) for k, v in retained.items())
        )
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


def run_episode(
    seed: int,
    scenario: str,
    arm: str,
    *,
    retain_trace=False,
    interface_config: PandaInterfaceConfig | None = None,
    controller_override: PandaAdaptiveController | None = None,
    task_start=None,
    task_goal=None,
    prepared_env: PandaReachEnv | None = None,
    prepared_interface: PandaControlInterface | None = None,
    phase_split_step: int | None = None,
):
    if arm not in ARMS:
        raise ValueError(f"unknown Panda controller arm: {arm}")
    parameters = sample_parameters(seed, scenario)
    if prepared_env is None:
        env = PandaReachEnv(parameters=parameters)
        _, reset_info = env.reset(seed=seed, start=task_start, goal=task_goal)
    else:
        if (
            not isinstance(prepared_env, PandaReachEnv)
            or prepared_env.parameters != parameters
            or prepared_env.config != PandaReachConfig()
            or prepared_env.steps != 0
            or prepared_env._done
            or task_start is not None
            or task_goal is not None
        ):
            raise ValueError("prepared Panda environment must match the parameters and fresh task")
        env, reset_info = prepared_env, {"sampling_attempts": 1}
    if phase_split_step is not None and (
        type(phase_split_step) is not int or not 0 < phase_split_step < env.horizon
    ):
        raise ValueError("phase split must be an interior control-step index")
    if controller_override is not None:
        if arm != "adaptive" or not isinstance(controller_override, PandaAdaptiveController):
            raise ValueError("controller overrides must be adaptive Panda controllers")
        controller = controller_override
    elif arm == "adaptive":
        controller = PandaAdaptiveController()
    elif arm == "oracle":
        controller = PandaComputedTorqueController(PandaDynamics(parameters=parameters))
    else:
        controller = PandaComputedTorqueController()
    start, goal = env.start.copy(), env.goal.copy()
    if prepared_interface is not None:
        if (
            not isinstance(prepared_interface, PandaControlInterface)
            or prepared_interface.config != interface_config
            or prepared_interface.measurements != 0
            or prepared_interface.commands != 0
        ):
            raise ValueError("prepared Panda interface must match its config and be unused")
        interface = prepared_interface
    else:
        interface = (
            PandaControlInterface(interface_config, seed) if interface_config is not None else None
        )
    tracking = np.zeros(7)
    error_squares = np.zeros(7)
    phase_squares = np.zeros((2, 7)) if phase_split_step is not None else None
    saturated = 0
    trace = []
    for _ in range(env.horizon):
        reference = env.reference()
        q, qd = env.data.qpos.copy(), env.data.qvel.copy()
        error = q - reference[0]
        tracking = np.maximum(tracking, np.abs(error))
        error_squares += error**2
        if phase_squares is not None:
            phase_squares[int(env.steps >= phase_split_step)] += error**2
        measured_q, measured_qd = interface.measure(q, qd, env.steps) if interface else (q, qd)
        control_q, control_qd, control_reference = measured_q, measured_qd, reference
        if hasattr(controller, "prepare_control"):
            control_q, control_qd, lookahead = controller.prepare_control(measured_q, measured_qd)
            control_reference = quintic_reference(
                start,
                goal,
                env.steps * env.config.control_dt + lookahead,
                env.config.move_seconds,
            )
        torque = controller.command(control_q, control_qd, *control_reference)
        saturated += int(np.any(np.abs(torque - controller.last_raw_torque) > 1e-12))
        delivered = interface.deliver(torque, env.torque_limit) if interface else torque
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
            if interface:
                trace[-1].update(
                    measured_q=measured_q.tolist(),
                    measured_qd=measured_qd.tolist(),
                    delivered_torque_nm=delivered.tolist(),
                )
            if hasattr(controller, "control_diagnostics"):
                trace[-1].update(
                    controller.control_diagnostics(),
                    control_reference_q=control_reference[0].tolist(),
                )
        env.step_torque(delivered)
        next_q, next_qd = (
            interface.measure(env.data.qpos, env.data.qvel, env.steps)
            if interface
            else (env.data.qpos.copy(), env.data.qvel.copy())
        )
        if arm == "adaptive":
            controller.observe_transition(
                measured_q, measured_qd, next_q, next_qd, torque, env.config.control_dt
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
        if interface:
            trace[-1].update(
                measured_q=next_q.tolist(), measured_qd=next_qd.tolist(), delivered_torque_nm=None
            )
        if hasattr(controller, "control_diagnostics"):
            trace[-1].update(controller.control_diagnostics())
            trace[-1].update(
                control_state=None,
                control_reference_q=None,
                prediction_lookahead_s=0.0,
                prediction_fallback=False,
            )
    estimator = controller.estimator.diagnostics() if arm == "adaptive" else None
    if estimator is not None:
        estimator["parameter_error"] = (
            controller.estimator.parameters - parameters.vector()
        ).tolist()
    record = {
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
    }
    if interface:
        record.update(interface_config=asdict(interface_config), **interface.diagnostics())
    if phase_squares is not None:
        record["tracking_phase_rms_rad"] = {
            "before": np.sqrt(phase_squares[0] / phase_split_step).tolist(),
            "after": np.sqrt(phase_squares[1] / (env.horizon - phase_split_step)).tolist(),
        }
    if hasattr(controller, "delay_diagnostics"):
        record["delay_estimator"] = controller.delay_diagnostics()
    return record, trace


def summarize(records: list[dict], *, scenarios=SCENARIOS, arms=ARMS) -> dict:
    if not records:
        raise ValueError("the Panda adaptation pilot needs at least one episode")
    seeds = sorted({r["seed"] for r in records})
    expected = {(scenario, seed, arm) for scenario in scenarios for seed in seeds for arm in arms}
    keys = [(r["scenario"], r["seed"], r["arm"]) for r in records]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("pilot records must contain exactly one of every paired cell")
    cells = {}
    contrasts = {}
    for scenario in scenarios:
        by_arm = {}
        for arm in arms:
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
            if rows[0]["estimator"] is not None:
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
            matched = [by_arm[arm][index] for arm in arms]
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
