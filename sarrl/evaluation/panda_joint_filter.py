"""Paired engineering screen for delayed Panda joint filtering."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from sarrl.controllers.panda_delay import PandaDelayController
from sarrl.controllers.panda_filtered import PandaFilteredController
from sarrl.envs.panda_interface import PandaControlInterface, PandaInterfaceConfig
from sarrl.envs.panda_joint_reach import PandaJointReachEnv
from sarrl.envs.panda_reach import PandaReachConfig
from sarrl.evaluation.panda_adaptation import sample_parameters
from sarrl.evaluation.panda_posture_delay import sample_task
from sarrl.safety.panda_joint_filter import PandaJointFilterConfig

ARMS = ("bank_align", "bank_filtered")
CONDITIONS = tuple(
    f"{speed}_{noise}_d{delay:02d}"
    for speed in ("slow", "fast")
    for noise in ("clean", "noise")
    for delay in (0, 20, 40)
)
CONFIG = PandaJointFilterConfig()


def configurations(condition):
    if condition not in CONDITIONS:
        raise ValueError("unknown Panda filter condition")
    speed, noise, delay = condition.split("_")
    move = 2.0 if speed == "slow" else 0.25
    task = PandaReachConfig(move_seconds=move, settle_seconds=3.0 - move)
    scale = 0.001 if noise == "noise" else 0.0
    interface = PandaInterfaceConfig(scale, scale, int(delay[1:]) // 10)
    return task, interface


def joint_filter_protocol():
    return {
        "kind": "exploratory_panda_joint_filter",
        "arms": list(ARMS),
        "scenarios": list(CONDITIONS),
        "seeds": list(range(9805800, 9805820)),
        "tasks": {c: asdict(configurations(c)[0]) for c in CONDITIONS},
        "interfaces": {c: asdict(configurations(c)[1]) for c in CONDITIONS},
        "filter": asdict(CONFIG),
        "posture": "same existing wide task draw per paired seed",
        "delay_handling": "all constant ages zero through four, no applied-age input",
        "prediction": "frozen estimated affine acceleration, exact polynomial prefix checks",
        "fallback": "latched bounded estimated-model braking; never labelled safe",
        "plant_parameters": "same payload/friction draw as retained reference",
        "reproduction": {
            "episodes": 4000,
            "reference": "results/panda_reference_validation/episodes.json",
        },
        "gate": {
            "baseline_fast_unsafe_fraction_min": 0.10,
            "fast_unsafe_fraction_reduction_min": 0.10,
            "filtered_unsafe_episodes": 0,
            "success_per_cell": "no fewer than cumulative bank",
            "slow_aborts": 0,
            "slow_paired_joint_mean_rms_increase_rad_max": 0.005,
            "meaning": "all pass to advance to confirmation; no default change",
        },
    }


def run_episode(seed, scenario, arm, *, retain_trace=False):
    if arm not in ARMS:
        raise ValueError("unknown Panda filter arm")
    filtered = arm == "bank_filtered"
    factory = (
        (lambda: PandaFilteredController(CONFIG))
        if filtered
        else (lambda: PandaDelayController(predict=False))
    )
    return episode(seed, scenario, arm, factory, filtered=filtered, retain_trace=retain_trace)


def episode(seed, scenario, arm, make_controller, *, filtered, retain_trace=False):
    """One paired episode; the controller is built where the v2.6 runner built it."""
    task_cfg, interface_cfg = configurations(scenario)
    parameters = sample_parameters(seed, "payload_friction")
    task = sample_task(seed, "wide")
    env = PandaJointReachEnv(task_cfg, parameters=parameters)
    env.reset(seed=seed, start=task.start, goal=task.goal)
    interface = PandaControlInterface(interface_cfg, seed)
    controller = make_controller()
    squared, trace, changes = np.zeros(7), [], 0
    for step in range(env.horizon):
        q, v = env.data.qpos.copy(), env.data.qvel.copy()
        reference = env.reference()
        squared += (q - reference[0]) ** 2
        mq, mv = interface.measure(q, v, step)
        candidate = controller.command(mq, mv, *reference)
        changes += int(np.any(np.abs(candidate - controller.last_raw_torque) > 1e-12))
        delivered = interface.deliver(candidate, env.torque_limit)
        if retain_trace:
            sample = {
                "time_s": step * task_cfg.control_dt,
                "q": q.tolist(),
                "qd": v.tolist(),
                "measured_q": mq.tolist(),
                "measured_qd": mv.tolist(),
                "reference": reference[0].tolist(),
                "torque_nm": candidate.tolist(),
                "delivered_torque_nm": delivered.tolist(),
                "estimated_parameters": controller.estimator.parameters.tolist(),
                "delay_scores": controller.scores.tolist(),
            }
            if filtered:
                sample.update(
                    filter=controller.last_filter,
                    nominal_command_nm=controller.last_nominal_command.tolist(),
                )
            trace.append(sample)
        env.step_torque(delivered)
        if retain_trace:
            trace[-1]["physical_substeps"] = env.last_substeps
        nq, nv = interface.measure(env.data.qpos, env.data.qvel, step + 1)
        controller.observe_transition(mq, mv, nq, nv, candidate, task_cfg.control_dt)
    if retain_trace:
        trace.append(
            {
                "time_s": 3.0,
                "q": env.data.qpos.tolist(),
                "qd": env.data.qvel.tolist(),
                "measured_q": nq.tolist(),
                "measured_qd": nv.tolist(),
            }
        )
    row = {
        **env.diagnostics(),
        "seed": seed,
        "scenario": scenario,
        "arm": arm,
        "task_success": env.diagnostics()["success"],
        "task_config": asdict(task_cfg),
        "task_sampling": asdict(task),
        "start_q": env.start.tolist(),
        "goal_q": env.goal.tolist(),
        "final_q": env.data.qpos.tolist(),
        "final_qd": env.data.qvel.tolist(),
        "true_parameters": asdict(parameters),
        "interface_config": asdict(interface_cfg),
        **interface.diagnostics(),
        "physical_steps": env.horizon * env.substeps,
        "tracking_rms_rad": np.sqrt(squared / env.horizon).tolist(),
        "command_changes_from_raw_nominal": changes,
        "estimator": controller.estimator.diagnostics(),
        "delay_estimator": controller.delay_diagnostics(),
        "prediction_virtual_substeps": 0,
    }
    row["delay_estimator"]["final_delay_correct"] = (
        controller.selected_delay_steps == interface_cfg.command_delay_steps
    )
    if filtered:
        row["filter"] = controller.filter_diagnostics()
        row["success"] = bool(row["success"] and not controller.aborted)
    return row, trace


def evaluate_gate(records):
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in records}
    seeds = sorted({r["seed"] for r in records})
    expected = {(c, s, a) for c in CONDITIONS for s in seeds for a in ARMS}
    if not seeds or len(lookup) != len(records) or set(lookup) != expected:
        raise ValueError("filter gate requires a complete paired factorial")
    for row in records:
        rms = np.asarray(row["tracking_rms_rad"])
        if rms.shape != (7,) or not np.all(np.isfinite(rms)) or np.any(rms < 0):
            raise ValueError("filter gate requires finite nonnegative joint RMS")
    failures = []
    fast = [c for c in CONDITIONS if c.startswith("fast_")]
    pairs = [
        (lookup[c, s, "bank_align"], lookup[c, s, "bank_filtered"]) for c in fast for s in seeds
    ]
    baseline = sum(a["unsafe_episode"] for a, b in pairs) / len(pairs)
    reduction = sum(a["unsafe_episode"] - b["unsafe_episode"] for a, b in pairs) / len(pairs)
    if baseline < 0.10:
        failures.append(
            {"criterion": "uninformative_fast_grid", "baseline_unsafe_fraction": baseline}
        )
    if reduction < 0.10:
        failures.append({"criterion": "fast_unsafe_reduction", "reduction": reduction})
    for condition in CONDITIONS:
        base = [lookup[condition, s, "bank_align"] for s in seeds]
        filtered = [lookup[condition, s, "bank_filtered"] for s in seeds]
        if any(r["unsafe_episode"] for r in filtered):
            failures.append({"condition": condition, "criterion": "filtered_unsafe"})
        if sum(r["success"] for r in filtered) < sum(r["success"] for r in base):
            failures.append({"condition": condition, "criterion": "success_count"})
        if condition.startswith("slow_"):
            if any(r["filter"]["aborted"] for r in filtered):
                failures.append({"condition": condition, "criterion": "slow_abort"})
            increase = np.mean(
                [
                    np.array(b["tracking_rms_rad"]) - a["tracking_rms_rad"]
                    for a, b in zip(base, filtered, strict=True)
                ],
                axis=0,
            )
            for joint, value in enumerate(increase):
                if value > 0.005:
                    failures.append(
                        {
                            "condition": condition,
                            "criterion": "slow_joint_rms",
                            "joint": joint + 1,
                            "difference_rad": float(value),
                        }
                    )
    return {
        "filter_advances": not failures,
        "reference_controller": "bank_align",
        "baseline_fast_unsafe_fraction": baseline,
        "fast_unsafe_fraction_reduction": reduction,
        "failures": failures,
        "meaning": "engineering screen; no general safety certificate",
    }


def summarize(records):
    decision = evaluate_gate(records)
    cells = {}
    for condition in CONDITIONS:
        for arm in ARMS:
            selected = [r for r in records if r["scenario"] == condition and r["arm"] == arm]
            cells[f"{condition}/{arm}"] = {
                "episodes": len(selected),
                "successes": sum(r["success"] for r in selected),
                "unsafe_episodes": sum(r["unsafe_episode"] for r in selected),
                "aborts": sum(r.get("filter", {}).get("aborted", False) for r in selected),
                "mean_tracking_rms_rad": np.mean(
                    [r["tracking_rms_rad"] for r in selected], axis=0
                ).tolist(),
                "mean_final_error_rad": float(
                    np.mean([r["joint_error_max_rad"] for r in selected])
                ),
            }
    return {
        "episodes": len(records),
        "physical_steps": sum(r["physical_steps"] for r in records),
        "cells": cells,
        "exploratory_gate": decision,
    }
