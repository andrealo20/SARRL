"""Descriptive closed-loop checks of the backup candidate on already opened tasks."""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter_ns

import numpy as np

from sarrl.controllers.panda_backup import PandaBackupController
from sarrl.envs.panda_interface import PandaControlInterface
from sarrl.envs.panda_joint_reach import PandaJointReachEnv
from sarrl.evaluation.panda_adaptation import sample_parameters
from sarrl.evaluation.panda_joint_filter import CONDITIONS, configurations
from sarrl.evaluation.panda_posture_delay import sample_task


def selected_tasks(records):
    tasks = {
        (r["scenario"], r["seed"])
        for r in records
        if r["arm"] == "bank_filtered" and r["unsafe_episode"]
    }
    tasks |= {(condition, 9805800) for condition in CONDITIONS}
    if len(tasks) != 32:
        raise ValueError(
            "integration requires the twelve first-seed tasks and twenty retained failures"
        )
    available = {(r["scenario"], r["seed"], r["arm"]) for r in records}
    if any(
        (c, s, arm) not in available for c, s in tasks for arm in ("bank_align", "bank_filtered")
    ):
        raise ValueError("every integration task must have both original records")
    return sorted(tasks)


def timing_summary(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (300,) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("timing requires all three hundred finite nonnegative cycles")
    return {
        "cycles": 300,
        "median_ms": float(np.median(values)),
        "p99_ms": float(np.quantile(values, 0.99)),
        "maximum_ms": float(np.max(values)),
        "over_10_ms": int(np.sum(values > 10)),
    }


def run_episode(seed, scenario):
    task_cfg, interface_cfg = configurations(scenario)
    parameters, task = sample_parameters(seed, "payload_friction"), sample_task(seed, "wide")
    env = PandaJointReachEnv(task_cfg, parameters=parameters)
    env.reset(seed=seed, start=task.start, goal=task.goal)
    interface = PandaControlInterface(interface_cfg, seed)
    started = perf_counter_ns()
    controller = PandaBackupController()
    construction_ms = (perf_counter_ns() - started) / 1e6
    squared, trace, cycles = np.zeros(7), [], []
    for step in range(env.horizon):
        q, v = env.data.qpos.copy(), env.data.qvel.copy()
        reference = env.reference()
        squared += (q - reference[0]) ** 2
        mq, mv = interface.measure(q, v, step)
        started = perf_counter_ns()
        sent = controller.command(mq, mv, *reference)
        command_ms = (perf_counter_ns() - started) / 1e6
        delivered = interface.deliver(sent, env.torque_limit)
        sample = {
            "time_s": step * task_cfg.control_dt,
            "q": q.tolist(),
            "qd": v.tolist(),
            "measured_q": mq.tolist(),
            "measured_qd": mv.tolist(),
            "reference": reference[0].tolist(),
            "torque_nm": sent.tolist(),
            "delivered_torque_nm": delivered.tolist(),
            "estimated_parameters": controller.estimator.parameters.tolist(),
            "delay_scores": controller.scores.tolist(),
            "supervisor": controller.supervisor.last,
            "nominal_command_nm": (
                controller.last_nominal_command.tolist()
                if controller.last_nominal_command is not None
                else None
            ),
            "command_ms": command_ms,
        }
        env.step_torque(delivered)
        sample["physical_substeps"] = env.last_substeps
        nq, nv = interface.measure(env.data.qpos, env.data.qvel, step + 1)
        started = perf_counter_ns()
        controller.observe_transition(mq, mv, nq, nv, sent, task_cfg.control_dt)
        sample["observe_ms"] = (perf_counter_ns() - started) / 1e6
        sample["control_cycle_ms"] = command_ms + sample["observe_ms"]
        cycles.append(sample["control_cycle_ms"])
        trace.append(sample)
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
        "arm": "bank_backup",
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
        "estimator": controller.estimator.diagnostics(),
        "delay_estimator": controller.delay_diagnostics(),
        "backup": controller.backup_diagnostics(),
        "control_timing": timing_summary(cycles),
        "construction_ms": construction_ms,
    }
    row["success"] = bool(row["success"] and not controller.aborted)
    row["delay_estimator"]["final_delay_correct"] = (
        controller.selected_delay_steps == interface_cfg.command_delay_steps
    )
    return row, trace


def summarize(rows):
    if len(rows) != 32 or len({(r["scenario"], r["seed"]) for r in rows}) != 32:
        raise ValueError("integration summary requires every fixed task once")
    return {
        "episodes": 32,
        "new_task_seeds": [],
        "successes": sum(r["success"] for r in rows),
        "task_successes_without_abort_rule": sum(r["task_success"] for r in rows),
        "unsafe_episodes": sum(r["unsafe_episode"] for r in rows),
        "aborted_episodes": sum(r["backup"]["aborted"] for r in rows),
        "emergency_episodes": sum(r["backup"]["emergency_steps"] > 0 for r in rows),
        "control_cycles_over_10_ms": sum(r["control_timing"]["over_10_ms"] for r in rows),
        "candidate_advances": False,
        "scope": "selected used-task integration; no confirmation or general safety conclusion",
        "cells": {
            c: {
                "episodes": len(group := [r for r in rows if r["scenario"] == c]),
                "successes": sum(r["success"] for r in group),
                "unsafe": sum(r["unsafe_episode"] for r in group),
                "aborts": sum(r["backup"]["aborted"] for r in group),
            }
            for c in CONDITIONS
        },
    }
