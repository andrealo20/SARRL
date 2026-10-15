"""Event ordering in reproduced Panda failure traces."""

from __future__ import annotations

import numpy as np

from sarrl.evaluation.control_diagnosis import panda_trace


def failure_timing(row, frames, lower, upper, velocity):
    for frame in frames[:-1]:
        for sample in frame["physical_substeps"]:
            for key in ("q", "qd"):
                value = np.asarray(sample[key])
                if value.shape != (7,) or not np.all(np.isfinite(value)):
                    raise ValueError("invalid physical substep state")
    result = panda_trace(row, frames, lower, upper, velocity)
    first = result["first_physical_violation"]
    abort = result["abort_step"]
    if first is None or abort is None or row["arm"] != "bank_filtered":
        raise ValueError("expected an unsafe, aborted filtered case")
    lag = row["interface_config"]["command_delay_steps"]
    arrival = abort + lag
    step = first["control_step"]
    origin = step - lag
    if origin < 0:
        kind = "initial_zero"
    elif frames[origin]["filter"].get("braking"):
        kind = "braking"
    else:
        kind = "pre_abort_control"
    phases = {"before_abort": 0, "pending_after_abort": 0, "braking_delivered": 0}
    last_time = None
    first_joints = None
    for index, frame in enumerate(frames[:-1]):
        for subindex, sample in enumerate(frame["physical_substeps"]):
            q, v = np.asarray(sample["q"]), np.asarray(sample["qd"])
            bad = (
                np.any(q < lower)
                or np.any(q > upper)
                or np.any(np.abs(v) > velocity)
                or sample["contacts"] > 0
            )
            if not bad:
                continue
            phase = (
                "before_abort"
                if index < abort
                else "pending_after_abort"
                if index < arrival
                else "braking_delivered"
            )
            phases[phase] += 1
            last_time = (
                index * row["task_config"]["control_dt"]
                + (subindex + 1) * row["task_config"]["timestep"]
            )
            if first_joints is None:
                first_joints = (np.flatnonzero(np.abs(v) > velocity) + 1).tolist()
    result.update(
        first_braking_delivery_step=arrival,
        first_braking_delivery_time_s=arrival * row["task_config"]["control_dt"],
        first_violation_delivered_command_origin_step=origin,
        first_violation_delivered_command_kind=kind,
        first_violation_velocity_joint_indices=first_joints,
        first_violation_phase=(
            "before_abort"
            if step < abort
            else "pending_after_abort"
            if step < arrival
            else "braking_delivered"
        ),
        first_violation_minus_braking_delivery_s=(
            first["time_s"] - arrival * row["task_config"]["control_dt"]
        ),
        unsafe_substeps_by_phase=phases,
        last_measured_violation_time_s=last_time,
        scope="recorded event ordering; delivered command identity does not establish causation",
    )
    return result


def summarize_failures(reports):
    if not reports:
        raise ValueError("empty failure report")
    return {
        "reproduced_cases": len(reports),
        "first_violation_phases": {
            phase: sum(r["first_violation_phase"] == phase for r in reports)
            for phase in ("before_abort", "pending_after_abort", "braking_delivered")
        },
        "first_violation_command_kinds": {
            kind: sum(r["first_violation_delivered_command_kind"] == kind for r in reports)
            for kind in ("initial_zero", "pre_abort_control", "braking")
        },
        "earliest_first_violation_s": min(r["first_physical_violation"]["time_s"] for r in reports),
        "latest_first_violation_s": max(r["first_physical_violation"]["time_s"] for r in reports),
        "new_task_seeds": [],
        "meaning": "reproduction on used seeds; no new candidate or confirmation claim",
    }
