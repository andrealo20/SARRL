"""Causal controller, witness, FIFO and physical-record auditing for retained backup traces."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from sarrl.controllers.panda_backup import PandaBackupController
from sarrl.envs.panda_joint_reach import PandaJointReachEnv
from sarrl.envs.panda_reach import PandaReachConfig, quintic_reference
from sarrl.evaluation.panda_adaptation import sample_parameters
from sarrl.evaluation.panda_backup_audit import audit_witness
from sarrl.evaluation.panda_backup_integration import timing_summary
from sarrl.robots.panda import VELOCITY_LIMIT, mujoco
from tools.audit_panda_estimator_memory import check_noise
from tools.recover_panda_failures import mismatches


def compare(actual, expected):
    difference = mismatches(
        json.loads(json.dumps(actual, allow_nan=False)),
        json.loads(json.dumps(expected, allow_nan=False)),
    )
    if difference:
        raise ValueError(f"causal backup replay differs: {difference[:5]}")


def without_hashes(value):
    if isinstance(value, dict):
        return {
            key: None if key.endswith("sha256") else without_hashes(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [without_hashes(item) for item in value]
    return value


def audit_episode(row, frames, projection_audit):
    if len(frames) != 301 or row["physical_steps"] != 1500:
        raise ValueError("integration trace horizon differs")
    check_noise(frames, row["seed"], row["interface_config"], row)
    control = PandaBackupController()
    bounds = control.joint_filter
    policy = emergency_policy = digest = None
    witnesses = delay_cases = intervals = projections = 0
    lag = row["interface_config"]["command_delay_steps"]
    cycles, squared = [], np.zeros(7)
    for index, frame in enumerate(frames[:-1]):
        q, v = np.asarray(frame["measured_q"]), np.asarray(frame["measured_qd"])
        reference = quintic_reference(
            row["start_q"], row["goal_q"], frame["time_s"], row["task_config"]["move_seconds"]
        )
        np.testing.assert_allclose(reference[0], frame["reference"], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(
            control.estimator.parameters, frame["estimated_parameters"], rtol=1e-9, atol=1e-10
        )
        np.testing.assert_allclose(control.scores, frame["delay_scores"], rtol=1e-9, atol=1e-10)
        history = [
            np.asarray(previous["torque_nm"]) for previous in frames[max(0, index - 4) : index]
        ]
        sent = control.command(q, v, *reference)
        record = frame["supervisor"]
        compare(without_hashes(control.supervisor.last), without_hashes(record))
        np.testing.assert_allclose(sent, frame["torque_nm"], rtol=1e-9, atol=1e-10)
        compare(
            control.last_nominal_command.tolist()
            if control.last_nominal_command is not None
            else None,
            frame["nominal_command_nm"],
        )
        if record["initial_policy"] is not None:
            emergency_policy = record["initial_policy"]
        for key in ("initial_witness", "proposal_witness"):
            witness = record[key]
            if witness is None:
                continue
            if key == "proposal_witness":
                np.testing.assert_array_equal(
                    witness["candidate_torque_nm"], record["projection"]["torque"]
                )
            else:
                initial = witness["policy"]
                expected_initial = np.asarray(initial["equilibrium_torque"]) - np.asarray(
                    initial["mass"]
                ) @ (initial["config"]["kd"] * v)
                np.testing.assert_allclose(
                    witness["candidate_torque_nm"], expected_initial, rtol=1e-9, atol=1e-10
                )
            report = audit_witness(
                witness,
                q,
                v,
                np.asarray(witness["candidate_torque_nm"]),
                history,
                bounds.lower,
                bounds.upper,
                bounds.velocity,
                bounds.limit,
            )
            witnesses += report["witnesses"]
            delay_cases += report["delay_cases"]
            intervals += report["checked_intervals"]
            if witness["accepted"]:
                policy, digest = witness["policy"], witness["policy_sha256"]
        if record["retained_policy_sha256"] != digest:
            raise ValueError("retained backup context changed without an accepted witness")
        if record["mode"] != "task":
            context = policy or emergency_policy
            raw = (
                (
                    np.asarray(context["equilibrium_torque"])
                    - np.asarray(context["mass"])
                    @ (
                        context["config"]["kp"] * (q - context["anchor"])
                        + context["config"]["kd"] * v
                    )
                )
                if context
                else np.zeros(7)
            )
            np.testing.assert_allclose(record["raw_torque_nm"], raw, rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(
                frame["torque_nm"], np.clip(raw, -bounds.limit, bounds.limit), rtol=1e-9, atol=1e-10
            )
        projection = record["projection"]
        if projection is not None and projection["predicted_feasible"]:
            projection_audit(control, q, v, projection["torque"], frame["nominal_command_nm"])
            projections += 1
        recorded_sent = np.asarray(frame["torque_nm"])
        expected_delivery = frames[index - lag]["torque_nm"] if index >= lag else np.zeros(7)
        np.testing.assert_array_equal(frame["delivered_torque_nm"], expected_delivery)
        if len(frame["physical_substeps"]) != 5:
            raise ValueError("missing physical substeps")
        for key in ("q", "qd"):
            np.testing.assert_array_equal(
                frame["physical_substeps"][-1][key], frames[index + 1][key]
            )
        # Use the retained sent torque after checking the controller's floating replay above.
        control.pending_sent = recorded_sent.copy()
        control.observe_transition(
            q,
            v,
            frames[index + 1]["measured_q"],
            frames[index + 1]["measured_qd"],
            recorded_sent,
            0.01,
        )
        squared += (np.asarray(frame["q"]) - reference[0]) ** 2
        np.testing.assert_allclose(
            frame["control_cycle_ms"], frame["command_ms"] + frame["observe_ms"], rtol=0, atol=1e-12
        )
        if min(frame["command_ms"], frame["observe_ms"]) < 0:
            raise ValueError("negative timing sample")
        cycles.append(frame["control_cycle_ms"])
    compare(control.backup_diagnostics(), row["backup"])
    compare(control.estimator.diagnostics(), row["estimator"])
    delay = control.delay_diagnostics()
    delay["final_delay_correct"] = control.selected_delay_steps == lag
    compare(delay, row["delay_estimator"])
    compare(timing_summary(cycles), row["control_timing"])
    np.testing.assert_allclose(
        np.sqrt(squared / 300), row["tracking_rms_rad"], rtol=1e-9, atol=1e-10
    )
    parameters = sample_parameters(row["seed"], "payload_friction")
    compare(asdict(parameters), row["true_parameters"])
    env = PandaJointReachEnv(PandaReachConfig(**row["task_config"]), parameters=parameters)
    env.reset(seed=row["seed"], start=row["start_q"], goal=row["goal_q"])
    samples = [s for frame in frames[:-1] for s in frame["physical_substeps"]]
    positions, velocities = (
        np.array([s["q"] for s in samples]),
        np.array([s["qd"] for s in samples]),
    )
    if (
        positions.shape != (1500, 7)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(velocities))
    ):
        raise ValueError("nonfinite physical samples")
    env.position_excess = float(
        max(0, np.max(np.maximum(env.lower - positions, positions - env.upper)))
    )
    env.velocity_excess = float(max(0, np.max(np.abs(velocities) - VELOCITY_LIMIT)))
    env.contact_substeps = sum(s["contacts"] > 0 for s in samples)
    env.max_speed = np.max(np.abs(velocities), axis=0)
    env.max_torque = np.max(np.abs([f["delivered_torque_nm"] for f in frames[:-1]]), axis=0)
    env.steps, env._done = 300, True
    env.data.qpos[:], env.data.qvel[:] = frames[-1]["q"], frames[-1]["qd"]
    mujoco.mj_forward(env.model, env.data)
    physical = env.diagnostics()
    for key, value in physical.items():
        compare(value, row["task_success"] if key == "success" else row[key])
    if row["success"] != (physical["success"] and not control.aborted):
        raise ValueError("abort must prevent controller success")
    np.testing.assert_array_equal(row["final_q"], frames[-1]["q"])
    np.testing.assert_array_equal(row["final_qd"], frames[-1]["qd"])
    return {
        "episodes": 1,
        "control_cycles": 300,
        "physical_trace_substeps": 1500,
        "witnesses": witnesses,
        "delay_cases": delay_cases,
        "independent_backup_intervals": intervals,
        "independent_feasible_projection_checks": projections,
    }
