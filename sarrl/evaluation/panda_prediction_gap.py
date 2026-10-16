"""Saved-state velocity identities for accepted frozen Panda backup predictions."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from sarrl.robots.panda import vector7


def retained_witness(row, frames):
    abort = row["backup"]["abort_step"]
    if abort is None or not 0 <= abort < len(frames) - 1:
        raise ValueError("prediction-gap review requires an aborted retained episode")
    record = frames[abort]["supervisor"]
    start = record["policy_step"]
    origin = record["policy_origin"]
    key = {"initial": "initial_witness", "proposal": "proposal_witness"}.get(origin)
    if key is None or start is None or not 0 <= start <= abort:
        raise ValueError("missing accepted backup provenance")
    witness = frames[start]["supervisor"][key]
    if not witness or not witness["accepted"]:
        raise ValueError("retained context must have an accepted witness")
    digest = hashlib.sha256(
        json.dumps(
            witness["policy"], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    if digest != witness["policy_sha256"] or digest != record["retained_policy_sha256"]:
        raise ValueError("retained policy hash differs")
    if origin == "proposal" and abort != start + 1:
        raise ValueError("last accepted proposal must immediately precede latched rejection")
    for frame in frames[abort:-1]:
        if (
            frame["supervisor"]["mode"] != "backup"
            or frame["supervisor"]["retained_policy_sha256"] != digest
        ):
            raise ValueError("fallback context or mode changed after abort")
    np.testing.assert_array_equal(witness["candidate_torque_nm"], frames[start]["torque_nm"])
    return start, witness


def prediction_gap(row, frames, lower, upper, velocity):
    if len(frames) != 301:
        raise ValueError("all three hundred intervals and the final boundary are required")
    start, witness = retained_witness(row, frames)
    policy = witness["policy"]
    cfg = policy["config"]
    dt, subdt, lag = (
        row["task_config"]["control_dt"],
        row["task_config"]["timestep"],
        row["interface_config"]["command_delay_steps"],
    )
    if (
        dt != cfg["control_dt"]
        or not 0 <= lag <= cfg["max_delay_steps"]
        or not np.isclose(dt, 5 * subdt)
    ):
        raise ValueError("retained transport or sample timing differs")
    matrix, mass, drift, anchor, equilibrium = (
        np.asarray(policy[name])
        for name in ("matrix", "mass", "drift", "anchor", "equilibrium_torque")
    )
    limit = np.asarray(policy["torque_limit"])
    q = vector7(frames[start]["measured_q"], "measured_q").copy()
    v = vector7(frames[start]["measured_qd"], "measured_qd").copy()
    saved_v = v.copy()
    saved_q = q.copy()
    initial_position_offset = vector7(frames[start]["q"], "physical_q") - q
    initial_offset = vector7(frames[start]["qd"], "physical_qd") - v
    model_gap, command_gap = np.zeros(7), np.zeros(7)
    model_position_gap, command_position_gap = np.zeros(7), np.zeros(7)
    history = [
        np.asarray(frames[i]["torque_nm"]) if i >= 0 else np.zeros(7)
        for i in range(start - lag, start)
    ]
    first = None
    maximum_closure = maximum_local_error = 0.0
    maximum_position_closure = 0.0
    predicted_position_excess = predicted_velocity_excess = 0.0
    actual_max, predicted_max = np.zeros(7), np.zeros(7)
    curve = []
    for index in range(start, 300):
        frame = frames[index]
        sent = (
            np.asarray(witness["candidate_torque_nm"])
            if index == start
            else equilibrium - mass @ (cfg["kp"] * (q - anchor) + cfg["kd"] * v)
        )
        if np.any(np.abs(sent) > limit + 1e-9):
            raise ValueError("accepted prediction requires saturation")
        if lag:
            delivered = history.pop(0)
            history.append(sent)
        else:
            delivered = sent
        acceleration = matrix @ delivered + drift
        recorded = vector7(frame["delivered_torque_nm"], "delivered_torque")
        expected_delivery = frames[index - lag]["torque_nm"] if index >= lag else np.zeros(7)
        np.testing.assert_array_equal(recorded, expected_delivery)
        saved_acceleration = matrix @ recorded + drift
        previous_v = vector7(frame["qd"], "physical_qd")
        previous_q = vector7(frame["q"], "physical_q")
        if len(frame["physical_substeps"]) != 5:
            raise ValueError("missing physical substeps")
        for subindex, sample in enumerate(frame["physical_substeps"]):
            elapsed = (subindex + 1) * subdt
            predicted_v = v + elapsed * acceleration
            predicted_q = q + elapsed * v + 0.5 * elapsed**2 * acceleration
            recorded_model_v = saved_v + elapsed * saved_acceleration
            recorded_model_q = saved_q + elapsed * saved_v + 0.5 * elapsed**2 * saved_acceleration
            actual_v, actual_q = (
                vector7(sample["qd"], "physical_qd"),
                vector7(sample["q"], "physical_q"),
            )
            observed_acceleration = (actual_v - previous_v) / subdt
            acceleration_difference = matrix @ (recorded - delivered)
            model_position_gap += (
                subdt * model_gap
                + actual_q
                - previous_q
                - subdt * previous_v
                - 0.5 * subdt**2 * saved_acceleration
            )
            command_position_gap += subdt * command_gap + 0.5 * subdt**2 * acceleration_difference
            model_gap += actual_v - previous_v - subdt * saved_acceleration
            command_gap += subdt * acceleration_difference
            closure = actual_v - predicted_v - initial_offset - model_gap - command_gap
            maximum_closure = max(maximum_closure, float(np.max(np.abs(closure))))
            np.testing.assert_allclose(closure, np.zeros(7), rtol=0, atol=1e-9)
            total_elapsed = (index - start) * dt + elapsed
            position_closure = (
                actual_q
                - predicted_q
                - initial_position_offset
                - total_elapsed * initial_offset
                - model_position_gap
                - command_position_gap
            )
            maximum_position_closure = max(
                maximum_position_closure, float(np.max(np.abs(position_closure)))
            )
            np.testing.assert_allclose(position_closure, np.zeros(7), rtol=0, atol=1e-9)
            position_excess = float(
                max(0, np.max(np.maximum(lower - predicted_q, predicted_q - upper)))
            )
            speed_excess = float(max(0, np.max(np.abs(predicted_v) - velocity)))
            predicted_position_excess = max(predicted_position_excess, position_excess)
            predicted_velocity_excess = max(predicted_velocity_excess, speed_excess)
            actual_max = np.maximum(actual_max, np.abs(actual_v))
            predicted_max = np.maximum(predicted_max, np.abs(predicted_v))
            unsafe = bool(
                np.any(actual_q < lower)
                or np.any(actual_q > upper)
                or np.any(np.abs(actual_v) > velocity)
                or sample["contacts"] > 0
            )
            time = index * dt + elapsed
            if unsafe and first is None:
                first = {
                    "time_s": time,
                    "control_step": index,
                    "substep": subindex,
                    "velocity_joint_indices": (
                        np.flatnonzero(np.abs(actual_v) > velocity) + 1
                    ).tolist(),
                    "position_joint_indices": (
                        np.flatnonzero((actual_q < lower) | (actual_q > upper)) + 1
                    ).tolist(),
                    "actual_position_rad": actual_q.tolist(),
                    "witness_position_rad": predicted_q.tolist(),
                    "recorded_command_model_position_rad": recorded_model_q.tolist(),
                    "initial_position_offset_rad": initial_position_offset.tolist(),
                    "initial_velocity_propagation_rad": (total_elapsed * initial_offset).tolist(),
                    "accumulated_model_position_discrepancy_rad": model_position_gap.tolist(),
                    "accumulated_command_position_difference_rad": command_position_gap.tolist(),
                    "position_identity_residual_rad": position_closure.tolist(),
                    "actual_velocity_rad_s": actual_v.tolist(),
                    "witness_velocity_rad_s": predicted_v.tolist(),
                    "recorded_command_model_velocity_rad_s": recorded_model_v.tolist(),
                    "initial_velocity_offset_rad_s": initial_offset.tolist(),
                    "accumulated_model_discrepancy_rad_s": model_gap.tolist(),
                    "accumulated_command_difference_rad_s": command_gap.tolist(),
                    "identity_residual_rad_s": closure.tolist(),
                    "interval_start_q": previous_q.tolist(),
                    "interval_start_qd": previous_v.tolist(),
                    "recorded_delivered_torque_nm": recorded.tolist(),
                    "observed_substep_acceleration_rad_s2": observed_acceleration.tolist(),
                }
            curve.append(
                {
                    "time_s": time,
                    "actual_velocity": actual_v.tolist(),
                    "actual_position": actual_q.tolist(),
                    "witness_position": predicted_q.tolist(),
                    "recorded_command_model_position": recorded_model_q.tolist(),
                    "witness_velocity": predicted_v.tolist(),
                    "recorded_command_model_velocity": recorded_model_v.tolist(),
                }
            )
            previous_v, previous_q = actual_v, actual_q
        maximum_local_error = max(
            maximum_local_error,
            float(np.max(np.abs(previous_v - np.asarray(frame["qd"]) - dt * saved_acceleration))),
        )
        q = q + dt * v + 0.5 * dt**2 * acceleration
        v = v + dt * acceleration
        saved_q += dt * saved_v + 0.5 * dt**2 * saved_acceleration
        saved_v += dt * saved_acceleration
    if bool(first) != row["unsafe_episode"]:
        raise ValueError("saved first violation must lie after retained policy acceptance")
    if predicted_position_excess > 1e-8 or predicted_velocity_excess > 1e-8:
        raise ValueError("accepted witness trajectory leaves physical limits")
    return {
        "cell": [row["scenario"], row["seed"], row["arm"]],
        "abort_step": row["backup"]["abort_step"],
        "policy_step": start,
        "policy_sha256": witness["policy_sha256"],
        "recorded_delay_steps_for_diagnosis_only": lag,
        "initial_velocity_offset_rad_s": initial_offset.tolist(),
        "initial_position_offset_rad": initial_position_offset.tolist(),
        "first_violation": first,
        "maximum_identity_residual_rad_s": maximum_closure,
        "maximum_position_identity_residual_rad": maximum_position_closure,
        "maximum_local_one_period_velocity_discrepancy_rad_s": maximum_local_error,
        "actual_max_speed_rad_s": actual_max.tolist(),
        "witness_max_speed_rad_s": predicted_max.tolist(),
        "predicted_physical_position_excess_rad": predicted_position_excess,
        "predicted_physical_velocity_excess_rad_s": predicted_velocity_excess,
        "analyzed_substeps": len(curve),
    }, curve


def acceleration_replacements(report, row, frames, regressor):
    point = report["first_violation"]
    if point is None:
        return None
    start = report["policy_step"]
    origin = frames[row["backup"]["abort_step"]]["supervisor"]["policy_origin"]
    witness = frames[start]["supervisor"][origin + "_witness"]
    policy = witness["policy"]
    theta = np.asarray(frames[start]["estimated_parameters"])
    true = row["true_parameters"]
    oracle_theta = np.array([true["payload_mass"], *true["viscous_extra"]])
    q, v, torque = (
        np.asarray(point[name])
        for name in ("interval_start_q", "interval_start_qd", "recorded_delivered_torque_nm")
    )

    def evaluate(position, speed, parameters):
        mass = regressor.mass_matrix(position, parameters)
        bias = regressor.inverse_dynamics(position, speed, np.zeros(7), parameters)
        return np.linalg.solve(mass, torque - bias)

    measured_q, measured_v = (
        np.asarray(frames[start][name]) for name in ("measured_q", "measured_qd")
    )
    np.testing.assert_allclose(
        evaluate(measured_q, measured_v, theta),
        np.asarray(policy["matrix"]) @ torque + policy["drift"],
        rtol=1e-9,
        atol=1e-10,
    )
    frozen = np.asarray(policy["matrix"]) @ torque + policy["drift"]
    refreshed = evaluate(q, v, theta)
    oracle = evaluate(q, v, oracle_theta)
    observed = np.asarray(point["observed_substep_acceleration_rad_s2"])
    return {
        "retained_estimated_parameters": theta.tolist(),
        "oracle_parameters_for_diagnosis_only": oracle_theta.tolist(),
        "frozen_acceleration_rad_s2": frozen.tolist(),
        "refreshed_state_acceleration_rad_s2": refreshed.tolist(),
        "oracle_acceleration_rad_s2": oracle.tolist(),
        "state_refresh_difference_rad_s2": (refreshed - frozen).tolist(),
        "parameter_replacement_difference_rad_s2": (oracle - refreshed).tolist(),
        "remaining_observed_difference_rad_s2": (observed - oracle).tolist(),
        "identity_residual_rad_s2": (
            observed - frozen - (refreshed - frozen) - (oracle - refreshed) - (observed - oracle)
        ).tolist(),
    }
