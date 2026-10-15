"""Descriptive analysis of retained control outcomes and recorded transitions."""

from __future__ import annotations

from collections import Counter

import numpy as np


def planar_pairs(rows):
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate episode identity")
    result = {}
    for scenario in sorted({r["scenario"] for r in rows}):
        selected = [
            r for r in rows if r["scenario"] == scenario and r["arm"].startswith("residual_")
        ]
        counts, losses = Counter(), Counter()
        differences = {key: [] for key in ("reward", "safety_intervention_fraction", "steps")}
        for row in selected:
            reference = lookup[scenario, row["seed"], "adaptive"]
            if row["true_parameters"] != reference["true_parameters"]:
                raise ValueError("unpaired physical parameters")
            label = (
                "both_success"
                if row["success"] and reference["success"]
                else "gain"
                if row["success"]
                else "loss"
                if reference["success"]
                else "both_fail"
            )
            counts[label] += 1
            if label == "loss":
                losses[row["outcome"]] += 1
            for key in differences:
                differences[key].append(row[key] - reference[key])
        result[scenario] = {
            "paired_model_tasks": len(selected),
            "contingency": {k: counts[k] for k in ("both_success", "gain", "loss", "both_fail")},
            "lost_success_outcomes": dict(sorted(losses.items())),
            "mean_paired_differences": {k: float(np.mean(v)) for k, v in differences.items()},
            "residual_outcomes": dict(sorted(Counter(r["outcome"] for r in selected).items())),
        }
    return result


def planar_trace(row, frames):
    """Reconstruct the declared unit-link reward from physical saved states."""
    if not frames:
        raise ValueError("empty planar trace")

    def position(q):
        return np.array([np.cos(q[0]) + np.cos(q[0] + q[1]), np.sin(q[0]) + np.sin(q[0] + q[1])])

    target = position(frames[0]["q_des"])
    costs = {k: 0.0 for k in ("distance", "speed", "torque", "success_bonus")}
    saturation, changed, executed = 0, 0, 0
    residuals, corrections, late_residuals = [], [], []
    distances, speeds = [], []
    for index, frame in enumerate(frames):
        raw = np.asarray(frame["raw_residual"], dtype=float)
        sent = np.asarray(frame["torque"], dtype=float)
        candidate = np.asarray(frame["baseline_torque"], dtype=float) + raw
        if raw.shape != (2,) or not np.all(np.isfinite(raw)) or np.any(np.abs(raw) > 8):
            raise ValueError("invalid bounded residual")
        residuals.append(float(raw @ raw))
        saturation += int(np.any(np.abs(raw) >= 0.95 * 8))
        if index >= len(frames) // 2:
            late_residuals.append(raw.tolist())
        if not frame["executed"]:
            continue
        executed += 1
        corrected = float(np.linalg.norm(sent - candidate))
        if not np.isclose(corrected, frame["correction"], rtol=1e-8, atol=1e-8):
            raise ValueError("projection correction differs from recorded commands")
        corrections.append(corrected)
        changed += int(corrected > 1e-9)
        following = np.asarray(frame["physical_next_state"])
        distance = float(np.linalg.norm(target - position(following[:2])))
        speed = float(np.linalg.norm(following[2:]))
        distances.append(distance)
        speeds.append(speed)
        costs["distance"] += distance
        costs["speed"] += 0.01 * speed**2
        costs["torque"] += 0.0002 * float(sent @ sent)
        costs["success_bonus"] += 10.0 * (distance <= 0.05 and speed <= 0.35)
    reward = costs["success_bonus"] - costs["distance"] - costs["speed"] - costs["torque"]
    if executed != row["steps"] or not np.isclose(reward, row["reward"], rtol=1e-8, atol=1e-7):
        raise ValueError("physical trace does not reconstruct episode reward/steps")
    final = np.asarray(frames[-1]["physical_next_state"])
    final_distance = float(np.linalg.norm(target - position(final[:2])))
    final_speed = float(np.linalg.norm(final[2:]))
    if not np.isclose(final_distance, row["final_distance"], rtol=1e-8, atol=1e-8):
        raise ValueError("trace endpoint differs from episode")
    if row["outcome"] == "timeout":
        blocker = (
            "distance_and_speed"
            if final_distance > 0.05 and final_speed > 0.35
            else "distance"
            if final_distance > 0.05
            else "speed"
        )
    else:
        blocker = row["outcome"]
    return {
        "cell": [row["scenario"], row["seed"], row["arm"]],
        "outcome": row["outcome"],
        "executed_steps": executed,
        "reward_terms": costs,
        "reconstructed_reward": reward,
        "training_penalty_adjusted_return": reward - 500.0 * row["safety_infeasible"],
        "final_distance_m": final_distance,
        "final_speed_norm_rad_s": final_speed,
        "timeout_blocker": blocker,
        "closest_distance_m": min(distances) if distances else final_distance,
        "position_reached_but_too_fast_steps": sum(
            d <= 0.05 and s > 0.35 for d, s in zip(distances, speeds, strict=True)
        ),
        "residual_rms_nm": float(np.sqrt(np.mean(residuals))),
        "residual_near_bound_attempts": saturation,
        "attempts": len(frames),
        "projected_executed_steps": changed,
        "mean_projection_correction_nm": float(np.mean(corrections)) if corrections else 0.0,
        "last_half_mean_residual_nm": np.mean(late_residuals, axis=0).tolist(),
    }


def panda_rows(rows):
    result = {}
    for scenario in sorted({r["scenario"] for r in rows}):
        chosen = [r for r in rows if r["scenario"] == scenario and r["arm"] == "bank_filtered"]
        reasons = Counter()
        abort_steps = []
        for row in chosen:
            reasons.update(row["filter"]["reasons"])
            if row["filter"]["aborted"]:
                abort_steps.append(row["filter"]["abort_step"])
        result[scenario] = {
            "episodes": len(chosen),
            "aborts": len(abort_steps),
            "abort_step_range": [min(abort_steps), max(abort_steps)] if abort_steps else None,
            "filter_attempt_reasons": dict(sorted(reasons.items())),
            "unsafe_episodes": sum(r["unsafe_episode"] for r in chosen),
            "position_violation_episodes": sum(r["joint_position_excess_rad"] > 0 for r in chosen),
            "velocity_violation_episodes": sum(
                r["joint_velocity_excess_rad_s"] > 0 for r in chosen
            ),
            "contact_episodes": sum(r["contact_substeps"] > 0 for r in chosen),
        }
    return result


def panda_trace(row, frames, lower, upper, velocity):
    """Order actual substep violations relative to the latched command abort."""
    dt, subdt = row["task_config"]["control_dt"], row["task_config"]["timestep"]
    first = None
    before = after = 0
    extrema = {k: 0.0 for k in ("position_rad", "velocity_rad_s")}
    abort = row.get("filter", {}).get("abort_step")
    for step, frame in enumerate(frames[:-1]):
        for substep, sample in enumerate(frame["physical_substeps"]):
            q, v = np.asarray(sample["q"]), np.asarray(sample["qd"])
            position = float(max(0, np.max(np.maximum(lower - q, q - upper))))
            speed = float(max(0, np.max(np.abs(v) - velocity)))
            extrema["position_rad"] = max(extrema["position_rad"], position)
            extrema["velocity_rad_s"] = max(extrema["velocity_rad_s"], speed)
            if position > 0 or speed > 0 or sample["contacts"] > 0:
                if first is None:
                    first = {
                        "control_step": step,
                        "substep": substep,
                        "time_s": step * dt + (substep + 1) * subdt,
                        "position_rad": position,
                        "velocity_rad_s": speed,
                        "contacts": sample["contacts"],
                    }
                if abort is not None and step >= abort:
                    after += 1
                else:
                    before += 1
    if bool(first) != row["unsafe_episode"]:
        raise ValueError("trace violation flag differs from retained episode")
    for key, endpoint in (
        ("position_rad", "joint_position_excess_rad"),
        ("velocity_rad_s", "joint_velocity_excess_rad_s"),
    ):
        if not np.isclose(extrema[key], row[endpoint], rtol=1e-9, atol=1e-10):
            raise ValueError("substep violation maximum differs")
    reason = None
    if abort is not None:
        frame = frames[abort]
        if not frame["filter"].get("braking"):
            raise ValueError("abort index does not begin latched braking")
        reason = frame["filter"]["reason"]
    return {
        "cell": [row["scenario"], row["seed"], row["arm"]],
        "abort_step": abort,
        "abort_time_s": abort * dt if abort is not None else None,
        "abort_reason": reason,
        "first_physical_violation": first,
        "unsafe_substeps_before_abort_or_without_abort": before,
        "unsafe_substeps_at_or_after_abort": after,
        "violation_maxima": extrema,
    }


def panda_prefix_reasons(row, frames, regressor, filter_type, config_type):
    """Inspect smaller delay bounds on the same saved abort state, without a QP."""
    abort = row["filter"]["abort_step"]
    if abort is None:
        return None
    frame = frames[abort]
    q, v = np.asarray(frame["measured_q"]), np.asarray(frame["measured_qd"])
    theta = np.asarray(frame["estimated_parameters"])
    mass = regressor.mass_matrix(q, theta)
    matrix = np.linalg.solve(mass, np.eye(7))
    drift = -matrix @ regressor.inverse_dynamics(q, v, np.zeros(7), theta)
    bounds = regressor.nominal.model.jnt_range
    history = [f["torque_nm"] for f in frames[max(0, abort - 4) : abort]]
    reasons = []
    for lag in range(5):
        config = config_type(**{**row["filter"]["config"], "max_delay_steps": lag})
        joint_filter = filter_type(
            bounds[:, 0], bounds[:, 1], regressor.nominal.torque_limit, config
        )
        _, _, reason, _ = joint_filter.constraints(q, v, matrix, drift, history)
        reasons.append(reason)
    if reasons[-1] != frame["filter"]["reason"]:
        raise ValueError("saved abort reason does not reconstruct")
    return {
        "reasons_by_maximum_delay_steps": reasons,
        "actual_delay_bound_reason": reasons[row["interface_config"]["command_delay_steps"]],
        "scope": "same saved state; privileged delay diagnostic; no QP or trajectory claim",
    }
