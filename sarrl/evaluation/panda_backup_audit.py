"""Sequential audit of backup witnesses without the verifier's rollout or metric code."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=16)
def audit_metrics(dt, maximum, kp, kd):
    size = 2 + maximum
    weight = np.diag([kp, 1] + [dt**2] * maximum)
    result = []
    for lag in range(maximum + 1):

        def advance(state, lag=lag):
            sent = -kp * state[0] - kd * state[1]
            delivered = state[-lag] if lag else sent
            return np.array(
                [state[0] + dt * state[1] + 0.5 * dt**2 * delivered, state[1] + dt * delivered]
                + ([*state[3:], sent] if maximum else [])
            )

        transition = np.column_stack([advance(basis) for basis in np.eye(size)])
        if np.max(np.abs(np.linalg.eigvals(transition))) >= 1:
            raise ValueError("audited backup map is unstable")
        operator = np.eye(size**2) - np.kron(transition.T, transition.T)
        metric = np.linalg.solve(operator, weight.reshape(-1, order="F")).reshape(
            size, size, order="F"
        )
        np.linalg.cholesky(metric)
        np.testing.assert_allclose(transition.T @ metric @ transition - metric, -weight, atol=1e-9)
        result.append((metric, np.kron(np.linalg.inv(metric), np.eye(7))))
    return tuple(result)


def audit_radius(policy, lag, lower, upper, velocity, limit):
    cfg = policy["config"]
    metric, inverse = audit_metrics(
        cfg["control_dt"], cfg["max_delay_steps"], cfg["kp"], cfg["kd"]
    )[lag]
    size, dt = len(metric), cfg["control_dt"]
    anchor, mass, equilibrium = (
        np.asarray(policy[key]) for key in ("anchor", "mass", "equilibrium_torque")
    )
    slack = np.minimum(anchor - lower, upper - anchor)
    actuator_slack = limit - np.abs(equilibrium)
    if np.any(slack <= 0) or np.any(actuator_slack <= 0):
        return 0.0
    radii = []

    def support(row):
        value = row.ravel()
        return np.sqrt(value @ inverse @ value)

    for joint in range(7):
        position, speed, acceleration = (np.zeros((size, 7)) for _ in range(3))
        position[0, joint], speed[1, joint] = 1, 1
        if lag:
            acceleration[size - lag, joint] = 1
        else:
            acceleration[0, joint], acceleration[1, joint] = -cfg["kp"], -cfg["kd"]
        radii.append(
            slack[joint]
            / (support(position) + dt * support(speed) + 0.5 * dt**2 * support(acceleration))
        )
        radii.append(velocity[joint] / (support(speed) + dt * support(acceleration)))
        feedback = np.zeros((size, 7))
        feedback[0], feedback[1] = -cfg["kp"] * mass[joint], -cfg["kd"] * mass[joint]
        radii.append(actuator_slack[joint] / support(feedback))
        for index in range(2, size):
            queued = np.zeros((size, 7))
            queued[index] = mass[joint]
            radii.append(actuator_slack[joint] / support(queued))
    return float(min(radii) ** 2)


def interval_valid(q, v, acceleration, dt, lower, upper, velocity, tol=1e-8):
    for joint in range(7):
        times = [0.0, dt]
        if abs(acceleration[joint]) > 1e-14:
            turn = -v[joint] / acceleration[joint]
            if 0 < turn < dt:
                times.append(turn)
        positions = [q[joint] + t * v[joint] + 0.5 * t**2 * acceleration[joint] for t in times]
        if min(positions) < lower[joint] - tol or max(positions) > upper[joint] + tol:
            return False
        if max(abs(v[joint]), abs(v[joint] + dt * acceleration[joint])) > velocity[joint] + tol:
            return False
    return True


def audit_witness(record, q, v, candidate, history, lower, upper, velocity, limit):
    policy, cases = record["policy"], record["cases"]
    digest = hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if digest != record["policy_sha256"]:
        raise ValueError("witness policy hash differs")
    np.testing.assert_array_equal(record["candidate_torque_nm"], candidate)
    cfg = policy["config"]
    dt, maximum, horizon = cfg["control_dt"], cfg["max_delay_steps"], cfg["horizon_steps"]
    matrix, drift, mass, anchor, equilibrium = (
        np.asarray(policy[key], dtype=float)
        for key in ("matrix", "drift", "mass", "anchor", "equilibrium_torque")
    )
    if any(not np.all(np.isfinite(value)) for value in (matrix, drift, mass, anchor, equilibrium)):
        raise ValueError("nonfinite witness model")
    np.linalg.cholesky(matrix)
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(matrix @ mass, np.eye(7), rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(equilibrium, -mass @ drift, rtol=1e-9, atol=1e-10)
    np.testing.assert_array_equal(policy["torque_limit"], limit)
    np.testing.assert_array_equal(anchor, q)
    if [case["lag"] for case in cases] != list(range(maximum + 1)):
        raise ValueError("witness must cover each configured delay once")
    if any(np.any(np.abs(command) > limit) for command in [candidate, *history]):
        raise ValueError("unbounded witness input")
    checked = accepted = 0
    for lag, case in enumerate(cases):
        position, speed = np.array(q, copy=True), np.array(v, copy=True)
        queue = [np.zeros(7) for _ in range(max(0, maximum - len(history)))]
        if maximum:
            queue += [np.array(u) for u in history[-maximum:]]
        radius = audit_radius(policy, lag, lower, upper, velocity, limit)
        energy = None
        reason, intervals = (
            ("empty_terminal_set", 0) if radius <= 0 else ("terminal_not_reached", horizon)
        )
        if radius > 0:
            metric = audit_metrics(dt, maximum, cfg["kp"], cfg["kd"])[lag][0]
            for step in range(horizon):
                sent = (
                    candidate
                    if not step
                    else equilibrium - mass @ (cfg["kp"] * (position - anchor) + cfg["kd"] * speed)
                )
                if np.any(np.abs(sent) > limit):
                    reason, intervals = "backup_torque_limit", step
                    break
                delivered = queue[-lag] if lag else sent
                acceleration = matrix @ delivered + drift
                if not interval_valid(position, speed, acceleration, dt, lower, upper, velocity):
                    reason = (
                        "pending_command_violation"
                        if step < lag
                        else "candidate_interval_violation"
                        if step == lag
                        else "backup_interval_violation"
                    )
                    intervals = step + 1
                    break
                position = position + dt * speed + 0.5 * dt**2 * acceleration
                speed = speed + dt * acceleration
                if maximum:
                    queue = queue[1:] + [sent]
                state = np.array([position - anchor, speed] + [matrix @ u + drift for u in queue])
                energy = float(np.trace(state.T @ metric @ state))
                if energy <= radius:
                    reason, intervals = "terminal_set_reached", step + 1
                    break
        is_accepted = reason == "terminal_set_reached"
        if (case["reason"], case["checked_intervals"], case["accepted"]) != (
            reason,
            intervals,
            is_accepted,
        ):
            raise ValueError("sequential backup witness decision differs")
        if reason in ("terminal_set_reached", "terminal_not_reached"):
            np.testing.assert_allclose(
                case["terminal_radius_squared"], radius, rtol=1e-9, atol=1e-10
            )
            np.testing.assert_allclose(case["terminal_energy"], energy, rtol=1e-9, atol=1e-10)
        elif case["terminal_energy"] is not None or case["terminal_radius_squared"] is not None:
            raise ValueError("unexpected terminal claim on a failed prefix")
        checked += intervals
        accepted += int(is_accepted)
    if record["accepted"] != (accepted == maximum + 1):
        raise ValueError("aggregate witness acceptance differs")
    return {"witnesses": 1, "delay_cases": maximum + 1, "checked_intervals": checked}
