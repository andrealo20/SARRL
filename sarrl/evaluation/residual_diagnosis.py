"""Read-only instrumentation of frozen residual controllers and their plant."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass

import numpy as np

from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm, PlanarArmParams
from sarrl.runtime import ControlStackConfig, SARRLControlStack
from sarrl.safety import HOCBFSafetyFilter

from .planar_v12 import planar_safety_config
from .safety_audit import evaluate_safety_episodes

TOLERANCE = 1e-10
CASES = {
    "id_reference": (9801800, 9801801),
    "ood_compound": (9801900, 9801901),
    "motor_fault": (9802000, 9802001),
}


def plain(value):
    """Copy numerical data into lossless JSON-compatible containers."""
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("non-finite diagnostic data")
    return value


def invalid_data(value):
    """Preserve failed diagnostics exactly, tagging only non-finite scalars."""
    if is_dataclass(value) and not isinstance(value, type):
        return invalid_data(asdict(value))
    if isinstance(value, np.ndarray):
        return invalid_data(value.tolist())
    if isinstance(value, np.generic):
        return invalid_data(value.item())
    if isinstance(value, dict):
        return {k: invalid_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [invalid_data(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def close(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.allclose(
        actual, expected, rtol=TOLERANCE, atol=TOLERANCE, equal_nan=False
    ):
        raise ValueError(f"diagnostic mismatch: {label}")


def exact(actual, expected, label):
    if plain(actual) != plain(expected):
        raise ValueError(f"diagnostic mismatch: {label}")


class ZeroResidualPolicy:
    def act(self, observation, deterministic=True):
        del observation, deterministic
        return np.zeros(2, dtype=np.float32)


class DiagnosticFailure(ValueError):
    def __init__(self, message, partial):
        super().__init__(message)
        self.partial = partial


class RecordingFilter(HOCBFSafetyFilter):
    """Capture the original projection and the constraints it actually used."""

    def __init__(self, model, config):
        super().__init__(model, config)
        self.last = None
        self.calls = 0

    def constraints(self, state, obstacles=()):
        result = super().constraints(state, obstacles)
        self._constraints = deepcopy(result)
        return result

    def filter(self, state, candidate, obstacles=()):
        if obstacles:
            raise ValueError("diagnostic row mapping requires no obstacles")
        result = super().filter(state, candidate, obstacles)
        A, bounds, current_safe = self._constraints
        if A.shape != (12, 2) or not np.array_equal(
            A[8:], np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        ):
            raise ValueError("unexpected constraint row mapping")
        exact(result.current_safe, current_safe, "filter current_safe")
        margins = A @ result.torque - bounds
        self.last = deepcopy(
            {
                "state": state,
                "candidate": candidate,
                "A": A,
                "bounds": bounds,
                "result": asdict(result),
                "candidate_margins": A @ candidate - bounds,
                "output_margins": margins,
                "near_active_rows": np.flatnonzero(np.abs(margins) <= self.config.feasibility_tol),
                "row_kinds": ["state"] * 8 + ["torque"] * 4,
            }
        )
        self.calls += 1
        return result


class _RecordingEnv:
    def __init__(self, env):
        self.env = env
        self.initial = None
        self.initial_observation = None
        self.last_reward = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        observation, info = self.env.reset(*args, **kwargs)
        self.initial = deepcopy(self.env.state_dict())
        self.initial_observation = observation.copy()
        return observation, info

    def step_torque(self, *args, **kwargs):
        result = self.env.step_torque(*args, **kwargs)
        self.last_reward = result[1]
        return result


class _RecordingStack:
    def __init__(self, stack, env):
        self.stack, self.env = stack, env
        self.before = None
        self.observation = None

    def __getattr__(self, name):
        return getattr(self.stack, name)

    def command(self, observation, state, q_des, **kwargs):
        self.before = deepcopy(self.env.state_dict())
        self.observation = np.asarray(observation).copy()
        return self.stack.command(observation, state, q_des, **kwargs)


def _unsaturated_baseline(baseline, state, q_des):
    q, qd = np.asarray(state)[:2], np.asarray(state)[2:]
    delta = np.asarray(q_des) - q
    acceleration = -baseline.kd * qd + baseline.kp * np.arctan2(np.sin(delta), np.cos(delta))
    return baseline.model.inverse_dynamics(q, qd, acceleration, include_friction=True)


def physical_balance(before, after, command, info, nominal, b_star):
    """Verify the observed transition, including fault timing and queue evolution."""
    state = np.asarray(before["state"])
    close(info["pre_step_state"], state, "pre-step state")
    b, r = command.baseline_torque, command.raw_residual
    c, h = b + r, command.torque
    u = np.asarray(info["commanded_torque_exact"])
    d = np.asarray(info["delayed_torque_exact"])
    a = np.asarray(info["plant_input_torque"])
    limit = before["constructor_config"]["torque_limit"]
    close(u, np.clip(h, -limit, limit), "plant clipping")
    queue = deepcopy(before["command_queue"])
    if before["action_delay"]:
        expected_delayed = queue.pop(0)
        queue.append(u.copy())
    else:
        expected_delayed = u
    close(d, expected_delayed, "delayed command")
    exact(after["command_queue"], queue, "command queue")
    gain = np.asarray(before["motor_gain"]).copy()
    params = deepcopy(before["arm_params"])
    fault = before["constructor_config"]["fault"]
    active = before["fault_active"]
    if fault and not active and before["steps"] >= fault["start_step"]:
        gain *= fault["motor_gain_multiplier"]
        params["payload_mass"] += fault["payload_delta"]
        active = True
    exact(after["fault_active"], active, "fault activation")
    exact(after["arm_params"], params, "transition plant parameters")
    exact(after["motor_gain"], gain, "transition motor gain")
    exact(after["steps"], before["steps"] + 1, "physical step count")
    close(a, d * gain, "applied torque")
    close(info["actuator_scaled_torque"], a, "actuator torque alias")
    terms = {
        "nominal_unclipped": b_star,
        "nominal_saturation": b - b_star,
        "residual": r,
        "projection": h - c,
        "plant_clipping": u - h,
        "delay": d - u,
        "motor_gain": a - d,
    }
    close(sum(terms.values()), a, "torque decomposition")
    real = PlanarArm(PlanarArmParams(**params))
    q, qd = state[:2], state[2:]
    load = real.coriolis_matrix(q, qd) @ qd + real.gravity_vector(q) + real.friction(qd)
    nominal_load = (
        nominal.coriolis_matrix(q, qd) @ qd + nominal.gravity_vector(q) + nominal.friction(qd)
    )
    net = a - load
    qdd = np.linalg.solve(real.mass_matrix(q), net)
    close(qdd, info["pre_step_acceleration"], "initial acceleration")
    close(real.step_rk4(state, a, before["dt"]), after["state"], "RK4 transition")
    return {
        "terms": terms,
        "commanded": u,
        "delayed": d,
        "applied": a,
        "mass_matrix": real.mass_matrix(q),
        "load": load,
        "nominal_load": nominal_load,
        "load_mismatch": load - nominal_load,
        "net": net,
        "qdd": qdd,
        "reward": None,
        "checks_passed": True,
    }


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def describe(record):
    traces = record["trace"]
    physical = [t for t in traces if t["executed"]]
    initial = record["initial"]
    arm = PlanarArm(PlanarArmParams(**initial["arm_params"]))
    initial_distance = float(
        np.linalg.norm(np.asarray(initial["target"]) - arm.forward_kinematics(initial["state"][:2]))
    )
    summary = {
        "outcome": "abort"
        if record["safety"]["safety_infeasible"]
        else ("success" if record["episode"]["success"] else "timeout"),
        "initial_distance_m": initial_distance,
        "minimum_distance_m": min(initial_distance, *(t["distance_m"] for t in traces)),
        "minimum_callback_distance_m": min(t["distance_m"] for t in traces),
        "final_distance_m": record["episode"]["final_distance"],
        "final_speed_rad_s": traces[-1]["speed_rad_s"],
        "tail50": None,
        "off_target_stop": False,
    }
    if len(physical) >= 50:
        tail = physical[-50:]
        distances = [t["distance_m"] for t in tail]
        speeds = [t["speed_rad_s"] for t in tail]
        states = np.asarray([t["after"]["state"] for t in tail])
        summary["tail50"] = {
            "distance_m": _stats(distances),
            "speed_rad_s": _stats(speeds),
            "state_range": np.ptp(states, axis=0).tolist(),
            "distance_range_m": float(np.ptp(distances)),
        }
        for name in ("qdd", "net"):
            norms = np.linalg.norm([t["physical"][name] for t in tail], axis=1)
            summary["tail50"][name + "_norm"] = {
                "max": float(np.max(norms)),
                "rms": float(np.sqrt(np.mean(norms**2))),
            }
        summary["off_target_stop"] = bool(
            summary["outcome"] == "timeout" and min(distances) > 0.05 and max(speeds) <= 0.06
        )
    return summary


def evaluate_recorded(policy, env, *, seed, scenario, controller):
    """Evaluate one episode with the canonical evaluator and an independent trace."""
    if env.mode != "torque":
        raise ValueError("diagnosis requires torque mode")
    nominal = PlanarArm()
    safety = RecordingFilter(nominal, planar_safety_config())
    baseline = ComputedTorqueController(nominal)
    inner = SARRLControlStack(
        baseline, policy, ControlStackConfig(require_safety=True), safety_filter=safety
    )
    wrapped = _RecordingEnv(env)
    stack = _RecordingStack(inner, env)
    trace = []
    pending = {}

    def observe(event):
        before, command = stack.before, event["command"]
        after = deepcopy(env.state_dict())
        pending.clear()
        pending.update(
            {
                "event": event,
                "before": before,
                "after": after,
                "observation": stack.observation,
                "filter": safety.last,
            }
        )
        exact(event["step"], len(trace), "attempt index")
        exact(safety.calls, len(trace) + 1, "single filter call")
        exact(event["state"], before["state"], "command input state")
        exact(safety.last["state"], before["state"], "filter input state")
        close(
            safety.last["candidate"],
            command.baseline_torque + command.raw_residual,
            "filter candidate",
        )
        b_star = _unsaturated_baseline(baseline, before["state"], before["q_des"])
        close(
            np.clip(b_star, -baseline.torque_limit, baseline.torque_limit),
            command.baseline_torque,
            "nominal saturation",
        )
        info = event["info"]
        dynamics = None
        if info is None:
            exact(after, before, "abort preserves complete plant state and RNG")
            exact(command.executable, False, "abort executable")
        else:
            exact(command.safety_certified, True, "required certificate")
            dynamics = physical_balance(before, after, command, info, nominal, b_star)
            state = np.asarray(after["state"])
            distance = float(np.linalg.norm(env.target - env.arm.forward_kinematics(state[:2])))
            expected_reward = (
                -distance
                - 0.01 * float(state[2:] @ state[2:])
                - 0.0002 * float(dynamics["commanded"] @ dynamics["commanded"])
                + (10.0 if info["success"] else 0.0)
            )
            close(wrapped.last_reward, expected_reward, "physical reward")
            dynamics["reward"] = wrapped.last_reward
        trace.append(
            plain(
                {
                    "attempt": event["step"],
                    "physical_step": before["steps"],
                    "executed": info is not None,
                    "before": before,
                    "after": after,
                    "observation": stack.observation,
                    "command": asdict(command),
                    "normalized_action": command.raw_residual
                    / np.asarray(inner.config.residual_limit),
                    "nominal_unclipped": b_star,
                    "filter": safety.last,
                    "physical": dynamics,
                    "terminated": event["terminated"],
                    "truncated": event["truncated"],
                    "distance_m": float(
                        np.linalg.norm(env.target - env.arm.forward_kinematics(env.state[:2]))
                    ),
                    "speed_rad_s": float(np.linalg.norm(env.state[2:])),
                }
            )
        )

    try:
        outcomes, diagnostics = evaluate_safety_episodes(
            stack,
            HOCBFSafetyFilter(nominal, planar_safety_config()),
            wrapped,
            episodes=1,
            seed=seed,
            scenario=scenario,
            controller=controller,
            transition_callback=observe,
        )
    except Exception as exc:
        # Failed values retain full precision; tagged non-finite values cannot look valid.
        raise DiagnosticFailure(
            str(exc),
            {
                "status": "invalid",
                "trace": trace,
                "pending": invalid_data(pending),
                "initial": invalid_data(wrapped.initial),
                "final": invalid_data(env.state_dict()),
            },
        ) from exc
    safety_row = asdict(diagnostics[0])
    if not outcomes[0].steps:
        safety_row["executed_constraint_margin_min"] = None
    record = plain(
        {
            "schema_version": 1,
            "status": "valid",
            "initial": wrapped.initial,
            "initial_observation": wrapped.initial_observation,
            "final": env.state_dict(),
            "episode": asdict(outcomes[0]),
            "safety": safety_row,
            "configuration": {
                "nominal": asdict(nominal.params),
                "safety": asdict(safety.config),
                "stack": asdict(inner.config),
                "kp": baseline.kp,
                "kd": baseline.kd,
            },
            "trace": trace,
        }
    )
    try:
        close(
            sum(t["physical"]["reward"] for t in trace if t["executed"]),
            record["episode"]["reward"],
            "episode reward",
        )
        exact(len(trace), safety_row["command_attempts"], "attempt count")
        exact(sum(t["executed"] for t in trace), record["episode"]["steps"], "physical count")
        record["summary"] = describe(record)
    except Exception as exc:
        record["status"] = "invalid"
        raise DiagnosticFailure(str(exc), record) from exc
    return record


def compare_historical(record, historical):
    """Compare only fields actually retained by the previous callback."""
    for name in ("controller", "scenario", "seed", "steps", "success"):
        exact(record["episode"][name], historical["episode"][name], "historical " + name)
    close(
        record["episode"]["final_distance"],
        historical["episode"]["final_distance"],
        "historical final distance",
    )
    exact(
        record["safety"]["safety_infeasible"],
        historical["safety"]["safety_infeasible"],
        "historical abort",
    )
    exact(len(record["trace"]), len(historical["trace"]), "historical trace length")
    for new, old in zip(record["trace"], historical["trace"], strict=True):
        for name in ("attempt", "executed"):
            exact(new[name], old[name], "historical " + name)
        for name in ("distance_m", "speed_rad_s"):
            close(new[name], old[name], "historical " + name)
        close(new["command"]["raw_residual"], old["raw_residual"], "historical residual")
        close(new["command"]["safety_correction"], old["correction"], "historical correction")


def paired_contrast(learned, zero):
    """Reference one shared zero-control cell without replicating its evidence."""
    for field in ("scenario", "seed"):
        exact(learned["episode"][field], zero["episode"][field], "paired " + field)
    exact(learned["initial"], zero["initial"], "paired initial plant and RNG")
    exact(learned["initial_observation"], zero["initial_observation"], "paired observation")
    return {
        "learned_key": [learned["episode"][k] for k in ("controller", "scenario", "seed")],
        "zero_key": [zero["episode"][k] for k in ("scenario", "seed")],
        "learned_outcome": learned["summary"]["outcome"],
        "zero_outcome": zero["summary"]["outcome"],
        "learned_summary": learned["summary"],
        "zero_summary": zero["summary"],
        "same_initial_state": True,
        "final_distance_difference_m": (
            learned["episode"]["final_distance"] - zero["episode"]["final_distance"]
        ),
        "common_terminal_time": learned["episode"]["steps"] == zero["episode"]["steps"],
    }
