"""Separate recording adapter and fixed decision rule for the integral nominal study."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import numpy as np

from sarrl.controllers.integral_nominal import IntegralNominalController
from sarrl.dynamics import PlanarArm
from sarrl.runtime import ControlStackConfig, SARRLControlStack
from sarrl.safety import HOCBFSafetyFilter

from .planar_v12 import planar_safety_config
from .residual_diagnosis import (
    CASES,
    DiagnosticFailure,
    RecordingFilter,
    ZeroResidualPolicy,
    _RecordingEnv,
    _RecordingStack,
    close,
    describe,
    evaluate_recorded,
    exact,
    invalid_data,
    physical_balance,
    plain,
)
from .safety_audit import evaluate_safety_episodes


class _CountingZeroPolicy(ZeroResidualPolicy):
    def __init__(self):
        self.calls = 0

    def act(self, observation, deterministic=True):
        self.calls += 1
        return super().act(observation, deterministic)


class _IntegralStack(_RecordingStack):
    def command(self, *args, **kwargs):
        self.last_command = super().command(*args, **kwargs)
        return self.last_command


class IntegralRecorder:
    """Observe canonical callbacks; commit nominal state only after verified execution."""

    def __init__(self, env):
        if env.mode != "torque" or env.dt != 0.02 or env.torque_limit != 40.0:
            raise ValueError("integral study requires torque mode, dt=0.02 and limit=40")
        self.env = env
        self.nominal = PlanarArm()
        self.baseline = IntegralNominalController(self.nominal)
        self.safety = RecordingFilter(self.nominal, planar_safety_config())
        self.policy = _CountingZeroPolicy()
        self.inner = SARRLControlStack(
            self.baseline,
            self.policy,
            ControlStackConfig(require_safety=True),
            safety_filter=self.safety,
        )
        self.wrapped = _RecordingEnv(env)
        self.stack = _IntegralStack(self.inner, env)
        self.trace = []
        self.pending = {}
        self._started = False

    def observe(self, event):
        before, command = self.stack.before, event["command"]
        after = deepcopy(self.env.state_dict())
        transaction = self.baseline.pending
        self.pending = {
            "event": event,
            "before": before,
            "after": after,
            "observation": self.stack.observation,
            "filter": self.safety.last,
            "integral": transaction,
        }
        if command is not self.stack.last_command or transaction is None:
            raise ValueError("callback does not match pending command")
        exact(event["step"], len(self.trace), "attempt index")
        exact(self.safety.calls, len(self.trace) + 1, "single filter call")
        exact(self.policy.calls, len(self.trace) + 1, "single policy call")
        exact(event["state"], before["state"], "command input state")
        exact(self.safety.last["state"], before["state"], "filter input state")
        exact(command.raw_residual, np.zeros(2), "zero policy residual")
        exact(command.gated_residual, np.zeros(2), "zero gated residual")
        close(self.safety.last["candidate"], transaction["b"], "filter candidate")
        close(command.baseline_torque, transaction["b"], "nominal clipping")
        raw = transaction["tau_raw"]
        close(
            transaction["nominal_pd_unclipped"] + transaction["integral_torque"],
            raw,
            "integral torque decomposition",
        )
        info = event["info"]
        dynamics = None
        # Validate and serialize the physical evidence before changing integral state.
        if info is None:
            exact(after, before, "abort preserves complete plant state and RNG")
            exact(command.executable, False, "abort executable")
            exact(command.safety_certified, False, "abort certificate")
        else:
            exact(command.executable, True, "executable command")
            exact(command.safety_certified, True, "required certificate")
            close(command.torque, self.safety.last["result"]["torque"], "certified torque")
            dynamics = physical_balance(before, after, command, info, self.nominal, raw)
            state = np.asarray(after["state"])
            distance = float(
                np.linalg.norm(self.env.target - self.env.arm.forward_kinematics(state[:2]))
            )
            reward = (
                -distance
                - 0.01 * float(state[2:] @ state[2:])
                - 0.0002 * float(dynamics["commanded"] @ dynamics["commanded"])
                + (10.0 if info["success"] else 0.0)
            )
            close(self.wrapped.last_reward, reward, "physical reward")
            dynamics["reward"] = self.wrapped.last_reward
        row = plain(
            {
                "attempt": event["step"],
                "physical_step": before["steps"],
                "executed": info is not None,
                "before": before,
                "after": after,
                "observation": self.stack.observation,
                "command": asdict(command),
                "normalized_action": command.raw_residual
                / np.asarray(self.inner.config.residual_limit),
                "nominal_unclipped": raw,
                "nominal_pd_unclipped": transaction["nominal_pd_unclipped"],
                "integral_torque": transaction["integral_torque"],
                "filter": self.safety.last,
                "physical": dynamics,
                "terminated": event["terminated"],
                "truncated": event["truncated"],
                "distance_m": float(
                    np.linalg.norm(
                        self.env.target - self.env.arm.forward_kinematics(self.env.state[:2])
                    )
                ),
                "speed_rad_s": float(np.linalg.norm(self.env.state[2:])),
            }
        )
        if info is None:
            integral = self.baseline.abort(transaction["token"])
        else:
            # Never feed delayed/applied torque, gain or payload into the controller.
            sent = np.clip(command.torque, -self.env.torque_limit, self.env.torque_limit)
            exact(sent, info["commanded_torque_exact"], "exact sent command")
            integral = self.baseline.commit(transaction["token"], command.torque, sent)
        row["integral"] = plain(integral)
        self.trace.append(row)
        exact(self.safety.calls, len(self.trace), "recorded filter calls")
        self.pending = {}

    def evaluate(self, *, seed, scenario):
        if self._started:
            raise ValueError("an integral recorder can evaluate only one episode")
        self._started = True
        try:
            outcomes, diagnostics = evaluate_safety_episodes(
                self.stack,
                HOCBFSafetyFilter(self.nominal, planar_safety_config()),
                self.wrapped,
                episodes=1,
                seed=seed,
                scenario=scenario,
                controller="I1",
                transition_callback=self.observe,
            )
            safety_row = asdict(diagnostics[0])
            if not outcomes[0].steps:
                safety_row["executed_constraint_margin_min"] = None
            record = plain(
                {
                    "schema_version": 1,
                    "status": "valid",
                    "model_identity": None,
                    "initial": self.wrapped.initial,
                    "initial_observation": self.wrapped.initial_observation,
                    "final": self.env.state_dict(),
                    "episode": asdict(outcomes[0]),
                    "safety": safety_row,
                    "configuration": {
                        "nominal": asdict(self.nominal.params),
                        "safety": asdict(self.safety.config),
                        "stack": asdict(self.inner.config),
                        "kp": self.baseline.kp,
                        "kd": self.baseline.kd,
                    },
                    "integral_configuration": {
                        "ki": self.baseline.ki,
                        "kaw": self.baseline.kaw,
                        "dt": self.baseline.dt,
                        "z_limit": self.baseline.z_limit,
                    },
                    "integral_final": self.baseline.state_dict(),
                    "trace": self.trace,
                }
            )
            close(
                sum(t["physical"]["reward"] for t in self.trace if t["executed"]),
                record["episode"]["reward"],
                "episode reward",
            )
            exact(len(self.trace), safety_row["command_attempts"], "attempt count")
            exact(sum(t["executed"] for t in self.trace), record["episode"]["steps"], "step count")
            record["summary"] = describe(record)
            record["integral_summary"] = integral_summary(self.trace)
            return record
        except Exception as exc:
            raise DiagnosticFailure(
                str(exc),
                invalid_data(
                    {
                        "status": "invalid",
                        "trace": self.trace,
                        "pending": self.pending,
                        "initial": self.wrapped.initial,
                        "final": self.env.state_dict(),
                        "integral_final": self.baseline.state_dict(),
                        "command_context": {
                            "before": self.stack.before,
                            "observation": self.stack.observation,
                            "command": getattr(self.stack, "last_command", None),
                            "filter": self.safety.last,
                        },
                    }
                ),
            ) from exc


def evaluate_nominal(env, *, seed, scenario, controller):
    if controller == "R0":
        return evaluate_recorded(
            ZeroResidualPolicy(), env, seed=seed, scenario=scenario, controller="R0"
        )
    if controller == "I1":
        return IntegralRecorder(env).evaluate(seed=seed, scenario=scenario)
    raise ValueError("unknown nominal controller")


def integral_summary(trace):
    physical = [row["integral"] for row in trace if row["executed"]]

    def stats(rows):
        result = {"steps": len(rows)}
        for field in ("aw_clip", "aw_hocbf", "aw_sendclip", "z_after", "integral_clipping"):
            if not rows:
                result[field] = None
                continue
            vectors = np.asarray([row[field] for row in rows])
            norms = np.linalg.norm(vectors, axis=1)
            result[field] = {
                "mean_vector": vectors.mean(axis=0),
                "rms_norm": float(np.sqrt(np.mean(norms**2))),
                "max_norm": float(norms.max()),
                "nonzero_attempts": int(np.count_nonzero(np.any(vectors != 0.0, axis=1))),
            }
        return result

    return plain({"episode": stats(physical), "tail_available": stats(physical[-50:])})


COMMON_FIELDS = (
    "episode",
    "safety",
    "initial",
    "initial_observation",
    "final",
    "configuration",
    "trace",
    "summary",
)
# Additions are excluded only at these explicit paths, never by substring matching.
INTEGRAL_TOP_FIELDS = {"integral_configuration", "integral_final", "integral_summary"}
INTEGRAL_TRACE_FIELDS = {"integral", "nominal_pd_unclipped", "integral_torque"}


def compare_tree(actual, expected, path="record"):
    """Strict JSON structure and discrete values; tolerant floats except inside RNG state."""
    if type(actual) is not type(expected):
        raise ValueError(f"type mismatch at {path}")
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"keys mismatch at {path}")
        for key in actual:
            if "rng" in key.lower():
                # Recursion below with exact floats also enforces bool/int type distinctions.
                compare_tree_exact(actual[key], expected[key], f"{path}.{key}")
            else:
                compare_tree(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise ValueError(f"length mismatch at {path}")
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            compare_tree(a, b, f"{path}[{index}]")
    elif isinstance(actual, float):
        close(actual, expected, path)
    elif actual != expected:
        raise ValueError(f"value mismatch at {path}")


def compare_tree_exact(actual, expected, path):
    if type(actual) is not type(expected):
        raise ValueError(f"type mismatch at {path}")
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"keys mismatch at {path}")
        for key in actual:
            compare_tree_exact(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise ValueError(f"length mismatch at {path}")
        for a, b in zip(actual, expected, strict=True):
            compare_tree_exact(a, b, path)
    elif actual != expected:
        raise ValueError(f"value mismatch at {path}")


def compare_reference(record, historical):
    current, old = deepcopy(record), deepcopy(historical)
    for item, label in ((current, "R0"), (old, "Z")):
        exact(item["status"], "valid", "reference status")
        if item.get("model_identity") is not None or item.get("identity") is not None:
            raise ValueError("reference must have no model identity")
        for section in ("episode", "safety"):
            exact(item[section]["controller"], label, "reference label")
        allowed = (
            set(COMMON_FIELDS)
            | {
                "schema_version",
                "status",
                "model_identity",
                "identity",
            }
            | INTEGRAL_TOP_FIELDS
        )
        if set(item) - allowed:
            raise ValueError("unexpected reference top-level fields")
        for row in item["trace"]:
            for field in INTEGRAL_TRACE_FIELDS:
                row.pop(field, None)
    current["episode"]["controller"] = "Z"
    current["safety"]["controller"] = "Z"
    for field in COMMON_FIELDS:
        compare_tree(current[field], old[field], field)


def decision(records):
    """Apply the ordered exploratory rule only to the full fixed matrix."""
    expected = {(s, seed, c) for s, seeds in CASES.items() for seed in seeds for c in ("R0", "I1")}
    try:
        cells = {
            (r["episode"]["scenario"], r["episode"]["seed"], r["episode"]["controller"]): r
            for r in records
        }
        if len(records) != 12 or set(cells) != expected:
            return "incomplete"
        if any(r["status"] != "valid" for r in records):
            return "incomplete"
        for scenario, seeds in CASES.items():
            for seed in seeds:
                r0, i1 = (cells[scenario, seed, c]["safety"] for c in ("R0", "I1"))
                if (
                    (i1["unsafe_episode"] and not r0["unsafe_episode"])
                    or (i1["safety_infeasible"] and not r0["safety_infeasible"])
                    or (
                        r0["unsafe_episode"]
                        and i1["normalized_violation_max"] > r0["normalized_violation_max"] + 1e-10
                    )
                ):
                    return "no_go_safety"
        if all(
            cells[s, seed, "I1"]["episode"]["success"]
            for s, seed in (
                ("id_reference", 9801800),
                ("motor_fault", 9802001),
            )
        ):
            return "target_cases_passed"
        return "hypothesis_not_met"
    except (KeyError, TypeError):
        return "incomplete"
