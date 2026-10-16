"""Latched Panda command supervision with immutable accepted backup context."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict

import numpy as np

from sarrl.robots.panda import vector7
from sarrl.safety.panda_backup import PandaBackupConfig, PandaBackupWitness


def policy_record(policy):
    return {
        name: getattr(policy, name).tolist()
        for name in ("anchor", "matrix", "drift", "mass", "equilibrium_torque", "torque_limit")
    } | {"config": asdict(policy.config)}


def policy_identity(policy):
    value = json.dumps(
        policy_record(policy), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(value.encode()).hexdigest()


def witness_record(result, candidate):
    return {
        "accepted": result.accepted,
        "candidate_torque_nm": np.asarray(candidate).tolist(),
        "policy": policy_record(result.policy),
        "policy_sha256": policy_identity(result.policy),
        "cases": [asdict(case) for case in result.cases],
    }


class PandaBackupSupervisor:
    def __init__(self, lower, upper, torque_limit, config=None):
        self.config = config or PandaBackupConfig()
        self.witness = PandaBackupWitness(lower, upper, torque_limit, self.config)
        self.joint_filter = self.witness.bounds
        self.retained_policy = self.emergency_policy = None
        self.policy_step = self.policy_origin = None
        self.aborted = False
        self.abort_step = self.abort_reason = None
        self.steps = self.accepted_proposals = self.witness_calls = 0
        self.backup_steps = self.emergency_steps = self.saturated_steps = 0
        self.reasons = Counter()
        self.last = None

    def _latch(self, reason):
        if not self.aborted:
            self.aborted, self.abort_step, self.abort_reason = True, self.steps, reason
            self.reasons[reason] += 1

    def _verify(self, q, v, torque, matrix, drift, history):
        self.witness_calls += 1
        return self.witness.verify(q, v, torque, matrix, drift, history)

    def _fallback(self, q, v):
        policy = self.retained_policy or self.emergency_policy
        raw = policy.requested_torque(q, v) if policy is not None else np.zeros(7)
        if not np.all(np.isfinite(raw)):
            raise ValueError("fallback torque is nonfinite")
        sent = np.clip(raw, -self.joint_filter.limit, self.joint_filter.limit)
        self.last.update(
            mode="backup" if self.retained_policy is not None else "emergency",
            raw_torque_nm=raw.tolist(),
            saturated=bool(np.any(raw != sent)),
        )
        self.backup_steps += int(self.retained_policy is not None)
        self.emergency_steps += int(self.retained_policy is None)
        self.saturated_steps += int(np.any(raw != sent))
        return sent

    def _finish(self, sent):
        self.last.update(
            step=self.steps,
            aborted=self.aborted,
            abort_step=self.abort_step,
            abort_reason=self.abort_reason,
            policy_step=self.policy_step,
            policy_origin=self.policy_origin,
            retained_policy_sha256=(
                policy_identity(self.retained_policy) if self.retained_policy is not None else None
            ),
            torque_nm=sent.tolist(),
        )
        self.steps += 1
        return sent.copy()

    def command(self, q, v, nominal, matrix, drift, history, *, model_error=None):
        q, v = vector7(q, "q"), vector7(v, "v")
        self.last = {
            "initial_policy": None,
            "initial_witness": None,
            "proposal_witness": None,
            "projection": None,
            "model_error": model_error,
        }
        if self.aborted:
            return self._finish(self._fallback(q, v))
        if model_error is not None:
            self._latch("estimated_model_error")
            return self._finish(self._fallback(q, v))
        nominal = vector7(nominal, "nominal")
        if not self.steps:
            initial_policy = self.witness.policy(q, matrix, drift)
            self.last["initial_policy"] = policy_record(initial_policy)
            self.emergency_policy = initial_policy
            raw = initial_policy.requested_torque(q, v)
            if np.any(np.abs(raw) > self.joint_filter.limit):
                self._latch("unrecoverable_initial_torque")
                return self._finish(self._fallback(q, v))
            initial = self._verify(q, v, raw, matrix, drift, history)
            self.last["initial_witness"] = witness_record(initial, raw)
            if not initial.accepted:
                self._latch("unrecoverable_initial_witness")
                return self._finish(self._fallback(q, v))
            self.retained_policy, self.policy_step, self.policy_origin = (
                initial.policy,
                0,
                "initial",
            )
        projection = self.joint_filter.filter(q, v, nominal, matrix, drift, history)
        self.last["projection"] = projection.diagnostics()
        if not projection.predicted_feasible:
            self._latch("projection_" + projection.reason)
            return self._finish(self._fallback(q, v))
        result = self._verify(q, v, projection.torque, matrix, drift, history)
        self.last["proposal_witness"] = witness_record(result, projection.torque)
        if not result.accepted:
            self._latch("proposal_witness_rejected")
            return self._finish(self._fallback(q, v))
        self.retained_policy, self.policy_step, self.policy_origin = (
            result.policy,
            self.steps,
            "proposal",
        )
        self.accepted_proposals += 1
        self.last.update(mode="task", raw_torque_nm=projection.torque.tolist(), saturated=False)
        return self._finish(projection.torque)

    def diagnostics(self):
        return {
            "config": asdict(self.config),
            "aborted": self.aborted,
            "abort_step": self.abort_step,
            "abort_reason": self.abort_reason,
            "steps": self.steps,
            "accepted_proposals": self.accepted_proposals,
            "witness_calls": self.witness_calls,
            "backup_steps": self.backup_steps,
            "emergency_steps": self.emergency_steps,
            "saturated_steps": self.saturated_steps,
            "reasons": dict(self.reasons),
            "claim": "frozen estimated-model witness; no physical safety certificate",
        }
