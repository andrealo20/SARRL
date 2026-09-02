from types import SimpleNamespace

import numpy as np
import pytest

from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm
from sarrl.envs import PlanarReachEnv
from sarrl.evaluation import (
    evaluate_safety_episodes,
    paired_diagnostic_difference,
    safety_envelope_violation,
)
from sarrl.runtime import ControlStackConfig, SARRLControlStack
from sarrl.safety import HOCBFSafetyFilter, SafetyConfig, SafetyResult


class ZeroPolicy:
    def act(self, obs, deterministic=True):
        del obs, deterministic
        return np.zeros(2, dtype=np.float32)


def _stack(*, safety: bool = False):
    nominal = PlanarArm()
    return SARRLControlStack(
        ComputedTorqueController(nominal),
        ZeroPolicy(),
        ControlStackConfig(require_safety=safety),
        safety_filter=HOCBFSafetyFilter(nominal) if safety else None,
    )


def test_safety_envelope_violation_reports_physical_units_and_normalized_excess():
    config = SafetyConfig()
    unsafe, position, velocity, normalized = safety_envelope_violation(
        np.array([3.15, 0.0, 7.7, 0.0]), config
    )

    assert unsafe
    assert position == pytest.approx(0.10)
    assert velocity == pytest.approx(0.70)
    assert normalized == pytest.approx(0.10)


def test_safety_audit_observes_initial_and_every_transition():
    env = PlanarReachEnv(mode="torque")
    observer = HOCBFSafetyFilter(PlanarArm())
    captured = []
    rows, diagnostics = evaluate_safety_episodes(
        _stack(),
        observer,
        env,
        episodes=2,
        seed=50_000,
        transition_callback=captured.append,
    )

    assert len(rows) == len(diagnostics) == 2
    assert len(captured) == sum(row.command_attempts for row in diagnostics)
    assert all(item["state"].shape == (4,) for item in captured)
    assert all(item["info"] is not None for item in captured)
    assert captured[-1]["terminated"] or captured[-1]["truncated"]
    assert all(row.state_observations == row.steps + 1 for row in diagnostics)
    assert all(row.command_attempts == row.steps for row in diagnostics)
    assert all(not row.safety_enabled for row in diagnostics)
    assert all(row.safety_certified_steps == 0 for row in diagnostics)


def test_safety_audit_retains_rejected_required_safety_attempt(monkeypatch):
    stack = _stack(safety=True)
    captured = []

    def infeasible(state, candidate, obstacles=()):
        del state, obstacles
        return SafetyResult(
            torque=np.asarray(candidate, dtype=np.float64),
            success=False,
            correction_norm=1.5,
            min_margin=-2.0,
            active_constraints=(),
            current_safe=True,
        )

    monkeypatch.setattr(stack.safety_filter, "filter", infeasible)
    rows, diagnostics = evaluate_safety_episodes(
        stack,
        HOCBFSafetyFilter(PlanarArm()),
        PlanarReachEnv(mode="torque"),
        episodes=1,
        seed=50_000,
        transition_callback=captured.append,
    )

    assert rows[0].steps == 0
    assert not rows[0].success
    assert diagnostics[0].state_observations == 1
    assert diagnostics[0].command_attempts == 1
    assert diagnostics[0].safety_infeasible
    assert np.isnan(diagnostics[0].executed_constraint_margin_min)
    assert len(captured) == 1
    assert captured[0]["info"] is None
    assert not captured[0]["command"].executable


def test_paired_diagnostic_difference_requires_unique_matching_seeds():
    filtered = [SimpleNamespace(seed=1, value=0.0), SimpleNamespace(seed=2, value=0.5)]
    reference = [SimpleNamespace(seed=1, value=1.0), SimpleNamespace(seed=2, value=0.5)]
    difference, low, high = paired_diagnostic_difference(
        filtered,
        reference,
        lambda row: row.value,
        bootstrap=500,
        seed=7,
    )

    assert difference == pytest.approx(-0.5)
    assert low <= difference <= high
    with pytest.raises(ValueError, match="unique episode seeds"):
        paired_diagnostic_difference(
            [filtered[0], filtered[0]], reference, lambda row: row.value
        )
