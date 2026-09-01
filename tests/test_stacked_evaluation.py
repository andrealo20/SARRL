import numpy as np

from sarrl.adaptation import AdaptiveContextEnv, ContextConfig, DynamicsContextEncoder
from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm
from sarrl.envs import PlanarReachEnv
from sarrl.evaluation import evaluate_stack_episodes
from sarrl.runtime import ControlStackConfig, SARRLControlStack
from sarrl.safety import HOCBFSafetyFilter, SafetyResult


class ZeroPolicy:
    def act(self, obs, deterministic=True):
        del obs, deterministic
        return np.zeros(2, dtype=np.float32)


def _safe_stack():
    nominal = PlanarArm()
    return SARRLControlStack(
        ComputedTorqueController(nominal),
        ZeroPolicy(),
        ControlStackConfig(require_safety=True),
        safety_filter=HOCBFSafetyFilter(nominal),
    )


def test_safe_stack_evaluation_retains_outcomes_and_diagnostics():
    rows, diagnostics = evaluate_stack_episodes(
        _safe_stack(),
        PlanarReachEnv(mode="torque"),
        episodes=2,
        seed=40_000,
    )

    assert [row.seed for row in rows] == [40_000, 40_001]
    assert len(diagnostics) == 2
    assert all(row.command_attempts >= row.steps for row in diagnostics)
    assert all(row.safety_certified_steps == row.command_attempts for row in diagnostics)
    assert all(not row.safety_infeasible for row in diagnostics)


def test_safe_stack_evaluation_aborts_infeasible_episode(monkeypatch):
    stack = _safe_stack()

    def infeasible(state, candidate, obstacles=()):
        del state, obstacles
        return SafetyResult(
            torque=np.asarray(candidate, dtype=np.float64),
            success=False,
            correction_norm=0.0,
            min_margin=-1.0,
            active_constraints=(),
            current_safe=True,
        )

    monkeypatch.setattr(stack.safety_filter, "filter", infeasible)
    rows, diagnostics = evaluate_stack_episodes(
        stack,
        PlanarReachEnv(mode="torque"),
        episodes=1,
        seed=40_000,
    )

    assert rows[0].steps == 0
    assert not rows[0].success
    assert diagnostics[0].safety_infeasible
    assert diagnostics[0].command_attempts == 1


def test_full_stack_evaluation_updates_context_after_filtered_torque():
    base = PlanarReachEnv(mode="torque")
    context = AdaptiveContextEnv(
        base,
        DynamicsContextEncoder(ContextConfig(latent_dim=4, hidden_dim=12, history=3)),
        device="cpu",
    )

    rows, diagnostics = evaluate_stack_episodes(
        _safe_stack(),
        context,
        episodes=1,
        seed=40_000,
        context_residual_limit=8.0,
    )

    assert len(rows) == len(diagnostics) == 1
    assert diagnostics[0].context_latent_norm_max > 0.0
