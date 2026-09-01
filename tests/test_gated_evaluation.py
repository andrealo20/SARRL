import numpy as np

from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm
from sarrl.envs import PlanarReachEnv
from sarrl.evaluation import evaluate_gated_policy_episodes
from sarrl.models import UncertaintyGate
from sarrl.runtime import SARRLControlStack


class FullPolicy:
    def act(self, obs, deterministic=True):
        del obs, deterministic
        return np.ones(2, dtype=np.float32)


class FixedUncertaintyEnsemble:
    def predict(self, state, action, device="cpu"):
        del state, action, device
        return (
            np.zeros(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
        )


def test_gated_evaluation_retains_outcomes_and_gate_diagnostics():
    nominal = PlanarArm()
    stack = SARRLControlStack(
        ComputedTorqueController(nominal),
        FullPolicy(),
        dynamics_ensemble=FixedUncertaintyEnsemble(),
        uncertainty_gate=UncertaintyGate(gain=5.0, min_scale=0.1),
    )
    env = PlanarReachEnv(mode="torque", max_steps=5)

    rows, diagnostics = evaluate_gated_policy_episodes(
        stack,
        env,
        episodes=2,
        seed=40_000,
        scenario="heldout",
        controller="A4_test",
    )

    assert [row.seed for row in rows] == [40_000, 40_001]
    assert [row.seed for row in diagnostics] == [40_000, 40_001]
    assert all(row.controller == "A4_test" for row in rows)
    assert all(row.steps == 5 for row in diagnostics)
    assert all(0.1 <= row.uncertainty_scale_min < 1.0 for row in diagnostics)
    assert all(
        row.gated_residual_norm_mean < row.raw_residual_norm_mean
        for row in diagnostics
    )


def test_gated_evaluation_rejects_invalid_episode_protocol():
    nominal = PlanarArm()
    stack = SARRLControlStack(
        ComputedTorqueController(nominal),
        FullPolicy(),
        dynamics_ensemble=FixedUncertaintyEnsemble(),
        uncertainty_gate=UncertaintyGate(),
    )
    env = PlanarReachEnv(mode="torque", max_steps=2)

    for episodes, seed in ((0, 40_000), (1, -1)):
        try:
            evaluate_gated_policy_episodes(stack, env, episodes, seed)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid gated evaluation protocol was accepted")
