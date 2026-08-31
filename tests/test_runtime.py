import numpy as np

from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm
from sarrl.models import UncertaintyGate
from sarrl.runtime import ControlStackConfig, SARRLControlStack
from sarrl.safety import HOCBFSafetyFilter


class ZeroPolicy:
    def act(self, obs, deterministic=True):
        del obs, deterministic
        return np.zeros(2, dtype=np.float32)


class FullPolicy:
    def act(self, obs, deterministic=True):
        del obs, deterministic
        return np.ones(2, dtype=np.float32)


class FixedUncertaintyEnsemble:
    def __init__(self, std):
        self.std = np.asarray(std, dtype=np.float32)

    def predict(self, state, action, device="cpu"):
        del state, action, device
        return np.zeros(2, dtype=np.float32), self.std.copy()


def _state_target():
    return np.array([0.2, 0.5, 0.0, 0.0]), np.array([0.4, 0.7])


def test_zero_residual_stack_reproduces_nominal_baseline():
    arm = PlanarArm()
    baseline = ComputedTorqueController(arm)
    stack = SARRLControlStack(baseline, ZeroPolicy())
    state, target = _state_target()
    result = stack.command(np.zeros(8), state, target)
    np.testing.assert_allclose(result.torque, baseline.command(state[:2], state[2:], target))
    np.testing.assert_array_equal(result.raw_residual, np.zeros(2))
    assert result.executable and not result.safety_certified


def test_required_safety_stack_returns_certified_feasible_torque():
    arm = PlanarArm()
    baseline = ComputedTorqueController(arm)
    safety = HOCBFSafetyFilter(arm)
    stack = SARRLControlStack(
        baseline,
        FullPolicy(),
        ControlStackConfig(require_safety=True),
        safety_filter=safety,
    )
    state, target = _state_target()
    result = stack.command(np.zeros(8), state, target)
    assert result.executable and result.safety_certified
    A, b, _ = safety.constraints(state)
    assert np.min(A @ result.torque - b) >= -2e-8


def test_uncertainty_gate_reduces_residual_authority_inside_stack():
    arm = PlanarArm()
    baseline = ComputedTorqueController(arm)
    ensemble = FixedUncertaintyEnsemble([1.0, 1.0])
    gate = UncertaintyGate(gain=5.0, min_scale=0.1)
    stack = SARRLControlStack(
        baseline,
        FullPolicy(),
        dynamics_ensemble=ensemble,
        uncertainty_gate=gate,
    )
    state, target = _state_target()
    result = stack.command(np.zeros(8), state, target)
    assert result.uncertainty_scale < 1.0
    assert np.linalg.norm(result.gated_residual) < np.linalg.norm(result.raw_residual)
