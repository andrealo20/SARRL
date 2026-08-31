from pathlib import Path

import numpy as np

from sarrl.dynamics import PlanarArm, PlanarArmParams
from sarrl.models import (
    ResidualDynamicsConfig,
    ResidualDynamicsEnsemble,
    UncertaintyGate,
    residual_acceleration_target,
    train_residual_ensemble,
)


def test_residual_target_is_zero_when_nominal_matches_plant():
    arm = PlanarArm()
    state = np.array([0.3, -0.4, 0.5, -0.2])
    torque = np.array([4.0, -1.0])
    qdd = arm.forward_dynamics(state[:2], state[2:], torque)
    target = residual_acceleration_target(arm, state, torque, qdd)
    np.testing.assert_allclose(target, np.zeros(2), atol=1e-7)


def test_residual_target_detects_payload_model_error():
    nominal = PlanarArm()
    actual = PlanarArm(PlanarArmParams(payload_mass=1.2))
    state = np.array([0.2, 0.7, -0.3, 0.4])
    torque = np.array([8.0, 3.0])
    qdd = actual.forward_dynamics(state[:2], state[2:], torque)
    target = residual_acceleration_target(nominal, state, torque, qdd)
    assert np.linalg.norm(target) > 0.1


def test_ensemble_training_reduces_synthetic_residual_loss():
    rng = np.random.default_rng(2)
    cfg = ResidualDynamicsConfig(hidden=(24, 24), ensemble_size=3, learning_rate=3e-3)
    x = rng.normal(size=(240, 4)).astype(np.float32)
    u = rng.normal(size=(240, 2)).astype(np.float32)
    y = np.stack(
        [0.3 * x[:, 0] - 0.2 * u[:, 0], -0.4 * x[:, 2] + 0.1 * u[:, 1]], axis=1
    ).astype(np.float32)
    ensemble = ResidualDynamicsEnsemble(cfg, seed=4)
    stats = train_residual_ensemble(
        ensemble, x, u, y, steps=180, batch_size=64, seed=4, device="cpu"
    )
    assert stats.final_loss < 0.25 * stats.initial_loss
    mean, std = ensemble.predict(x[:8], u[:8])
    assert mean.shape == (8, 2) and std.shape == (8, 2)
    assert np.all(std >= 0.0)


def test_ensemble_checkpoint_round_trip(tmp_path: Path):
    cfg = ResidualDynamicsConfig(hidden=(12, 12), ensemble_size=2)
    a = ResidualDynamicsEnsemble(cfg, seed=7)
    state = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    action = np.array([1.0, -2.0], dtype=np.float32)
    expected = a.predict(state, action)
    path = tmp_path / "ensemble.pt"
    a.save(path)
    b = ResidualDynamicsEnsemble.load(path)
    got = b.predict(state, action)
    np.testing.assert_allclose(got[0], expected[0], atol=0.0)
    np.testing.assert_allclose(got[1], expected[1], atol=0.0)


def test_uncertainty_gate_is_monotonic_and_has_floor():
    gate = UncertaintyGate(gain=5.0, min_scale=0.2)
    low = gate.scale(np.array([0.01, 0.01]))
    high = gate.scale(np.array([1.0, 1.0]))
    huge = gate.scale(np.array([100.0, 100.0]))
    assert 1.0 >= low > high >= 0.2
    assert huge == 0.2
