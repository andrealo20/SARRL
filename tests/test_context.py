from pathlib import Path

import numpy as np
import torch

from sarrl.adaptation import (
    AdaptiveContextEnv,
    ContextConfig,
    DynamicsContextEncoder,
    train_context_encoder,
)
from sarrl.envs import PlanarReachEnv


def test_context_encoder_shapes_and_bounds():
    cfg = ContextConfig(hidden_dim=24, latent_dim=6, history=5)
    model = DynamicsContextEncoder(cfg)
    x = torch.randn(7, 5, cfg.transition_dim)
    latent, pred = model(x)
    assert latent.shape == (7, 6)
    assert pred.shape == (7, 8)
    assert torch.all(latent <= 1.0) and torch.all(latent >= -1.0)


def test_context_checkpoint_round_trip(tmp_path: Path):
    cfg = ContextConfig(hidden_dim=16, latent_dim=5, history=4)
    model = DynamicsContextEncoder(cfg)
    x = torch.randn(2, 4, cfg.transition_dim)
    with torch.no_grad():
        expected = model(x)[0]
    path = tmp_path / "ctx.pt"
    model.save(path)
    restored = DynamicsContextEncoder.load(path)
    with torch.no_grad():
        got = restored(x)[0]
    torch.testing.assert_close(got, expected, atol=0.0, rtol=0.0)


def test_adaptive_wrapper_is_causal_and_does_not_query_ground_truth_context():
    env = PlanarReachEnv(mode="residual")
    cfg = ContextConfig(latent_dim=4, hidden_dim=12, history=3)
    model = DynamicsContextEncoder(cfg)
    wrapped = AdaptiveContextEnv(env, model, device="cpu")
    obs, _ = wrapped.reset(seed=4)
    assert obs.shape == (12,)
    np.testing.assert_array_equal(obs[-4:], np.zeros(4))

    # Runtime adaptation must work even if the diagnostic label API is unavailable.
    env.dynamics_context = lambda: (_ for _ in ()).throw(RuntimeError("must not be called"))
    next_obs, _, _, _, info = wrapped.step(np.array([0.2, -0.1], dtype=np.float32))
    assert next_obs.shape == (12,)
    assert info["context_latent"].shape == (4,)
    assert np.all(np.isfinite(next_obs))


def test_adaptive_wrapper_supports_filtered_torque_with_residual_context_action():
    env = PlanarReachEnv(mode="torque")
    cfg = ContextConfig(latent_dim=4, hidden_dim=12, history=3)
    wrapped = AdaptiveContextEnv(env, DynamicsContextEncoder(cfg), device="cpu")
    obs, _ = wrapped.reset(seed=9)

    next_obs, _, _, _, info = wrapped.step_torque(
        np.array([2.0, -1.0]),
        baseline=np.array([1.5, -0.5]),
        context_action=np.array([0.25, -0.125]),
    )

    assert obs.shape == next_obs.shape == (12,)
    assert info["context_latent"].shape == (4,)
    np.testing.assert_array_equal(wrapped.q_des, env.q_des)
    assert np.linalg.norm(wrapped.latent) > 0.0


def test_supervised_context_training_reduces_loss_on_synthetic_identifiable_data():
    rng = np.random.default_rng(8)
    cfg = ContextConfig(
        obs_dim=2,
        action_dim=1,
        context_dim=2,
        latent_dim=4,
        hidden_dim=12,
        history=4,
        learning_rate=3e-3,
    )
    n = 160
    x = rng.normal(size=(n, cfg.history, cfg.transition_dim)).astype(np.float32)
    # Smooth target encoded in the history; this checks the training path rather
    # than claiming physical identifiability from a synthetic dataset.
    y = np.stack([x[:, :, 0].mean(1), x[:, :, 1].mean(1)], axis=1).astype(np.float32)
    model = DynamicsContextEncoder(cfg)
    stats = train_context_encoder(model, x, y, steps=180, batch_size=48, seed=3, device="cpu")
    assert stats.final_loss < 0.55 * stats.initial_loss


def test_adaptive_wrapper_state_round_trip_preserves_next_transition():
    torch.manual_seed(13)
    cfg = ContextConfig(latent_dim=4, hidden_dim=12, history=3)

    original = AdaptiveContextEnv(
        PlanarReachEnv(mode="residual"),
        DynamicsContextEncoder(cfg),
        device="cpu",
    )

    original.reset(seed=17)
    original.step(np.array([0.20, -0.10], dtype=np.float32))
    original.step(np.array([-0.15, 0.25], dtype=np.float32))

    restored = AdaptiveContextEnv.from_state_dict(
        original.state_dict(),
        device="cpu",
    )

    np.testing.assert_array_equal(restored.state, original.state)
    np.testing.assert_array_equal(restored.latent, original.latent)
    assert restored.constructor_config() == original.constructor_config()

    action = np.array([0.12, -0.18], dtype=np.float32)
    expected = original.step(action)
    got = restored.step(action)

    np.testing.assert_array_equal(got[0], expected[0])
    assert got[1] == expected[1]
    assert got[2] == expected[2]
    assert got[3] == expected[3]
    assert got[4]["success"] == expected[4]["success"]
    assert got[4]["distance"] == expected[4]["distance"]
    np.testing.assert_array_equal(restored.state, original.state)
    np.testing.assert_array_equal(restored.latent, original.latent)


def test_context_dataset_collection_is_exactly_reproducible():
    from tools.train_context import collect

    x1, y1 = collect(
        samples=4,
        history=3,
        data_seed=100000,
    )
    x2, y2 = collect(
        samples=4,
        history=3,
        data_seed=100000,
    )

    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)

    assert x1.shape == (4, 3, 18)
    assert y1.shape == (4, 8)


def test_context_dataset_changes_with_data_seed():
    from tools.train_context import collect

    x1, y1 = collect(
        samples=3,
        history=3,
        data_seed=100000,
    )
    x2, y2 = collect(
        samples=3,
        history=3,
        data_seed=110000,
    )

    assert not np.array_equal(x1, x2)
    assert not np.array_equal(y1, y2)
