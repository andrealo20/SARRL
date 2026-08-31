from pathlib import Path

import numpy as np
import torch

from sarrl.rl import SACAgent, SACConfig
from sarrl.rl.networks import SquashedGaussianActor


def test_actor_actions_bounded_and_log_prob_finite():
    actor = SquashedGaussianActor(8, 2, hidden=(32, 32))
    obs = torch.randn(512, 8)
    action, logp, deterministic = actor.sample(obs)
    assert torch.all(action <= 1.0) and torch.all(action >= -1.0)
    assert torch.all(deterministic <= 1.0) and torch.all(deterministic >= -1.0)
    assert torch.isfinite(logp).all()


def test_actor_rescoring_matches_sample_log_prob():
    torch.manual_seed(5)
    actor = SquashedGaussianActor(4, 2, hidden=(16, 16))
    obs = torch.randn(128, 4)
    action, sampled_logp, _ = actor.sample(obs)
    rescored = actor.log_prob(obs, action)
    torch.testing.assert_close(sampled_logp, rescored, atol=3e-5, rtol=3e-5)


def test_terminal_bellman_target_does_not_bootstrap():
    agent = SACAgent(3, 1, SACConfig(hidden=(16, 16)), seed=0)
    rewards = torch.tensor([[2.5]], device=agent.device)
    dones = torch.ones((1, 1), device=agent.device)
    next_q = torch.tensor([[100.0]], device=agent.device)
    next_logp = torch.tensor([[-7.0]], device=agent.device)
    target = agent.compute_bellman_target(rewards, dones, next_q, next_logp)
    torch.testing.assert_close(target, rewards)


def test_nonterminal_bellman_target_matches_formula():
    cfg = SACConfig(gamma=0.9, init_alpha=0.2, hidden=(16, 16))
    agent = SACAgent(3, 1, cfg, seed=0)
    r = torch.tensor([[1.0]], device=agent.device)
    d = torch.zeros((1, 1), device=agent.device)
    q = torch.tensor([[4.0]], device=agent.device)
    lp = torch.tensor([[-0.5]], device=agent.device)
    got = agent.compute_bellman_target(r, d, q, lp)
    expected = r + 0.9 * (q - agent.alpha.detach() * lp)
    torch.testing.assert_close(got, expected)


def test_soft_target_update_moves_toward_online_network():
    cfg = SACConfig(tau=0.25, hidden=(8, 8))
    agent = SACAgent(2, 1, cfg, seed=1)
    with torch.no_grad():
        for p in agent.q1.parameters():
            p.add_(1.0)
    before = [p.detach().clone() for p in agent.q1_target.parameters()]
    source = [p.detach().clone() for p in agent.q1.parameters()]
    agent.soft_update_targets()
    for b, s, a in zip(before, source, agent.q1_target.parameters(), strict=True):
        torch.testing.assert_close(a, 0.75 * b + 0.25 * s)


def test_sac_update_metrics_are_finite():
    agent = SACAgent(4, 2, SACConfig(hidden=(32, 32)), seed=3)
    rng = np.random.default_rng(3)
    batch = {
        "obs": rng.normal(size=(64, 4)).astype(np.float32),
        "actions": rng.uniform(-1, 1, size=(64, 2)).astype(np.float32),
        "rewards": rng.normal(size=(64, 1)).astype(np.float32),
        "next_obs": rng.normal(size=(64, 4)).astype(np.float32),
        "dones": rng.integers(0, 2, size=(64, 1)).astype(np.float32),
    }
    metrics = agent.update(batch)
    assert agent.update_steps == 1
    assert metrics and all(np.isfinite(v) for v in metrics.values())


def test_checkpoint_round_trip(tmp_path: Path):
    cfg = SACConfig(hidden=(16, 16))
    a = SACAgent(4, 2, cfg, seed=11)
    obs = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    expected = a.act(obs, deterministic=True)
    path = tmp_path / "agent.pt"
    a.save(path)
    b = SACAgent(4, 2, cfg, seed=99)
    meta = b.load(path)
    np.testing.assert_allclose(b.act(obs, deterministic=True), expected, atol=0.0)
    assert meta["checkpoint_version"] == 1


def test_from_checkpoint_reconstructs_nondefault_architecture(tmp_path: Path):
    cfg = SACConfig(hidden=(23, 17))
    a = SACAgent(5, 2, cfg, seed=8)
    obs = np.linspace(-0.2, 0.2, 5, dtype=np.float32)
    expected = a.act(obs, deterministic=True)
    path = tmp_path / "custom.pt"
    a.save(path)
    b = SACAgent.from_checkpoint(path, seed=99, load_optimizers=False)
    assert b.config.hidden == (23, 17)
    np.testing.assert_array_equal(b.act(obs, deterministic=True), expected)


def test_deterministic_action_does_not_advance_torch_rng():
    import torch

    agent = SACAgent(4, 2, SACConfig(hidden=(16, 16)), seed=12)
    obs = np.zeros(4, dtype=np.float32)
    before = torch.get_rng_state().clone()
    a1 = agent.act(obs, deterministic=True)
    after = torch.get_rng_state().clone()
    a2 = agent.act(obs, deterministic=True)
    assert torch.equal(before, after)
    np.testing.assert_array_equal(a1, a2)
