from pathlib import Path

import numpy as np

from sarrl.envs import PlanarReachEnv
from sarrl.rl import (
    ReplayBuffer,
    SACAgent,
    SACConfig,
    load_training_checkpoint,
    load_training_session,
    save_training_checkpoint,
)


def _build(seed):
    env = PlanarReachEnv(mode="residual", max_steps=40)
    agent = SACAgent(8, 2, SACConfig(hidden=(16, 16)), seed=seed)
    replay = ReplayBuffer(8, 2, 100, seed=seed)
    return env, agent, replay


def test_training_checkpoint_restores_environment_replay_and_rng_exactly(tmp_path: Path):
    env, agent, replay = _build(5)
    obs, _ = env.reset(seed=5)
    for _ in range(20):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        replay.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    path = tmp_path / "training.pt"
    save_training_checkpoint(path, agent, replay, env, {"step": 20, "obs": obs})

    # Reference continuation after the save.
    import random

    random_ref = random.random()
    action_ref = agent.act(obs)
    next_ref = env.step(action_ref)
    batch_ref = replay.sample(8)
    metrics_ref = agent.update(batch_ref)

    env2, agent2, replay2 = _build(999)
    loop = load_training_checkpoint(path, agent2, replay2, env2)
    random_got = random.random()
    assert random_got == random_ref
    obs2 = np.asarray(loop["obs"], dtype=np.float32)
    action_got = agent2.act(obs2)
    next_got = env2.step(action_got)
    batch_got = replay2.sample(8)
    metrics_got = agent2.update(batch_got)

    np.testing.assert_array_equal(action_got, action_ref)
    np.testing.assert_array_equal(next_got[0], next_ref[0])
    assert next_got[1:4] == next_ref[1:4]
    for key in batch_ref:
        np.testing.assert_array_equal(batch_got[key], batch_ref[key])
    for key in metrics_ref:
        np.testing.assert_allclose(metrics_got[key], metrics_ref[key], rtol=0.0, atol=0.0)


def test_training_session_reconstructs_nondefault_components(tmp_path: Path):
    from sarrl.envs import DomainRandomization, FaultSpec

    env = PlanarReachEnv(
        mode="torque",
        dt=0.01,
        max_steps=77,
        torque_limit=31.0,
        residual_limit=5.0,
        success_radius=0.07,
        randomization=DomainRandomization(
            mass_fraction=0.1,
            friction_fraction=0.2,
            motor_gain_fraction=0.05,
            payload_range=(0.1, 0.9),
            sensor_noise_std=0.002,
            action_delay_max=2,
        ),
        fault=FaultSpec(start_step=12, motor_gain_multiplier=(1.0, 0.7), payload_delta=0.2),
    )
    agent = SACAgent(8, 2, SACConfig(hidden=(19, 13)), seed=4)
    replay = ReplayBuffer(8, 2, 123, seed=4)
    obs, _ = env.reset(seed=4)
    for _ in range(9):
        action = agent.act(obs)
        nxt, reward, terminated, truncated, _ = env.step(action)
        replay.add(obs, action, reward, nxt, terminated)
        obs = nxt
        if terminated or truncated:
            obs, _ = env.reset()

    path = tmp_path / "session.pt"
    save_training_checkpoint(path, agent, replay, env, {"step": 9, "obs": obs})
    a2, r2, e2, loop = load_training_session(path)

    assert a2.config.hidden == (19, 13)
    assert r2.capacity == 123 and len(r2) == len(replay)
    assert e2.constructor_config() == env.constructor_config()
    assert loop["step"] == 9 and loop["checkpoint_version"] == 2
    np.testing.assert_array_equal(e2.state, env.state)


def test_cuda_rng_restore_converts_checkpoint_states_to_cpu(monkeypatch):
    import torch

    agent = SACAgent(8, 2, SACConfig(hidden=(16, 16)), seed=7)
    payload = agent.state_dict(include_optimizers=False)

    class MappedCudaRngState:
        def __init__(self):
            self.cpu_called = False

        def cpu(self):
            self.cpu_called = True
            return torch.tensor([1, 2, 3], dtype=torch.uint8)

    mapped_state = MappedCudaRngState()
    payload["cuda_rng_state_all"] = [mapped_state]

    captured = {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def capture_rng_states(states):
        captured["states"] = states

    monkeypatch.setattr(torch.cuda, "set_rng_state_all", capture_rng_states)

    agent.load_state_dict(
        payload,
        load_optimizers=False,
        restore_rng=True,
    )

    assert mapped_state.cpu_called

    restored = captured["states"]
    assert len(restored) == 1
    assert isinstance(restored[0], torch.Tensor)
    assert restored[0].device.type == "cpu"
    assert restored[0].dtype == torch.uint8


def test_training_session_reconstructs_adaptive_context_exactly(tmp_path: Path):
    from sarrl.adaptation import (
        AdaptiveContextEnv,
        ContextConfig,
        DynamicsContextEncoder,
    )

    cfg = ContextConfig(latent_dim=4, hidden_dim=12, history=3)
    env = AdaptiveContextEnv(
        PlanarReachEnv(mode="residual", max_steps=40),
        DynamicsContextEncoder(cfg),
        device="cpu",
    )
    agent = SACAgent(12, 2, SACConfig(hidden=(16, 16)), seed=6)
    replay = ReplayBuffer(12, 2, 100, seed=6)

    obs, _ = env.reset(seed=6)
    for _ in range(12):
        action = agent.act(obs)
        nxt, reward, terminated, truncated, _ = env.step(action)
        replay.add(obs, action, reward, nxt, terminated)
        obs = nxt
        if terminated or truncated:
            obs, _ = env.reset()

    path = tmp_path / "adaptive_session.pt"
    save_training_checkpoint(
        path,
        agent,
        replay,
        env,
        {"step": 12, "obs": obs},
    )

    # Reference stochastic continuation from the exact saved state.
    action_ref = agent.act(obs)
    next_ref = env.step(action_ref)

    agent2, replay2, env2, loop = load_training_session(path)
    obs2 = np.asarray(loop["obs"], dtype=np.float32)

    action_got = agent2.act(obs2)
    next_got = env2.step(action_got)

    assert isinstance(env2, AdaptiveContextEnv)
    assert agent2.obs_dim == 12
    assert replay2.obs.shape[1] == 12
    assert replay2.next_obs.shape[1] == 12
    assert loop["step"] == 12

    np.testing.assert_array_equal(action_got, action_ref)
    np.testing.assert_array_equal(next_got[0], next_ref[0])
    assert next_got[1] == next_ref[1]
    assert next_got[2] == next_ref[2]
    assert next_got[3] == next_ref[3]
    assert next_got[4]["success"] == next_ref[4]["success"]
    assert next_got[4]["distance"] == next_ref[4]["distance"]


def test_adaptive_context_resume_matches_twenty_training_steps_exactly(tmp_path: Path):
    from sarrl.adaptation import (
        AdaptiveContextEnv,
        ContextConfig,
        DynamicsContextEncoder,
    )

    cfg = ContextConfig(latent_dim=4, hidden_dim=12, history=3)
    env = AdaptiveContextEnv(
        PlanarReachEnv(mode="residual", max_steps=18),
        DynamicsContextEncoder(cfg),
        device="cpu",
    )
    agent = SACAgent(12, 2, SACConfig(hidden=(16, 16)), seed=21)
    replay = ReplayBuffer(12, 2, 200, seed=21)

    obs, _ = env.reset(seed=21)

    # Build some history before the checkpoint.
    for _ in range(12):
        action = agent.act(obs)
        nxt, reward, terminated, truncated, _ = env.step(action)
        replay.add(obs, action, reward, nxt, terminated)
        obs = nxt
        if terminated or truncated:
            obs, _ = env.reset()

    checkpoint = tmp_path / "adaptive_exact_resume.pt"
    save_training_checkpoint(
        checkpoint,
        agent,
        replay,
        env,
        {"step": 12, "obs": obs},
    )

    def continuation(agent, replay, env, obs, steps=20):
        trace = []

        for _ in range(steps):
            action = agent.act(obs)
            nxt, reward, terminated, truncated, info = env.step(action)

            replay.add(
                obs,
                action,
                reward,
                nxt,
                terminated,
            )

            batch = replay.sample(8)
            metrics = agent.update(batch)

            if terminated or truncated:
                post_obs, _ = env.reset()
            else:
                post_obs = nxt

            trace.append(
                {
                    "action": action.copy(),
                    "next_obs": nxt.copy(),
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "success": info["success"],
                    "distance": info["distance"],
                    "batch": {key: value.copy() for key, value in batch.items()},
                    "metrics": dict(metrics),
                    "post_obs": post_obs.copy(),
                    "latent": env.latent.copy(),
                }
            )

            obs = post_obs

        return trace, obs

    # Reference continuation from the original live session.
    expected, expected_obs = continuation(
        agent,
        replay,
        env,
        obs,
    )

    # Reload the checkpoint and repeat the exact same continuation.
    agent2, replay2, env2, loop = load_training_session(checkpoint)
    obs2 = np.asarray(loop["obs"], dtype=np.float32)

    got, got_obs = continuation(
        agent2,
        replay2,
        env2,
        obs2,
    )

    assert len(expected) == len(got) == 20

    for expected_step, got_step in zip(expected, got, strict=True):
        np.testing.assert_array_equal(
            got_step["action"],
            expected_step["action"],
        )
        np.testing.assert_array_equal(
            got_step["next_obs"],
            expected_step["next_obs"],
        )
        np.testing.assert_array_equal(
            got_step["post_obs"],
            expected_step["post_obs"],
        )
        np.testing.assert_array_equal(
            got_step["latent"],
            expected_step["latent"],
        )

        assert got_step["reward"] == expected_step["reward"]
        assert got_step["terminated"] == expected_step["terminated"]
        assert got_step["truncated"] == expected_step["truncated"]
        assert got_step["success"] == expected_step["success"]
        assert got_step["distance"] == expected_step["distance"]

        for key in expected_step["batch"]:
            np.testing.assert_array_equal(
                got_step["batch"][key],
                expected_step["batch"][key],
            )

        for key in expected_step["metrics"]:
            np.testing.assert_allclose(
                got_step["metrics"][key],
                expected_step["metrics"][key],
                rtol=0.0,
                atol=0.0,
            )

    np.testing.assert_array_equal(got_obs, expected_obs)
    np.testing.assert_array_equal(env2.state, env.state)
    np.testing.assert_array_equal(env2.latent, env.latent)
