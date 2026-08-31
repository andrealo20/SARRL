"""Deterministic policy evaluation on fixed seed sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import EpisodeResult


@dataclass(frozen=True)
class PolicyEvaluation:
    episodes: int
    successes: int
    success_rate: float
    reward_mean: float
    reward_std: float
    final_distance_mean: float

    @property
    def selection_key(self) -> tuple[float, float]:
        """Lexicographic model-selection key: success first, then reward."""
        return self.success_rate, self.reward_mean


def evaluate_policy_episodes(
    agent,
    env,
    episodes: int,
    seed: int,
    scenario: str = "evaluation",
    controller: str = "policy",
) -> list[EpisodeResult]:
    """Return retained per-episode records for deterministic evaluation."""
    if episodes <= 0 or seed < 0:
        raise ValueError("episodes must be positive and seed non-negative")
    rows: list[EpisodeResult] = []
    for ep in range(episodes):
        episode_seed = seed + ep
        obs, _ = env.reset(seed=episode_seed)
        total = 0.0
        max_speed = 0.0
        max_torque = 0.0
        fault_seen = False
        steps = 0
        while True:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total += float(reward)
            max_speed = max(max_speed, float(np.max(np.abs(env.state[2:]))))
            if "commanded_torque" in info:
                max_torque = max(
                    max_torque,
                    float(np.max(np.abs(np.asarray(info["commanded_torque"], dtype=np.float64)))),
                )
            fault_seen = fault_seen or bool(info.get("fault_active", False))
            if terminated or truncated:
                rows.append(
                    EpisodeResult(
                        scenario=scenario,
                        controller=controller,
                        seed=episode_seed,
                        reward=total,
                        steps=steps,
                        success=bool(info["success"]),
                        final_distance=float(info["distance"]),
                        max_speed=max_speed,
                        max_command_torque=max_torque,
                        fault_seen=fault_seen,
                    )
                )
                break
    return rows


def evaluate_policy(agent, env, episodes: int, seed: int) -> PolicyEvaluation:
    """Evaluate a deterministic policy on seeds ``seed .. seed+episodes-1``.

    The environment supplied here should be separate from the training
    environment. `SACAgent.act(..., deterministic=True)` is RNG-free, so this
    function does not perturb the agent's stochastic action stream.
    """
    rows = evaluate_policy_episodes(agent, env, episodes, seed)
    successes = sum(int(row.success) for row in rows)
    rewards = np.asarray([row.reward for row in rows], dtype=np.float64)
    dist = np.asarray([row.final_distance for row in rows], dtype=np.float64)
    return PolicyEvaluation(
        episodes=episodes,
        successes=successes,
        success_rate=successes / episodes,
        reward_mean=float(rewards.mean()),
        reward_std=float(rewards.std()),
        final_distance_mean=float(dist.mean()),
    )
