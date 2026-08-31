"""Deterministic policy evaluation on fixed seed sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def evaluate_policy(agent, env, episodes: int, seed: int) -> PolicyEvaluation:
    """Evaluate a deterministic policy on seeds ``seed .. seed+episodes-1``.

    The environment supplied here should be separate from the training
    environment. `SACAgent.act(..., deterministic=True)` is RNG-free, so this
    function does not perturb the agent's stochastic action stream.
    """
    if episodes <= 0 or seed < 0:
        raise ValueError("episodes must be positive and seed non-negative")
    successes = 0
    rewards: list[float] = []
    distances: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        while True:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            if terminated or truncated:
                successes += int(info["success"])
                rewards.append(total)
                distances.append(float(info["distance"]))
                break
    arr = np.asarray(rewards, dtype=np.float64)
    dist = np.asarray(distances, dtype=np.float64)
    return PolicyEvaluation(
        episodes=episodes,
        successes=successes,
        success_rate=successes / episodes,
        reward_mean=float(arr.mean()),
        reward_std=float(arr.std()),
        final_distance_mean=float(dist.mean()),
    )
