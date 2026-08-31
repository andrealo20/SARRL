"""Fixed-capacity replay buffer with reproducible sampling."""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int, seed: int = 0):
        if obs_dim <= 0 or action_dim <= 0 or capacity <= 0:
            raise ValueError("buffer dimensions and capacity must be positive")
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward: float, next_obs, done: bool) -> None:
        self.obs[self.ptr] = np.asarray(obs, dtype=np.float32)
        self.actions[self.ptr] = np.asarray(action, dtype=np.float32)
        self.rewards[self.ptr, 0] = float(reward)
        self.next_obs[self.ptr] = np.asarray(next_obs, dtype=np.float32)
        self.dones[self.ptr, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size <= 0 or batch_size > self.size:
            raise ValueError("batch_size must be in [1, current_size]")
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }

    def __len__(self) -> int:
        return self.size
