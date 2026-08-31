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


    @classmethod
    def from_state_dict(cls, state: dict) -> ReplayBuffer:
        obs = np.asarray(state["obs"])
        actions = np.asarray(state["actions"])
        if obs.ndim != 2 or actions.ndim != 2 or obs.shape[0] != actions.shape[0]:
            raise ValueError("invalid replay checkpoint arrays")
        obj = cls(obs.shape[1], actions.shape[1], int(state["capacity"]), seed=0)
        obj.load_state_dict(state)
        return obj

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

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "obs": self.obs.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "next_obs": self.next_obs.copy(),
            "dones": self.dones.copy(),
            "ptr": self.ptr,
            "size": self.size,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("replay checkpoint capacity does not match")
        for name in ("obs", "actions", "rewards", "next_obs", "dones"):
            src = np.asarray(state[name], dtype=getattr(self, name).dtype)
            dst = getattr(self, name)
            if src.shape != dst.shape:
                raise ValueError(f"replay checkpoint {name} shape does not match")
            dst[...] = src
        ptr, size = int(state["ptr"]), int(state["size"])
        if not 0 <= ptr < self.capacity or not 0 <= size <= self.capacity:
            raise ValueError("invalid replay checkpoint indices")
        self.ptr, self.size = ptr, size
        self.rng.bit_generator.state = state["rng_state"]

    def __len__(self) -> int:
        return self.size
