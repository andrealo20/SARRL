"""Online dynamics-context estimation from transition history.

The runtime encoder is causal: it consumes only observations, executed actions
and observation changes. Ground-truth dynamics parameters are optional labels
used to train or audit the representation; they are never required by the
wrapper during control.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sarrl.utils.spaces import BoxSpace


@dataclass(frozen=True)
class ContextConfig:
    obs_dim: int = 8
    action_dim: int = 2
    context_dim: int = 8
    latent_dim: int = 16
    hidden_dim: int = 64
    history: int = 16
    learning_rate: float = 1e-3

    @property
    def transition_dim(self) -> int:
        return self.obs_dim + self.action_dim + self.obs_dim

    def validate(self) -> None:
        if any(
            value <= 0
            for value in (
                self.obs_dim,
                self.action_dim,
                self.context_dim,
                self.latent_dim,
                self.hidden_dim,
                self.history,
            )
        ):
            raise ValueError("context dimensions and history must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("context learning rate must be positive and finite")


class DynamicsContextEncoder(nn.Module):
    CHECKPOINT_VERSION = 1

    def __init__(self, config: ContextConfig | None = None):
        super().__init__()
        self.config = config or ContextConfig()
        self.config.validate()
        self.gru = nn.GRU(
            input_size=self.config.transition_dim,
            hidden_size=self.config.hidden_dim,
            batch_first=True,
        )
        self.latent_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.latent_dim),
            nn.Tanh(),
        )
        self.context_head = nn.Linear(self.config.latent_dim, self.config.context_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.orthogonal_(self.latent_head[0].weight, gain=1.0)
        nn.init.zeros_(self.latent_head[0].bias)
        nn.init.orthogonal_(self.context_head.weight, gain=0.01)
        nn.init.zeros_(self.context_head.bias)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim != 3 or sequence.shape[-1] != self.config.transition_dim:
            raise ValueError("sequence must have shape [batch, time, transition_dim]")
        if sequence.shape[1] == 0:
            raise ValueError("context sequence must contain at least one transition")
        output, _ = self.gru(sequence)
        latent = self.latent_head(output[:, -1])
        prediction = self.context_head(latent)
        return latent, prediction

    @staticmethod
    def transition_feature(obs, action, next_obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        if obs.ndim != 1 or action.ndim != 1 or next_obs.shape != obs.shape:
            raise ValueError("transition components must be one-dimensional and compatible")
        if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(action)) or not np.all(
            np.isfinite(next_obs)
        ):
            raise ValueError("transition features must be finite")
        return np.concatenate([obs, action, next_obs - obs]).astype(np.float32)

    def encode_numpy(self, sequence: np.ndarray, device: str | torch.device = "cpu") -> np.ndarray:
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim != 2:
            raise ValueError("sequence must have shape [time, transition_dim]")
        self.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
            latent, _ = self.to(device)(tensor)
        return latent.squeeze(0).cpu().numpy().astype(np.float32)

    def save(self, path) -> None:
        torch.save(
            {
                "checkpoint_version": self.CHECKPOINT_VERSION,
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
            },
            Path(path),
        )

    @classmethod
    def load(cls, path, map_location: str | torch.device = "cpu") -> DynamicsContextEncoder:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if payload.get("checkpoint_version") != cls.CHECKPOINT_VERSION:
            raise ValueError("unsupported context checkpoint version")
        cfg = ContextConfig(**payload["config"])
        model = cls(cfg)
        model.load_state_dict(payload["state_dict"])
        return model


@dataclass(frozen=True)
class ContextTrainingStats:
    initial_loss: float
    final_loss: float
    steps: int


def train_context_encoder(
    model: DynamicsContextEncoder,
    sequences: np.ndarray,
    targets: np.ndarray,
    steps: int = 500,
    batch_size: int = 64,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> ContextTrainingStats:
    """Supervise the context head; gradients shape the latent representation too."""
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    x = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    cfg = model.config
    if x.ndim != 3 or x.shape[1:] != (cfg.history, cfg.transition_dim):
        raise ValueError("sequences have incompatible shape")
    if y.shape != (x.shape[0], cfg.context_dim):
        raise ValueError("targets have incompatible shape")
    if x.shape[0] == 0:
        raise ValueError("training data must not be empty")

    rng = np.random.default_rng(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()

    def full_loss() -> float:
        model.eval()
        with torch.no_grad():
            _, pred = model(torch.as_tensor(x, device=dev))
            value = loss_fn(pred, torch.as_tensor(y, device=dev)).item()
        model.train()
        return float(value)

    initial = full_loss()
    for _ in range(steps):
        idx = rng.integers(0, x.shape[0], size=min(batch_size, x.shape[0]))
        xb = torch.as_tensor(x[idx], device=dev)
        yb = torch.as_tensor(y[idx], device=dev)
        _, pred = model(xb)
        loss = loss_fn(pred, yb)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite context loss")
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    final = full_loss()
    return ContextTrainingStats(initial, final, steps)


class AdaptiveContextEnv:
    """Append a causal learned dynamics latent to a base environment observation."""

    def __init__(self, env, encoder: DynamicsContextEncoder, device=None):
        self.env = env
        self.encoder = encoder
        self.config = encoder.config
        if env.observation_space.shape != (self.config.obs_dim,):
            raise ValueError("encoder observation dimension does not match environment")
        if env.action_space.shape != (self.config.action_dim,):
            raise ValueError("encoder action dimension does not match environment")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder.to(self.device).eval()
        self.action_space = env.action_space
        self.observation_space = BoxSpace(
            -np.ones(self.config.obs_dim + self.config.latent_dim),
            np.ones(self.config.obs_dim + self.config.latent_dim),
        )
        self._history: deque[np.ndarray] = deque(maxlen=self.config.history)
        self._last_obs: np.ndarray | None = None
        self._latent = np.zeros(self.config.latent_dim, dtype=np.float32)

    def _augmented(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(obs, dtype=np.float32), self._latent]).astype(np.float32)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self._history.clear()
        self._latent.fill(0.0)
        self._last_obs = np.asarray(obs, dtype=np.float32).copy()
        return self._augmented(obs), info

    def step(self, action):
        if self._last_obs is None:
            raise RuntimeError("reset must be called before step")
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        feature = self.encoder.transition_feature(self._last_obs, action, next_obs)
        self._history.append(feature)
        padded = np.zeros((self.config.history, self.config.transition_dim), dtype=np.float32)
        hist = np.asarray(self._history, dtype=np.float32)
        padded[-len(hist) :] = hist
        self._latent = self.encoder.encode_numpy(padded, device=self.device)
        self._last_obs = np.asarray(next_obs, dtype=np.float32).copy()
        info = dict(info)
        info["context_latent"] = self._latent.copy()
        return self._augmented(next_obs), reward, terminated, truncated, info

    @property
    def latent(self) -> np.ndarray:
        return self._latent.copy()
