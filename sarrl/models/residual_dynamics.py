"""Learned residual dynamics and epistemic-uncertainty ensemble."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sarrl.dynamics import PlanarArm


@dataclass(frozen=True)
class ResidualDynamicsConfig:
    state_dim: int = 4
    action_dim: int = 2
    output_dim: int = 2
    hidden: tuple[int, int] = (128, 128)
    ensemble_size: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6

    def validate(self) -> None:
        if any(v <= 0 for v in (self.state_dim, self.action_dim, self.output_dim, self.ensemble_size)):
            raise ValueError("residual dynamics dimensions and ensemble size must be positive")
        if not self.hidden or any(v <= 0 for v in self.hidden):
            raise ValueError("residual dynamics hidden layers must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid residual dynamics optimiser settings")


class ResidualAccelerationModel(nn.Module):
    def __init__(self, config: ResidualDynamicsConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        last = config.state_dim + config.action_dim
        for width in config.hidden:
            layer = nn.Linear(last, width)
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(layer.bias)
            layers.extend([layer, nn.SiLU()])
            last = width
        out = nn.Linear(last, config.output_dim)
        nn.init.orthogonal_(out.weight, gain=0.01)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


@dataclass(frozen=True)
class EnsembleTrainingStats:
    initial_loss: float
    final_loss: float
    steps: int


class ResidualDynamicsEnsemble(nn.Module):
    CHECKPOINT_VERSION = 1

    def __init__(self, config: ResidualDynamicsConfig | None = None, seed: int = 0):
        super().__init__()
        self.config = config or ResidualDynamicsConfig()
        self.config.validate()
        torch.manual_seed(seed)
        self.models = nn.ModuleList(
            [ResidualAccelerationModel(self.config) for _ in range(self.config.ensemble_size)]
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return [ensemble, batch, output_dim] predictions."""
        return torch.stack([model(state, action) for model in self.models], dim=0)

    def predict(self, state, action, device: str | torch.device = "cpu") -> tuple[np.ndarray, np.ndarray]:
        state_arr = np.asarray(state, dtype=np.float32)
        action_arr = np.asarray(action, dtype=np.float32)
        single = state_arr.ndim == 1
        if single:
            state_arr = state_arr[None, :]
            action_arr = action_arr[None, :]
        if state_arr.ndim != 2 or action_arr.ndim != 2:
            raise ValueError("state and action must be vectors or batches")
        if state_arr.shape[0] != action_arr.shape[0]:
            raise ValueError("state and action batches must align")
        dev = torch.device(device)
        self.to(dev).eval()
        with torch.no_grad():
            pred = self(
                torch.as_tensor(state_arr, device=dev),
                torch.as_tensor(action_arr, device=dev),
            )
            mean = pred.mean(dim=0)
            std = pred.std(dim=0, unbiased=False)
        mean_np = mean.cpu().numpy().astype(np.float32)
        std_np = std.cpu().numpy().astype(np.float32)
        return (mean_np[0], std_np[0]) if single else (mean_np, std_np)

    def corrected_acceleration(
        self,
        nominal_model: PlanarArm,
        state,
        torque,
        device: str | torch.device = "cpu",
    ) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=np.float64)
        torque = np.asarray(torque, dtype=np.float64)
        if state.shape != (4,) or torque.shape != (2,):
            raise ValueError("state/torque shapes must be (4,) and (2,)")
        nominal = nominal_model.forward_dynamics(state[:2], state[2:], torque)
        residual, uncertainty = self.predict(state.astype(np.float32), torque.astype(np.float32), device)
        return nominal + residual.astype(np.float64), uncertainty.astype(np.float64)

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
    def load(cls, path, map_location: str | torch.device = "cpu") -> "ResidualDynamicsEnsemble":
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if payload.get("checkpoint_version") != cls.CHECKPOINT_VERSION:
            raise ValueError("unsupported residual dynamics checkpoint version")
        cfg = ResidualDynamicsConfig(**payload["config"])
        obj = cls(cfg)
        obj.load_state_dict(payload["state_dict"])
        return obj


def residual_acceleration_target(
    nominal_model: PlanarArm,
    state,
    applied_torque,
    observed_acceleration,
) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    torque = np.asarray(applied_torque, dtype=np.float64)
    observed = np.asarray(observed_acceleration, dtype=np.float64)
    if state.shape != (4,) or torque.shape != (2,) or observed.shape != (2,):
        raise ValueError("residual target expects state(4), torque(2), acceleration(2)")
    nominal = nominal_model.forward_dynamics(state[:2], state[2:], torque)
    return (observed - nominal).astype(np.float32)


def train_residual_ensemble(
    ensemble: ResidualDynamicsEnsemble,
    states: np.ndarray,
    actions: np.ndarray,
    targets: np.ndarray,
    steps: int = 1000,
    batch_size: int = 128,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> EnsembleTrainingStats:
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    x = np.asarray(states, dtype=np.float32)
    u = np.asarray(actions, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    cfg = ensemble.config
    if x.ndim != 2 or x.shape[1] != cfg.state_dim:
        raise ValueError("states have incompatible shape")
    if u.shape != (x.shape[0], cfg.action_dim) or y.shape != (x.shape[0], cfg.output_dim):
        raise ValueError("actions/targets have incompatible shape")
    if x.shape[0] == 0:
        raise ValueError("training data must not be empty")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ensemble.to(dev).train()
    optimizers = [
        torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        for model in ensemble.models
    ]
    rngs = [np.random.default_rng(seed + 1009 * i) for i in range(cfg.ensemble_size)]
    loss_fn = nn.MSELoss()

    tx = torch.as_tensor(x, device=dev)
    tu = torch.as_tensor(u, device=dev)
    ty = torch.as_tensor(y, device=dev)

    def full_loss() -> float:
        ensemble.eval()
        with torch.no_grad():
            pred = ensemble(tx, tu)
            value = torch.mean((pred - ty.unsqueeze(0)) ** 2).item()
        ensemble.train()
        return float(value)

    initial = full_loss()
    for _ in range(steps):
        for model, optimizer, rng in zip(ensemble.models, optimizers, rngs, strict=True):
            # Independent bootstrap minibatches are what make disagreement a
            # useful epistemic signal instead of K identical networks.
            idx = rng.integers(0, x.shape[0], size=min(batch_size, x.shape[0]))
            pred = model(tx[idx], tu[idx])
            loss = loss_fn(pred, ty[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite residual dynamics loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    final = full_loss()
    return EnsembleTrainingStats(initial, final, steps)


@dataclass(frozen=True)
class UncertaintyGate:
    """Reduce learned residual authority as ensemble disagreement increases.

    This is a robustness heuristic, not a safety certificate. Hard constraints
    remain the responsibility of the HOCBF safety layer.
    """

    gain: float = 4.0
    min_scale: float = 0.1

    def __post_init__(self) -> None:
        if self.gain < 0.0 or not 0.0 <= self.min_scale <= 1.0:
            raise ValueError("invalid uncertainty gate parameters")

    def scale(self, uncertainty) -> float:
        u = np.asarray(uncertainty, dtype=np.float64)
        if not np.all(np.isfinite(u)) or np.any(u < 0.0):
            raise ValueError("uncertainty must be finite and non-negative")
        scalar = float(np.linalg.norm(u))
        return float(max(self.min_scale, 1.0 / (1.0 + self.gain * scalar)))

    def apply(self, residual, uncertainty) -> tuple[np.ndarray, float]:
        residual = np.asarray(residual, dtype=np.float64)
        if residual.shape != (2,) or not np.all(np.isfinite(residual)):
            raise ValueError("residual must be a finite 2-vector")
        scale = self.scale(uncertainty)
        return residual * scale, scale
