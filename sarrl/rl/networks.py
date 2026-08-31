"""Neural networks for Soft Actor-Critic."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for width in hidden:
        layers.extend([nn.Linear(last, width), nn.ReLU()])
        last = width
    layers.append(nn.Linear(last, out_dim))
    net = nn.Sequential(*layers)
    for module in net:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(module.bias)
    nn.init.orthogonal_(net[-1].weight, gain=0.01)
    return net


class SquashedGaussianActor(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        if not hidden:
            raise ValueError("actor needs at least one hidden layer")
        trunk_layers: list[nn.Module] = []
        last = obs_dim
        for width in hidden:
            layer = nn.Linear(last, width)
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
            trunk_layers.extend([layer, nn.ReLU()])
            last = width
        self.trunk = nn.Sequential(*trunk_layers)
        self.mean = nn.Linear(last, action_dim)
        self.log_std = nn.Linear(last, action_dim)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)
        nn.init.zeros_(self.log_std.bias)

    def distribution_params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    @staticmethod
    def _log_det_tanh(raw: torch.Tensor) -> torch.Tensor:
        # log(1 - tanh(x)^2), stable for large |x|.
        return 2.0 * (math.log(2.0) - raw - F.softplus(-2.0 * raw))

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Mean action without sampling or advancing any RNG state."""
        mean, _ = self.distribution_params(obs)
        return torch.tanh(mean)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution_params(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = (dist.log_prob(raw) - self._log_det_tanh(raw)).sum(-1, keepdim=True)
        deterministic = torch.tanh(mean)
        return action, log_prob, deterministic

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        bounded = torch.clamp(action, -1.0 + eps, 1.0 - eps)
        raw = torch.atanh(bounded)
        mean, log_std = self.distribution_params(obs)
        dist = torch.distributions.Normal(mean, log_std.exp())
        return (dist.log_prob(raw) - self._log_det_tanh(raw)).sum(-1, keepdim=True)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = _mlp(obs_dim + action_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))
