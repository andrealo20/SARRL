"""Soft Actor-Critic implemented directly in PyTorch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sarrl.rl.networks import QNetwork, SquashedGaussianActor
from sarrl.utils import seed_everything


@dataclass(frozen=True)
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.2
    hidden: tuple[int, int] = (256, 256)

    def validate(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must lie in (0, 1]")
        if any(v <= 0.0 for v in (self.actor_lr, self.critic_lr, self.alpha_lr, self.init_alpha)):
            raise ValueError("learning rates and init_alpha must be positive")


class SACAgent:
    CHECKPOINT_VERSION = 1

    def __init__(self, obs_dim: int, action_dim: int, config: SACConfig | None = None, seed=0):
        self.config = config or SACConfig()
        self.config.validate()
        seed_everything(seed)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden = self.config.hidden
        self.actor = SquashedGaussianActor(obs_dim, action_dim, hidden).to(self.device)
        self.q1 = QNetwork(obs_dim, action_dim, hidden).to(self.device)
        self.q2 = QNetwork(obs_dim, action_dim, hidden).to(self.device)
        self.q1_target = QNetwork(obs_dim, action_dim, hidden).to(self.device)
        self.q2_target = QNetwork(obs_dim, action_dim, hidden).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for net in (self.q1_target, self.q2_target):
            for param in net.parameters():
                param.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=self.config.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=self.config.critic_lr)
        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(self.config.init_alpha), dtype=torch.float32, device=self.device)
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
        self.target_entropy = -float(action_dim)
        self.update_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, det = self.actor.sample(x)
            chosen = det if deterministic else action
        return chosen.squeeze(0).cpu().numpy().astype(np.float32)

    def compute_bellman_target(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_q: torch.Tensor,
        next_log_prob: torch.Tensor,
    ) -> torch.Tensor:
        return rewards + self.config.gamma * (1.0 - dones) * (
            next_q - self.alpha.detach() * next_log_prob
        )

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs)
            next_q = torch.minimum(
                self.q1_target(next_obs, next_action), self.q2_target(next_obs, next_action)
            )
            target = self.compute_bellman_target(rewards, dones, next_q, next_log_prob)

        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)
        q1_loss = torch.mean((q1 - target) ** 2)
        q2_loss = torch.mean((q2 - target) ** 2)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        new_action, log_prob, _ = self.actor.sample(obs)
        q_pi = torch.minimum(self.q1(obs, new_action), self.q2(obs, new_action))
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.soft_update_targets()
        self.update_steps += 1

        metrics = {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.detach().item()),
            "q_target_mean": float(target.mean().item()),
            "entropy_proxy": float((-log_prob).mean().item()),
        }
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError("non-finite SAC metric")
        return metrics

    def soft_update_targets(self) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for target, source in ((self.q1_target, self.q1), (self.q2_target, self.q2)):
                for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
                    tp.mul_(1.0 - tau).add_(sp, alpha=tau)

    def save(self, path) -> None:
        payload = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "q1_opt": self.q1_opt.state_dict(),
            "q2_opt": self.q2_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "update_steps": self.update_steps,
        }
        torch.save(payload, Path(path))

    def load(self, path, load_optimizers: bool = True) -> dict:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if payload.get("checkpoint_version") != self.CHECKPOINT_VERSION:
            raise ValueError("unsupported SARRL SAC checkpoint version")
        if payload["obs_dim"] != self.obs_dim or payload["action_dim"] != self.action_dim:
            raise ValueError("checkpoint dimensions do not match this agent")
        for key, net in (
            ("actor", self.actor),
            ("q1", self.q1),
            ("q2", self.q2),
            ("q1_target", self.q1_target),
            ("q2_target", self.q2_target),
        ):
            net.load_state_dict(payload[key])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"].to(self.device))
        if load_optimizers:
            self.actor_opt.load_state_dict(payload["actor_opt"])
            self.q1_opt.load_state_dict(payload["q1_opt"])
            self.q2_opt.load_state_dict(payload["q2_opt"])
            self.alpha_opt.load_state_dict(payload["alpha_opt"])
        self.update_steps = int(payload.get("update_steps", 0))
        return {
            "checkpoint_version": payload["checkpoint_version"],
            "update_steps": self.update_steps,
        }
