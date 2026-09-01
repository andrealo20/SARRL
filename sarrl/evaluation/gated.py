"""Held-out evaluation for residual policies with uncertainty gating."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import EpisodeResult


@dataclass(frozen=True)
class GateEpisodeDiagnostics:
    """Per-episode diagnostics retained alongside A4 outcome metrics."""

    scenario: str
    controller: str
    seed: int
    steps: int
    uncertainty_scale_mean: float
    uncertainty_scale_min: float
    uncertainty_norm_mean: float
    uncertainty_norm_max: float
    raw_residual_norm_mean: float
    gated_residual_norm_mean: float


def evaluate_gated_policy_episodes(
    stack,
    env,
    episodes: int,
    seed: int,
    scenario: str = "heldout",
    controller: str = "gated_policy",
) -> tuple[list[EpisodeResult], list[GateEpisodeDiagnostics]]:
    """Evaluate one composed policy and retain outcome and gate diagnostics."""
    if episodes <= 0 or seed < 0:
        raise ValueError("episodes must be positive and seed non-negative")

    rows: list[EpisodeResult] = []
    diagnostics: list[GateEpisodeDiagnostics] = []

    for episode in range(episodes):
        episode_seed = seed + episode
        obs, _ = env.reset(seed=episode_seed)
        total = 0.0
        max_speed = 0.0
        max_torque = 0.0
        fault_seen = False
        steps = 0
        scales: list[float] = []
        uncertainty_norms: list[float] = []
        raw_residual_norms: list[float] = []
        gated_residual_norms: list[float] = []

        while True:
            result = stack.command(
                obs,
                env.state,
                env.q_des,
                deterministic=True,
            )
            if not result.executable:
                raise RuntimeError("A4 stack produced a non-executable command")

            scales.append(float(result.uncertainty_scale))
            uncertainty_norms.append(float(np.linalg.norm(result.uncertainty)))
            raw_residual_norms.append(float(np.linalg.norm(result.raw_residual)))
            gated_residual_norms.append(float(np.linalg.norm(result.gated_residual)))

            obs, reward, terminated, truncated, info = env.step_torque(
                result.torque,
                baseline=result.baseline_torque,
            )
            steps += 1
            total += float(reward)
            max_speed = max(max_speed, float(np.max(np.abs(env.state[2:]))))
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
                diagnostics.append(
                    GateEpisodeDiagnostics(
                        scenario=scenario,
                        controller=controller,
                        seed=episode_seed,
                        steps=steps,
                        uncertainty_scale_mean=float(np.mean(scales)),
                        uncertainty_scale_min=float(np.min(scales)),
                        uncertainty_norm_mean=float(np.mean(uncertainty_norms)),
                        uncertainty_norm_max=float(np.max(uncertainty_norms)),
                        raw_residual_norm_mean=float(np.mean(raw_residual_norms)),
                        gated_residual_norm_mean=float(np.mean(gated_residual_norms)),
                    )
                )
                break

    return rows, diagnostics
