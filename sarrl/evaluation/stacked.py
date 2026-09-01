"""Held-out evaluation for safety-filtered and fully composed planar stacks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import EpisodeResult


@dataclass(frozen=True)
class StackEpisodeDiagnostics:
    """Per-episode gate, context and HOCBF diagnostics."""

    scenario: str
    controller: str
    seed: int
    steps: int
    command_attempts: int
    safety_infeasible: bool
    safety_certified_steps: int
    safety_intervention_steps: int
    unsafe_state_steps: int
    safety_correction_mean: float
    safety_correction_max: float
    safety_correction_sum: float
    safety_constraint_margin_min: float
    uncertainty_scale_mean: float
    uncertainty_scale_min: float
    uncertainty_norm_mean: float
    uncertainty_norm_max: float
    raw_residual_norm_mean: float
    gated_residual_norm_mean: float
    context_latent_norm_mean: float
    context_latent_norm_max: float


def _physical_env(env):
    return getattr(env, "env", env)


def _distance(env) -> float:
    base = _physical_env(env)
    position = base.arm.forward_kinematics(base.state[:2])
    return float(np.linalg.norm(base.target - position))


def evaluate_stack_episodes(
    stack,
    env,
    episodes: int,
    seed: int,
    scenario: str = "heldout",
    controller: str = "safe_stack",
    *,
    context_residual_limit: float | None = None,
    intervention_tolerance: float = 1e-9,
) -> tuple[list[EpisodeResult], list[StackEpisodeDiagnostics]]:
    """Evaluate a required-safety stack with optional causal context updates."""
    if episodes <= 0 or seed < 0:
        raise ValueError("episodes must be positive and seed non-negative")
    if stack.safety_filter is None or not stack.config.require_safety:
        raise ValueError("stack evaluation requires a hard safety filter")
    if context_residual_limit is not None and context_residual_limit <= 0.0:
        raise ValueError("context residual limit must be positive")
    if intervention_tolerance < 0.0:
        raise ValueError("intervention tolerance must be non-negative")

    rows: list[EpisodeResult] = []
    diagnostics: list[StackEpisodeDiagnostics] = []

    for episode in range(episodes):
        episode_seed = seed + episode
        obs, _ = env.reset(seed=episode_seed)
        total = 0.0
        max_speed = 0.0
        max_torque = 0.0
        fault_seen = False
        steps = 0
        attempts = 0
        infeasible = False
        certified_steps = 0
        unsafe_state_steps = 0
        corrections: list[float] = []
        constraint_margins: list[float] = []
        scales: list[float] = []
        uncertainty_norms: list[float] = []
        raw_residual_norms: list[float] = []
        gated_residual_norms: list[float] = []
        context_latent_norms: list[float] = []
        final_info = None

        while True:
            state = np.asarray(env.state, dtype=np.float64)
            A, b, current_safe = stack.safety_filter.constraints(state)
            if not current_safe:
                unsafe_state_steps += 1

            result = stack.command(
                obs,
                state,
                env.q_des,
                deterministic=True,
            )
            attempts += 1
            corrections.append(float(result.safety_correction))
            constraint_margins.append(float(np.min(A @ result.torque - b)))
            scales.append(float(result.uncertainty_scale))
            uncertainty_norms.append(float(np.linalg.norm(result.uncertainty)))
            raw_residual_norms.append(float(np.linalg.norm(result.raw_residual)))
            gated_residual_norms.append(float(np.linalg.norm(result.gated_residual)))
            context_latent_norms.append(
                float(np.linalg.norm(getattr(env, "latent", np.zeros(1))))
            )

            if result.safety_certified:
                certified_steps += 1
            if not result.executable:
                infeasible = True
                break

            if context_residual_limit is None:
                obs, reward, terminated, truncated, info = env.step_torque(
                    result.torque,
                    baseline=result.baseline_torque,
                )
            else:
                context_action = np.clip(
                    result.raw_residual / context_residual_limit,
                    -1.0,
                    1.0,
                )
                obs, reward, terminated, truncated, info = env.step_torque(
                    result.torque,
                    baseline=result.baseline_torque,
                    context_action=context_action,
                )

            steps += 1
            total += float(reward)
            max_speed = max(max_speed, float(np.max(np.abs(env.state[2:]))))
            max_torque = max(
                max_torque,
                float(np.max(np.abs(np.asarray(info["commanded_torque"], dtype=np.float64)))),
            )
            fault_seen = fault_seen or bool(info.get("fault_active", False))
            final_info = info

            if terminated or truncated:
                break

        success = (
            bool(final_info["success"])
            if final_info is not None and not infeasible
            else False
        )
        final_distance = (
            float(final_info["distance"])
            if final_info is not None and not infeasible
            else _distance(env)
        )
        rows.append(
            EpisodeResult(
                scenario=scenario,
                controller=controller,
                seed=episode_seed,
                reward=total,
                steps=steps,
                success=success,
                final_distance=final_distance,
                max_speed=max_speed,
                max_command_torque=max_torque,
                fault_seen=fault_seen,
            )
        )
        diagnostics.append(
            StackEpisodeDiagnostics(
                scenario=scenario,
                controller=controller,
                seed=episode_seed,
                steps=steps,
                command_attempts=attempts,
                safety_infeasible=infeasible,
                safety_certified_steps=certified_steps,
                safety_intervention_steps=sum(
                    correction > intervention_tolerance for correction in corrections
                ),
                unsafe_state_steps=unsafe_state_steps,
                safety_correction_mean=float(np.mean(corrections)),
                safety_correction_max=float(np.max(corrections)),
                safety_correction_sum=float(np.sum(corrections)),
                safety_constraint_margin_min=float(np.min(constraint_margins)),
                uncertainty_scale_mean=float(np.mean(scales)),
                uncertainty_scale_min=float(np.min(scales)),
                uncertainty_norm_mean=float(np.mean(uncertainty_norms)),
                uncertainty_norm_max=float(np.max(uncertainty_norms)),
                raw_residual_norm_mean=float(np.mean(raw_residual_norms)),
                gated_residual_norm_mean=float(np.mean(gated_residual_norms)),
                context_latent_norm_mean=float(np.mean(context_latent_norms)),
                context_latent_norm_max=float(np.max(context_latent_norms)),
            )
        )

    return rows, diagnostics
