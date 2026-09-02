"""Paired trajectory-level safety evaluation for planar control stacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sarrl.safety import HOCBFSafetyFilter, SafetyConfig

from .protocol import EpisodeResult


@dataclass(frozen=True)
class SafetyEpisodeDiagnostics:
    """Per-episode physical-envelope and command-certificate diagnostics."""

    scenario: str
    controller: str
    seed: int
    steps: int
    state_observations: int
    unsafe_state_observations: int
    unsafe_state_fraction: float
    unsafe_episode: bool
    unsafe_entry_count: int
    first_unsafe_observation: int
    joint_position_violation_max_rad: float
    joint_velocity_violation_max_rad_s: float
    normalized_violation_mean: float
    normalized_violation_max: float
    normalized_violation_integral: float
    command_attempts: int
    candidate_constraint_violation_steps: int
    candidate_constraint_violation_fraction: float
    candidate_constraint_margin_min: float
    executed_constraint_margin_min: float
    safety_enabled: bool
    safety_infeasible: bool
    safety_certified_steps: int
    safety_intervention_steps: int
    safety_intervention_fraction: float
    safety_correction_mean: float
    safety_correction_max: float
    fault_seen: bool
    success: bool


def _physical_env(env):
    return getattr(env, "env", env)


def _distance(env) -> float:
    base = _physical_env(env)
    position = base.arm.forward_kinematics(base.state[:2])
    return float(np.linalg.norm(base.target - position))


def safety_envelope_violation(
    state, config: SafetyConfig
) -> tuple[bool, float, float, float]:
    """Return unsafe, position excess, velocity excess and normalized excess."""
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (4,) or not np.all(np.isfinite(state)):
        raise ValueError("state must be a finite vector of shape (4,)")
    config.validate()
    q = state[:2]
    qd = state[2:]
    lower = np.asarray(config.joint_lower, dtype=np.float64)
    upper = np.asarray(config.joint_upper, dtype=np.float64)
    velocity = np.asarray(config.velocity_limit, dtype=np.float64)

    lower_excess = np.maximum(lower - q, 0.0)
    upper_excess = np.maximum(q - upper, 0.0)
    position_excess = float(np.max(np.maximum(lower_excess, upper_excess)))
    velocity_excess = float(np.max(np.maximum(np.abs(qd) - velocity, 0.0)))

    lower_scale = np.maximum(np.abs(lower), np.finfo(np.float64).eps)
    upper_scale = np.maximum(np.abs(upper), np.finfo(np.float64).eps)
    position_normalized = float(
        np.max(np.maximum(lower_excess / lower_scale, upper_excess / upper_scale))
    )
    velocity_normalized = float(np.max(np.maximum(np.abs(qd) / velocity - 1.0, 0.0)))
    normalized = max(position_normalized, velocity_normalized)
    return normalized > 0.0, position_excess, velocity_excess, normalized


def paired_diagnostic_difference(
    filtered: list[SafetyEpisodeDiagnostics],
    reference: list[SafetyEpisodeDiagnostics],
    metric: Callable[[SafetyEpisodeDiagnostics], float],
    *,
    bootstrap: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for a diagnostic mean difference."""
    if bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    filtered_map = {row.seed: float(metric(row)) for row in filtered}
    reference_map = {row.seed: float(metric(row)) for row in reference}
    if len(filtered_map) != len(filtered) or len(reference_map) != len(reference):
        raise ValueError("paired comparisons require unique episode seeds")
    if filtered_map.keys() != reference_map.keys() or not filtered_map:
        raise ValueError("paired comparisons require the same non-empty seed set")
    seeds = sorted(filtered_map)
    differences = np.asarray(
        [filtered_map[item] - reference_map[item] for item in seeds], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(bootstrap, len(differences)))
    distribution = differences[draws].mean(axis=1)
    return (
        float(differences.mean()),
        float(np.quantile(distribution, 0.025)),
        float(np.quantile(distribution, 0.975)),
    )


def _append_state_violation(
    state,
    config: SafetyConfig,
    unsafe_flags: list[bool],
    position_violations: list[float],
    velocity_violations: list[float],
    normalized_violations: list[float],
) -> None:
    unsafe, position, velocity, normalized = safety_envelope_violation(state, config)
    unsafe_flags.append(unsafe)
    position_violations.append(position)
    velocity_violations.append(velocity)
    normalized_violations.append(normalized)


def evaluate_safety_episodes(
    stack,
    observer: HOCBFSafetyFilter,
    env,
    episodes: int,
    seed: int,
    scenario: str = "heldout",
    controller: str = "controller",
    *,
    context_residual_limit: float | None = None,
    intervention_tolerance: float = 1e-9,
) -> tuple[list[EpisodeResult], list[SafetyEpisodeDiagnostics]]:
    """Evaluate filtered or unfiltered stacks against one fixed safety envelope."""
    if episodes <= 0 or seed < 0:
        raise ValueError("episodes must be positive and seed non-negative")
    if context_residual_limit is not None and context_residual_limit <= 0.0:
        raise ValueError("context residual limit must be positive")
    if intervention_tolerance < 0.0:
        raise ValueError("intervention tolerance must be non-negative")

    config = observer.config
    results: list[EpisodeResult] = []
    diagnostics: list[SafetyEpisodeDiagnostics] = []
    safety_enabled = stack.safety_filter is not None

    for episode in range(episodes):
        episode_seed = seed + episode
        obs, _ = env.reset(seed=episode_seed)
        total_reward = 0.0
        max_speed = 0.0
        max_torque = 0.0
        fault_seen = False
        steps = 0
        attempts = 0
        infeasible = False
        certified_steps = 0
        intervention_steps = 0
        candidate_violation_steps = 0
        candidate_margins: list[float] = []
        executed_margins: list[float] = []
        corrections: list[float] = []
        unsafe_flags: list[bool] = []
        position_violations: list[float] = []
        velocity_violations: list[float] = []
        normalized_violations: list[float] = []
        final_info = None

        _append_state_violation(
            env.state,
            config,
            unsafe_flags,
            position_violations,
            velocity_violations,
            normalized_violations,
        )
        while True:
            state = np.asarray(env.state, dtype=np.float64)
            constraints, bounds, _ = observer.constraints(state)
            command = stack.command(obs, state, env.q_des, deterministic=True)
            attempts += 1

            candidate = command.baseline_torque + command.gated_residual
            candidate_margin = float(np.min(constraints @ candidate - bounds))
            candidate_margins.append(candidate_margin)
            candidate_violation_steps += int(candidate_margin < -config.feasibility_tol)
            corrections.append(float(command.safety_correction))
            intervention_steps += int(command.safety_correction > intervention_tolerance)
            certified_steps += int(command.safety_certified)

            if not command.executable:
                infeasible = True
                break

            executed_margins.append(float(np.min(constraints @ command.torque - bounds)))
            if context_residual_limit is None:
                obs, reward, terminated, truncated, info = env.step_torque(
                    command.torque,
                    baseline=command.baseline_torque,
                )
            else:
                context_action = np.clip(
                    command.raw_residual / context_residual_limit,
                    -1.0,
                    1.0,
                )
                obs, reward, terminated, truncated, info = env.step_torque(
                    command.torque,
                    baseline=command.baseline_torque,
                    context_action=context_action,
                )
            steps += 1
            total_reward += float(reward)
            max_speed = max(max_speed, float(np.max(np.abs(env.state[2:]))))
            max_torque = max(
                max_torque,
                float(np.max(np.abs(np.asarray(info["commanded_torque"], dtype=np.float64)))),
            )
            fault_seen = fault_seen or bool(info.get("fault_active", False))
            final_info = info
            _append_state_violation(
                env.state,
                config,
                unsafe_flags,
                position_violations,
                velocity_violations,
                normalized_violations,
            )
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
        results.append(
            EpisodeResult(
                scenario=scenario,
                controller=controller,
                seed=episode_seed,
                reward=total_reward,
                steps=steps,
                success=success,
                final_distance=final_distance,
                max_speed=max_speed,
                max_command_torque=max_torque,
                fault_seen=fault_seen,
            )
        )

        unsafe_count = sum(unsafe_flags)
        unsafe_entries = sum(
            current and (index == 0 or not unsafe_flags[index - 1])
            for index, current in enumerate(unsafe_flags)
        )
        first_unsafe = next(
            (index for index, current in enumerate(unsafe_flags) if current), -1
        )
        diagnostics.append(
            SafetyEpisodeDiagnostics(
                scenario=scenario,
                controller=controller,
                seed=episode_seed,
                steps=steps,
                state_observations=len(unsafe_flags),
                unsafe_state_observations=unsafe_count,
                unsafe_state_fraction=unsafe_count / len(unsafe_flags),
                unsafe_episode=unsafe_count > 0,
                unsafe_entry_count=unsafe_entries,
                first_unsafe_observation=first_unsafe,
                joint_position_violation_max_rad=float(np.max(position_violations)),
                joint_velocity_violation_max_rad_s=float(np.max(velocity_violations)),
                normalized_violation_mean=float(np.mean(normalized_violations)),
                normalized_violation_max=float(np.max(normalized_violations)),
                normalized_violation_integral=float(
                    np.sum(normalized_violations) * _physical_env(env).dt
                ),
                command_attempts=attempts,
                candidate_constraint_violation_steps=candidate_violation_steps,
                candidate_constraint_violation_fraction=candidate_violation_steps / attempts,
                candidate_constraint_margin_min=float(np.min(candidate_margins)),
                executed_constraint_margin_min=(
                    float(np.min(executed_margins)) if executed_margins else float("nan")
                ),
                safety_enabled=safety_enabled,
                safety_infeasible=infeasible,
                safety_certified_steps=certified_steps,
                safety_intervention_steps=intervention_steps,
                safety_intervention_fraction=intervention_steps / attempts,
                safety_correction_mean=float(np.mean(corrections)),
                safety_correction_max=float(np.max(corrections)),
                fault_seen=fault_seen,
                success=success,
            )
        )

    return results, diagnostics
