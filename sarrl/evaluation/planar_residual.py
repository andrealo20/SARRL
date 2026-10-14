"""Controlled residual comparison on the retained identified/margined planar stack."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from sarrl.controllers import AdaptiveNominalConfig, AdaptiveNominalController, CommandRegressor
from sarrl.dynamics import PlanarArm
from sarrl.evaluation.adaptive_pilot import (
    MeasuredState,
    NominalStack,
    PilotEpisode,
    PlantOptions,
    _prediction_error,
    build_arm,
    make_env,
    probe_bank,
)
from sarrl.evaluation.margin_campaign import V22_PLANT_OPTIONS, v22_config
from sarrl.evaluation.planar_v12 import planar_safety_config
from sarrl.evaluation.planar_v13 import v13_scenarios
from sarrl.evaluation.safety_audit import evaluate_safety_episodes
from sarrl.safety import HOCBFSafetyFilter, MarginHOCBFSafetyFilter

TRAINING_SEEDS = tuple(range(9806000, 9806005))
EVALUATION_SEEDS = tuple(range(9806100, 9806200))
SCENARIOS = ("id_reference", "ood_compound", "motor_fault")
ARMS = ("fixed", "adaptive", *[f"residual_{seed}" for seed in TRAINING_SEEDS])
TRAINING_STEPS = 200000
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 260000


def study_protocol():
    return {
        "kind": "controlled_planar_adaptive_residual",
        "scenarios": list(SCENARIOS),
        "arms": list(ARMS),
        "training_seeds": list(TRAINING_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "training_steps_per_model": TRAINING_STEPS,
        "training": {
            "start_steps": 5000,
            "batch_size": 256,
            "hidden": [256, 256],
            "replay_capacity": 200000,
            "update_every": 1,
            "checkpoint_every": 25000,
            "selection": "final checkpoint only, no validation or early stopping",
            "residual_limit_nm": 8.0,
            "certificate_kappa": 0.25,
            "device": "cpu",
        },
        "plant": "mujoco",
        "plant_options": asdict(V22_PLANT_OPTIONS),
        "adaptive_config": asdict(v22_config()),
        "scenarios_config": [s.to_dict() for s in v13_scenarios()],
        "reproduction": {
            "records": 300,
            "arm": "adaptive_hocbf_margin",
            "seeds": list(range(56200, 56300)),
            "reference": "results/uncertainty_margin_v22/episodes.jsonl",
        },
        "runtime": {
            "python_major_minor": "3.12",
            "numpy": "2.4.6",
            "mujoco": "3.14.0",
            "torch": "2.12.0+cu130",
            "numerical_threads": 1,
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "crossed training-model and whole task-seed blocks across scenarios",
        },
        "gate": {
            "primary_scenarios": ["id_reference", "motor_fault"],
            "primary_success_gain_min": 0.03,
            "primary_success_lower95_above": 0.0,
            "each_scenario_success_lower95_min": -0.03,
            "each_scenario_unsafe_upper95_max": 0.01,
            "each_scenario_abort_point_increase_max": 0.03,
            "nonnegative_primary_models_min": 4,
            "meaning": "all criteria to advance; no default change or general safety claim",
        },
    }


def analyze(records, *, bootstrap_draws=BOOTSTRAP_DRAWS):
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in records}
    seeds = sorted({r["seed"] for r in records})
    expected = {(c, s, a) for c in SCENARIOS for s in seeds for a in ARMS}
    if not seeds or len(records) != len(lookup) or set(lookup) != expected:
        raise ValueError("residual analysis requires a complete paired factorial")
    if type(bootstrap_draws) is not int or bootstrap_draws < 1:
        raise ValueError("bootstrap draws must be a positive integer")
    for row in records:
        if not np.isfinite(row["final_distance"]) or row["final_distance"] < 0:
            raise ValueError("analysis requires finite nonnegative distances")
        for field in ("success", "unsafe_episode", "safety_infeasible"):
            if type(row[field]) is not bool:
                raise ValueError("analysis endpoints must be Boolean")
    arrays = {}
    for field in ("success", "unsafe_episode", "safety_infeasible"):
        arrays[field] = np.array(
            [
                [
                    [
                        float(lookup[c, s, arm][field]) - float(lookup[c, s, "adaptive"][field])
                        for s in seeds
                    ]
                    for c in SCENARIOS
                ]
                for arm in ARMS[2:]
            ]
        )
    # Draw both axes together; each sampled task keeps all three conditions.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    model_index = rng.integers(0, len(TRAINING_SEEDS), size=(bootstrap_draws, len(TRAINING_SEEDS)))
    task_index = rng.integers(0, len(seeds), size=(bootstrap_draws, len(seeds)))
    sampled = {
        field: np.array(
            [
                values[models][:, :, tasks].mean(axis=(0, 2))
                for models, tasks in zip(model_index, task_index, strict=True)
            ]
        )
        for field, values in arrays.items()
    }
    contrasts = {}
    failures = []
    for index, condition in enumerate(SCENARIOS):
        contrasts[condition] = {}
        for field in arrays:
            point = float(arrays[field][:, index, :].mean())
            lo, hi = np.quantile(sampled[field][:, index], [0.025, 0.975])
            contrasts[condition][field] = {
                "difference": point,
                "lower95": float(lo),
                "upper95": float(hi),
            }
        if contrasts[condition]["success"]["lower95"] < -0.03:
            failures.append({"condition": condition, "criterion": "success_noninferiority"})
        if contrasts[condition]["unsafe_episode"]["upper95"] > 0.01:
            failures.append({"condition": condition, "criterion": "unsafe_margin"})
        if contrasts[condition]["safety_infeasible"]["difference"] > 0.03:
            failures.append({"condition": condition, "criterion": "abort_margin"})
    primary_indices = [SCENARIOS.index(c) for c in ("id_reference", "motor_fault")]
    primary_point = float(arrays["success"][:, primary_indices, :].mean())
    primary_lo, primary_hi = np.quantile(
        sampled["success"][:, primary_indices].mean(axis=1), [0.025, 0.975]
    )
    per_model = arrays["success"][:, primary_indices, :].mean(axis=(1, 2))
    if primary_point < 0.03 or primary_lo <= 0:
        failures.append({"condition": "primary", "criterion": "success_gain"})
    if np.count_nonzero(per_model >= 0) < 4:
        failures.append({"condition": "primary", "criterion": "training_model_consistency"})
    cells = {}
    for condition in SCENARIOS:
        for arm in ARMS:
            selected = [lookup[condition, s, arm] for s in seeds]
            cells[f"{condition}/{arm}"] = {
                "episodes": len(selected),
                "successes": sum(r["success"] for r in selected),
                "unsafe_episodes": sum(r["unsafe_episode"] for r in selected),
                "aborts": sum(r["safety_infeasible"] for r in selected),
                "mean_final_distance_m": float(np.mean([r["final_distance"] for r in selected])),
            }
    return {
        "episodes": len(records),
        "cells": cells,
        "paired_contrasts": contrasts,
        "bootstrap_draws": bootstrap_draws,
        "primary_success": {
            "difference": primary_point,
            "lower95": float(primary_lo),
            "upper95": float(primary_hi),
            "per_model_difference": per_model.tolist(),
        },
        "decision": {
            "residual_advances": not failures,
            "reference_controller": "adaptive",
            "failures": failures,
            "meaning": "controlled comparison; no default change",
        },
    }


class RecordingStack(NominalStack):
    def __init__(self, *args, retain_trace=False, physical_state=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.retain_trace = retain_trace
        self.trace = []
        self.physical_state = physical_state

    def command(self, observation, state, q_des, obstacles=(), deterministic=True):
        result = super().command(observation, state, q_des, obstacles, deterministic)
        if self.retain_trace:
            self.trace.append(
                {
                    "observation": np.asarray(observation).tolist(),
                    "measured_state": np.asarray(state).tolist(),
                    "q_des": np.asarray(q_des).tolist(),
                    "torque": result.torque.tolist(),
                    "baseline_torque": result.baseline_torque.tolist(),
                    "raw_residual": result.raw_residual.tolist(),
                    "executable": result.executable,
                    "safety_certified": result.safety_certified,
                    "correction": result.safety_correction,
                    "physical_state": self.physical_state().tolist(),
                }
            )
        return result


def run_episode(seed, scenario, arm, *, policy=None, retain_trace=False):
    if arm not in ARMS or scenario not in SCENARIOS:
        raise ValueError("unknown planar residual cell")
    if arm.startswith("residual_") != (policy is not None):
        raise ValueError("residual arms require a policy and baselines cannot receive one")
    return _run_case(
        arm,
        scenario,
        seed,
        "heldout",
        v22_config(),
        True,
        "mujoco",
        V22_PLANT_OPTIONS,
        policy,
        0.0 if arm == "fixed" else 0.25,
        retain_trace,
    )


def _run_case(
    arm: str,
    scenario: str,
    seed: int,
    origin: str,
    config: AdaptiveNominalConfig,
    compensate_delay: bool = False,
    plant: str = "analytical",
    options: PlantOptions | None = None,
    policy=None,
    certificate_margin: float = 0.0,
    retain_trace=False,
):
    spec = {s.key: s for s in v13_scenarios()}[scenario]
    options = options or PlantOptions()
    env = make_env(plant, spec, options)
    effective = "fixed_hocbf" if arm == "fixed" else "adaptive_hocbf_margin"
    controller, certificate, estimator = build_arm(effective, config, certificate_margin)
    stack = RecordingStack(
        controller,
        certificate,
        compensate_delay=compensate_delay and arm != "fixed",
        policy=policy,
        estimator=estimator,
        retain_trace=retain_trace,
        physical_state=lambda: env.state.copy(),
    )
    # Estimator fields describe whichever estimator ran, the controller or an observer.
    estimator = stack.estimator if stack.estimator is not None else controller
    adaptive = isinstance(estimator, AdaptiveNominalController)
    if adaptive:
        controller = estimator
        controller.reset()
    converged_at = {"step": None}
    measured = MeasuredState(env) if options.sensor_noise_std > 0.0 else None
    residual_squares = []

    def observe(event):
        if retain_trace:
            stack.trace[-1].update(
                physical_next_state=env.state.tolist(), executed=event["info"] is not None
            )
            stack.trace[-1]["measured_next_state"] = (
                (measured() if measured is not None else env.state).tolist()
                if event["info"] is not None
                else event["state"].tolist()
            )
        if policy is not None:
            residual = np.asarray(event["command"].raw_residual, dtype=np.float64)
            residual_squares.append(float(residual @ residual))
        if adaptive and event["info"] is not None:
            after = measured() if measured is not None else env.state
            controller.observe(event["state"], after, event["command"].torque)
            if converged_at["step"] is None and controller.converged:
                converged_at["step"] = int(env.steps)

    outcomes, diagnostics = evaluate_safety_episodes(
        stack,
        HOCBFSafetyFilter(PlanarArm(), planar_safety_config()),
        env,
        episodes=1,
        seed=seed,
        scenario=scenario,
        controller=arm,
        transition_callback=observe,
        state_source=measured,
    )
    outcome, safety = outcomes[0], diagnostics[0]
    if safety.safety_infeasible:
        label = "abort"
    elif outcome.success:
        label = "success"
    else:
        label = "timeout"
    probes = probe_bank()
    true_parameters = CommandRegressor.parameters(env.arm.params, env.motor_gain).tolist()
    certificate = stack.safety_filter
    margined = isinstance(certificate, MarginHOCBFSafetyFilter)
    episode = PilotEpisode(
        arm=arm,
        scenario=scenario,
        seed=seed,
        origin=origin,
        plant=plant,
        outcome=label,
        steps=int(outcome.steps),
        final_distance=float(outcome.final_distance),
        reward=float(outcome.reward),
        max_speed=float(outcome.max_speed),
        max_command_torque=float(outcome.max_command_torque),
        fault_seen=bool(outcome.fault_seen),
        success=bool(outcome.success),
        unsafe_episode=bool(safety.unsafe_episode),
        normalized_violation_max=float(safety.normalized_violation_max),
        safety_infeasible=bool(safety.safety_infeasible),
        safety_intervention_fraction=float(safety.safety_intervention_fraction),
        true_delay=int(env.action_delay),
        selected_lag=int(controller.lag) if adaptive else None,
        converged_at_step=converged_at["step"],
        prediction_error_rms=_prediction_error(controller, env, probes) if adaptive else None,
        estimated_parameters=controller.parameters.tolist() if adaptive else None,
        true_parameters=true_parameters,
        model_fallbacks=int(controller.model_fallbacks) if adaptive else 0,
        prediction_fallbacks=int(controller.prediction_fallbacks) if adaptive else 0,
        model_fallback_steps=int(controller.model_fallback_steps) if adaptive else 0,
        control_steps=int(controller.control_steps) if adaptive else 0,
        selected_time_constant=float(controller.time_constant) if adaptive else None,
        true_time_constant=float(getattr(env, "actuator_time_constant", 0.0)),
        true_armature=float(getattr(env, "armature", 0.0)),
        residual_rms=float(np.sqrt(np.mean(residual_squares))) if residual_squares else None,
        obstacle_present=bool(env.obstacles),
        obstacle_violation_max_m=float(safety.obstacle_violation_max_m),
        obstacle_contact=bool(safety.obstacle_contact_episode),
        obstacle_contact_steps=int(safety.obstacle_contact_steps),
        obstacle_contact_geoms=tuple(safety.obstacle_contact_geoms),
        joint_position_violation_max_rad=float(safety.joint_position_violation_max_rad),
        joint_velocity_violation_max_rad_s=float(safety.joint_velocity_violation_max_rad_s),
        first_unsafe_step=int(safety.first_unsafe_observation),
        certificate_margin=float(certificate_margin),
        margin_steps=int(certificate.margin_steps) if margined else 0,
        margin_reduced_steps=int(certificate.reduced_steps) if margined else 0,
        margin_dropped_steps=int(certificate.dropped_steps) if margined else 0,
        margin_mean_command=(
            tuple(float(v) for v in certificate.applied / certificate.margin_steps)
            if margined and certificate.margin_steps
            else None
        ),
    )

    return asdict(episode), stack.trace
