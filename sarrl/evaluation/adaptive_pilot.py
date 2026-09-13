"""Exploratory pilot: fixed against adaptive nominal control, with and without HOCBF.

No learned residual is involved. Four arms run on shared initial conditions:

- ``fixed``: the frozen computed-torque nominal, unfiltered;
- ``fixed_hocbf``: the same nominal behind the HOCBF filter on the nominal model;
- ``adaptive``: computed torque on the online command-space estimate, unfiltered;
- ``adaptive_hocbf``: the adaptive nominal behind the HOCBF filter evaluated on
  the live estimated model.

The canonical evaluator and safety envelope are reused unchanged, so success,
abort and unsafe semantics are those of every retained campaign.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from sarrl.controllers import (
    AdaptiveNominalConfig,
    AdaptiveNominalController,
    CommandRegressor,
    ComputedTorqueController,
)
from sarrl.dynamics import PlanarArm
from sarrl.envs import PlanarReachEnv
from sarrl.envs.planar_reach import ObstacleSpec
from sarrl.evaluation.planar_v12 import planar_safety_config
from sarrl.evaluation.planar_v13 import v13_scenarios
from sarrl.evaluation.safety_audit import evaluate_safety_episodes
from sarrl.runtime import ControlStackConfig, ControlStackResult
from sarrl.safety import HOCBFSafetyFilter

# The two crossed arms separate the controller's model from the certificate's:
# `adaptive_hocbf_fixedmodel` drives the identified nominal but certifies with
# the fixed nominal model; `fixed_hocbf_adaptivemodel` drives the fixed nominal
# while an estimator that only observes supplies the certificate's model.
ARMS = (
    "fixed",
    "fixed_hocbf",
    "adaptive",
    "adaptive_hocbf",
    "adaptive_hocbf_fixedmodel",
    "fixed_hocbf_adaptivemodel",
)
# Arms that add a trained bounded residual to the filtered nominal command.
RESIDUAL_ARMS = ("fixed_hocbf_residual", "adaptive_hocbf_residual")
HISTORICAL_CASES = {
    "id_reference": (9801800, 9801801),
    "ood_compound": (9801900, 9801901),
    "motor_fault": (9802000, 9802001),
}
FRESH_SEED_BASE = {"id_reference": 9803000, "ood_compound": 9803100, "motor_fault": 9803200}
# One preregistered probe bank shared by every episode, so prediction errors are comparable.
PROBE_SEED = 190_001
PROBE_COUNT = 50


def pilot_cases(fresh_per_scenario: int, historical: bool = True) -> list[tuple[str, int, str]]:
    """Historical diagnostic seeds first, then fresh seeds outside every official range."""
    if fresh_per_scenario < 0 or fresh_per_scenario > 100:
        raise ValueError("fresh_per_scenario must lie in 0..100")
    cases = []
    for scenario in ("id_reference", "ood_compound", "motor_fault"):
        for seed in HISTORICAL_CASES[scenario] if historical else ():
            cases.append((scenario, seed, "historical"))
        for offset in range(fresh_per_scenario):
            cases.append((scenario, FRESH_SEED_BASE[scenario] + offset, "fresh"))
    return cases


class NominalStack:
    """Nominal command, optional bounded residual policy, optional hard projection."""

    def __init__(
        self,
        baseline,
        safety_filter=None,
        config=None,
        compensate_delay=False,
        policy=None,
        estimator=None,
    ):
        self.baseline = baseline
        self.safety_filter = safety_filter
        self.policy = policy
        # An estimator that is not the baseline still takes its per-step guard decision.
        self.estimator = estimator if estimator is not baseline else None
        self.config = config or ControlStackConfig(require_safety=safety_filter is not None)
        self.config.validate()
        if compensate_delay and not hasattr(baseline, "predict_state"):
            raise ValueError("delay compensation needs a controller with predict_state")
        self.compensate_delay = bool(compensate_delay)

    def command(self, observation, state, q_des, obstacles=(), deterministic=True):
        state = np.asarray(state, dtype=np.float64)
        if hasattr(self.baseline, "begin_step"):
            # One guard decision per control step, before any model query.
            self.baseline.begin_step(state[:2])
        if self.estimator is not None:
            self.estimator.begin_step(state[:2])
        if self.compensate_delay:
            # Evaluate control law and certificate where the new command will act.
            state = self.baseline.predict_state(state)
        baseline = self.baseline.command(state[:2], state[2:], q_des)
        zeros = np.zeros(2, dtype=np.float64)
        limit = np.asarray(self.config.torque_limit)
        residual = zeros
        if self.policy is not None:
            action = np.asarray(
                self.policy.act(observation, deterministic=deterministic), dtype=np.float64
            )
            if action.shape != (2,) or not np.all(np.isfinite(action)):
                raise ValueError("policy must return a finite 2-vector")
            residual = np.clip(action, -1.0, 1.0) * np.asarray(self.config.residual_limit)
        candidate = baseline + residual
        common = {
            "baseline_torque": baseline,
            "raw_residual": residual,
            "gated_residual": residual,
            "uncertainty": zeros,
            "uncertainty_scale": 1.0,
            "ensemble_mean": zeros,
            "ensemble_query_torque": candidate,
        }
        if self.safety_filter is None:
            return ControlStackResult(
                torque=np.clip(candidate, -limit, limit),
                safety_correction=0.0,
                safety_certified=False,
                executable=True,
                **common,
            )
        safety = self.safety_filter.filter(state, candidate, obstacles)
        if not safety.success:
            return ControlStackResult(
                torque=np.clip(candidate, -limit, limit),
                safety_correction=safety.correction_norm,
                safety_certified=False,
                executable=not self.config.require_safety,
                **common,
            )
        return ControlStackResult(
            torque=safety.torque,
            safety_correction=safety.correction_norm,
            safety_certified=True,
            executable=True,
            **common,
        )


@dataclass(frozen=True)
class PilotEpisode:
    arm: str
    scenario: str
    seed: int
    origin: str
    plant: str
    outcome: str
    steps: int
    final_distance: float
    reward: float
    max_speed: float
    max_command_torque: float
    fault_seen: bool
    success: bool
    unsafe_episode: bool
    normalized_violation_max: float
    safety_infeasible: bool
    safety_intervention_fraction: float
    true_delay: int
    selected_lag: int | None
    converged_at_step: int | None
    prediction_error_rms: tuple[float, float] | None
    estimated_parameters: list | None
    true_parameters: list
    model_fallbacks: int = 0
    prediction_fallbacks: int = 0
    model_fallback_steps: int = 0
    control_steps: int = 0
    selected_time_constant: float | None = None
    true_time_constant: float | None = None
    true_armature: float | None = None
    residual_rms: float | None = None
    obstacle_present: bool = False
    obstacle_violation_max_m: float = 0.0
    obstacle_contact: bool = False
    obstacle_contact_steps: int = 0
    obstacle_contact_geoms: tuple[str, ...] = ()
    # Severity of envelope violations, from the audit diagnostics.
    joint_position_violation_max_rad: float = 0.0
    joint_velocity_violation_max_rad_s: float = 0.0
    first_unsafe_step: int = -1


RESIDUAL_SUFFIX = "_residual"


def build_arm(arm: str, config: AdaptiveNominalConfig):
    """Return (controller, safety filter, estimator); the estimator may be the controller."""
    if arm.endswith(RESIDUAL_SUFFIX):
        arm = arm[: -len(RESIDUAL_SUFFIX)]
    nominal = PlanarArm()
    safety_config = planar_safety_config()
    if arm == "fixed":
        return ComputedTorqueController(nominal), None, None
    if arm == "fixed_hocbf":
        return ComputedTorqueController(nominal), HOCBFSafetyFilter(nominal, safety_config), None
    estimator = AdaptiveNominalController(nominal, config)
    if arm == "fixed_hocbf_adaptivemodel":
        return (
            ComputedTorqueController(nominal),
            HOCBFSafetyFilter(estimator.estimated_model(), safety_config),
            estimator,
        )
    if arm == "adaptive":
        return estimator, None, estimator
    if arm == "adaptive_hocbf":
        return estimator, HOCBFSafetyFilter(estimator.estimated_model(), safety_config), estimator
    if arm == "adaptive_hocbf_fixedmodel":
        return estimator, HOCBFSafetyFilter(nominal, safety_config), estimator
    raise ValueError(f"unknown arm {arm}")


def build_stack(
    arm: str, config: AdaptiveNominalConfig, compensate_delay: bool = False, policy=None
):
    """Arms ending in `_residual` add a bounded policy residual to the nominal command."""
    if arm.endswith(RESIDUAL_SUFFIX) != (policy is not None):
        raise ValueError("a residual arm needs a policy and a nominal arm must not carry one")
    controller, safety_filter, estimator = build_arm(arm, config)
    adaptive = isinstance(controller, AdaptiveNominalController)
    stack = NominalStack(
        controller,
        safety_filter,
        compensate_delay=compensate_delay and adaptive,
        policy=policy,
        estimator=estimator,
    )
    return controller, stack


def probe_bank():
    probe_rng = np.random.default_rng(PROBE_SEED)
    return [
        (
            probe_rng.uniform(-2.5, 2.5, 2),
            probe_rng.uniform(-3.0, 3.0, 2),
            probe_rng.uniform(-15.0, 15.0, 2),
        )
        for _ in range(PROBE_COUNT)
    ]


def _prediction_error(controller, env, probes):
    """RMS command error of the final estimate against the analytical-parameter oracle.

    The oracle is the plant's sampled inertial and friction parameters with
    its motor gains, expressed in command coordinates. Effects outside that
    parametrisation (armature, actuator dynamics, engine friction) are not in
    the oracle, so on a MuJoCo plant this measures distance from the best
    representable model, not from the plant.
    """
    truth = CommandRegressor.parameters(env.arm.params, env.motor_gain)
    estimate = controller.parameters
    errors = np.zeros((len(probes), 2))
    for index, (q, qd, qdd) in enumerate(probes):
        rows = controller.regressor.rows(q, qd, qdd)
        errors[index] = np.einsum("ij,ij->i", rows, estimate - truth)
    return tuple(float(v) for v in np.sqrt(np.mean(errors**2, axis=0)))


@dataclass(frozen=True)
class PlantOptions:
    """What the plant adds beyond the analytical benchmark; all zero reproduces it."""

    sensor_noise_std: float = 0.0
    armature: float = 0.0
    actuator_time_constant: float = 0.0
    armature_range: tuple[float, float] | None = None
    actuator_time_constant_range: tuple[float, float] | None = None
    obstacle: ObstacleSpec | None = None


def make_env(plant: str, spec, options: PlantOptions | None = None):
    """Analytical plant by default; MuJoCo when requested and installed."""
    options = options or PlantOptions()
    randomization = replace(spec.randomization, sensor_noise_std=options.sensor_noise_std)
    if plant == "analytical":
        if (
            options.armature
            or options.actuator_time_constant
            or options.armature_range
            or options.actuator_time_constant_range
        ):
            raise ValueError("armature and actuator dynamics need the mujoco plant")
        return PlanarReachEnv(
            mode="torque", randomization=randomization, fault=spec.fault, obstacle=options.obstacle
        )
    if plant == "mujoco":
        from sarrl.envs.mujoco_planar import MujocoPlanarReachEnv

        return MujocoPlanarReachEnv(
            mode="torque",
            randomization=randomization,
            fault=spec.fault,
            armature=options.armature,
            actuator_time_constant=options.actuator_time_constant,
            armature_range=options.armature_range,
            actuator_time_constant_range=options.actuator_time_constant_range,
            obstacle=options.obstacle,
        )
    raise ValueError(f"unknown plant {plant}")


class MeasuredState:
    """One noisy measurement of the plant state per step, shared by every consumer.

    The environment draws sensor noise from its own seeded generator; caching
    by step count makes the controller's command and the estimator's update
    see the same measurement, as one sample-and-hold sensor would provide.
    """

    def __init__(self, env):
        self.env = env
        self._step = None
        self._value = None

    def __call__(self, env=None) -> np.ndarray:
        if self._step != self.env.steps:
            self._value = np.asarray(self.env._sensed_state(), dtype=np.float64)
            self._step = self.env.steps
        return self._value.copy()


def run_case(
    arm: str,
    scenario: str,
    seed: int,
    origin: str,
    config: AdaptiveNominalConfig,
    compensate_delay: bool = False,
    plant: str = "analytical",
    options: PlantOptions | None = None,
    policy=None,
):
    spec = {s.key: s for s in v13_scenarios()}[scenario]
    options = options or PlantOptions()
    env = make_env(plant, spec, options)
    controller, stack = build_stack(arm, config, compensate_delay, policy)
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
    return PilotEpisode(
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
    )


def summarize(episodes: list[PilotEpisode]) -> dict:
    """Per scenario and arm counts; historical and fresh cases are reported apart."""
    table: dict = {}
    for origin in ("historical", "fresh", "all"):
        for scenario in ("id_reference", "ood_compound", "motor_fault"):
            for arm in ARMS + RESIDUAL_ARMS:
                rows = [
                    e
                    for e in episodes
                    if e.arm == arm
                    and e.scenario == scenario
                    and (origin == "all" or e.origin == origin)
                ]
                if not rows:
                    continue
                table[f"{origin}/{scenario}/{arm}"] = {
                    "episodes": len(rows),
                    "success": sum(e.success for e in rows),
                    "timeout": sum(e.outcome == "timeout" for e in rows),
                    "abort": sum(e.outcome == "abort" for e in rows),
                    "unsafe_episodes": sum(e.unsafe_episode for e in rows),
                    "median_final_distance_m": float(np.median([e.final_distance for e in rows])),
                    "max_normalized_violation": float(
                        max(e.normalized_violation_max for e in rows)
                    ),
                    "lag_correct": (
                        sum(e.selected_lag == e.true_delay for e in rows)
                        if rows[0].selected_lag is not None
                        else None
                    ),
                    "episodes_with_model_fallback": sum(e.model_fallbacks > 0 for e in rows),
                    "episodes_with_prediction_fallback": sum(
                        e.prediction_fallbacks > 0 for e in rows
                    ),
                    "mean_residual_rms": (
                        float(np.mean([e.residual_rms for e in rows]))
                        if rows[0].residual_rms is not None
                        else None
                    ),
                    "obstacle_episodes": sum(e.obstacle_present for e in rows),
                    "obstacle_violation_episodes": sum(
                        e.obstacle_violation_max_m > 0.0 for e in rows
                    ),
                    "obstacle_contact_episodes": sum(e.obstacle_contact for e in rows),
                    "obstacle_tip_contact_episodes": sum(
                        "tip" in e.obstacle_contact_geoms for e in rows
                    ),
                    "obstacle_link_contact_episodes": sum(
                        any(g.startswith("link") for g in e.obstacle_contact_geoms) for e in rows
                    ),
                }
    return table


def episodes_to_records(episodes: list[PilotEpisode]) -> list[dict]:
    return [asdict(e) for e in episodes]
