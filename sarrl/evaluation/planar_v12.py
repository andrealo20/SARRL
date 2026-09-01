"""Frozen protocol constants for the SARRL v1.2 planar ablation study."""

from __future__ import annotations

from sarrl.envs import DomainRandomization
from sarrl.safety import SafetyConfig

V12_TRAINING_SEEDS = (0, 1, 2, 3, 4)

V12_VALIDATION_SEED = 20_000
V12_VALIDATION_EPISODES = 30

V12_HELDOUT_SEED = 40_000
V12_HELDOUT_EPISODES = 100

V12_A3_TRAINING_COMMIT = "3068a858ae46d55a43705963ede6e0d72b66492d"
V12_A4_ENSEMBLE_COMMIT = "22fde136682013990157b9a11d42b923d20afa3e"
V12_SAFETY_INTERVENTION_TOLERANCE = 1e-9

V12_CONTEXT_SAMPLES = 2_000
V12_CONTEXT_HISTORY = 16
V12_CONTEXT_TRAINING_STEPS = 1_500

# Dedicated, disjoint context-data namespace for each A3 training seed.
V12_CONTEXT_DATA_SEED_BASE = 100_000
V12_CONTEXT_DATA_SEED_STRIDE = 10_000

V12_ENSEMBLE_SAMPLES = 10_000
V12_ENSEMBLE_TRAINING_STEPS = 2_000
V12_ENSEMBLE_BATCH_SIZE = 128

# Dedicated, disjoint residual-dynamics data namespace for each A4 seed.
V12_ENSEMBLE_DATA_SEED_BASE = 200_000
V12_ENSEMBLE_DATA_SEED_STRIDE = 20_000


def planar_id_randomization() -> DomainRandomization:
    """Return the frozen randomized planar benchmark distribution."""
    return DomainRandomization(
        mass_fraction=0.15,
        friction_fraction=0.30,
        motor_gain_fraction=0.15,
        payload_range=(0.0, 1.0),
        action_delay_max=2,
    )


def planar_id_randomization_dict() -> dict:
    """JSON-friendly representation of the frozen randomization protocol."""
    return {
        "mass_fraction": 0.15,
        "friction_fraction": 0.30,
        "motor_gain_fraction": 0.15,
        "payload_range": [0.0, 1.0],
        "action_delay_max": 2,
    }


def planar_ensemble_randomization() -> DomainRandomization:
    """Return the frozen residual-dynamics identification distribution."""
    return DomainRandomization(
        mass_fraction=0.25,
        friction_fraction=0.40,
        motor_gain_fraction=0.20,
        payload_range=(0.0, 1.5),
    )


def planar_ensemble_randomization_dict() -> dict:
    """JSON-friendly representation of A4 ensemble-data randomization."""
    return {
        "mass_fraction": 0.25,
        "friction_fraction": 0.40,
        "motor_gain_fraction": 0.20,
        "payload_range": [0.0, 1.5],
        "action_delay_max": 0,
    }


def planar_safety_config() -> SafetyConfig:
    """Return the frozen model-relative HOCBF configuration for A5/A6."""
    return SafetyConfig()


def planar_safety_config_dict() -> dict:
    """JSON-friendly representation of the frozen A5/A6 safety protocol."""
    return {
        "torque_limit": [40.0, 40.0],
        "joint_lower": [-3.05, -3.05],
        "joint_upper": [3.05, 3.05],
        "velocity_limit": [7.0, 7.0],
        "joint_gamma1": 5.0,
        "joint_gamma2": 5.0,
        "velocity_dt": 0.02,
        "feasibility_tol": 2e-8,
        "obstacles": [],
        "require_safety": True,
        "infeasible_semantics": "abort_episode_unsuccessful",
        "intervention_tolerance": V12_SAFETY_INTERVENTION_TOLERANCE,
        "guarantee_scope": "nominal_model_relative",
    }


def context_data_seed(training_seed: int) -> int:
    """Return the start of the dedicated context-data seed range."""
    if training_seed < 0:
        raise ValueError("training seed must be non-negative")
    return V12_CONTEXT_DATA_SEED_BASE + V12_CONTEXT_DATA_SEED_STRIDE * training_seed


def validate_context_data_range(
    training_seed: int,
    samples: int,
) -> tuple[int, int]:
    """Validate and return the inclusive context-data episode seed range."""
    if samples <= 0:
        raise ValueError("context sample count must be positive")
    if samples > V12_CONTEXT_DATA_SEED_STRIDE:
        raise ValueError("context sample count exceeds the reserved per-seed namespace")

    start = context_data_seed(training_seed)
    end = start + samples - 1
    return start, end


def ensemble_data_seed(training_seed: int) -> int:
    """Return the start of the dedicated A4 ensemble-data seed range."""
    if training_seed < 0:
        raise ValueError("training seed must be non-negative")
    return V12_ENSEMBLE_DATA_SEED_BASE + V12_ENSEMBLE_DATA_SEED_STRIDE * training_seed


def validate_ensemble_data_range(
    training_seed: int,
    samples: int,
) -> tuple[int, int]:
    """Validate and return the inclusive A4 ensemble-data seed range."""
    if samples <= 0:
        raise ValueError("ensemble sample count must be positive")
    if samples > V12_ENSEMBLE_DATA_SEED_STRIDE:
        raise ValueError("ensemble sample count exceeds the reserved per-seed namespace")

    start = ensemble_data_seed(training_seed)
    end = start + samples - 1
    return start, end
