"""Frozen protocol constants for the SARRL v1.2 planar ablation study."""

from __future__ import annotations

from sarrl.envs import DomainRandomization

V12_TRAINING_SEEDS = (0, 1, 2, 3, 4)

V12_VALIDATION_SEED = 20_000
V12_VALIDATION_EPISODES = 30

V12_HELDOUT_SEED = 40_000
V12_HELDOUT_EPISODES = 100

V12_CONTEXT_SAMPLES = 2_000
V12_CONTEXT_HISTORY = 16
V12_CONTEXT_TRAINING_STEPS = 1_500

# Dedicated, disjoint context-data namespace for each A3 training seed.
V12_CONTEXT_DATA_SEED_BASE = 100_000
V12_CONTEXT_DATA_SEED_STRIDE = 10_000


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
