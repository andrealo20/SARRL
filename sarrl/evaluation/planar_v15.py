"""Frozen Phase-A protocol for SARRL v1.5 uncertainty-gate calibration."""

from __future__ import annotations

from .planar_v12 import planar_id_randomization_dict

V15_PHASE_A_EVALUATION_SEED = 60_000
V15_PHASE_A_EPISODES = 100
V15_PHASE_A_TRAINING_SEEDS = (0, 1, 2, 3, 4)
V15_PHASE_A_POLICIES = ("A2", "A3")
V15_PHASE_A_BOOTSTRAP_SEED = 150_000
V15_PHASE_A_BOOTSTRAP_SAMPLES = 10_000
V15_PHASE_A_MIN_FINITE_PAIRS = 10
V15_PHASE_A_MIN_COMMON_EPISODES = 90
V15_PHASE_A_SCREENING_THRESHOLD = 0.2


def v15_phase_a_protocol_dict() -> dict:
    """Return the preregistered, JSON-friendly Phase-A protocol."""
    return {
        "release_target": "v1.5.0",
        "campaign": "uncertainty_gate_calibration_phase_a",
        "training": "reuse_v1.2_artifacts_without_retraining",
        "policies": list(V15_PHASE_A_POLICIES),
        "training_seeds": list(V15_PHASE_A_TRAINING_SEEDS),
        "pairing": "policy_training_seed_i_with_ensemble_seed_i",
        "randomization": planar_id_randomization_dict(),
        "evaluation": {
            "seed_start": V15_PHASE_A_EVALUATION_SEED,
            "episodes_per_cell": V15_PHASE_A_EPISODES,
            "cells": len(V15_PHASE_A_POLICIES) * len(V15_PHASE_A_TRAINING_SEEDS),
            "total_episodes": (
                len(V15_PHASE_A_POLICIES)
                * len(V15_PHASE_A_TRAINING_SEEDS)
                * V15_PHASE_A_EPISODES
            ),
        },
        "measurement": {
            "ensemble_torque_input": "commanded_torque",
            "residual_target_nominal_torque_input": "commanded_torque",
            "observed_acceleration": "exact_pre_rk4_continuous_time_derivative",
            "uncertainty_scalar": "l2_norm_population_std",
            "error_scalar": "l2_norm_ensemble_mean_minus_residual_target_float64",
        },
        "statistics": {
            "primary": "median_of_10_cell_median_within_episode_spearman_rho",
            "minimum_finite_pairs_per_episode": V15_PHASE_A_MIN_FINITE_PAIRS,
            "constant_variable_rho": 0.0,
            "minimum_global_common_episode_ids": V15_PHASE_A_MIN_COMMON_EPISODES,
            "bootstrap": "paired_episode_seed_percentile",
            "bootstrap_seed": V15_PHASE_A_BOOTSTRAP_SEED,
            "bootstrap_samples": V15_PHASE_A_BOOTSTRAP_SAMPLES,
            "interval_quantiles": [0.025, 0.975],
            "screening_threshold": V15_PHASE_A_SCREENING_THRESHOLD,
            "scope": "conditional_on_five_frozen_artifact_pairs",
        },
    }
