"""Frozen protocol helpers for v1.7 safety-aware policy training."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .planar_v12 import planar_id_randomization_dict, planar_safety_config_dict
from .planar_v13 import v13_scenarios

V17_CONDITIONS = ("C0_posthoc_hocbf", "C1_inloop_hocbf")
V17_TRAINING_SEEDS = (20, 21, 22, 23, 24)
V17_TRAINING_STEPS = 200_000
V17_START_STEPS = 5_000
V17_VALIDATION_SEED = 630_000
V17_VALIDATION_EPISODES = 30
V17_EVALUATION = {
    "id_reference": {"seed": 620_000, "episodes": 500},
    "ood_compound": {"seed": 621_000, "episodes": 100},
    "motor_fault": {"seed": 622_000, "episodes": 100},
}
V17_BOOTSTRAP_SAMPLES = 20_000
V17_BOOTSTRAP_SEED = 170_000
V17_INFEASIBLE_REWARD = -500.0
V17_SAFETY_MARGIN = 0.02
V17_PROMISING_SUCCESS_GAIN = 0.03


@dataclass(frozen=True)
class PairedInterval:
    difference: float
    ci95_low: float
    ci95_high: float
    upper_one_sided_95: float


def v17_protocol_dict() -> dict:
    """Return the machine-readable v1.7 training and evaluation protocol."""
    return {
        "release_target": "v1.7.0",
        "campaign": "safety_aware_policy_training",
        "conditions": {
            "C0_posthoc_hocbf": {
                "training_hocbf": False,
                "validation_hocbf": True,
                "evaluation_hocbf": True,
            },
            "C1_inloop_hocbf": {
                "training_hocbf": True,
                "validation_hocbf": True,
                "evaluation_hocbf": True,
            },
        },
        "training": {
            "seeds": list(V17_TRAINING_SEEDS),
            "steps_per_seed": V17_TRAINING_STEPS,
            "start_steps": V17_START_STEPS,
            "batch_size": 256,
            "hidden": [256, 256],
            "update_every": 1,
            "replay_capacity": 200_000,
            "mode": "residual",
            "randomization": planar_id_randomization_dict(),
            "infeasible_reward": V17_INFEASIBLE_REWARD,
            "replay_action": "raw_normalized_policy_action",
            "correction_penalty": None,
        },
        "validation": {
            "seed_start": V17_VALIDATION_SEED,
            "episodes": V17_VALIDATION_EPISODES,
            "every_steps": 25_000,
            "hocbf_required_for_both_conditions": True,
            "selection": "success_rate_then_reward_then_earliest_step",
        },
        "evaluation": {
            key: dict(value) for key, value in V17_EVALUATION.items()
        },
        "scenarios": [
            json.loads(json.dumps(scenario.to_dict())) for scenario in v13_scenarios()
        ],
        "safety": planar_safety_config_dict(),
        "primary": {
            "scenario": "id_reference",
            "metric": "success_rate",
            "estimand": "C1_inloop_hocbf_minus_C0_posthoc_hocbf",
        },
        "statistics": {
            "bootstrap": "paired_two_level_training_seed_then_episode_seed",
            "bootstrap_samples": V17_BOOTSTRAP_SAMPLES,
            "bootstrap_seed": V17_BOOTSTRAP_SEED,
            "two_sided_interval": 0.95,
            "safety_upper_bound": 0.95,
        },
        "decision": {
            "safety_margin": V17_SAFETY_MARGIN,
            "promising_success_gain": V17_PROMISING_SUCCESS_GAIN,
            "minimum_nonnegative_seed_pairs": 4,
        },
    }


def two_level_paired_interval(
    treatment: dict[int, dict[int, float]],
    control: dict[int, dict[int, float]],
    *,
    bootstrap: int = V17_BOOTSTRAP_SAMPLES,
    seed: int = V17_BOOTSTRAP_SEED,
) -> PairedInterval:
    """Bootstrap paired training seeds, then paired episode seeds within them."""
    if bootstrap <= 0 or seed < 0:
        raise ValueError("bootstrap must be positive and seed non-negative")
    if treatment.keys() != control.keys() or not treatment:
        raise ValueError("conditions require identical non-empty training seed sets")

    training_seeds = sorted(treatment)
    differences = []
    expected_episode_seeds = None
    for training_seed in training_seeds:
        treatment_rows = treatment[training_seed]
        control_rows = control[training_seed]
        if treatment_rows.keys() != control_rows.keys() or not treatment_rows:
            raise ValueError("conditions require identical non-empty episode seed sets")
        episode_seeds = tuple(sorted(treatment_rows))
        if expected_episode_seeds is None:
            expected_episode_seeds = episode_seeds
        elif episode_seeds != expected_episode_seeds:
            raise ValueError("all training seeds require the same episode seed set")
        difference = np.asarray(
            [
                float(treatment_rows[item]) - float(control_rows[item])
                for item in episode_seeds
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(difference)):
            raise ValueError("paired values must be finite")
        differences.append(difference)

    matrix = np.stack(differences)
    point = float(matrix.mean())
    rng = np.random.default_rng(seed)
    distribution = np.empty(bootstrap, dtype=np.float64)
    chunk_size = 256
    n_training, n_episodes = matrix.shape

    for start in range(0, bootstrap, chunk_size):
        stop = min(start + chunk_size, bootstrap)
        size = stop - start
        outer = rng.integers(0, n_training, size=(size, n_training))
        inner = rng.integers(
            0,
            n_episodes,
            size=(size, n_training, n_episodes),
        )
        sampled = matrix[outer[:, :, None], inner]
        distribution[start:stop] = sampled.mean(axis=(1, 2))

    return PairedInterval(
        difference=point,
        ci95_low=float(np.quantile(distribution, 0.025)),
        ci95_high=float(np.quantile(distribution, 0.975)),
        upper_one_sided_95=float(np.quantile(distribution, 0.95)),
    )


def classify_v17_result(
    success: PairedInterval,
    unsafe: PairedInterval,
    infeasible: PairedInterval,
    per_training_seed_success: dict[int, float],
) -> str:
    """Apply the frozen ID decision rule."""
    if set(per_training_seed_success) != set(V17_TRAINING_SEEDS):
        raise ValueError("decision requires all five frozen training seeds")
    if not all(np.isfinite(value) for value in per_training_seed_success.values()):
        raise ValueError("per-training-seed success effects must be finite")

    nonnegative = sum(value >= 0.0 for value in per_training_seed_success.values())
    safety_point_pass = (
        unsafe.difference <= V17_SAFETY_MARGIN
        and infeasible.difference <= V17_SAFETY_MARGIN
    )
    safety_bound_pass = (
        unsafe.upper_one_sided_95 <= V17_SAFETY_MARGIN
        and infeasible.upper_one_sided_95 <= V17_SAFETY_MARGIN
    )

    if success.ci95_low > 0.0 and safety_bound_pass and nonnegative >= 4:
        return "advance"
    if success.difference <= 0.0 or not safety_point_pass:
        return "no_go"
    if (
        success.difference >= V17_PROMISING_SUCCESS_GAIN
        and nonnegative >= 4
        and safety_point_pass
    ):
        return "promising_but_inconclusive"
    return "inconclusive"
