"""Frozen protocol for SARRL v1.4 quantified planar safety."""

from __future__ import annotations

from .planar_v12 import planar_safety_config_dict
from .planar_v13 import V13_EPISODES, V13_EVALUATION_SEED, V13_TRAINING_SEEDS, v13_scenarios

V14_EVALUATION_SEED = V13_EVALUATION_SEED
V14_EPISODES = V13_EPISODES
V14_TRAINING_SEEDS = V13_TRAINING_SEEDS
V14_CONDITIONS = ("A2_unfiltered", "A5_hocbf", "A6_prefilter", "A6_hocbf")
V14_PAIRINGS = (
    ("A5_hocbf", "A2_unfiltered", "A2_to_A5_hocbf"),
    ("A6_hocbf", "A6_prefilter", "A6_hocbf_effect"),
)


def v14_protocol_dict() -> dict:
    """Return the JSON-friendly frozen v1.4 safety protocol."""
    return {
        "release_target": "v1.4.0",
        "campaign": "planar_quantified_safety",
        "training": "reuse_v1.2_artifacts_without_retraining",
        "evaluation_population": "reuse_v1.3_seeds_for_direct_paired_safety_audit",
        "training_seeds": list(V14_TRAINING_SEEDS),
        "evaluation": {
            "seed_start": V14_EVALUATION_SEED,
            "episodes_per_model_scenario": V14_EPISODES,
            "paired_across_filter_variants": True,
            "state_observation_semantics": "initial_state_plus_every_executed_transition",
        },
        "conditions": {
            "A2_unfiltered": "retained A2 residual policy without HOCBF",
            "A5_hocbf": "same A2 policy with required hard HOCBF",
            "A6_prefilter": "A3 context policy plus frozen uncertainty gate without HOCBF",
            "A6_hocbf": "same A6 pre-filter stack with required hard HOCBF",
        },
        "pairings": [
            {"filtered": filtered, "reference": reference, "label": label}
            for filtered, reference, label in V14_PAIRINGS
        ],
        "scenarios": [scenario.to_dict() for scenario in v13_scenarios()],
        "safety_envelope": planar_safety_config_dict(),
        "metrics": {
            "state": [
                "unsafe_episode_rate",
                "unsafe_state_fraction",
                "unsafe_entry_count",
                "joint_position_violation_max_rad",
                "joint_velocity_violation_max_rad_s",
                "normalized_violation_mean",
                "normalized_violation_max",
                "normalized_violation_integral",
            ],
            "command": [
                "candidate_constraint_violation_fraction",
                "executed_constraint_margin_min",
                "safety_intervention_fraction",
                "safety_correction_mean",
                "safety_infeasible_rate",
            ],
            "task": ["success_rate"],
        },
        "statistics": {
            "multi_seed_spread": "sample_sd_ddof_1",
            "episode_rate_interval": "wilson_95",
            "paired_filter_difference": "paired_bootstrap_95",
            "bootstrap_samples": 10_000,
        },
        "guarantee_scope": {
            "certificate": "nominal_instantaneous_command_model_only",
            "excluded": [
                "actuator_delay",
                "randomized_mass_friction_motor_gain_and_payload",
                "injected_faults",
                "discretization_and_hardware",
            ],
        },
    }
