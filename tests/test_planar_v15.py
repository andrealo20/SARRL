import numpy as np

from sarrl.evaluation import (
    V15_PHASE_A_EPISODES,
    V15_PHASE_A_EVALUATION_SEED,
    V15_PHASE_A_TRAINING_SEEDS,
    analyze_phase_a,
    spearman_or_zero,
    summarize_episode,
    v15_phase_a_protocol_dict,
)


def _episode(policy: str, training_seed: int, episode_seed: int, reverse: bool = False):
    uncertainty = np.arange(12, dtype=np.float64)
    error = uncertainty[::-1] if reverse else uncertainty
    return summarize_episode(
        policy=policy,
        training_seed=training_seed,
        ensemble_seed=training_seed,
        episode_seed=episode_seed,
        uncertainty=uncertainty,
        error=error,
        attempted_pairs=12,
        terminated=False,
        truncated=True,
    )


def test_v15_phase_a_protocol_freezes_seed_pairing_and_budget():
    protocol = v15_phase_a_protocol_dict()
    assert V15_PHASE_A_EVALUATION_SEED == 60_000
    assert V15_PHASE_A_EPISODES == 100
    assert V15_PHASE_A_TRAINING_SEEDS == (0, 1, 2, 3, 4)
    assert protocol["pairing"] == "policy_training_seed_i_with_ensemble_seed_i"
    assert protocol["randomization"]["action_delay_max"] == 2
    assert protocol["evaluation"]["cells"] == 10
    assert protocol["evaluation"]["total_episodes"] == 1_000


def test_spearman_constant_variable_is_retained_as_zero():
    rho, zero_variance = spearman_or_zero(np.ones(12), np.arange(12))
    assert rho == 0.0
    assert zero_variance


def test_episode_requires_ten_finite_pairs():
    row = summarize_episode(
        policy="A2",
        training_seed=0,
        ensemble_seed=0,
        episode_seed=60_000,
        uncertainty=np.arange(9),
        error=np.arange(9),
        attempted_pairs=11,
        terminated=False,
        truncated=True,
    )
    assert not row.qualifies
    assert row.excluded_nonfinite_pairs == 2
    assert row.spearman_rho is None


def test_phase_a_analysis_uses_all_ten_fixed_cells_and_paired_common_seeds():
    episodes = [
        _episode(policy, training_seed, episode_seed)
        for policy in ("A2", "A3")
        for training_seed in V15_PHASE_A_TRAINING_SEEDS
        for episode_seed in range(60_000, 60_100)
    ]
    result = analyze_phase_a(episodes, bootstrap_samples=50)
    assert result["common_episode_count"] == 100
    assert len(result["cells"]) == 10
    assert result["target_median_rho"] == 1.0
    assert result["ci95_low"] == 1.0
    assert result["ci95_high"] == 1.0
    assert result["decision"] == "proceed_phase_b"


def test_phase_a_fails_closed_when_common_set_is_below_ninety():
    episodes = [
        _episode(policy, training_seed, episode_seed, reverse=True)
        for policy in ("A2", "A3")
        for training_seed in V15_PHASE_A_TRAINING_SEEDS
        for episode_seed in range(60_000, 60_089)
    ]
    result = analyze_phase_a(episodes, bootstrap_samples=10)
    assert result["common_episode_count"] == 89
    assert result["decision"] == "inconclusive"
    assert result["target_median_rho"] is None
