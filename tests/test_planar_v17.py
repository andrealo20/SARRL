import numpy as np
import pytest

from sarrl.evaluation import (
    V17_EVALUATION,
    V17_TRAINING_SEEDS,
    PairedInterval,
    classify_v17_result,
    two_level_paired_interval,
    v17_protocol_dict,
)


def _paired(value: float, episodes: int = 20):
    return {
        training_seed: {episode: value for episode in range(episodes)}
        for training_seed in V17_TRAINING_SEEDS
    }


def _interval(difference, low=None, high=None, upper=None):
    return PairedInterval(
        difference=difference,
        ci95_low=difference if low is None else low,
        ci95_high=difference if high is None else high,
        upper_one_sided_95=difference if upper is None else upper,
    )


def test_v17_protocol_has_disjoint_fresh_evaluation_ranges():
    protocol = v17_protocol_dict()
    assert protocol["training"]["seeds"] == list(V17_TRAINING_SEEDS)
    ranges = []
    for config in V17_EVALUATION.values():
        ranges.append(set(range(config["seed"], config["seed"] + config["episodes"])))
    assert all(a.isdisjoint(b) for index, a in enumerate(ranges) for b in ranges[index + 1 :])
    assert sum(item["episodes"] for item in V17_EVALUATION.values()) == 700


def test_two_level_interval_recovers_constant_paired_difference():
    control = _paired(0.25)
    treatment = _paired(0.35)
    result = two_level_paired_interval(treatment, control, bootstrap=250, seed=7)
    assert np.isclose(result.difference, 0.1)
    assert np.isclose(result.ci95_low, 0.1)
    assert np.isclose(result.ci95_high, 0.1)
    assert np.isclose(result.upper_one_sided_95, 0.1)


def test_two_level_interval_is_deterministic():
    control = {
        seed: {episode: float((seed + episode) % 3 == 0) for episode in range(25)}
        for seed in V17_TRAINING_SEEDS
    }
    treatment = {
        seed: {episode: float((seed + episode) % 4 == 0) for episode in range(25)}
        for seed in V17_TRAINING_SEEDS
    }
    first = two_level_paired_interval(treatment, control, bootstrap=500, seed=91)
    second = two_level_paired_interval(treatment, control, bootstrap=500, seed=91)
    assert first == second


def test_two_level_interval_rejects_mismatched_episode_seeds():
    control = _paired(0.0)
    treatment = _paired(1.0)
    treatment[V17_TRAINING_SEEDS[0]].pop(0)
    with pytest.raises(ValueError, match="episode seed sets"):
        two_level_paired_interval(treatment, control, bootstrap=10)


def test_v17_advance_requires_efficacy_safety_and_seed_consistency():
    success = _interval(0.05, low=0.01, high=0.09)
    unsafe = _interval(0.0, low=-0.01, high=0.01, upper=0.01)
    infeasible = _interval(0.0, low=-0.01, high=0.01, upper=0.01)
    per_seed = {seed: 0.05 for seed in V17_TRAINING_SEEDS}
    assert classify_v17_result(success, unsafe, infeasible, per_seed) == "advance"


def test_v17_positive_point_with_open_interval_is_promising():
    success = _interval(0.04, low=-0.01, high=0.09)
    safe = _interval(0.0, low=-0.02, high=0.02, upper=0.03)
    per_seed = {seed: 0.04 for seed in V17_TRAINING_SEEDS}
    assert (
        classify_v17_result(success, safe, safe, per_seed)
        == "promising_but_inconclusive"
    )


def test_v17_safety_point_violation_is_no_go():
    success = _interval(0.08, low=0.04, high=0.12)
    unsafe = _interval(0.03, low=0.01, high=0.05, upper=0.05)
    infeasible = _interval(0.0)
    per_seed = {seed: 0.08 for seed in V17_TRAINING_SEEDS}
    assert classify_v17_result(success, unsafe, infeasible, per_seed) == "no_go"
