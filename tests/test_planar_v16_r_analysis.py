import math

import numpy as np
import pytest

from tools.run_planar_v16_r_analysis import (
    ANALYSIS_RNG_SEED,
    BOOTSTRAP_DRAWS,
    ONSET_FIRST_STEP,
    ONSET_LAST_STEP,
    ONSET_ROWS,
    PRIMARY_SCENARIOS,
    THRESHOLD,
    analyse,
    as_matrices,
    auc,
    decide,
)


def _rows(n_seeds=100, artifacts=5, event_rate=0.2, separation=0.0, seed=0):
    """Synthetic landmark rows with a controllable predictor/endpoint signal."""
    rng = np.random.default_rng(seed)
    out = []
    for scenario in PRIMARY_SCENARIOS:
        for a in range(artifacts):
            for s in range(n_seeds):
                y = int(rng.random() < event_rate)
                x = rng.normal() + separation * y
                out.append(
                    {
                        "population": "safety",
                        "condition": "A6c_gate_off_control",
                        "training_seed": str(a),
                        "ensemble_seed": str(a),
                        "scenario": scenario,
                        "episode_seed": str(50000 + s),
                        "uncertainty_norm_landmark_median": repr(x),
                        "uncertainty_ratio_landmark_median": repr(x / 2),
                        "unsafe_episode": "True" if y else "False",
                        "safety_infeasible": "False",
                        "operational_failure": "True" if y else "False",
                    }
                )
    return out


def test_auc_is_half_for_a_pure_coin_flip_predictor():
    rng = np.random.default_rng(1)
    n = 20000
    y = (rng.random(n) < 0.2).astype(np.int8)
    x = rng.normal(size=n)
    assert auc(x, y) == pytest.approx(0.5, abs=0.02)


def test_auc_is_one_when_the_predictor_separates_perfectly():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    assert auc(x, y) == pytest.approx(1.0)


def test_auc_is_zero_when_the_predictor_is_perfectly_inverted():
    x = np.array([3.0, 2.0, 1.0, 0.0])
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    assert auc(x, y) == pytest.approx(0.0)


def test_auc_uses_midranks_and_handles_degenerate_labels():
    assert auc(np.array([1.0, 1.0]), np.array([0, 1], dtype=np.int8)) == pytest.approx(0.5)
    assert math.isnan(auc(np.array([1.0, 2.0]), np.array([0, 0], dtype=np.int8)))


def test_as_matrices_has_the_arm_shape_and_rejects_gaps():
    rows = _rows()
    x, y, seeds, artifacts = as_matrices(
        rows, "id_reference", "uncertainty_norm_landmark_median", "operational_failure"
    )
    assert x.shape == (5, 100)
    assert y.shape == (5, 100)
    assert len(seeds) == 100 and len(artifacts) == 5
    with pytest.raises(RuntimeError, match="incomplete matrix"):
        as_matrices(
            rows[:-1], "ood_compound", "uncertainty_norm_landmark_median", "operational_failure"
        )


def test_decision_requires_both_components_and_uses_strict_comparison():
    boundary = {
        s: {"lower_bound": THRESHOLD, "upper_bound": 0.9} for s in PRIMARY_SCENARIOS
    }
    assert decide(boundary) == "inconclusive"  # 0.60 belongs to the null
    above = {s: {"lower_bound": 0.601, "upper_bound": 0.9} for s in PRIMARY_SCENARIOS}
    assert decide(above) == "positive"
    mixed = dict(above)
    mixed["id_reference"] = {"lower_bound": 0.4, "upper_bound": 0.55}
    assert decide(mixed) == "inconclusive"
    below = {s: {"lower_bound": 0.3, "upper_bound": 0.55} for s in PRIMARY_SCENARIOS}
    assert decide(below) == "negative"


def test_null_data_does_not_produce_a_positive():
    """A predictor with no signal must not clear the threshold."""
    rows = _rows(separation=0.0, seed=7)
    rng = np.random.default_rng(ANALYSIS_RNG_SEED)
    res = analyse(
        rows, "uncertainty_norm_landmark_median", "operational_failure",
        {s: 2.5 for s in PRIMARY_SCENARIOS}, rng,
    )
    for s in PRIMARY_SCENARIOS:
        assert res[s]["auc"] == pytest.approx(0.5, abs=0.05)
    assert decide(res) != "positive"


def test_strong_signal_is_detected():
    rows = _rows(separation=2.0, seed=3)
    rng = np.random.default_rng(ANALYSIS_RNG_SEED)
    res = analyse(
        rows, "uncertainty_norm_landmark_median", "operational_failure",
        {s: 2.5 for s in PRIMARY_SCENARIOS}, rng,
    )
    for s in PRIMARY_SCENARIOS:
        assert res[s]["auc"] > 0.85
    assert decide(res) == "positive"


def test_bounds_bracket_the_point_estimate_and_report_counts():
    rows = _rows(separation=1.0, seed=5)
    rng = np.random.default_rng(ANALYSIS_RNG_SEED)
    res = analyse(
        rows, "uncertainty_norm_landmark_median", "operational_failure",
        {s: 2.0 for s in PRIMARY_SCENARIOS}, rng,
    )
    for s in PRIMARY_SCENARIOS:
        d = res[s]
        assert d["lower_bound"] <= d["auc"] <= d["upper_bound"]
        assert d["episodes"] == 500
        assert 0 < d["events"] < 500
        assert len(d["per_cell_auc"]) == 5


def test_bootstrap_is_deterministic_under_the_frozen_seed():
    rows = _rows(separation=1.0, seed=11)
    critical = {s: 2.5 for s in PRIMARY_SCENARIOS}
    a = analyse(
        rows, "uncertainty_norm_landmark_median", "operational_failure",
        critical, np.random.default_rng(ANALYSIS_RNG_SEED),
    )
    b = analyse(
        rows, "uncertainty_norm_landmark_median", "operational_failure",
        critical, np.random.default_rng(ANALYSIS_RNG_SEED),
    )
    for s in PRIMARY_SCENARIOS:
        assert a[s]["lower_bound"] == b[s]["lower_bound"]


def test_frozen_analysis_constants():
    assert THRESHOLD == 0.60
    assert BOOTSTRAP_DRAWS == 10_000
    assert ANALYSIS_RNG_SEED == 160_000
    assert (ONSET_FIRST_STEP, ONSET_LAST_STEP, ONSET_ROWS) == (21, 45, 25)
