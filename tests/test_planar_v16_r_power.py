import math

import numpy as np
import pytest
from scipy.stats import norm

from tools.run_planar_v16_r_power import (
    CALIBRATION_CONFIGS,
    EPISODE_SEEDS,
    PREVALENCE,
    PRIMARY_SCENARIOS,
    THRESHOLD,
    TRAINING_SEEDS,
    auc_from_arrays,
    bootstrap_lower_bounds,
    decide,
    mu_for_auc,
    simulate_replicate,
    solve_tau,
    variance_components,
    wilson_interval,
)


@pytest.mark.parametrize("icc", [0.0, 0.05, 0.10, 0.20])
def test_variance_components_invert_the_latent_icc(icc):
    var_seed, var_artifact = variance_components(icc)
    assert var_artifact == pytest.approx(var_seed / 2.0)
    recovered = var_seed / (var_seed + var_artifact + 1.0)
    assert recovered == pytest.approx(icc, abs=1e-12)


def test_variance_components_reject_out_of_range():
    with pytest.raises(ValueError):
        variance_components(-0.01)
    with pytest.raises(ValueError):
        variance_components(2.0 / 3.0)


@pytest.mark.parametrize("prevalence", [0.048, 0.184, 0.24])
@pytest.mark.parametrize("icc", [0.0, 0.20])
def test_solve_tau_reproduces_the_target_prevalence(prevalence, icc):
    var_seed, var_artifact = variance_components(icc)
    var_total = var_seed + var_artifact + 1.0
    tau = solve_tau(prevalence, var_total)
    achieved = 1.0 - norm.cdf(tau / math.sqrt(var_total))
    assert achieved == pytest.approx(prevalence, abs=1e-9)


def test_mu_imposes_the_target_auc_in_a_large_sample():
    rng = np.random.default_rng(11)
    for target in (0.55, 0.70, 0.80):
        mu = mu_for_auc(target)
        n = 200_000
        x0 = rng.normal(0.0, 1.0, size=n)
        x1 = rng.normal(mu, 1.0, size=n)
        x = np.concatenate([x0, x1])
        y = np.concatenate([np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)])
        assert auc_from_arrays(x, y) == pytest.approx(target, abs=0.005)


def test_auc_is_invariant_to_how_the_endpoint_was_generated():
    """The binormal construction fixes AUC regardless of endpoint clustering."""
    rng = np.random.default_rng(5)
    for icc in (0.0, 0.20):
        aucs = []
        for _ in range(40):
            rep = simulate_replicate(rng, 0.70, icc)
            y = rep["y"]["ood_compound"].ravel()
            if y.sum() in (0, y.size):
                continue
            aucs.append(auc_from_arrays(rep["x"]["ood_compound"].ravel(), y))
        assert float(np.mean(aucs)) == pytest.approx(0.70, abs=0.02)


def test_auc_handles_degenerate_labels():
    x = np.array([0.1, 0.2, 0.3])
    assert math.isnan(auc_from_arrays(x, np.zeros(3, dtype=np.int8)))
    assert math.isnan(auc_from_arrays(x, np.ones(3, dtype=np.int8)))


def test_auc_uses_midranks_for_ties():
    x = np.array([1.0, 1.0])
    y = np.array([0, 1], dtype=np.int8)
    assert auc_from_arrays(x, y) == pytest.approx(0.5)


def test_replicate_has_the_frozen_design_shape():
    rng = np.random.default_rng(3)
    rep = simulate_replicate(rng, 0.70, 0.10)
    for scenario in PRIMARY_SCENARIOS:
        assert rep["y"][scenario].shape == (TRAINING_SEEDS, EPISODE_SEEDS)
        assert rep["x"][scenario].shape == (TRAINING_SEEDS, EPISODE_SEEDS)


def test_random_effects_are_shared_across_the_two_scenarios():
    """Seed and artifact effects are common; only the idiosyncratic term differs."""
    rng = np.random.default_rng(7)
    rep = simulate_replicate(rng, 0.70, 0.20)
    a, b = (rep["latent"][s] for s in PRIMARY_SCENARIOS)
    # Removing the shared structure must leave uncorrelated residuals, while the
    # raw liabilities stay positively correlated through u_s + v_a.
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.05


def test_bootstrap_applies_one_joint_index_and_returns_both_scenarios():
    rng = np.random.default_rng(13)
    rep = simulate_replicate(rng, 0.80, 0.05)
    bounds = bootstrap_lower_bounds(rng, rep, draws=64)
    assert set(bounds) == set(PRIMARY_SCENARIOS)
    assert all(0.0 <= v <= 1.0 for v in bounds.values())


def test_decision_requires_both_components_to_exceed_the_threshold():
    assert decide({"id_reference": 0.61, "ood_compound": 0.62})
    assert not decide({"id_reference": THRESHOLD, "ood_compound": 0.62})
    assert not decide({"id_reference": 0.59, "ood_compound": 0.99})


def test_prevalences_match_the_retained_gate_off_arm():
    assert PREVALENCE["id_reference"] == pytest.approx(24 / 500)
    assert PREVALENCE["ood_compound"] == pytest.approx(92 / 500)


def test_per_scenario_auc_is_honoured():
    """Asymmetric composite-null cells need a different AUC in each scenario."""
    rng = np.random.default_rng(21)
    config = {"id_reference": 0.60, "ood_compound": 0.85}
    aucs = {s: [] for s in PRIMARY_SCENARIOS}
    for _ in range(40):
        rep = simulate_replicate(rng, config, 0.10)
        for s in PRIMARY_SCENARIOS:
            y = rep["y"][s].ravel()
            if 0 < y.sum() < y.size:
                aucs[s].append(auc_from_arrays(rep["x"][s].ravel(), y))
    assert float(np.mean(aucs["id_reference"])) == pytest.approx(0.60, abs=0.03)
    assert float(np.mean(aucs["ood_compound"])) == pytest.approx(0.85, abs=0.03)


def test_scalar_auc_still_applies_to_both_scenarios():
    rng = np.random.default_rng(22)
    rep = simulate_replicate(rng, 0.70, 0.10)
    assert set(rep["x"]) == set(PRIMARY_SCENARIOS)


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson_interval(100, 2000)
    assert lo < 0.05 < hi
    lo, hi = wilson_interval(140, 2000)
    assert lo > 0.05  # 7.0% excludes the nominal level at n=2000


def test_calibration_configs_place_exactly_one_component_on_the_boundary():
    for config in CALIBRATION_CONFIGS:
        on_boundary = [s for s, a in config.items() if a == THRESHOLD]
        assert len(on_boundary) == 1
        other = [a for s, a in config.items() if s not in on_boundary]
        assert all(a > THRESHOLD for a in other)
