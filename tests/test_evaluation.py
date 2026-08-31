import numpy as np
import pytest

from sarrl.envs import PlanarReachEnv
from sarrl.evaluation import (
    EpisodeResult,
    aggregate,
    evaluate_policy,
    evaluate_policy_episodes,
    paired_success_difference,
    seed_ranges_overlap,
    wilson_interval,
)


def _rows(successes):
    return [
        EpisodeResult("s", "c", i, float(i), 10 + i, ok, 0.1, 1.0, 2.0, False)
        for i, ok in enumerate(successes)
    ]


def test_wilson_interval_contains_observed_rate_and_known_42_of_50_case():
    lo, hi = wilson_interval(42, 50)
    assert lo < 0.84 < hi
    np.testing.assert_allclose([lo, hi], [0.7149, 0.9166], atol=5e-4)


def test_aggregate_computes_success_and_success_only_steps():
    rows = _rows([True, False, True, True])
    m = aggregate(rows)
    assert m.n == 4 and m.successes == 3 and m.success_rate == 0.75
    assert m.success_steps_mean == pytest.approx(np.mean([10, 12, 13]))


def test_paired_success_bootstrap_is_reproducible():
    a = _rows([True, True, False, True, False, True])
    b = _rows([False, True, False, False, False, True])
    one = paired_success_difference(a, b, bootstrap=2000, seed=9)
    two = paired_success_difference(a, b, bootstrap=2000, seed=9)
    assert one == two
    assert one[0] > 0.0


class _ZeroResidualPolicy:
    def act(self, obs, deterministic=False):
        return np.zeros(2, dtype=np.float32)


def test_policy_evaluation_is_reproducible_on_fixed_seeds():
    a = evaluate_policy(_ZeroResidualPolicy(), PlanarReachEnv(mode="residual"), 6, 700)
    b = evaluate_policy(_ZeroResidualPolicy(), PlanarReachEnv(mode="residual"), 6, 700)
    assert a == b
    assert a.episodes == 6 and a.successes == 6
    assert a.selection_key == (a.success_rate, a.reward_mean)


def test_policy_episode_records_are_auditable():
    rows = evaluate_policy_episodes(
        _ZeroResidualPolicy(), PlanarReachEnv(mode="residual"), 3, 900,
        scenario="nominal", controller="zero_residual",
    )
    assert [row.seed for row in rows] == [900, 901, 902]
    assert all(row.scenario == "nominal" and row.controller == "zero_residual" for row in rows)
    assert all(row.max_speed >= 0.0 and row.max_command_torque >= 0.0 for row in rows)


def test_seed_ranges_must_be_disjoint_for_validation_and_heldout():
    assert seed_ranges_overlap(100, 20, 119, 5)
    assert not seed_ranges_overlap(100, 20, 120, 5)
    with pytest.raises(ValueError):
        seed_ranges_overlap(-1, 2, 10, 2)
