import numpy as np
import pytest

from sarrl.evaluation import EpisodeResult, aggregate, paired_success_difference, wilson_interval


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
