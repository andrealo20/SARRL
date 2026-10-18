"""Snapshot identity, cache equivalence and retained admission-report guards."""

import gzip
import json
import sys

import numpy as np
import pytest

from sarrl.safety.panda_local_coefficients import PandaSnapshotDynamics, enclose_local_coefficients
from sarrl.safety.panda_state_region import EnclosureUnavailable, VectorBox


def snapshot(cache_size=2):
    pytest.importorskip("flint")
    pytest.importorskip("mujoco")
    from sarrl.robots.panda import HOME

    return PandaSnapshotDynamics(
        VectorBox(HOME - 0.5, HOME + 0.5),
        VectorBox(-np.ones(7), np.ones(7)),
        mass_remainder=np.zeros((7, 7)),
        bias_remainder=np.zeros(7),
        assumption="Normalized ideal model",
        cells=1,
        cache_size=cache_size,
    )


def boxes(radius=0.001):
    from sarrl.robots.panda import HOME

    return VectorBox(HOME - radius, HOME + radius), VectorBox(-np.ones(7) * 0.1, np.ones(7) * 0.1)


def test_snapshot_and_cache_are_bitwise_equal_to_fresh_coefficients():
    dynamics = snapshot()
    q, v = boxes()
    fresh, captured = enclose_local_coefficients(q, v), dynamics.coefficients(q, v)
    assert dynamics.coefficients(q, v) is captured
    for name in ("nominal", "payload"):
        for field in ("mass", "mass_q", "gravity", "bias"):
            a, b = getattr(getattr(fresh, name), field), getattr(getattr(captured, name), field)
            np.testing.assert_array_equal(a.lower, b.lower)
            np.testing.assert_array_equal(a.upper, b.upper)
            assert not b.lower.flags.writeable and not b.upper.flags.writeable
    assert fresh.compiled_constants_sha256 == captured.compiled_constants_sha256
    assert fresh.model_xml_sha256 == captured.model_xml_sha256


@pytest.mark.parametrize("cache_size", [0, 1])
def test_bounded_cache_eviction_and_disabled_cache(cache_size):
    dynamics = snapshot(cache_size)
    q, v = boxes()
    first = dynamics.coefficients(q, v)
    dynamics.coefficients(*boxes(0.002))
    assert len(dynamics._cache) == cache_size
    assert dynamics.coefficients(q, v) is not first


def test_snapshot_checks_compiled_constants_even_on_cache_hits():
    dynamics = snapshot()
    q, v = boxes()
    dynamics.coefficients(q, v)
    dynamics._snapshot.dof_damping[0] += 0.25
    with pytest.raises(ValueError, match="snapshot changed"):
        dynamics.coefficients(q, v)


def test_snapshot_rejects_uncovered_state_before_using_cache():
    dynamics = snapshot()
    q, _ = boxes()
    with pytest.raises(EnclosureUnavailable, match="state_region_not_covered"):
        dynamics.coefficients(q, VectorBox.point(np.full(7, 2)))


@pytest.mark.parametrize("cache_size", [-1, 513, True, 1.5])
def test_snapshot_rejects_invalid_cache_limits(cache_size):
    with pytest.raises(ValueError, match="cache_size"):
        snapshot(cache_size)


def test_snapshot_failure_restores_arithmetic_precision(monkeypatch):
    dynamics = snapshot()
    import flint

    import sarrl.safety.panda_local_coefficients as implementation

    previous = flint.ctx.prec

    def failure(*args, **kwargs):
        raise ValueError("injected snapshot failure")

    monkeypatch.setattr(implementation, "_evaluate", failure)
    with pytest.raises(ValueError, match="injected snapshot"):
        dynamics.coefficients(*boxes())
    assert flint.ctx.prec == previous


def test_terminal_screen_preserves_reports_and_rejects_changed_sources(tmp_path, monkeypatch):
    from tools import assess_panda_terminal_admission as tool

    path = tmp_path / "assessment.json.gz"
    path.write_bytes(gzip.compress(json.dumps({"source_sha256": {"wrong": "source"}}).encode()))
    original = path.read_bytes()
    monkeypatch.setattr(tool, "assessment", lambda: pytest.fail("unexpected numerical run"))
    monkeypatch.setattr(sys, "argv", ["screen", "--output", str(path)])
    with pytest.raises(FileExistsError):
        tool.main()
    monkeypatch.setattr(sys, "argv", ["screen", "--audit", "--output", str(path)])
    with pytest.raises(ValueError, match="fingerprints differ"):
        tool.main()
    assert path.read_bytes() == original


def test_terminal_screen_requires_the_exact_retained_baseline():
    from tools.assess_panda_terminal_admission import baseline

    retained = baseline()
    assert retained["decision"]["numerical_prefix_contained"]
    assert len(retained["prefix"]["cases"]) == 16
