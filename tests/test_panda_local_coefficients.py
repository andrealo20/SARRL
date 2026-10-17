"""Continuous state-box coefficients and parameter-coverage checks."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from sarrl.safety.panda_local_coefficients import enclose_local_coefficients
from sarrl.safety.panda_parameter_prefix import (
    PandaParameterPrefixCase,
    PandaParameterPrefixResult,
    audit_parameter_prefix,
    parameter_cells,
    verify_parameter_prefix,
)
from sarrl.safety.panda_state_region import VectorBox
from sarrl.safety.panda_structured_backup import PandaParameterBox, PandaStructuredContext
from sarrl.safety.panda_structured_prefix import PandaPrefixConfig, PandaPrefixErrors
from sarrl.safety.panda_validated_prefix import PandaValidatedPrefix


@pytest.mark.parametrize(
    "bounds", [(0.5, 2.5), (1.0, np.nextafter(1.0, np.inf)), (0.0, np.nextafter(0.0, np.inf))]
)
def test_cell_cover_uses_exact_endpoints_and_keeps_every_viscous_interval(bounds):
    full = PandaParameterBox(np.array([bounds[0], *[0.5] * 7]), np.array([bounds[1], *[3.0] * 7]))
    parts = parameter_cells(full, 3)
    low, high = map(Fraction, bounds)
    for index, part in enumerate(parts):
        assert Fraction(float(part.lower[0])) <= low + (high - low) * index / 3
        assert Fraction(float(part.upper[0])) >= low + (high - low) * (index + 1) / 3
        assert np.all(part.lower >= full.lower) and np.all(part.upper <= full.upper)
        np.testing.assert_array_equal(part.lower[1:], full.lower[1:])
        np.testing.assert_array_equal(part.upper[1:], full.upper[1:])
    assert all(a.upper[0] >= b.lower[0] for a, b in zip(parts[:-1], parts[1:], strict=True))


@pytest.mark.parametrize("field", ["q", "v"])
def test_bad_local_domain_is_rejected_before_loading_optional_backends(field):
    good = VectorBox.point(np.zeros(7))
    with pytest.raises(ValueError, match="VectorBox"):
        enclose_local_coefficients(None if field == "q" else good, None if field == "v" else good)


@pytest.fixture(scope="module")
def local():
    pytest.importorskip("flint")
    pytest.importorskip("mujoco")
    from sarrl.robots.panda import HOME

    return enclose_local_coefficients(
        VectorBox(HOME - 0.001, HOME + 0.001), VectorBox(-np.ones(7) * 0.1, np.ones(7) * 0.1)
    )


def test_local_coefficients_cover_point_enclosures_and_restore_precision(local):
    import flint

    from sarrl.safety.panda_center_arithmetic import enclose_panda_center

    before = flint.ctx.prec
    masks = [
        np.array([(pattern >> i) & 1 for i in range(7)], dtype=bool)
        for pattern in (0, 17, 42, 85, 110, 127)
    ]
    for mask in masks:
        q = np.where(mask, local.q_domain.lower, local.q_domain.upper)
        v = np.where(mask, local.v_domain.upper, local.v_domain.lower)
        inner = enclose_panda_center(q, v, precision_bits=256)
        for wide, narrow in ((local.nominal, inner.nominal), (local.payload, inner.payload)):
            for name in ("mass", "mass_q", "gravity", "bias"):
                a, b = getattr(wide, name), getattr(narrow, name)
                assert np.all(a.lower <= b.lower) and np.all(a.upper >= b.upper)
    assert flint.ctx.prec == before


def test_zero_width_local_path_preserves_center_results_without_dynamics_calls(monkeypatch):
    pytest.importorskip("flint")
    mujoco = pytest.importorskip("mujoco")
    from sarrl.robots.panda import HOME
    from sarrl.safety.panda_center_arithmetic import enclose_panda_center

    def prohibited(*args, **kwargs):
        raise RuntimeError("local coefficients must not evaluate dynamics or integrate")

    for name in ("mj_forward", "mj_inverse", "mj_mulM", "mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, prohibited)
    velocity = np.linspace(-0.2, 0.15, 7)
    direct = enclose_local_coefficients(VectorBox.point(HOME), VectorBox.point(velocity))
    center = enclose_panda_center(HOME, velocity)
    for a, b in ((direct.nominal, center.nominal), (direct.payload, center.payload)):
        for name in ("mass", "mass_q", "gravity", "bias"):
            np.testing.assert_array_equal(getattr(a, name).lower, getattr(b, name).lower)
            np.testing.assert_array_equal(getattr(a, name).upper, getattr(b, name).upper)
    assert direct.compiled_constants_sha256 == center.compiled_constants_sha256


@pytest.mark.parametrize("kind", ["source", "evaluation"])
def test_local_failures_restore_precision_and_reject_source_changes(monkeypatch, kind):
    flint = pytest.importorskip("flint")
    pytest.importorskip("mujoco")
    import sarrl.safety.panda_local_coefficients as implementation
    from sarrl.robots.panda import HOME

    before = flint.ctx.prec
    if kind == "source":
        calls = iter(("before", "after"))
        monkeypatch.setattr(
            implementation, "model_provenance", lambda: {"files": {"panda.xml": next(calls)}}
        )
    else:

        def failure(*args, **kwargs):
            raise ValueError("injected failure")

        monkeypatch.setattr(implementation, "_evaluate", failure)
    with pytest.raises(ValueError):
        enclose_local_coefficients(
            VectorBox.point(HOME), VectorBox.point(np.zeros(7)), precision_bits=256
        )
    assert flint.ctx.prec == before


def test_local_bias_avoids_global_absolute_derivative_inflation(local):
    from sarrl.safety.panda_regional_arithmetic import enclose_panda_region

    old = enclose_panda_region(local.q_domain, local.v_domain)
    assert np.max(local.nominal.bias.upper - local.nominal.bias.lower) < np.max(
        old.nominal.bias.upper - old.nominal.bias.lower
    )
    assert np.max(local.payload.bias.upper - local.payload.bias.lower) < np.max(
        old.payload.bias.upper - old.payload.bias.lower
    )


@pytest.fixture(scope="module")
def partitioned():
    pytest.importorskip("flint")
    from tests.test_panda_validated_prefix import Manufactured

    domain = VectorBox(-np.ones(7), np.ones(7))
    dynamics = Manufactured(
        domain,
        domain,
        mass_remainder=np.zeros((7, 7)),
        bias_remainder=np.zeros(7),
        assumption="Manufactured ideal model",
        cells=1,
    )
    parameters = PandaParameterBox(np.array([0.5, *[0.5] * 7]), np.array([2.5, *[3.0] * 7]))
    checker = PandaValidatedPrefix(
        dynamics,
        parameters,
        domain.lower,
        domain.upper,
        np.ones(7),
        np.full(7, 20),
        PandaPrefixErrors(*[np.zeros(7) for _ in range(5)]),
        PandaPrefixConfig(),
    )
    context = PandaStructuredContext(np.zeros(7), np.array([1.5, *[1.75] * 7]))
    q, v, candidate, history = np.zeros(7), np.zeros(7), np.full(7, 0.75), [np.full(7, 0.75)] * 4
    result = verify_parameter_prefix(
        checker, context, q, v, candidate, history, intervals=2, cells=3
    )
    assert result.prefix_contained and not result.terminal_verified
    return checker, context, q, v, candidate, history, result


def test_parameter_prefix_audits_all_branches_without_calling_search(partitioned, monkeypatch):
    checker, context, q, v, candidate, history, result = partitioned
    monkeypatch.setattr(
        PandaValidatedPrefix,
        "verify_prefix",
        lambda *a, **kw: pytest.fail("search called by auditor"),
    )
    report = audit_parameter_prefix(
        checker, context, q, v, candidate, history, result, intervals=2, cells=3
    )
    assert report["checked_parameter_cells"] == 3 and report["checked_intervals"] == 30
    assert not report["physical_binding_verified"]


@pytest.mark.parametrize("kind", ["missing", "mass_gap", "friction", "lag", "flag"])
def test_parameter_auditor_rejects_missing_coverage_and_altered_records(partitioned, kind):
    checker, context, q, v, candidate, history, result = partitioned
    first = result.cases[0]
    if kind == "missing":
        altered = replace(result, cases=result.cases[1:])
    else:
        if kind in ("mass_gap", "friction"):
            low = first.parameters.lower.copy()
            low[0 if kind == "mass_gap" else 1] += 0.01
            first = replace(first, parameters=PandaParameterBox(low, first.parameters.upper))
        elif kind == "lag":
            first = replace(first, prefix=replace(first.prefix, cases=first.prefix.cases[1:]))
        else:
            lag = replace(first.prefix.cases[0], prefix_contained=False)
            first = replace(
                first, prefix=replace(first.prefix, cases=(lag, *first.prefix.cases[1:]))
            )
        altered = replace(result, cases=(first, *result.cases[1:]))
    with pytest.raises(ValueError, match="prefix audit"):
        audit_parameter_prefix(
            checker, context, q, v, candidate, history, altered, intervals=2, cells=3
        )


def test_empty_or_one_failed_branch_can_never_pass(partitioned):
    assert not PandaParameterPrefixResult(()).prefix_contained
    _, _, _, _, _, _, result = partitioned
    bad = replace(result.cases[0].prefix.cases[0], intervals=(), prefix_contained=False)
    prefix = replace(result.cases[0].prefix, cases=(bad, *result.cases[0].prefix.cases[1:]))
    altered = PandaParameterPrefixResult(
        (PandaParameterPrefixCase(result.cases[0].parameters, prefix),)
    )
    assert not altered.prefix_contained and not altered.terminal_verified


def test_parameter_screen_preserves_reports_and_rejects_changed_sources(tmp_path, monkeypatch):
    import gzip
    import json
    import sys

    from tools import assess_panda_parameter_prefix as tool

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


def test_frozen_baseline_identity_is_checked():
    from tools.assess_panda_parameter_prefix import _baseline

    baseline = _baseline()
    assert not baseline["decision"]["numerical_prefix_contained"]
    assert len(baseline["prefix"]["cases"]) == 5
    assert all(not case["intervals"] for case in baseline["prefix"]["cases"])
