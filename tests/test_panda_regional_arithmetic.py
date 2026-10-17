"""Whole-box coefficient arithmetic and exact-radius regression checks."""

from fractions import Fraction

import numpy as np
import pytest

from sarrl.safety.panda_center_arithmetic import _export, enclose_panda_center
from sarrl.safety.panda_regional_arithmetic import (
    RegionalCoefficientEnclosure,
    _center,
    _radius,
    _serial_bounds,
    enclose_panda_region,
)
from sarrl.safety.panda_state_region import VectorBox


def test_radius_covers_midpoint_rounding_for_adjacent_binary64_endpoints():
    lower, upper = np.ones(7), np.full(7, np.nextafter(1.0, np.inf))
    domain = VectorBox(lower, upper)
    radius = _radius(domain, domain.center)
    for i in range(7):
        c = Fraction(float(domain.center[i]))
        assert Fraction(float(radius[i])) >= abs(Fraction(float(lower[i])) - c)
        assert Fraction(float(radius[i])) >= abs(Fraction(float(upper[i])) - c)
    assert np.all(radius > domain.radius)
    with pytest.raises(ValueError):
        radius[0] = 0


def test_exact_midpoint_stays_inside_a_subnormal_singleton():
    value = 3 * np.nextafter(0.0, np.inf)
    box = VectorBox.point(np.full(7, value))
    np.testing.assert_array_equal(_center(box), box.lower)
    assert np.any(box.center != box.lower)
    assert not np.any(_radius(box, _center(box)))


@pytest.mark.parametrize("kind", ["q", "v"])
def test_invalid_domains_fail_before_optional_backend(kind):
    good = VectorBox.point(np.zeros(7))
    with pytest.raises(ValueError, match="VectorBox"):
        enclose_panda_region(
            np.zeros(7) if kind == "q" else good, np.zeros(7) if kind == "v" else good
        )


def test_invalid_derivative_majorants_are_rejected():
    from sarrl.safety.panda_center_arithmetic import ArrayEnclosure

    with pytest.raises(ValueError, match="mass_q_upper"):
        RegionalCoefficientEnclosure(
            ArrayEnclosure(np.eye(7), np.eye(7)),
            ArrayEnclosure(np.zeros(7), np.zeros(7)),
            -np.ones((7, 7, 7)),
            np.zeros((7, 7, 7, 7)),
            np.zeros((7, 7)),
            np.zeros((7, 14)),
        )


def test_ball_product_rule_majorants_match_exact_two_link_calculation():
    flint = pytest.importorskip("flint")
    with flint.ctx.workprec(128):
        lengths = np.full((2, 7), flint.arb(0), dtype=object)
        lengths[0, 0], lengths[1, 0], lengths[1, 1] = flint.arb(1), flint.arb(2), flint.arb(1)
        d1, d2, g1 = _serial_bounds(
            lengths, [flint.arb(1)] * 2, [flint.arb(0)] * 2, flint.arb(10), [0, 1], flint.arb
        )
        # Independent exact product-rule sums; true M11=3+2*cos(q2).
        assert d1[0, 0, 1] == 4
        assert d2[0, 0, 1, 1] == 6
        assert g1[0, 0] == 30 and g1[0, 1] == 10
        assert _export(d2).upper[0, 0, 1, 1] >= 2


@pytest.fixture(scope="module")
def region():
    pytest.importorskip("flint")
    pytest.importorskip("mujoco")
    from sarrl.robots.panda import HOME

    return enclose_panda_region(
        VectorBox(HOME - 0.001, HOME + 0.001), VectorBox(-np.ones(7) * 0.01, np.ones(7) * 0.01)
    )


def test_regional_coefficients_cover_point_enclosures_at_eight_corner_patterns(region):
    import itertools

    for pattern in itertools.islice(itertools.product((-1, 1), repeat=7), 0, 128, 17):
        mask = np.array(pattern) > 0
        q = np.where(mask, region.q_domain.upper, region.q_domain.lower)
        v = np.where(mask, region.v_domain.upper, region.v_domain.lower)
        point = enclose_panda_center(q, v, precision_bits=256)
        for global_group, point_group in (
            (region.nominal, point.nominal),
            (region.payload, point.payload),
        ):
            for name in ("mass", "bias"):
                outer, inner = getattr(global_group, name), getattr(point_group, name)
                assert np.all(outer.lower <= inner.lower) and np.all(outer.upper >= inner.upper)
            magnitude = np.maximum(
                np.abs(point_group.mass_q.lower), np.abs(point_group.mass_q.upper)
            )
            assert np.all(magnitude <= global_group.mass_q_upper)


def test_reference_error_bounds_include_variation_and_point_error(region):
    from sarrl.controllers.panda_adaptive import PandaRegressor

    regressor = PandaRegressor()
    m0 = regressor.nominal.mass_matrix(region.center.q)
    mp = regressor.unit_payload.mass_matrix(region.center.q) - m0
    h0, rows = regressor.evaluate(region.center.q, region.center.velocity, np.zeros(7))
    for group, mass, bias in ((region.nominal, m0, h0), (region.payload, mp, rows[:, 0])):
        for enclosure, reference in ((group.mass, mass), (group.bias, bias)):
            error = enclosure.absolute_error(reference)
            for index in np.ndindex(reference.shape):
                value, allowance = Fraction(float(reference[index])), Fraction(float(error[index]))
                assert abs(Fraction(float(enclosure.lower[index])) - value) <= allowance
                assert abs(Fraction(float(enclosure.upper[index])) - value) <= allowance
    with pytest.raises(ValueError):
        region.nominal.mass_qq_upper[0, 0, 0, 0] = 0


def test_zero_radius_reduces_to_point_enclosure_without_dynamics_calls(monkeypatch):
    pytest.importorskip("flint")
    mujoco = pytest.importorskip("mujoco")
    from sarrl.robots.panda import HOME

    def prohibited(*args, **kwargs):
        raise RuntimeError("regional coefficient evaluation must not call dynamics or integration")

    for name in ("mj_forward", "mj_inverse", "mj_mulM", "mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, prohibited)
    result = enclose_panda_region(VectorBox.point(HOME), VectorBox.point(np.zeros(7)))
    assert not np.any(result.q_radius_upper) and not np.any(result.v_radius_upper)
    for outer, point in (
        (result.nominal, result.center.nominal),
        (result.payload, result.center.payload),
    ):
        for name in ("mass", "bias"):
            np.testing.assert_array_equal(getattr(outer, name).lower, getattr(point, name).lower)
            np.testing.assert_array_equal(getattr(outer, name).upper, getattr(point, name).upper)


def test_region_restores_precision_and_matches_model_provenance(region):
    import flint

    old_precision = flint.ctx.prec
    larger = enclose_panda_region(region.q_domain, region.v_domain, precision_bits=256)
    assert flint.ctx.prec == old_precision
    assert larger.center.compiled_constants_sha256 == region.center.compiled_constants_sha256
    for low, high in ((region.nominal, larger.nominal), (region.payload, larger.payload)):
        assert np.all(low.mass.lower <= high.mass.upper) and np.all(
            high.mass.lower <= low.mass.upper
        )


@pytest.mark.parametrize("kind", ["constants", "source", "evaluation"])
def test_model_mismatch_and_evaluation_failure_do_not_leave_precision_changed(kind, monkeypatch):
    flint = pytest.importorskip("flint")
    pytest.importorskip("mujoco")
    import sarrl.safety.panda_regional_arithmetic as implementation
    from sarrl.robots.panda import HOME
    from sarrl.safety.panda_analytic_region import _checked_packaged_model

    old_precision = flint.ctx.prec
    if kind == "constants":
        model = _checked_packaged_model()
        model.body_mass[model.jnt_bodyid[2]] += 0.01
        monkeypatch.setattr(implementation, "_checked_packaged_model", lambda: model)
    elif kind == "source":
        monkeypatch.setattr(
            implementation, "model_provenance", lambda: {"files": {"panda.xml": "other"}}
        )
    else:

        def failure(*args):
            raise ValueError("injected regional evaluation failure")

        monkeypatch.setattr(implementation, "_global_bounds", failure)
    with pytest.raises(ValueError):
        enclose_panda_region(
            VectorBox.point(HOME), VectorBox.point(np.zeros(7)), precision_bits=256
        )
    assert flint.ctx.prec == old_precision
