"""Exact endpoint conversion and optional normalized-model ball arithmetic."""

import sys
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from sarrl.safety.panda_center_arithmetic import ArrayEnclosure, _outward, enclose_panda_center


@pytest.mark.parametrize(
    "value",
    [
        Fraction(1, 3),
        Fraction(-1, 3),
        Fraction(1, 10**400),
        Fraction(-1, 10**400),
        Fraction(2**53 + 1),
        Fraction(0),
    ],
)
def test_binary64_export_encloses_exact_rational_including_subnormals(value):
    lower, upper = _outward(value, upper=False), _outward(value, upper=True)
    assert Fraction(lower) <= value <= Fraction(upper)
    if lower != upper:
        assert np.nextafter(lower, np.inf) == upper


def test_unrepresentable_endpoint_is_rejected():
    with pytest.raises(ValueError, match="binary64 range"):
        _outward(Fraction(10**400), upper=True)


def test_absolute_error_is_outward_and_immutable():
    lower, upper = np.array([0.1]), np.array([0.3])
    bounds = ArrayEnclosure(lower, upper)
    lower[0] = 9
    error = bounds.absolute_error([0.2])
    exact = max(abs(Fraction(0.2) - Fraction(0.1)), abs(Fraction(0.2) - Fraction(0.3)))
    assert Fraction(float(error[0])) >= exact
    assert bounds.lower[0] == 0.1
    with pytest.raises(ValueError):
        error[0] = 0


@pytest.mark.parametrize("kind", ["order", "shape", "nonfinite", "reference"])
def test_invalid_array_bounds_are_rejected(kind):
    with pytest.raises(ValueError):
        if kind == "order":
            ArrayEnclosure([1], [0])
        elif kind == "shape":
            ArrayEnclosure([0, 1], [2])
        elif kind == "nonfinite":
            ArrayEnclosure([0], [np.inf])
        else:
            ArrayEnclosure([0], [1]).absolute_error([np.nan])


@pytest.mark.parametrize("precision", [False, 64, 1025, 128.5])
def test_invalid_precision_fails_before_optional_import(precision):
    with pytest.raises(ValueError, match="precision_bits"):
        enclose_panda_center(np.zeros(7), np.zeros(7), precision_bits=precision)


@pytest.mark.parametrize("kind", ["absent", "version"])
def test_optional_backend_fails_explicitly(kind, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "flint",
        None if kind == "absent" else SimpleNamespace(__version__="unsupported"),
    )
    expected = ImportError if kind == "absent" else RuntimeError
    with pytest.raises(expected):
        enclose_panda_center(np.zeros(7), np.zeros(7))


def test_nonfinite_ball_cannot_be_exported():
    flint = pytest.importorskip("flint")
    from sarrl.safety.panda_center_arithmetic import _export

    with pytest.raises(ValueError, match="nonfinite ball"):
        _export(np.array([flint.arb.nan()], dtype=object))


@pytest.mark.parametrize("offset", [0.0, 0.2, -0.3])
def test_center_enclosures_bound_reference_discrepancies_and_restore_precision(offset, monkeypatch):
    flint = pytest.importorskip("flint")
    mujoco = pytest.importorskip("mujoco")
    from sarrl.controllers.panda_adaptive import PandaRegressor
    from sarrl.robots.panda import HOME
    from sarrl.safety.panda_kinematic_coefficients import packaged_kinematic_coefficients

    q, v = HOME + offset * np.sin(np.arange(7)), np.linspace(-0.2, 0.15, 7)
    nominal, payload, damping = packaged_kinematic_coefficients(q)
    regressor = PandaRegressor()
    m0 = regressor.nominal.mass_matrix(q)
    mp = regressor.unit_payload.mass_matrix(q) - m0
    h0, rows = regressor.evaluate(q, v, np.zeros(7))
    old_precision = flint.ctx.prec

    def prohibited(*args, **kwargs):
        raise RuntimeError("center enclosure must not evaluate dynamics or integrate")

    for name in ("mj_forward", "mj_inverse", "mj_mulM", "mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, prohibited)
    enclosure = enclose_panda_center(q, v)
    assert flint.ctx.prec == old_precision
    tighter = enclose_panda_center(q, v, precision_bits=256)
    assert flint.ctx.prec == old_precision
    for group, high, ordinary, direct_mass, direct_bias, friction in (
        (enclosure.nominal, tighter.nominal, nominal, m0, h0, damping),
        (enclosure.payload, tighter.payload, payload, mp, rows[:, 0], np.zeros(7)),
    ):
        for name, reference in (
            ("mass", ordinary.mass),
            ("mass_q", ordinary.mass_q),
            ("gravity", ordinary.gravity),
            ("bias", ordinary.bias(v, friction)),
            ("mass", direct_mass),
            ("bias", direct_bias),
        ):
            bounds, higher = getattr(group, name), getattr(high, name)
            assert np.all(bounds.lower <= higher.upper) and np.all(higher.lower <= bounds.upper)
            allowance = bounds.absolute_error(reference)
            for index in np.ndindex(reference.shape):
                actual = Fraction(float(reference[index]))
                allowed = Fraction(float(allowance[index]))
                assert abs(actual - Fraction(float(bounds.lower[index]))) <= allowed
                assert abs(actual - Fraction(float(bounds.upper[index]))) <= allowed
            assert np.max(allowance) < 1e-10
    original_q = q.copy()
    q[:] = 0
    np.testing.assert_array_equal(enclosure.q, original_q)
    with pytest.raises(ValueError):
        enclosure.velocity[0] = 1


def test_ball_identities_and_precision_restore_on_failure(monkeypatch):
    flint = pytest.importorskip("flint")
    pytest.importorskip("mujoco")
    import sarrl.safety.panda_center_arithmetic as implementation

    old_precision = flint.ctx.prec
    with flint.ctx.workprec(128):
        x = flint.arb(1) / 3
        assert (3 * x).contains(1)
        assert (x.sin() ** 2 + x.cos() ** 2).contains(1)
    assert flint.ctx.prec == old_precision

    def failure(*args):
        raise RuntimeError("injected evaluation failure")

    monkeypatch.setattr(implementation, "_evaluate", failure)
    with pytest.raises(RuntimeError, match="injected"):
        enclose_panda_center(np.zeros(7), np.zeros(7), precision_bits=256)
    assert flint.ctx.prec == old_precision
