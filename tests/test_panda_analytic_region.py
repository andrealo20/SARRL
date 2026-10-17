"""Analytic bounds, independent coefficient checks and conditional Panda queries."""

import itertools

import numpy as np
import pytest

from sarrl.safety.panda_analytic_region import (
    SerialCoefficientBounds,
    analytic_panda_region,
    packaged_coefficient_bounds,
    serial_coefficient_bounds,
)
from sarrl.safety.panda_state_region import EnclosureUnavailable, VectorBox
from sarrl.safety.panda_structured_backup import PandaParameterBox


def test_two_link_planar_derivative_amplitudes_are_bounded_for_every_angle():
    # Two unit rods, unit point masses at their ends, no rotational inertia.
    # M11=3+2*cos(q2), M12=1+cos(q2), M22=1.
    lengths = np.zeros((2, 7))
    lengths[0, 0], lengths[1, :2] = 1, [2, 1]
    bounds = serial_coefficient_bounds(lengths, np.ones(2), np.zeros(2), 10, [0, 1])
    amplitudes = np.array([[2, 1], [1, 0]])
    assert np.all(bounds.mass_q[:2, :2, 1] >= amplitudes)
    assert np.all(bounds.mass_qq[:2, :2, 1, 1] >= amplitudes)
    # g=[20*cos(q1)+10*cos(q1+q2), 10*cos(q1+q2)].
    assert np.all(bounds.gravity_q[:2, :2] >= [[30, 10], [10, 10]])
    # h_c=[-sin(q2)*(2*v1*v2+v2**2), sin(q2)*v1**2].
    speed = np.array([2, 3, 0, 0, 0, 0, 0])
    derivative = bounds.bias_derivative(speed, np.zeros(7))
    assert np.all(derivative[:2, 1] >= [10 + 2 * 2 * 3 + 3**2, 10 + 2**2])
    assert np.all(derivative[:2, 7:9] >= [[2 * 3, 2 * 2 + 2 * 3], [2 * 2, 0]])
    assert not np.any(bounds.mass_q[2:])
    assert not np.any(bounds.gravity_q[2:])


def test_pure_rotor_at_joint_origin_has_zero_derivatives():
    bounds = serial_coefficient_bounds(np.zeros((1, 7)), [2], [3], 9.81, [0])
    assert not np.any(bounds.mass_q)
    assert not np.any(bounds.mass_qq)
    assert not np.any(bounds.gravity_q)
    damping = np.arange(7)
    np.testing.assert_array_equal(
        bounds.bias_derivative(np.ones(7), damping)[:, 7:], np.diag(damping)
    )


def test_rotated_anisotropic_rotor_derivative_amplitudes_are_bounded():
    # Two intersecting hinge axes and a massless body with I=diag(1,2,3).
    # Its M11 can be 1*cos(q2)**2+3*sin(q2)**2, M22=2, M12=0.
    # First/second derivative amplitudes are 2 and 4 at every angle.
    bounds = serial_coefficient_bounds(np.zeros((1, 7)), [0], [3], 0, [1])
    assert bounds.mass_q[0, 0, 1] >= 2
    assert bounds.mass_qq[0, 0, 1, 1] >= 4
    assert not np.any(bounds.mass_q[:, :, 0])
    assert not np.any(bounds.mass_qq[:, :, 0, :])


@pytest.mark.parametrize("kind", ["offset", "spring", "fluid", "payload"])
def test_unsupported_packaged_model_changes_are_rejected(kind, monkeypatch):
    pytest.importorskip("mujoco")
    import sarrl.safety.panda_analytic_region as implementation
    from sarrl.robots.panda import load_panda_model

    model = load_panda_model()
    if kind == "offset":
        model.jnt_pos[0, 0] = 0.01
    elif kind == "spring":
        model.jnt_stiffness[0] = 1
    elif kind == "fluid":
        model.opt.density = 1
    else:
        model.body_mass[model.body("attachment").id] = 1
    monkeypatch.setattr(implementation, "load_panda_model", lambda: model)
    with pytest.raises(ValueError, match="serial hinge-chain"):
        packaged_coefficient_bounds()


@pytest.mark.parametrize("kind", ["length", "mass", "inertia", "gravity", "ancestor", "speed"])
def test_invalid_bounds_are_rejected(kind):
    lengths, masses, inertias, gravity, last = np.ones((1, 7)), [1], [1], 9.81, [6]
    if kind == "length":
        lengths[0, 0] = np.nan
    elif kind == "mass":
        masses = [-1]
    elif kind == "inertia":
        inertias = [np.inf]
    elif kind == "gravity":
        gravity = -1
    elif kind == "ancestor":
        last = [7]
    with pytest.raises(ValueError):
        bounds = serial_coefficient_bounds(lengths, masses, inertias, gravity, last)
        if kind == "speed":
            bounds.bias_derivative(-np.ones(7), np.zeros(7))


def test_bounds_copy_and_validate_inputs():
    d1, d2, g = np.ones((7, 7, 7)), np.ones((7, 7, 7, 7)), np.ones((7, 7))
    bounds = SerialCoefficientBounds(d1, d2, g)
    d1[:] = 5
    np.testing.assert_array_equal(bounds.mass_q, np.ones((7, 7, 7)))
    with pytest.raises(ValueError):
        bounds.mass_qq[0, 0, 0, 0] = 2
    with pytest.raises(ValueError):
        SerialCoefficientBounds(-d1, d2, g)


@pytest.fixture(scope="module")
def panda_bounds():
    pytest.importorskip("mujoco")
    return packaged_coefficient_bounds()


def test_packaged_derivatives_cover_independent_finite_differences(panda_bounds, monkeypatch):
    import mujoco

    from sarrl.controllers.panda_adaptive import PandaRegressor
    from sarrl.robots.panda import HOME

    def prohibited(*args, **kwargs):
        raise RuntimeError("physical integration is prohibited")

    for name in ("mj_step", "mj_step1", "mj_step2", "mj_Euler", "mj_implicit", "mj_RungeKutta"):
        monkeypatch.setattr(mujoco, name, prohibited)
    regressor = PandaRegressor()
    nominal, payload, damping = panda_bounds
    speed = np.full(7, 0.2)
    derivatives = [
        nominal.bias_derivative(speed, damping),
        payload.bias_derivative(speed, np.zeros(7)),
    ]

    def coefficients(q, v):
        m0 = regressor.nominal.mass_matrix(q)
        mp = regressor.unit_payload.mass_matrix(q) - m0
        h0 = regressor.nominal.inverse_dynamics(q, v, np.zeros(7))
        hp = regressor.unit_payload.inverse_dynamics(q, v, np.zeros(7)) - h0
        return (m0, mp), (h0, hp)

    for offset in (0.0, 0.15, -0.2):
        q, v = HOME + offset * np.sin(np.arange(7)), speed * np.cos(np.arange(7))
        for k in range(14):
            dx = np.eye(14)[k] * 2e-5
            plus = coefficients(q + dx[:7], v + dx[7:])
            minus = coefficients(q - dx[:7], v - dx[7:])
            for group, bound in enumerate((nominal, payload)):
                if k < 7:
                    actual = (plus[0][group] - minus[0][group]) / 4e-5
                    assert np.all(np.abs(actual) <= bound.mass_q[:, :, k] + 2e-7)
                actual = (plus[1][group] - minus[1][group]) / 4e-5
                assert np.all(np.abs(actual) <= derivatives[group][:, k] + 2e-7)
        assert regressor.nominal.data.time == 0 and regressor.unit_payload.data.time == 0


def test_packaged_region_encloses_coefficients_and_small_domain_acceleration(panda_bounds):
    from sarrl.robots.panda import HOME, PandaDynamics, PandaModelParameters

    radius = 1e-7
    region = analytic_panda_region(
        VectorBox(HOME - radius, HOME + radius),
        VectorBox(-np.ones(7) * radius, np.ones(7) * radius),
        mass_remainder=np.zeros((7, 7)),
        bias_remainder=np.zeros(7),
        assumption="Ideal unconstrained continuous-model diagnostic; no physical force claim.",
    )
    parameters = PandaParameterBox(np.array([0.5, *[0.5] * 7]), np.array([2.5, *[3.0] * 7]))
    torque = VectorBox.point(region.bias0 + region.payload_bias)
    acceleration = region.acceleration(parameters, region.q_domain, region.v_domain, torque)
    variations = region._variation(region.q_domain, region.v_domain)
    nominal = PandaDynamics()
    unit = PandaDynamics(parameters=PandaModelParameters(1))
    # Deterministic sign patterns are regression checks, not a source of derivative bounds.
    for pattern in itertools.islice(itertools.product((-1, 1), repeat=7), 0, 128, 17):
        q = HOME + radius * np.array(pattern)
        v = -radius * np.array(pattern)
        m0 = nominal.mass_matrix(q)
        mp = unit.mass_matrix(q) - m0
        h0 = nominal.inverse_dynamics(q, v, np.zeros(7))
        hp = unit.inverse_dynamics(q, v, np.zeros(7)) - h0
        for actual, center, variation in zip(
            (m0, mp, h0, hp),
            (region.mass0, region.payload_mass, region.bias0, region.payload_bias),
            variations[:4],
            strict=True,
        ):
            assert np.all(np.abs(actual - center) <= variation + 1e-10)
        for mass, friction in itertools.product((0.5, 1.5, 2.5), (0.5, 3)):
            actual = np.linalg.solve(m0 + mass * mp, torque.center - h0 - mass * hp - friction * v)
            assert acceleration.contains(VectorBox.point(actual))
    assert np.all(region.bias0_derivative >= 0)
    with pytest.raises(EnclosureUnavailable, match="state_region_not_covered"):
        region.acceleration(parameters, VectorBox.point(HOME + 1), region.v_domain, torque)


def test_force_remainders_and_assumption_are_required():
    with pytest.raises(TypeError):
        analytic_panda_region(VectorBox.point(np.zeros(7)), VectorBox.point(np.zeros(7)))
