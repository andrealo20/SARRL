"""Independent center dynamics and analytic regional derivative checks."""

import itertools

import numpy as np
import pytest

from sarrl.safety.panda_analytic_region import SerialCoefficientBounds
from sarrl.safety.panda_kinematic_coefficients import KinematicCoefficients, _local_transform
from sarrl.safety.panda_regional_bounds import local_mass_bounds, regional_panda_region
from sarrl.safety.panda_state_region import EnclosureUnavailable, VectorBox
from sarrl.safety.panda_structured_backup import PandaParameterBox


def test_center_plus_hessian_bounds_sine_derivative_on_entire_interval():
    # M11=2+sin(q1), centered at pi/2: dM11/dq1=0 and |d2M11/dq1²|<=1.
    d1, d2 = np.zeros((7, 7, 7)), np.zeros((7, 7, 7, 7))
    d1[0, 0, 0], d2[0, 0, 0, 0] = 1, 1
    global_bounds = SerialCoefficientBounds(d1, d2, np.zeros((7, 7)))
    previous = 0
    for radius in (0, 0.01, 0.1, 0.5, 2):
        local = local_mass_bounds(global_bounds, np.zeros((7, 7, 7)), np.ones(7) * radius)
        exact = np.sin(radius) if radius < np.pi / 2 else 1
        assert local.mass_q[0, 0, 0] >= exact
        assert np.all(local.mass_q <= global_bounds.mass_q)
        assert local.mass_q[0, 0, 0] >= previous
        previous = local.mass_q[0, 0, 0]


def test_transform_derivatives_have_expected_pivot_and_repeated_angle_terms():
    transform, first, second = _local_transform([1, 2, 3], [1, 0, 0, 0], 2, [0, 0, 1], 0)
    np.testing.assert_array_equal(transform[:3, 3], [1, 2, 3])
    np.testing.assert_array_equal(first[2, :3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(second[2, 2, :3, :3], np.diag([-1, -1, 0]))
    assert not np.any(first[:, :, 3])
    assert not np.any(second[0])


@pytest.mark.parametrize("kind", ["shape", "nonfinite", "negative_radius"])
def test_invalid_center_derivative_inputs_are_rejected(kind):
    bounds = SerialCoefficientBounds(np.ones((7, 7, 7)), np.ones((7, 7, 7, 7)), np.ones((7, 7)))
    center, radius = np.zeros((7, 7, 7)), np.ones(7)
    if kind == "shape":
        center = center[:, :, :6]
    elif kind == "nonfinite":
        center[0, 0, 0] = np.nan
    else:
        radius[0] = -1
    with pytest.raises(ValueError):
        local_mass_bounds(bounds, center, radius)


def test_kinematic_coefficients_are_immutable_and_validate_shapes():
    mass = np.eye(7)
    coefficients = KinematicCoefficients(mass, np.zeros((7, 7, 7)), np.zeros(7), np.zeros((7, 7)))
    mass[0, 0] = 4
    assert coefficients.mass[0, 0] == 1
    with pytest.raises(ValueError):
        coefficients.mass_q[0, 0, 0] = 1
    with pytest.raises(ValueError):
        KinematicCoefficients(np.ones((6, 6)), np.zeros((7, 7, 7)), np.zeros(7), np.zeros((7, 7)))


@pytest.mark.parametrize("offset", [0, 0.2, -0.3])
def test_analytic_coefficients_and_derivatives_match_independent_dynamics(offset, monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    from sarrl.controllers.panda_adaptive import PandaRegressor
    from sarrl.robots.panda import HOME
    from sarrl.safety.panda_kinematic_coefficients import packaged_kinematic_coefficients

    q, v = HOME + offset * np.sin(np.arange(7)), np.linspace(-0.2, 0.15, 7)
    regressor = PandaRegressor()

    def direct(position):
        m0 = regressor.nominal.mass_matrix(position)
        mp = regressor.unit_payload.mass_matrix(position) - m0
        g0 = regressor.nominal.inverse_dynamics(position, np.zeros(7), np.zeros(7))
        gp = regressor.unit_payload.inverse_dynamics(position, np.zeros(7), np.zeros(7)) - g0
        return ((m0, g0), (mp, gp))

    expected = direct(q)
    h0 = regressor.nominal.inverse_dynamics(q, v, np.zeros(7))
    hp = regressor.unit_payload.inverse_dynamics(q, v, np.zeros(7)) - h0
    finite_differences = []
    for direction in np.eye(7):
        plus, minus = direct(q + 2e-5 * direction), direct(q - 2e-5 * direction)
        finite_differences.append(
            [
                ((plus[j][0] - minus[j][0]) / 4e-5, (plus[j][1] - minus[j][1]) / 4e-5)
                for j in range(2)
            ]
        )

    def prohibited(*args, **kwargs):
        raise RuntimeError(
            "dynamics and physical integration are prohibited in analytic evaluation"
        )

    # Loading model constants is allowed; evaluating derivatives must use the chain algebra.
    for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse", "mj_mulM"):
        monkeypatch.setattr(mujoco, name, prohibited)
    nominal, payload, damping = packaged_kinematic_coefficients(q)
    for j, (coefficients, bias, friction) in enumerate(
        ((nominal, h0, damping), (payload, hp, np.zeros(7)))
    ):
        np.testing.assert_allclose(coefficients.mass, expected[j][0], atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(coefficients.gravity, expected[j][1], atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(coefficients.bias(v, friction), bias, atol=2e-11, rtol=2e-11)
        for k in range(7):
            np.testing.assert_allclose(
                coefficients.mass_q[:, :, k], finite_differences[k][j][0], atol=2e-8, rtol=2e-8
            )
            np.testing.assert_allclose(
                coefficients.gravity_q[:, k], finite_differences[k][j][1], atol=2e-8, rtol=2e-8
            )


def test_regional_bounds_cover_corner_derivatives_and_direct_acceleration():
    pytest.importorskip("mujoco")
    from sarrl.controllers.panda_adaptive import PandaRegressor
    from sarrl.robots.panda import HOME
    from sarrl.safety.panda_analytic_region import analytic_panda_region
    from sarrl.safety.panda_kinematic_coefficients import packaged_kinematic_coefficients

    radius = 0.001
    qbox = VectorBox(HOME - radius, HOME + radius)
    vbox = VectorBox(-np.ones(7) * 0.01, np.ones(7) * 0.01)
    kwargs = dict(
        mass_remainder=np.zeros((7, 7)),
        bias_remainder=np.zeros(7),
        assumption="Ideal unconstrained coefficient diagnostic only.",
    )
    region, global_region = (
        regional_panda_region(qbox, vbox, **kwargs),
        analytic_panda_region(qbox, vbox, **kwargs),
    )
    assert np.all(region.mass0_q_derivative <= global_region.mass0_q_derivative)
    assert np.all(region.payload_mass_q_derivative <= global_region.payload_mass_q_derivative)
    assert np.max(region.mass0_q_derivative) < 0.5 * np.max(global_region.mass0_q_derivative)
    parameters = PandaParameterBox(np.array([0.5, *[0.5] * 7]), np.array([2.5, *[3.0] * 7]))
    torque = VectorBox.point(region.bias0 + 1.5 * region.payload_bias)
    acceleration = region.acceleration(parameters, qbox, vbox, torque)
    regressor = PandaRegressor()
    variation = region._variation(qbox, vbox)
    for pattern in itertools.islice(itertools.product((-1, 1), repeat=7), 0, 128, 17):
        q, v = HOME + radius * np.array(pattern), 0.01 * np.array(pattern)
        c0, cp, _ = packaged_kinematic_coefficients(q)
        assert np.all(np.abs(c0.mass_q) <= region.mass0_q_derivative + 1e-10)
        assert np.all(np.abs(cp.mass_q) <= region.payload_mass_q_derivative + 1e-10)
        m0 = regressor.nominal.mass_matrix(q)
        mp = regressor.unit_payload.mass_matrix(q) - m0
        h0, rows = regressor.evaluate(q, v, np.zeros(7))
        for actual, center, allowance in zip(
            (m0, mp, h0, rows[:, 0]),
            (region.mass0, region.payload_mass, region.bias0, region.payload_bias),
            variation[:4],
            strict=True,
        ):
            assert np.all(np.abs(actual - center) <= allowance + 1e-10)
        for mass in (0.5, 1.5, 2.5):
            for friction in (0.5, 3):
                actual = np.linalg.solve(
                    m0 + mass * mp, torque.center - h0 - mass * rows[:, 0] - friction * v
                )
                assert acceleration.contains(VectorBox.point(actual))


def test_broad_regional_query_is_inconclusive_and_outside_query_is_rejected():
    pytest.importorskip("mujoco")
    from sarrl.robots.panda import HOME

    region = regional_panda_region(
        VectorBox(HOME - 0.01, HOME + 0.01),
        VectorBox(-np.ones(7) * 0.01, np.ones(7) * 0.01),
        mass_remainder=np.zeros((7, 7)),
        bias_remainder=np.zeros(7),
        assumption="Ideal unconstrained coefficient diagnostic only.",
    )
    parameters = PandaParameterBox(np.array([0.5, *[0.5] * 7]), np.array([2.5, *[3.0] * 7]))
    torque = VectorBox.point(region.bias0 + 1.5 * region.payload_bias)
    with pytest.raises(EnclosureUnavailable, match="mass_variation_unverified"):
        region.acceleration(parameters, region.q_domain, region.v_domain, torque)
    with pytest.raises(EnclosureUnavailable, match="state_region_not_covered"):
        region.acceleration(parameters, VectorBox.point(HOME + 0.02), region.v_domain, torque)


def test_detached_base_chain_is_rejected(monkeypatch):
    pytest.importorskip("mujoco")
    import sarrl.safety.panda_analytic_region as implementation
    from sarrl.robots.panda import load_panda_model
    from sarrl.safety.panda_kinematic_coefficients import packaged_kinematic_coefficients

    model = load_panda_model()
    model.body_parentid[model.jnt_bodyid[0]] = 0
    monkeypatch.setattr(implementation, "load_panda_model", lambda: model)
    with pytest.raises(ValueError, match="serial hinge-chain"):
        packaged_kinematic_coefficients(np.zeros(7))
