import itertools

import numpy as np
import pytest

from sarrl.safety.panda_structured_backup import (
    PandaParameterBox,
    PandaStructuredContext,
    point_acceleration_interval,
)


def fixture():
    direction = np.linspace(-0.2, 0.3, 7)
    mass0 = np.diag(np.linspace(0.7, 1.4, 7)) + np.outer(direction, direction)
    columns = np.sin(np.arange(21).reshape(7, 3)) * 0.15
    payload = columns @ columns.T
    bias0, bias_payload = np.linspace(-0.1, 0.2, 7), np.linspace(0.3, -0.2, 7)
    q, v = np.linspace(-0.04, 0.03, 7), np.linspace(-0.1, 0.2, 7)
    context = PandaStructuredContext(np.zeros(7), np.array([1.2, *np.linspace(0.7, 2, 7)]))
    box = PandaParameterBox(np.array([0.5, *[0.5] * 7]), np.array([2.5, *[3.0] * 7]))
    return context, box, q, v, mass0, payload, bias0, bias_payload


@pytest.mark.parametrize("mass_estimate", [0.0, 1.2, 3.5])
@pytest.mark.parametrize("input_difference", [np.zeros(7), np.linspace(-0.3, 0.3, 7)])
def test_continuous_mass_enclosure_contains_all_friction_corners_on_independent_grid(
    mass_estimate, input_difference
):
    context, box, q, v, m0, mp, h0, hp = fixture()
    estimate = context.parameters.copy()
    estimate[0] = mass_estimate
    context = PandaStructuredContext(context.anchor, estimate)
    desired, torque = context.point(q, v, m0, mp, h0, hp)
    lower, upper = point_acceleration_interval(
        m0, mp, hp, v, desired, estimate, box, input_difference=input_difference
    )
    friction = np.array(list(itertools.product((0.5, 3.0), repeat=7)))
    for mass in np.linspace(0.5, 2.5, 33):
        rhs = torque + input_difference - h0 - mass * hp - friction * v
        actual = np.linalg.solve(m0 + mass * mp, rhs.T).T
        assert np.all(actual >= lower - 1e-10)
        assert np.all(actual <= upper + 1e-10)
        residual = (
            (mass_estimate - mass) * (mp @ desired + hp)
            + (estimate[1:] - friction) * v
            + input_difference
        )
        reconstructed = desired + np.linalg.solve(m0 + mass * mp, residual.T).T
        np.testing.assert_allclose(actual, reconstructed, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("matched", [False, True])
def test_singleton_parameter_box_collapses_to_the_exact_acceleration(matched):
    context, _, q, v, m0, mp, h0, hp = fixture()
    truth = context.parameters if matched else np.array([2.3, *[2.7] * 7])
    box = PandaParameterBox(truth, truth)
    desired, torque = context.point(q, v, m0, mp, h0, hp)
    lower, upper = point_acceleration_interval(
        m0, mp, hp, v, desired, context.parameters, box, input_difference=np.zeros(7)
    )
    actual = np.linalg.solve(m0 + truth[0] * mp, torque - h0 - truth[0] * hp - truth[1:] * v)
    np.testing.assert_allclose(lower, actual, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(upper, actual, rtol=1e-10, atol=1e-10)
    if matched:
        np.testing.assert_allclose(actual, desired, rtol=1e-12, atol=1e-12)


def test_refining_mass_cells_retains_or_tightens_the_enclosure():
    context, box, q, v, m0, mp, h0, hp = fixture()
    desired, _ = context.point(q, v, m0, mp, h0, hp)
    coarse = point_acceleration_interval(
        m0, mp, hp, v, desired, context.parameters, box, input_difference=np.zeros(7), mass_cells=1
    )
    refined = point_acceleration_interval(
        m0, mp, hp, v, desired, context.parameters, box, input_difference=np.zeros(7)
    )
    assert np.all(refined[0] >= coarse[0] - 1e-10)
    assert np.all(refined[1] <= coarse[1] + 1e-10)


def test_viscous_uncertainty_has_no_effect_at_zero_velocity():
    context, box, q, _, m0, mp, h0, hp = fixture()
    v = np.zeros(7)
    desired, _ = context.point(q, v, m0, mp, h0, hp)
    fixed_friction = PandaParameterBox(np.array([0.5, *[1.0] * 7]), np.array([2.5, *[1.0] * 7]))
    a = point_acceleration_interval(
        m0, mp, hp, v, desired, context.parameters, box, input_difference=np.zeros(7)
    )
    b = point_acceleration_interval(
        m0, mp, hp, v, desired, context.parameters, fixed_friction, input_difference=np.zeros(7)
    )
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("alteration", ["delayed", "clipped"])
def test_an_altered_delivered_command_needs_the_additional_input_term(alteration):
    context, _, q, v, m0, mp, h0, hp = fixture()
    truth = context.parameters
    box = PandaParameterBox(truth, truth)
    desired, torque = context.point(q, v, m0, mp, h0, hp)
    lower, upper = point_acceleration_interval(
        m0, mp, hp, v, desired, truth, box, input_difference=np.zeros(7)
    )
    delivered = torque + 1 if alteration == "delayed" else np.clip(torque, -0.01, 0.01)
    mass = m0 + truth[0] * mp
    actual = np.linalg.solve(mass, delivered - h0 - truth[0] * hp - truth[1:] * v)
    assert np.any((actual < lower) | (actual > upper))
    np.testing.assert_allclose(actual, desired + np.linalg.solve(mass, delivered - torque))
    corrected = point_acceleration_interval(
        m0, mp, hp, v, desired, truth, box, input_difference=delivered - torque
    )
    assert np.all(actual >= corrected[0]) and np.all(actual <= corrected[1])


@pytest.mark.parametrize("cells", [0, True, 257, 1.5])
def test_invalid_partition_is_rejected(cells):
    context, box, q, v, m0, mp, h0, hp = fixture()
    desired, _ = context.point(q, v, m0, mp, h0, hp)
    with pytest.raises(ValueError):
        point_acceleration_interval(
            m0,
            mp,
            hp,
            v,
            desired,
            context.parameters,
            box,
            input_difference=np.zeros(7),
            mass_cells=cells,
        )


def test_nonpositive_mass_family_is_rejected_even_when_the_center_is_positive():
    box = PandaParameterBox(np.zeros(8), np.array([2.5, *np.ones(7)]))
    with pytest.raises(ValueError, match="mass family"):
        point_acceleration_interval(
            np.eye(7),
            -0.5 * np.eye(7),
            np.zeros(7),
            np.zeros(7),
            np.zeros(7),
            np.zeros(8),
            box,
            input_difference=np.zeros(7),
        )


@pytest.mark.parametrize("kind", ["shape", "negative", "nonfinite", "reversed", "gain", "matrix"])
def test_invalid_context_or_parameter_model_is_rejected(kind):
    context, box, q, v, m0, mp, h0, hp = fixture()
    with pytest.raises(ValueError):
        if kind == "shape":
            PandaParameterBox(np.zeros(7), np.ones(8))
        elif kind == "negative":
            PandaStructuredContext(q, -np.ones(8))
        elif kind == "nonfinite":
            PandaStructuredContext(np.full(7, np.nan), np.ones(8))
        elif kind == "reversed":
            PandaParameterBox(np.ones(8), np.zeros(8))
        elif kind == "gain":
            PandaStructuredContext(q, np.ones(8), kp=0)
        else:
            context.point(q, v, m0[:, :6], mp, h0, hp)


def test_context_and_parameter_box_copy_their_inputs():
    anchor, parameters = np.zeros(7), np.ones(8)
    context, box = (
        PandaStructuredContext(anchor, parameters),
        PandaParameterBox(parameters, parameters),
    )
    anchor[:], parameters[:] = 9, 9
    np.testing.assert_array_equal(context.anchor, np.zeros(7))
    np.testing.assert_array_equal(context.parameters, np.ones(8))
    np.testing.assert_array_equal(box.lower, np.ones(8))
    for array in (context.anchor, context.parameters, box.lower, box.upper):
        with pytest.raises(ValueError):
            array[0] = 3


@pytest.mark.parametrize("offset", [0.0, 0.08])
def test_packaged_panda_forward_dynamics_matches_the_structured_identity_without_stepping(
    offset, monkeypatch
):
    mujoco = pytest.importorskip("mujoco")
    from sarrl.controllers.panda_adaptive import PandaRegressor
    from sarrl.robots.panda import HOME, PandaDynamics, PandaModelParameters

    def prohibited(*args, **kwargs):
        raise RuntimeError("physical integration is prohibited in the structured point test")

    for name in ("mj_step", "mj_step1", "mj_step2", "mj_Euler", "mj_implicit", "mj_RungeKutta"):
        monkeypatch.setattr(mujoco, name, prohibited)
    q, v = HOME + offset * np.sin(np.arange(7)), np.linspace(-0.05, 0.08, 7)
    regressor = PandaRegressor()
    m0 = regressor.nominal.mass_matrix(q)
    mp = regressor.unit_payload.mass_matrix(q) - m0
    h0 = regressor.nominal.inverse_dynamics(q, v, np.zeros(7))
    hp = regressor.unit_payload.inverse_dynamics(q, v, np.zeros(7)) - h0
    context = PandaStructuredContext(HOME, np.array([1.2, *[1.0] * 7]))
    desired, torque = context.point(q, v, m0, mp, h0, hp)
    box = PandaParameterBox(np.array([0.5, *[0.5] * 7]), np.array([2.5, *[3.0] * 7]))
    input_difference = np.linspace(0.03, -0.02, 7)
    lower, upper = point_acceleration_interval(
        m0, mp, hp, v, desired, context.parameters, box, input_difference=input_difference
    )
    for payload, friction in itertools.product((0.5, 1.5, 2.5), (0.5, 3.0)):
        dynamics = PandaDynamics(parameters=PandaModelParameters(payload, (friction,) * 7))
        assert np.all(np.abs(torque + input_difference) <= dynamics.torque_limit)
        dynamics.update(q, v)
        dynamics.data.ctrl[:] = torque + input_difference
        mujoco.mj_forward(dynamics.model, dynamics.data)
        assert dynamics.data.time == 0 and dynamics.data.nefc == 0 and dynamics.data.ncon == 0
        np.testing.assert_array_equal(dynamics.data.qpos, q)
        np.testing.assert_array_equal(dynamics.data.qvel, v)
        actual = dynamics.data.qacc.copy()
        affine = np.linalg.solve(
            m0 + payload * mp, torque + input_difference - h0 - payload * hp - friction * v
        )
        residual = (context.parameters[0] - payload) * (mp @ desired + hp)
        residual += (context.parameters[1:] - friction) * v + input_difference
        reconstructed = desired + np.linalg.solve(m0 + payload * mp, residual)
        np.testing.assert_allclose(actual, affine, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(actual, reconstructed, rtol=1e-9, atol=1e-9)
        assert np.all(actual >= lower - 1e-9) and np.all(actual <= upper + 1e-9)
