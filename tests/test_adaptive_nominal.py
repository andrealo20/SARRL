import numpy as np
import pytest

from sarrl.controllers import (
    AdaptiveNominalConfig,
    AdaptiveNominalController,
    CommandRegressor,
    ComputedTorqueController,
)
from sarrl.dynamics import PlanarArm, PlanarArmParams
from sarrl.safety import HOCBFSafetyFilter, SafetyConfig


def _random_plant(rng):
    return PlanarArmParams(
        m1=rng.uniform(0.7, 1.3),
        m2=rng.uniform(0.7, 1.3),
        i1=rng.uniform(0.05, 0.12),
        i2=rng.uniform(0.05, 0.12),
        viscous=(rng.uniform(0.02, 0.08), rng.uniform(0.02, 0.08)),
        coulomb=(rng.uniform(0.01, 0.04), rng.uniform(0.01, 0.04)),
        payload_mass=rng.uniform(0.0, 1.5),
    )


def test_regressor_reproduces_inverse_dynamics_for_random_plants():
    rng = np.random.default_rng(3)
    regressor = CommandRegressor()
    for _ in range(50):
        params = _random_plant(rng)
        arm = PlanarArm(params)
        q, qd, qdd = rng.normal(size=(3, 2)) * np.array([[2.0], [3.0], [20.0]])
        expected = arm.inverse_dynamics(q, qd, qdd, include_friction=True)
        theta = CommandRegressor.parameters(params)
        actual = np.einsum("ij,ij->i", regressor.rows(q, qd, qdd), theta)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_command_space_parameters_absorb_the_motor_gain():
    rng = np.random.default_rng(5)
    params = _random_plant(rng)
    arm = PlanarArm(params)
    gain = np.array([0.8, 0.55])
    theta = CommandRegressor.parameters(params, gain)
    q, qd, qdd = rng.normal(size=(3, 2))
    command = np.einsum("ij,ij->i", CommandRegressor().rows(q, qd, qdd), theta)
    np.testing.assert_allclose(command * gain, arm.inverse_dynamics(q, qd, qdd), rtol=1e-12)


def test_controller_with_prior_matches_fixed_computed_torque():
    nominal = PlanarArm()
    adaptive = AdaptiveNominalController(nominal)
    fixed = ComputedTorqueController(nominal)
    rng = np.random.default_rng(9)
    for _ in range(20):
        q, qd, q_des = rng.uniform(-2.0, 2.0, size=(3, 2))
        np.testing.assert_allclose(
            adaptive.command(q, qd, q_des), fixed.command(q, qd, q_des), rtol=1e-12, atol=1e-12
        )


def _simulate(controller, params, gain, delay, steps, rng, dt=0.02, excite=True):
    """Closed loop on a plant with gain and delay, mirroring PlanarReachEnv.step_torque."""
    arm = PlanarArm(params)
    state = np.concatenate([rng.uniform(-0.25, 0.25, 2), rng.uniform(-0.05, 0.05, 2)])
    queue = [np.zeros(2) for _ in range(delay)]
    q_des = rng.uniform(-1.5, 1.5, 2)
    for step in range(steps):
        if excite and step % 40 == 0:
            q_des = rng.uniform(-1.5, 1.5, 2)
        sent = controller.command(state[:2], state[2:], q_des)
        queue.append(sent.copy())
        applied = queue.pop(0) * gain
        after = arm.step_rk4(state, applied, dt)
        controller.observe(state, after, sent)
        state = after
    return state


@pytest.mark.parametrize("delay", [0, 2, 3])
def test_estimator_identifies_lag_and_predictive_parameters(delay):
    rng = np.random.default_rng(11 + delay)
    params = _random_plant(rng)
    gain = rng.uniform(0.6, 1.2, size=2)
    controller = AdaptiveNominalController(PlanarArm())
    _simulate(controller, params, gain, delay, steps=400, rng=rng)
    assert controller.lag == delay
    truth = CommandRegressor.parameters(params, gain)
    regressor = CommandRegressor()
    # Predictive check: the estimated map from acceleration demand to command
    # must agree with the true plant across random states.
    for _ in range(20):
        q, qd, qdd = rng.normal(size=(3, 2)) * np.array([[1.5], [2.0], [10.0]])
        rows = regressor.rows(q, qd, qdd)
        predicted = np.einsum("ij,ij->i", rows, controller.parameters)
        expected = np.einsum("ij,ij->i", rows, truth)
        np.testing.assert_allclose(predicted, expected, rtol=0.05, atol=0.3)


def test_estimator_recovers_after_an_abrupt_gain_loss():
    rng = np.random.default_rng(21)
    params = _random_plant(rng)
    controller = AdaptiveNominalController(PlanarArm())
    arm = PlanarArm(params)
    gain = np.array([1.0, 1.0])
    state = np.concatenate([rng.uniform(-0.25, 0.25, 2), rng.uniform(-0.05, 0.05, 2)])
    q_des = rng.uniform(-1.0, 1.0, 2)
    errors = []
    for step in range(300):
        if step == 100:
            gain = np.array([1.0, 0.55])
        if step % 40 == 0:
            # Moderate targets keep the motion inside the regime the one-step
            # finite-difference acceleration represents well.
            q_des = rng.uniform(-1.0, 1.0, 2)
        sent = controller.command(state[:2], state[2:], q_des)
        after = arm.step_rk4(state, sent * gain, 0.02)
        report = controller.observe(state, after, sent)
        errors.append(float(np.abs(report["innovation"]).max()))
        state = after
    assert max(errors[50:100]) < 0.5, "the estimator must have settled before the fault"
    assert max(errors[100:103]) > 5.0, "the fault must be visible in the innovation"
    assert max(errors[150:]) < 1.0, "the estimator must settle again after the fault"
    truth = CommandRegressor.parameters(params, gain)
    regressor = CommandRegressor()
    for _ in range(20):
        q, qd, qdd = rng.normal(size=(3, 2)) * np.array([[1.5], [2.0], [10.0]])
        rows = regressor.rows(q, qd, qdd)
        np.testing.assert_allclose(
            rows[1] @ controller.parameters[1], rows[1] @ truth[1], rtol=0.05, atol=0.5
        )


def test_estimated_model_matches_plant_forward_dynamics_in_command_units():
    rng = np.random.default_rng(31)
    params = _random_plant(rng)
    gain = np.array([0.9, 0.7])
    controller = AdaptiveNominalController(PlanarArm())
    controller.theta[:] = CommandRegressor.parameters(params, gain)
    model = controller.estimated_model()
    arm = PlanarArm(params)
    for _ in range(10):
        q, qd, command = rng.normal(size=(3, 2)) * np.array([[1.5], [2.0], [15.0]])
        np.testing.assert_allclose(
            model.forward_dynamics(q, qd, command),
            arm.forward_dynamics(q, qd, command * gain),
            rtol=1e-9,
            atol=1e-9,
        )
        np.testing.assert_allclose(model.forward_kinematics(q), arm.forward_kinematics(q))


def test_hocbf_filter_accepts_the_estimated_model():
    controller = AdaptiveNominalController(PlanarArm())
    safety = HOCBFSafetyFilter(controller.estimated_model(), SafetyConfig())
    state = np.array([0.5, -0.3, 0.2, 0.1])
    result = safety.filter(state, np.array([5.0, -3.0]))
    assert result.success
    assert result.current_safe
    reference = HOCBFSafetyFilter(PlanarArm(), SafetyConfig()).filter(state, np.array([5.0, -3.0]))
    np.testing.assert_allclose(result.torque, reference.torque, rtol=1e-9, atol=1e-9)


def test_config_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        AdaptiveNominalConfig(forgetting=0.0).validate()
    with pytest.raises(ValueError):
        AdaptiveNominalConfig(process_noise=-1.0).validate()
    with pytest.raises(ValueError):
        AdaptiveNominalConfig(lower=(1.0,) * 7, upper=(0.5,) * 7).validate()
    with pytest.raises(ValueError):
        AdaptiveNominalConfig(max_lag=-1).validate()


def test_reset_restores_the_prior_and_clears_history():
    rng = np.random.default_rng(41)
    controller = AdaptiveNominalController(PlanarArm())
    _simulate(controller, _random_plant(rng), np.ones(2), 1, steps=30, rng=rng)
    assert controller.updates == 30
    controller.reset()
    assert controller.updates == 0 and controller.lag == 0
    np.testing.assert_array_equal(controller.parameters, controller.prior)
