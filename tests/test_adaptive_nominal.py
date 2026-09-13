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
    assert controller.lag == 0
    assert max(errors[150:]) < 2.0, "the estimator must settle again after the fault"
    assert np.mean(errors[150:]) < 0.2
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


def test_predict_state_walks_the_queued_commands_through_the_estimated_model():
    rng = np.random.default_rng(61)
    params = _random_plant(rng)
    gain = np.array([0.9, 1.1])
    controller = AdaptiveNominalController(PlanarArm(), AdaptiveNominalConfig(lag_min_updates=1))
    controller.theta[:] = CommandRegressor.parameters(params, gain)
    arm = PlanarArm(params)
    state = np.array([0.2, -0.1, 0.5, -0.3])
    sent = [rng.uniform(-10.0, 10.0, 2) for _ in range(3)]
    controller._sent = [s.copy() for s in sent]
    controller.updates = 5
    controller.error_score[:] = (1.0, 1.0, 0.0, 1.0)  # lag 2 selected
    assert controller.lag == controller.prediction_lag == 2
    expected = state.copy()
    for command in sent[-2:]:
        expected = arm.step_rk4(expected, command * gain, 0.02)
    np.testing.assert_allclose(controller.predict_state(state), expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(controller.predict_state(state, steps=0), state)


def test_reset_restores_the_prior_and_clears_history():
    rng = np.random.default_rng(41)
    controller = AdaptiveNominalController(PlanarArm())
    _simulate(controller, _random_plant(rng), np.ones(2), 1, steps=30, rng=rng)
    assert controller.updates == 30
    early = AdaptiveNominalController(PlanarArm(), AdaptiveNominalConfig(lag_min_updates=10))
    _simulate(early, _random_plant(rng), np.ones(2), 3, steps=5, rng=rng)
    assert early.prediction_lag == 0, "no lag is trusted for prediction before lag_min_updates"
    controller.reset()
    assert controller.updates == 0 and controller.lag == 0
    np.testing.assert_array_equal(controller.parameters, controller.prior)


def test_payload_prior_shifts_only_the_payload_column():
    controller = AdaptiveNominalController(PlanarArm(), AdaptiveNominalConfig(payload_prior=0.5))
    assert np.all(controller.parameters[:, 4] == 0.5)
    reference = AdaptiveNominalController(PlanarArm())
    np.testing.assert_array_equal(controller.parameters[:, :4], reference.parameters[:, :4])
    np.testing.assert_array_equal(controller.parameters[:, 5:], reference.parameters[:, 5:])


def test_filter_gate_uses_the_nominal_model_until_convergence():
    rng = np.random.default_rng(51)
    config = AdaptiveNominalConfig(filter_gate=True, gate_threshold=1.0, gate_min_updates=5)
    controller = AdaptiveNominalController(PlanarArm(), config)
    model = controller.estimated_model()
    nominal = PlanarArm()
    q, qd = np.array([0.4, -0.6]), np.array([0.3, 0.2])
    assert model.uses_nominal and not controller.converged
    np.testing.assert_array_equal(model.mass_matrix(q), nominal.mass_matrix(q))
    np.testing.assert_allclose(
        model.forward_dynamics(q, qd, np.array([3.0, 1.0])),
        nominal.forward_dynamics(q, qd, np.array([3.0, 1.0])),
        rtol=1e-12,
        atol=1e-12,
    )
    _simulate(controller, _random_plant(rng), np.array([0.9, 0.8]), 1, steps=120, rng=rng)
    assert controller.converged and not model.uses_nominal
    assert not np.array_equal(model.mass_matrix(q), nominal.mass_matrix(q))
    # The latch survives a burst of innovation, as after an in-episode fault.
    controller.recent_score[:] = 1e6
    assert controller.converged
    controller.reset()
    assert not controller.converged
    with pytest.raises(ValueError):
        AdaptiveNominalConfig(gate_threshold=0.0).validate()
    with pytest.raises(ValueError):
        AdaptiveNominalConfig(payload_prior=-0.1).validate()


def test_actuator_time_constant_hypotheses_select_the_true_lag_and_constant():
    rng = np.random.default_rng(71)
    params = _random_plant(rng)
    gain = np.array([0.9, 1.1])
    grid = (0.0, 0.01, 0.03, 0.06)
    controller = AdaptiveNominalController(
        PlanarArm(), AdaptiveNominalConfig(actuator_time_constants=grid)
    )
    assert len(controller.hypotheses) == 4 * len(grid)
    arm = PlanarArm(params)
    dt, substeps, tau, delay = 0.02, 10, 0.03, 1
    alpha = (dt / substeps) / (tau + dt / substeps)
    state = np.concatenate([rng.uniform(-0.25, 0.25, 2), rng.uniform(-0.05, 0.05, 2)])
    queue = [np.zeros(2) for _ in range(delay)]
    delivered = np.zeros(2)
    q_des = rng.uniform(-1.0, 1.0, 2)
    for step in range(300):
        if step % 40 == 0:
            q_des = rng.uniform(-1.0, 1.0, 2)
        sent = controller.command(state[:2], state[2:], q_des)
        queue.append(sent.copy())
        applied = queue.pop(0) * gain
        before = state.copy()
        # Plant with a first-order actuator advanced on ten sub-steps, as in MuJoCo.
        for _ in range(substeps):
            delivered = delivered + alpha * (applied - delivered)
            state = arm.step_rk4(state, delivered, dt / substeps)
        controller.observe(before, state, sent)
    assert controller.lag == delay
    assert controller.time_constant == pytest.approx(tau)
    # The default grid keeps the v1.9 behaviour: hypotheses are the lags alone.
    assert len(AdaptiveNominalController(PlanarArm()).hypotheses) == 4
    with pytest.raises(ValueError):
        AdaptiveNominalConfig(actuator_time_constants=(-0.01,)).validate()
