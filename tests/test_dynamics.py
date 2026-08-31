import numpy as np
import pytest

from sarrl.dynamics import PlanarArm, PlanarArmParams


def test_mass_matrix_is_symmetric_positive_definite():
    arm = PlanarArm()
    rng = np.random.default_rng(1)
    for _ in range(100):
        q = rng.uniform(-np.pi, np.pi, size=2)
        M = arm.mass_matrix(q)
        np.testing.assert_allclose(M, M.T, atol=1e-12)
        assert np.min(np.linalg.eigvalsh(M)) > 1e-6


def test_mdot_minus_2c_is_skew_symmetric():
    arm = PlanarArm()
    rng = np.random.default_rng(2)
    h = 1e-6
    for _ in range(50):
        q = rng.uniform(-2.0, 2.0, size=2)
        qd = rng.uniform(-1.5, 1.5, size=2)
        mdot = np.zeros((2, 2))
        for k in range(2):
            dq = np.zeros(2)
            dq[k] = h
            dmdq = (arm.mass_matrix(q + dq) - arm.mass_matrix(q - dq)) / (2.0 * h)
            mdot += dmdq * qd[k]
        S = mdot - 2.0 * arm.coriolis_matrix(q, qd)
        np.testing.assert_allclose(S + S.T, np.zeros((2, 2)), atol=2e-8)


def test_forward_inverse_dynamics_round_trip():
    arm = PlanarArm()
    rng = np.random.default_rng(3)
    for _ in range(100):
        q = rng.uniform(-2.0, 2.0, 2)
        qd = rng.uniform(-2.0, 2.0, 2)
        qdd = rng.uniform(-4.0, 4.0, 2)
        tau = arm.inverse_dynamics(q, qd, qdd)
        got = arm.forward_dynamics(q, qd, tau)
        np.testing.assert_allclose(got, qdd, rtol=1e-10, atol=1e-10)


def test_jacobian_matches_finite_difference():
    arm = PlanarArm()
    q = np.array([0.7, -0.9])
    h = 1e-7
    Jn = np.zeros((2, 2))
    for k in range(2):
        dq = np.zeros(2)
        dq[k] = h
        Jn[:, k] = (arm.forward_kinematics(q + dq) - arm.forward_kinematics(q - dq)) / (2 * h)
    np.testing.assert_allclose(arm.jacobian(q), Jn, atol=1e-8)


def test_inverse_kinematics_round_trip():
    arm = PlanarArm()
    for target in (np.array([1.2, 0.4]), np.array([0.4, 1.1]), np.array([-0.8, 0.7])):
        q = arm.inverse_kinematics(target)
        np.testing.assert_allclose(arm.forward_kinematics(q), target, atol=1e-10)


def test_rk4_energy_drift_small_for_conservative_plant():
    p = PlanarArmParams(gravity=0.0, viscous=(0.0, 0.0), coulomb=(0.0, 0.0))
    arm = PlanarArm(p)
    x = np.array([0.3, -0.7, 1.0, -0.4])
    e0 = arm.energy(x[:2], x[2:])
    for _ in range(1000):
        x = arm.step_rk4(x, np.zeros(2), 0.001, include_friction=False)
    e1 = arm.energy(x[:2], x[2:])
    assert abs(e1 - e0) < 2e-8


def test_parameter_validation_rejects_impossible_com():
    with pytest.raises(ValueError):
        PlanarArm(PlanarArmParams(lc1=2.0))


def test_rk4_rejects_nonpositive_dt():
    arm = PlanarArm()
    with pytest.raises(ValueError):
        arm.step_rk4(np.zeros(4), np.zeros(2), 0.0)


def test_payload_preserves_mass_matrix_and_coriolis_identity():
    arm = PlanarArm(PlanarArmParams(payload_mass=1.7))
    q = np.array([0.3, -1.1])
    qd = np.array([0.8, 0.4])
    eig = np.linalg.eigvalsh(arm.mass_matrix(q))
    assert np.all(eig > 0.0)
    h = 1e-6
    mdot = np.zeros((2, 2))
    for k in range(2):
        dq = np.zeros(2)
        dq[k] = h
        mdot += ((arm.mass_matrix(q + dq) - arm.mass_matrix(q - dq)) / (2 * h)) * qd[k]
    skew = mdot - 2.0 * arm.coriolis_matrix(q, qd)
    np.testing.assert_allclose(skew + skew.T, np.zeros((2, 2)), atol=3e-8)


def test_payload_increases_gravity_load_in_expected_direction():
    base = PlanarArm()
    loaded = PlanarArm(PlanarArmParams(payload_mass=1.0))
    q = np.array([0.0, 0.0])
    assert np.all(loaded.gravity_vector(q) > base.gravity_vector(q))


def test_jacobian_dot_times_qd_matches_finite_difference():
    arm = PlanarArm()
    q = np.array([0.6, -0.8])
    qd = np.array([0.9, -0.35])
    h = 1e-7
    numeric = ((arm.jacobian(q + h * qd) - arm.jacobian(q - h * qd)) / (2 * h)) @ qd
    np.testing.assert_allclose(arm.jacobian_dot_times_qd(q, qd), numeric, atol=2e-8)
