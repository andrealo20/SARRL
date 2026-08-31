import numpy as np

from sarrl.dynamics import PlanarArm
from sarrl.safety import CircularObstacle, HOCBFSafetyFilter, SafetyConfig, project_polytope_2d


def test_exact_projection_onto_single_halfspace():
    A = np.array([[1.0, 0.0]])
    b = np.array([1.0])
    result = project_polytope_2d(np.array([0.0, 2.0]), A, b)
    assert result.success
    np.testing.assert_allclose(result.x, np.array([1.0, 2.0]), atol=1e-12)
    assert result.active == (0,)


def test_exact_projection_detects_infeasible_polytope():
    A = np.array([[1.0, 0.0], [-1.0, 0.0]])
    b = np.array([1.0, 1.0])
    result = project_polytope_2d(np.zeros(2), A, b)
    assert not result.success
    assert result.min_margin < 0.0


def test_safety_filter_output_satisfies_all_hard_constraints_when_successful():
    arm = PlanarArm()
    filt = HOCBFSafetyFilter(arm, SafetyConfig(torque_limit=(25.0, 25.0)))
    state = np.array([0.3, -0.5, 0.2, -0.1])
    candidate = np.array([25.0, -25.0])
    result = filt.filter(state, candidate)
    assert result.success
    A, b, _ = filt.constraints(state)
    assert np.min(A @ result.torque - b) >= -2e-8
    assert np.all(np.abs(result.torque) <= 25.0 + 1e-8)


def test_obstacle_hocbf_intervenes_on_torque_accelerating_into_obstacle():
    arm = PlanarArm()
    q = np.array([0.0, np.pi / 2.0])
    state = np.concatenate([q, np.zeros(2)])
    ee = arm.forward_kinematics(q)
    obstacle = CircularObstacle(center=(float(ee[0] + 0.22), float(ee[1])), radius=0.1, margin=0.05)
    filt = HOCBFSafetyFilter(arm)

    # Search a bounded candidate that most violates the obstacle row, then
    # verify the projection changes it and makes every hard row feasible.
    A, b, _ = filt.constraints(state, [obstacle])
    obstacle_row = A[-5]  # obstacle row appears immediately before 4 torque-box rows
    limit = np.array([40.0, 40.0])
    candidate = -np.sign(obstacle_row) * limit
    result = filt.filter(state, candidate, [obstacle])
    assert result.success
    assert result.correction_norm > 1e-5
    assert np.min(A @ result.torque - b) >= -2e-8


def test_filter_reports_current_geometric_violation_separately_from_qp_feasibility():
    arm = PlanarArm()
    state = np.array([0.0, np.pi / 2.0, 0.0, 0.0])
    ee = arm.forward_kinematics(state[:2])
    obstacle = CircularObstacle(center=(float(ee[0]), float(ee[1])), radius=0.2)
    result = HOCBFSafetyFilter(arm).filter(state, np.zeros(2), [obstacle])
    assert not result.current_safe
    # A state already inside an obstacle can make the hard recovery problem
    # feasible or infeasible depending on actuator authority; current_safe is
    # deliberately not conflated with solver success.
    assert isinstance(result.success, bool)
