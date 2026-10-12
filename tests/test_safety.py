import numpy as np
import pytest

from sarrl.dynamics import PlanarArm
from sarrl.safety import (
    CircularObstacle,
    HOCBFSafetyFilter,
    MarginHOCBFSafetyFilter,
    SafetyConfig,
    project_polytope_2d,
)


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


def test_margin_filter_without_margin_or_uncertainty_matches_the_base_filter():
    arm = PlanarArm()
    state = np.array([2.8, -0.5, 2.0, -0.4])
    candidate = np.array([30.0, -10.0])
    reference = HOCBFSafetyFilter(arm).filter(state, candidate)

    def unused(state, candidate):
        raise AssertionError("a zero margin must not query the uncertainty")

    for kappa, uncertainty in ((0.0, unused), (2.0, lambda state, candidate: None)):
        result = MarginHOCBFSafetyFilter(arm, kappa=kappa, uncertainty=uncertainty).filter(
            state, candidate
        )
        np.testing.assert_array_equal(result.torque, reference.torque)
        assert result.active_constraints == reference.active_constraints


def test_margin_filter_keeps_every_model_row_for_command_errors_within_the_margin():
    arm = PlanarArm()
    sigma = np.array([0.5, 1.5])
    kappa = 2.0
    obstacle = CircularObstacle(center=(0.2, 1.4), radius=0.1)
    margin = MarginHOCBFSafetyFilter(arm, kappa=kappa, uncertainty=lambda s, c: sigma)
    state = np.array([2.8, -0.5, 2.0, -0.4])
    candidate = np.array([30.0, -10.0])
    A, b, _ = margin.constraints(state, [obstacle])
    base = HOCBFSafetyFilter(arm).filter(state, candidate, [obstacle])
    result = margin.filter(state, candidate, [obstacle])
    assert base.success and result.success
    model_rows = A.shape[0] - 4
    slack = A @ result.torque - b
    # Each model row keeps kappa * |A_r| sigma of slack; the torque box keeps none.
    assert np.all(slack[:model_rows] >= kappa * (np.abs(A[:model_rows]) @ sigma) - 2e-8)
    assert np.all(slack[model_rows:] >= -2e-8)
    # The base projection sits on a row the margin moves it away from.
    assert np.min((A @ base.torque - b)[:model_rows]) < 1e-6
    assert result.correction_norm > base.correction_norm
    # So the rows hold for every command error inside the box |du_j| <= kappa sigma_j.
    rng = np.random.default_rng(3)
    for du in rng.uniform(-1.0, 1.0, size=(200, 2)) * kappa * sigma:
        assert np.min((A @ (result.torque - du) - b)[:model_rows]) >= -2e-8


def test_margin_filter_rejects_bad_margins_and_uncertainties():
    arm = PlanarArm()
    for kappa in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            MarginHOCBFSafetyFilter(arm, kappa=kappa, uncertainty=lambda s, c: np.zeros(2))
    state = np.array([0.3, -0.5, 0.2, -0.1])
    for sigma in (np.zeros(3), np.array([-1.0, 0.0]), np.array([np.nan, 0.0])):
        margin = MarginHOCBFSafetyFilter(arm, kappa=1.0, uncertainty=lambda s, c, v=sigma: v)
        with pytest.raises(ValueError):
            margin.filter(state, np.zeros(2))


def test_margin_filter_halves_an_infeasible_margin_and_drops_it_last():
    arm = PlanarArm()
    state = np.array([2.8, -0.5, 2.0, -0.4])
    candidate = np.array([30.0, -10.0])
    reference = HOCBFSafetyFilter(arm).filter(state, candidate)
    moderate = MarginHOCBFSafetyFilter(arm, kappa=1.0, uncertainty=lambda s, c: np.ones(2))
    assert moderate.filter(state, candidate).success
    assert (moderate.margin_steps, moderate.reduced_steps, moderate.dropped_steps) == (1, 0, 0)
    np.testing.assert_array_equal(moderate.applied, np.ones(2))
    # A margin the torque box cannot meet is halved until it can.
    large = MarginHOCBFSafetyFilter(arm, kappa=1.0, uncertainty=lambda s, c: np.full(2, 40.0))
    result = large.filter(state, candidate)
    assert result.success and result.correction_norm > reference.correction_norm
    assert (large.margin_steps, large.reduced_steps, large.dropped_steps) == (1, 1, 0)
    halvings = np.log2(40.0 / large.applied[0])
    assert large.applied[0] == large.applied[1] and halvings == round(halvings) >= 1
    # Past the last halving the projection falls back to the rows without margin.
    huge = MarginHOCBFSafetyFilter(arm, kappa=1.0, uncertainty=lambda s, c: np.full(2, 1e6))
    result = huge.filter(state, candidate)
    np.testing.assert_array_equal(result.torque, reference.torque)
    assert (huge.margin_steps, huge.reduced_steps, huge.dropped_steps) == (1, 0, 1)
    np.testing.assert_array_equal(huge.applied, np.zeros(2))
    huge.reset_counters()
    assert huge.margin_steps == huge.dropped_steps == 0
