from dataclasses import replace

import numpy as np
import pytest

from sarrl.safety.panda_joint_filter import (
    PandaJointFilter,
    PandaJointFilterConfig,
    project_torque,
)


@pytest.mark.parametrize(
    "config",
    [
        PandaJointFilterConfig(max_delay_steps=True),
        PandaJointFilterConfig(control_dt=np.nan),
        PandaJointFilterConfig(gamma=100),
        PandaJointFilterConfig(solver_iterations=0),
    ],
)
def test_invalid_filter_settings(config):
    with pytest.raises(ValueError):
        config.validate()


def test_weighted_projection_matches_independent_box_solution():
    limit = np.arange(1.0, 8.0)
    candidate = np.array([-8.0, -3.0, -0.5, 0.0, 2.0, 7.0, 10.0])
    result = project_torque(candidate, np.empty((0, 7)), np.empty(0), limit)
    assert result.predicted_feasible
    np.testing.assert_allclose(result.torque, np.clip(candidate, -limit, limit), atol=1e-10)
    assert np.all(np.abs(result.torque) <= limit)
    assert result.stationarity_residual < 1e-7


def test_coupled_constraint_has_analytic_solution_despite_duplicate_rows():
    row = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = project_torque(np.zeros(7), np.stack((row, row)), np.array([1.0, 1.0]), np.ones(7))
    assert result.predicted_feasible
    np.testing.assert_allclose(result.torque, [0.5, 0.5, 0, 0, 0, 0, 0], atol=1e-10)


def test_empty_polytope_is_never_reported_feasible():
    A = np.array([[1.0, 0, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0, 0]])
    result = project_torque(np.zeros(7), A, np.array([0.8, 0.8]), np.ones(7))
    assert not result.predicted_feasible and result.reason == "infeasible"
    zero = project_torque(np.zeros(7), np.zeros((1, 7)), np.ones(1), np.ones(7))
    assert not zero.predicted_feasible


def test_solver_flag_alone_cannot_accept_constraint_violation(monkeypatch):
    from types import SimpleNamespace

    import sarrl.safety.panda_joint_filter as module

    monkeypatch.setattr(
        module, "minimize", lambda *args, **kwargs: SimpleNamespace(x=np.zeros(7), success=True)
    )
    result = project_torque(np.zeros(7), np.eye(7), np.full(7, 0.5), np.ones(7))
    assert not result.predicted_feasible and result.reason == "solver_failure"


def test_frozen_acceleration_constraints_cover_all_command_ages():
    filter = PandaJointFilter(np.full(7, -2.0), np.full(7, 2.0), np.full(7, 20.0))
    q, v = np.zeros(7), np.full(7, 0.1)
    history = [np.full(7, index) for index in range(4)]
    A, b, reason, arrivals = filter.constraints(q, v, np.eye(7), np.zeros(7), history)
    assert reason == "ready" and A.shape == (140, 7)
    for lag, (position, velocity) in enumerate(arrivals):
        expected_q, expected_v = q.copy(), v.copy()
        for command in history[4 - lag :]:
            expected_q += expected_v * 0.01 + 0.5 * command * 0.01**2
            expected_v += command * 0.01
        np.testing.assert_allclose(position, expected_q)
        np.testing.assert_allclose(velocity, expected_v)
    result = filter.filter(q, v, np.full(7, 100.0), np.eye(7), np.zeros(7), history)
    assert result.predicted_feasible
    assert min(A @ result.torque - b) >= -1e-7


def test_pending_command_violation_cannot_be_repaired_by_a_new_command():
    config = replace(PandaJointFilterConfig(), max_delay_steps=1)
    filter = PandaJointFilter(np.full(7, -2.0), np.full(7, 2.0), np.full(7, 20.0), config)
    q, v = np.zeros(7), np.zeros(7)
    v[0] = filter.velocity[0] - 0.01
    result = filter.filter(q, v, np.full(7, -20.0), np.eye(7), np.zeros(7), [np.full(7, 20.0)])
    assert not result.predicted_feasible and result.reason == "pending_command_violation"


def test_polynomial_interval_check_includes_interior_position_extrema():
    filter = PandaJointFilter(np.full(7, -2.0), np.full(7, 2.0), np.full(7, 20.0))
    q, v, a = np.zeros(7), np.zeros(7), np.zeros(7)
    q[0], v[0], a[0] = filter.upper[0] - 0.001, 1.0, -200.0
    assert q[0] + v[0] * 0.01 + 0.5 * a[0] * 0.01**2 < filter.upper[0]
    assert not filter._inside_interval(q, v, a)


def test_substep_instrumentation_preserves_original_physics_exactly():
    pytest.importorskip("mujoco")
    from sarrl.envs.panda_joint_reach import PandaJointReachEnv
    from sarrl.envs.panda_reach import PandaReachEnv
    from sarrl.robots.panda import PandaModelParameters

    parameters = PandaModelParameters(payload_mass=1.0, viscous_extra=(1.0,) * 7)
    original, instrumented = (
        PandaReachEnv(parameters=parameters),
        PandaJointReachEnv(parameters=parameters),
    )
    original.reset(seed=9804100)
    instrumented.reset(seed=9804100)
    for step in range(5):
        torque = np.sin(np.arange(7) + step)
        first, second = original.step_torque(torque), instrumented.step_torque(torque)
        np.testing.assert_array_equal(first[0], second[0])
        assert first[1:] == second[1:]
        assert len(instrumented.last_substeps) == 5
        np.testing.assert_array_equal(instrumented.last_substeps[-1]["q"], original.data.qpos)


def test_braking_is_latched_and_never_claimed_feasible():
    pytest.importorskip("mujoco")
    from sarrl.controllers.panda_filtered import PandaFilteredController
    from sarrl.robots.panda import HOME

    controller = PandaFilteredController()
    velocity = np.full(7, 4.0)
    sent = controller.command(HOME, velocity, HOME)
    assert controller.aborted and not controller.last_filter["predicted_feasible"]
    assert controller.last_filter["braking"]
    assert np.all(np.abs(sent) <= controller.dynamics.torque_limit)
    controller.command(HOME, np.zeros(7), HOME)
    assert controller.attempts == 1 and controller.braking_steps == 2


def test_safety_gain_cannot_hide_lost_task_successes():
    from sarrl.evaluation.panda_joint_filter import ARMS, CONDITIONS, evaluate_gate

    records = [
        {
            "scenario": c,
            "seed": 9804100,
            "arm": a,
            "tracking_rms_rad": [0.0] * 7,
            "success": a == "bank_filtered" or c.startswith("slow_"),
            "unsafe_episode": a == "bank_align" and c.startswith("fast_"),
            "filter": {"aborted": False},
        }
        for c in CONDITIONS
        for a in ARMS
    ]
    assert evaluate_gate(records)["filter_advances"]
    target = next(
        r for r in records if r["arm"] == "bank_filtered" and r["scenario"].startswith("slow_")
    )
    target["success"] = False
    target["filter"]["aborted"] = True
    decision = evaluate_gate(records)
    assert not decision["filter_advances"]
    assert {f["criterion"] for f in decision["failures"]} == {"success_count", "slow_abort"}
    with pytest.raises(ValueError, match="factorial"):
        evaluate_gate(records[:-1])


def test_uninformative_grid_cannot_advance_filter():
    from sarrl.evaluation.panda_joint_filter import ARMS, CONDITIONS, evaluate_gate

    rows = [
        {
            "scenario": c,
            "seed": 9804100,
            "arm": a,
            "tracking_rms_rad": [0.0] * 7,
            "success": True,
            "unsafe_episode": False,
            "filter": {"aborted": False},
        }
        for c in CONDITIONS
        for a in ARMS
    ]
    decision = evaluate_gate(rows)
    assert not decision["filter_advances"]
    assert {f["criterion"] for f in decision["failures"]} == {
        "uninformative_fast_grid",
        "fast_unsafe_reduction",
    }


def test_runner_refuses_alternate_outputs_and_invalid_workers(tmp_path):
    from tools.run_panda_joint_filter import main

    for args in (["--output", str(tmp_path / "alternate")], ["--workers", "5"]):
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code == 2
    assert not (tmp_path / "alternate").exists()


def test_filtered_episode_retains_fifo_noise_and_physical_substeps():
    pytest.importorskip("mujoco")
    from sarrl.controllers.panda_filtered import PandaFilteredController
    from sarrl.evaluation.panda_joint_filter import run_episode
    from tools.audit_panda_estimator_memory import check_noise
    from tools.audit_panda_joint_filter import independent_projection_check

    row, trace = run_episode(9804100, "slow_noise_d20", "bank_filtered", retain_trace=True)
    assert row["physical_steps"] == 1500 and len(trace) == 301
    check_noise(trace, row["seed"], row["interface_config"], row)
    substeps = [s for frame in trace[:-1] for s in frame["physical_substeps"]]
    assert len(substeps) == 1500
    np.testing.assert_allclose(
        row["max_speed_rad_s"], np.max(np.abs([s["qd"] for s in substeps]), axis=0)
    )
    assert row["contact_substeps"] == sum(s["contacts"] > 0 for s in substeps)
    assert row["success"] == (row["task_success"] and not row["filter"]["aborted"])
    controller = PandaFilteredController()
    for index, frame in enumerate(trace[:3]):
        following = trace[index + 1]
        np.testing.assert_array_equal(
            frame["delivered_torque_nm"],
            np.zeros(7) if index < 2 else trace[index - 2]["torque_nm"],
        )
        if frame["filter"]["predicted_feasible"]:
            independent_projection_check(
                controller,
                frame["measured_q"],
                frame["measured_qd"],
                frame["torque_nm"],
                frame["nominal_command_nm"],
            )
        controller.observe_transition(
            frame["measured_q"],
            frame["measured_qd"],
            following["measured_q"],
            following["measured_qd"],
            frame["torque_nm"],
            0.01,
        )
