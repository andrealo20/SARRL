import json
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sarrl.controllers.panda_adaptive import (  # noqa: E402
    PandaAdaptiveController,
    PandaEstimatorConfig,
    PandaRegressor,
    PandaRLS,
)
from sarrl.evaluation.panda_adaptation import (  # noqa: E402
    reproduction_mismatches,
    run_episode,
    sample_parameters,
)
from sarrl.evaluation.panda_nominal import run_episode as nominal_episode  # noqa: E402
from sarrl.robots.panda import HOME, PandaDynamics, PandaModelParameters  # noqa: E402


def test_panda_payload_and_friction_regressor_matches_engine_inverse_dynamics():
    parameters = PandaModelParameters(1.7, (0.7, 1.1, 1.5, 2.0, 2.3, 1.7, 0.9))
    actual = PandaDynamics(parameters=parameters)
    regressor = PandaRegressor()
    rng = np.random.default_rng(9804200)
    for _ in range(20):
        q = HOME + rng.uniform(-0.2, 0.2, 7)
        qd, qdd = rng.uniform(-0.5, 0.5, (2, 7))
        expected = actual.inverse_dynamics(q, qd, qdd)
        result = regressor.inverse_dynamics(q, qd, qdd, parameters.vector())
        np.testing.assert_allclose(result, expected, atol=1e-11)
        np.testing.assert_allclose(
            regressor.mass_matrix(q, parameters.vector()), actual.mass_matrix(q), atol=1e-12
        )


def test_panda_payload_and_friction_forward_dynamics_roundtrip():
    dynamics = PandaDynamics(parameters=PandaModelParameters(2.0, (1.3,) * 7))
    rng = np.random.default_rng(9804201)
    q, qd = HOME + 0.1, rng.uniform(-0.2, 0.2, 7)
    desired = rng.uniform(-0.3, 0.3, 7)
    torque = dynamics.inverse_dynamics(q, qd, desired)
    dynamics.data.ctrl[:] = torque
    mujoco.mj_forward(dynamics.model, dynamics.data)
    assert dynamics.data.ncon == 0
    np.testing.assert_allclose(dynamics.data.qacc, desired, atol=1e-10)


def test_panda_rls_recovers_excited_parameters_without_access_to_the_plant():
    regressor, estimator = PandaRegressor(), PandaRLS()
    theta = np.array([1.6, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 2.9])
    rng = np.random.default_rng(9804202)
    for _ in range(200):
        q = HOME + rng.uniform(-0.3, 0.3, 7)
        qd, qdd = rng.uniform(-1, 1, (2, 7))
        _, rows = regressor.evaluate(q, qd, qdd)
        estimator.update(rows, rows @ theta)
    np.testing.assert_allclose(estimator.parameters, theta, atol=0.01)
    np.testing.assert_allclose(estimator.covariance, estimator.covariance.T, atol=1e-14)
    assert np.linalg.eigvalsh(estimator.covariance).min() > 0.0
    assert estimator.diagnostics()["information_rank"] == 8


def test_panda_rls_keeps_unobserved_parameters_at_their_prior():
    estimator = PandaRLS()
    before = estimator.covariance.copy()
    estimator.update(np.zeros((7, 8)), np.ones(7))
    np.testing.assert_array_equal(estimator.parameters, np.zeros(8))
    np.testing.assert_array_equal(estimator.covariance, before)
    assert estimator.diagnostics()["information_rank"] == 0


def test_panda_estimator_bounds_and_invalid_observation_are_checked():
    estimator = PandaRLS()
    rows = np.eye(8)[:7]
    estimator.update(rows, np.full(7, 1e6))
    assert np.all(estimator.parameters >= 0)
    assert np.all(estimator.parameters <= estimator.upper)
    previous = estimator.parameters.copy()
    with pytest.raises(ValueError):
        estimator.update(np.full((7, 8), np.nan), np.ones(7))
    np.testing.assert_array_equal(estimator.parameters, previous)
    with pytest.raises(ValueError):
        PandaRLS(PandaEstimatorConfig(innovation_variance=0.0))
    with pytest.raises(ValueError):
        PandaModelParameters(payload_mass=-1).validate()
    with pytest.raises(ValueError):
        PandaModelParameters(viscous_extra=(-1.0,) * 7).validate()


def test_panda_adaptive_controller_starts_from_the_unidentified_nominal():
    from sarrl.controllers.panda_nominal import PandaComputedTorqueController

    adaptive = PandaAdaptiveController()
    fixed = PandaComputedTorqueController()
    np.testing.assert_array_equal(
        adaptive.command(HOME, np.zeros(7), HOME), fixed.command(HOME, np.zeros(7), HOME)
    )
    assert adaptive.estimator.updates == 0
    with pytest.raises(ValueError):
        adaptive.observe_transition(HOME, np.zeros(7), HOME, np.zeros(7), np.zeros(7), 0)


def test_panda_scenarios_reuse_parameter_draws_without_changing_task_draws():
    seed = 9804203
    payload = sample_parameters(seed, "payload")
    friction = sample_parameters(seed, "friction")
    combined = sample_parameters(seed, "payload_friction")
    assert payload.payload_mass == combined.payload_mass
    assert friction.viscous_extra == combined.viscous_extra
    assert sample_parameters(seed, "nominal").vector().tolist() == [0.0] * 8
    with pytest.raises(ValueError):
        sample_parameters(seed, "missing")


def test_panda_zero_parameters_match_the_original_xml_loading_path(monkeypatch):
    import sarrl.envs.panda_reach as reach
    import sarrl.robots.panda as robot

    actual, actual_trace = nominal_episode(9804208, retain_trace=True)

    def original_loader(timestep=0.002, *, parameters=None):
        model = mujoco.MjModel.from_xml_path(str(robot.ASSETS / "panda.xml"))
        model.opt.timestep = timestep
        return model

    monkeypatch.setattr(robot, "load_panda_model", original_loader)
    monkeypatch.setattr(reach, "load_panda_model", original_loader)
    original, original_trace = nominal_episode(9804208, retain_trace=True)
    assert actual == original
    assert actual_trace == original_trace


def test_panda_nominal_reproduction_matches_the_retained_reference_runtime():
    runtime = json.loads(Path("results/panda_nominal_pilot/manifest.json").read_text())["runtime"]
    if mujoco.__version__ != runtime["mujoco"] or np.__version__ != runtime["numpy"]:
        pytest.skip("strict retained reproduction requires its recorded MuJoCo and NumPy versions")
    rows = json.loads(Path("results/panda_nominal_pilot/episodes.json").read_text())
    actual, _ = nominal_episode(rows[0]["seed"])
    assert reproduction_mismatches(actual, rows[0]) == []


def test_panda_reproduction_accepts_roundoff_but_detects_changed_results():
    retained = {"error": 8e-8, "steps": 300, "success": True, "state": [0.0, 0.5]}
    assert reproduction_mismatches({**retained, "error": 8e-8 + 1e-12}, retained) == []
    assert reproduction_mismatches({**retained, "error": 1e-5}, retained) == ["error"]
    assert reproduction_mismatches({**retained, "steps": 299}, retained) == ["steps"]
    assert reproduction_mismatches({**retained, "success": False}, retained) == ["success"]


def test_panda_adaptive_rollout_uses_paired_states_and_improves_loaded_tracking():
    fixed, _ = run_episode(9804204, "payload_friction", "fixed")
    adaptive, trace = run_episode(9804204, "payload_friction", "adaptive", retain_trace=True)
    oracle, _ = run_episode(9804204, "payload_friction", "oracle")
    for field in ("true_parameters", "start_q", "goal_q", "target_xyz_m"):
        assert fixed[field] == adaptive[field] == oracle[field]
    assert adaptive["joint_error_max_rad"] < fixed["joint_error_max_rad"] / 5
    assert adaptive["success"]
    assert oracle["success"]
    assert not fixed["privileged_parameters"]
    assert not adaptive["privileged_parameters"]
    assert oracle["privileged_parameters"]
    assert trace[0]["estimated_parameters"] == [0.0] * 8
    assert adaptive["estimator"]["updates"] == 300
    assert not adaptive["unsafe_episode"]


def test_panda_adaptive_episode_repeats_exactly():
    first, first_trace = run_episode(9804205, "payload", "adaptive", retain_trace=True)
    second, second_trace = run_episode(9804205, "payload", "adaptive", retain_trace=True)
    assert first == second
    assert first_trace == second_trace


def test_panda_adaptation_resume_keeps_completed_cells_and_rejects_corruption(tmp_path):
    from tools.run_panda_adaptation import execute_cells

    planned = [("nominal", 9804206, "fixed"), ("nominal", 9804206, "adaptive")]
    calls = []

    def execute(seed, scenario, arm, *, retain_trace):
        calls.append((scenario, seed, arm))
        return {"scenario": scenario, "seed": seed, "arm": arm, "success": True}, []

    directory = tmp_path / "cells"
    first = execute_cells(directory, planned[:1], 9804206, execute=execute)
    assert len(calls) == 1
    resumed = execute_cells(directory, planned, 9804206, execute=execute)
    assert len(calls) == 2
    assert resumed[0] == first[0]
    execute_cells(directory, planned, 9804206, execute=execute)
    assert len(calls) == 2
    path = next(directory.glob("*.json"))
    data = json.loads(path.read_text())
    data["payload"]["record"]["success"] = False
    path.write_text(json.dumps(data))
    # A missing earlier cell must not be run before a corrupt later cell is noticed.
    other = next(p for p in directory.glob("*.json") if p != path)
    other.unlink()
    before = len(calls)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        execute_cells(directory, planned, 9804206, execute=execute)
    assert len(calls) == before


def test_panda_adaptation_runner_refuses_bad_seeds_and_existing_output(tmp_path):
    from tools.run_panda_adaptation import main

    with pytest.raises(SystemExit) as exc:
        main(["--seed-start", "50000", "--output", str(tmp_path / "new")])
    assert exc.value.code == 2
    assert not (tmp_path / "new").exists()
    with pytest.raises(SystemExit) as exc:
        main(["--output", str(tmp_path)])
    assert exc.value.code == 2


def test_panda_adaptation_summary_refuses_unpaired_cells():
    from sarrl.evaluation.panda_adaptation import summarize

    with pytest.raises(ValueError, match="paired cell"):
        summarize([{"seed": 9804207, "scenario": "nominal", "arm": "fixed"}])
