import copy

import numpy as np
import pytest

from sarrl.controllers.panda_adaptive import PandaEstimatorConfig, PandaRLS
from sarrl.controllers.panda_estimator_memory import (
    PandaEstimatorMemoryConfig,
    PandaForgettingRLS,
    PandaJointMemoryController,
)


@pytest.mark.parametrize("memory", [True, 1, 2.5, -1])
def test_parameter_memory_configuration_rejects_invalid_intervals(memory):
    with pytest.raises(ValueError):
        PandaForgettingRLS(memory_config=PandaEstimatorMemoryConfig(memory))


def test_unbounded_solution_matches_independent_weighted_batch_precision():
    cfg = PandaEstimatorConfig(innovation_variance=1.7)
    estimator = PandaForgettingRLS(cfg)
    initial_precision = np.linalg.inv(estimator.covariance)
    designs, observations = [], []
    truth = np.full(8, 0.25)
    for step in range(45):
        design = np.eye(8)[np.arange(7) + (step % 2)] * (1 + step / 100)
        values = design @ truth
        designs.append(design)
        observations.append(values)
        estimator.update(design, values)
        decay = estimator.memory_config.decay
        weights = decay ** np.arange(step, -1, -1)
        precision = decay ** (step + 1) * initial_precision
        rhs = np.zeros(8)
        information = np.zeros((8, 8))
        for weight, rows, residual in zip(weights, designs, observations, strict=True):
            information += weight * rows.T @ rows
            precision += weight * rows.T @ rows / cfg.innovation_variance
            rhs += weight * rows.T @ residual / cfg.innovation_variance
        np.testing.assert_allclose(estimator.covariance, np.linalg.inv(precision), rtol=1e-12)
        np.testing.assert_allclose(
            estimator.parameters, np.linalg.solve(precision, rhs), atol=1e-13
        )
        np.testing.assert_allclose(estimator.information, information, rtol=1e-12)
        assert estimator.bound_hits == 0


def test_none_memory_matches_original_arithmetic_exactly():
    original = PandaRLS()
    candidate = PandaForgettingRLS(memory_config=PandaEstimatorMemoryConfig(None))
    rng = np.random.default_rng(9805205)
    for _ in range(60):
        design, residual = rng.normal(size=(7, 8)), rng.normal(size=7)
        original.update(design, residual)
        candidate.update(design, residual)
        for field in ("parameters", "covariance", "information", "innovation_squared"):
            np.testing.assert_array_equal(getattr(candidate, field), getattr(original, field))
        assert candidate.updates == original.updates
        assert candidate.bound_hits == original.bound_hits


def test_invalid_and_overflowing_updates_preserve_all_state():
    estimator = PandaForgettingRLS()
    before = copy.deepcopy(estimator.__dict__)
    with pytest.raises(ValueError):
        estimator.update(np.zeros((6, 8)), np.zeros(7))
    for field in ("parameters", "covariance", "information", "innovation_squared"):
        np.testing.assert_array_equal(getattr(estimator, field), before[field])
    estimator.covariance[:] = np.eye(8) * 1.79e308
    before = copy.deepcopy(estimator.__dict__)
    with pytest.raises(FloatingPointError):
        estimator.update(np.zeros((7, 8)), np.zeros(7))
    for field in ("parameters", "covariance", "information", "innovation_squared"):
        np.testing.assert_array_equal(getattr(estimator, field), before[field])
    assert estimator.updates == estimator.bound_hits == 0


def test_loss_of_excitation_has_explicit_inverse_information_growth():
    estimator = PandaForgettingRLS()
    initial = estimator.covariance.copy()
    for _ in range(100):
        estimator.update(np.zeros((7, 8)), np.zeros(7))
    np.testing.assert_allclose(estimator.covariance, initial / estimator.memory_config.decay**100)
    assert estimator.diagnostics()["information_rank"] == 0
    np.testing.assert_array_equal(estimator.parameters, np.zeros(8))


def test_covariance_stays_positive_and_parameters_respect_bounds():
    estimator = PandaForgettingRLS()
    rng = np.random.default_rng(9805206)
    for _ in range(400):
        estimator.update(rng.normal(size=(7, 8)), rng.normal(0, 100, 7))
        assert np.linalg.eigvalsh(estimator.covariance).min() > 0
        assert np.all(estimator.parameters >= 0)
        assert np.all(estimator.parameters <= estimator.upper)
    assert estimator.bound_hits > 0


def test_controller_none_memory_matches_score_only_causal_prefix():
    pytest.importorskip("mujoco")
    from sarrl.controllers.panda_score_memory import PandaScoreMemoryController
    from sarrl.robots.panda import HOME

    original = PandaScoreMemoryController()
    candidate = PandaJointMemoryController(PandaEstimatorMemoryConfig(None))
    rng = np.random.default_rng(9805207)
    for _ in range(80):
        q, velocity = HOME + rng.normal(0, 0.05, 7), rng.normal(0, 0.1, 7)
        next_q, next_velocity = q + velocity * 0.01, velocity + rng.normal(0, 0.02, 7)
        sent = rng.uniform(-5, 5, 7)
        for controller in (original, candidate):
            controller.observe_transition(q, velocity, next_q, next_velocity, sent, 0.01)
        np.testing.assert_array_equal(original.scores, candidate.scores)
        assert original.selected_history == candidate.selected_history
        for old, new in zip(original.estimators, candidate.estimators, strict=True):
            np.testing.assert_array_equal(old.parameters, new.parameters)
            np.testing.assert_array_equal(old.covariance, new.covariance)
    assert not candidate.predict and candidate.known_delay_steps is None
    assert candidate.score_memory.config.memory_steps == 50


def gate_records():
    from sarrl.evaluation.panda_estimator_memory import ARMS, CONDITIONS

    return [
        {
            "scenario": condition,
            "seed": 9805208,
            "arm": arm,
            "success": True,
            "unsafe_episode": False,
            "joint_error_max_rad": 0.0,
            "tracking_phase_rms_rad": {"after": [0.0] * 7},
            "delay_estimator": {
                "final_delay_correct": (
                    arm == "bank_joint_memory" or condition.endswith(("fixed10", "fixed40"))
                )
            },
        }
        for condition in CONDITIONS
        for arm in ARMS
    ]


def test_identification_gain_does_not_override_tracking_or_physical_failure():
    from sarrl.evaluation.panda_estimator_memory import evaluate_gate

    records = gate_records()
    assert evaluate_gate(records)["candidate_advances"]
    target = next(
        row
        for row in records
        if row["scenario"] == "warm_noise_down" and row["arm"] == "bank_joint_memory"
    )
    target["tracking_phase_rms_rad"]["after"] = [0.00011] + [0.0] * 6
    result = evaluate_gate(records)
    assert not result["candidate_advances"]
    assert result["final_age_gains_over_score_only"]["all_changing"] == 1
    assert any(failure["criterion"] == "after_tracking_rms_rad" for failure in result["failures"])
    assert all(failure["joint"] == 1 for failure in result["failures"])
    target["tracking_phase_rms_rad"]["after"] = [0.0] * 7
    target["unsafe_episode"] = True
    assert not evaluate_gate(records)["candidate_advances"]


def test_gate_rejects_missing_duplicates_nonfinite_metrics_and_constant_regressions():
    from sarrl.evaluation.panda_estimator_memory import evaluate_gate

    records = gate_records()
    with pytest.raises(ValueError, match="factorial"):
        evaluate_gate(records[:-1])
    with pytest.raises(ValueError, match="factorial"):
        evaluate_gate(records + records[:1])
    target = next(
        row
        for row in records
        if row["scenario"] == "cold_clean_fixed10" and row["arm"] == "bank_joint_memory"
    )
    target["joint_error_max_rad"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_gate(records)
    target["joint_error_max_rad"] = 0.0
    target["delay_estimator"]["final_delay_correct"] = False
    result = evaluate_gate(records)
    assert not result["candidate_advances"]
    assert any(failure["criterion"] == "constant_final_age_count" for failure in result["failures"])


def test_future_packet_profile_cannot_change_pre_switch_joint_memory_commands():
    pytest.importorskip("mujoco")
    from sarrl.evaluation.panda_estimator_memory import run_episode

    changing_row, changing = run_episode(
        9805208, "warm_noise_down", "bank_joint_memory", retain_trace=True
    )
    constant_row, constant = run_episode(
        9805208, "warm_noise_fixed40", "bank_joint_memory", retain_trace=True
    )
    assert changing_row["measurement_noise_sha256"] == constant_row["measurement_noise_sha256"]
    assert changing_row["delay_estimator"]["parameter_decay"] == 49 / 51
    assert not changing_row["privileged_delay"]
    for first, second in zip(changing[:101], constant[:101], strict=True):
        for field in ("q", "qd", "torque_nm", "estimated_parameters", "delay_scores"):
            assert first[field] == second[field]


def test_reproduction_mismatch_blocks_fresh_estimator_memory_seeds(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from tools import run_panda_estimator_memory as runner

    rows = json.loads(Path("results/panda_score_memory_pilot/episodes.json").read_text())
    payloads = [{"record": copy.deepcopy(row), "trace": []} for row in rows]
    payloads[0]["record"]["measurement_noise_sha256"] = "changed"
    monkeypatch.setattr(runner, "execute_cells", lambda *args, **kwargs: payloads)
    with pytest.raises(RuntimeError, match="no fresh seed opened"):
        runner.reproduce_previous(tmp_path)
    assert json.loads((tmp_path / "reproduction.json").read_text())["matched"] == 1919


def test_runner_refuses_alternate_outputs_and_invalid_parallelism(tmp_path):
    from tools.run_panda_estimator_memory import main

    for arguments in (["--output", str(tmp_path / "alternate")], ["--workers", "5"]):
        with pytest.raises(SystemExit) as error:
            main(arguments)
        assert error.value.code == 2
    assert not (tmp_path / "alternate").exists()
