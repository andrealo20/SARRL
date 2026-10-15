import copy

import numpy as np
import pytest

from sarrl.evaluation.control_diagnosis import (
    panda_prefix_reasons,
    panda_rows,
    panda_trace,
    planar_pairs,
    planar_trace,
)


def test_diagnosis_rejects_changed_input_completion_identity(tmp_path, monkeypatch):
    from tools import diagnose_control_failures as command

    monkeypatch.setattr(command, "ROOT", tmp_path)
    source = tmp_path / "results/planar_residual_v26"
    source.mkdir(parents=True)
    (source / "complete.json").write_text("{}\n")
    with pytest.raises(ValueError, match="input identity"):
        command.evidence("planar_residual_v26")


def test_diagnosis_never_overwrites_existing_output(tmp_path, monkeypatch):
    from tools import diagnose_control_failures as command

    monkeypatch.setattr(command, "OUTPUT", tmp_path)
    with pytest.raises(SystemExit) as stopped:
        command.main(["--output", str(tmp_path)])
    assert stopped.value.code == 2


def record(arm, seed=1, outcome="success"):
    return {
        "scenario": "id_reference",
        "seed": seed,
        "arm": arm,
        "success": outcome == "success",
        "outcome": outcome,
        "true_parameters": [1.0],
        "reward": 0.0,
        "safety_intervention_fraction": 0.0,
        "steps": 1,
        "safety_infeasible": outcome == "abort",
        "final_distance": 0.0,
    }


def frame():
    return {
        "q_des": [0.0, 0.0],
        "raw_residual": [0.0, 0.0],
        "torque": [0.0, 0.0],
        "baseline_torque": [0.0, 0.0],
        "executed": True,
        "correction": 0.0,
        "physical_next_state": [0.0, 0.0, 0.4, 0.0],
    }


def test_paired_losses_are_distinct_from_abort_and_independent_model_counts():
    baseline = record("adaptive")
    rows = [
        baseline,
        record("residual_1", outcome="timeout"),
        record("residual_2", outcome="abort"),
        record("residual_3"),
    ]
    report = planar_pairs(rows)["id_reference"]
    assert report["paired_model_tasks"] == 3
    assert report["contingency"] == {"both_success": 1, "gain": 0, "loss": 2, "both_fail": 0}
    assert report["lost_success_outcomes"] == {"abort": 1, "timeout": 1}
    with pytest.raises(ValueError, match="duplicate"):
        planar_pairs(rows + [baseline])
    changed = copy.deepcopy(rows)
    changed[1]["true_parameters"] = [2.0]
    with pytest.raises(ValueError, match="unpaired"):
        planar_pairs(changed)


def test_planar_trace_separates_speed_failure_reward_and_command_projection():
    row = record("residual_1", outcome="timeout")
    row["reward"] = -0.01 * 0.4**2 - 0.0002
    sample = frame()
    sample.update(raw_residual=[8.0, 0.0], torque=[1.0, 0.0], correction=7.0)
    result = planar_trace(row, [sample])
    assert result["timeout_blocker"] == "speed"
    assert result["position_reached_but_too_fast_steps"] == 1
    assert result["residual_near_bound_attempts"] == 1
    assert result["mean_projection_correction_nm"] == 7.0
    assert result["projected_executed_steps"] == 1
    assert result["reward_terms"]["success_bonus"] == 0
    sample["correction"] = 6.0
    with pytest.raises(ValueError, match="correction"):
        planar_trace(row, [sample])


def test_abort_return_exposes_training_penalty_without_an_executed_step():
    row = record("residual_1", outcome="abort")
    row["steps"] = 0
    sample = frame()
    sample["executed"] = False
    result = planar_trace(row, [sample])
    assert result["reconstructed_reward"] == 0
    assert result["training_penalty_adjusted_return"] == -500
    assert result["executed_steps"] == 0
    assert result["projected_executed_steps"] == 0


def test_reward_reconstruction_rejects_changed_episode_and_residual_bound():
    row = record("residual_1", outcome="timeout")
    with pytest.raises(ValueError, match="reward/steps"):
        planar_trace(row, [frame()])
    sample = frame()
    sample["raw_residual"] = [8.01, 0.0]
    with pytest.raises(ValueError, match="bounded"):
        planar_trace(row, [sample])


def panda_record(abort=0):
    return {
        "scenario": "fast_clean_d40",
        "seed": 1,
        "arm": "bank_filtered",
        "task_config": {"control_dt": 0.01, "timestep": 0.002},
        "interface_config": {"command_delay_steps": 4},
        "filter": {
            "abort_step": abort,
            "aborted": abort is not None,
            "reasons": {"pending_command_violation": 1},
            "config": {},
        },
        "unsafe_episode": True,
        "joint_position_excess_rad": 0.0,
        "joint_velocity_excess_rad_s": 1.0,
        "contact_substeps": 0,
    }


def test_panda_violation_in_the_abort_interval_is_after_command_latch():
    row = panda_record()
    frames = [
        {
            "filter": {"braking": True, "reason": "pending_command_violation"},
            "physical_substeps": [{"q": [0.0] * 7, "qd": [2.0] * 7, "contacts": 0}],
        },
        {},
    ]
    result = panda_trace(row, frames, np.full(7, -1.0), np.ones(7), np.ones(7))
    assert result["first_physical_violation"]["time_s"] == 0.002
    assert result["unsafe_substeps_at_or_after_abort"] == 1
    assert result["unsafe_substeps_before_abort_or_without_abort"] == 0
    assert panda_rows([row])[row["scenario"]]["velocity_violation_episodes"] == 1
    row["unsafe_episode"] = False
    with pytest.raises(ValueError, match="violation flag"):
        panda_trace(row, frames, -np.ones(7), np.ones(7), np.ones(7))


def test_panda_prefix_is_a_saved_state_diagnostic_and_never_solves_a_qp():
    class Model:
        jnt_range = np.array([[-1.0, 1.0]] * 7)

    class Nominal:
        model = Model()
        torque_limit = np.ones(7)

    class Regressor:
        nominal = Nominal()

        def mass_matrix(self, q, parameters):
            return np.eye(7)

        def inverse_dynamics(self, q, v, acceleration, parameters):
            return np.zeros(7)

    class Config:
        def __init__(self, **kwargs):
            self.bound = kwargs["max_delay_steps"]

    class Filter:
        def __init__(self, lower, upper, limit, config):
            self.config = config

        def constraints(self, *args):
            return None, None, "ready" if self.config.bound < 2 else "pending_command_violation", []

    row = panda_record()
    row["interface_config"]["command_delay_steps"] = 0
    sample = {
        "measured_q": [0.0] * 7,
        "measured_qd": [0.0] * 7,
        "estimated_parameters": [0.0] * 8,
        "filter": {"reason": "pending_command_violation"},
    }
    report = panda_prefix_reasons(row, [sample], Regressor(), Filter, Config)
    assert report["reasons_by_maximum_delay_steps"] == [
        "ready",
        "ready",
        "pending_command_violation",
        "pending_command_violation",
        "pending_command_violation",
    ]
    assert report["actual_delay_bound_reason"] == "ready"
