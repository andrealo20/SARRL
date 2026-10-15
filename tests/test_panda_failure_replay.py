import gzip
import json

import numpy as np
import pytest

from sarrl.evaluation.panda_failure_timing import failure_timing
from tools.recover_panda_failures import mismatches, read_items, select_cases


def selected():
    return [
        {"scenario": scenario, "seed": 9805800 + i, "arm": "bank_filtered", "unsafe_episode": True}
        for scenario in ("fast_clean_d40", "fast_noise_d40")
        for i in range(10)
    ]


def test_replay_comparison_preserves_discrete_types_and_original_float_tolerance():
    expected = {"seed": 9805801, "unsafe": True, "nested": [[1.0, 0.0]]}
    actual = {**expected, "nested": [[1.0 + 1e-12, 1e-12]]}
    assert not mismatches(actual, expected)
    assert mismatches({**actual, "unsafe": 1}, expected) == [".unsafe"]
    assert mismatches({**actual, "seed": 9805801.0}, expected) == [".seed"]
    assert mismatches({**actual, "nested": [[1.0, 1e-5]]}, expected)
    assert mismatches({**actual, "nested": [[1.0, float("nan")]]}, expected)


def test_selection_contains_every_retained_unsafe_case_and_rejects_missing_cells():
    rows = selected()
    rows.append(
        {
            "scenario": "slow_clean_d00",
            "seed": 9805800,
            "arm": "bank_filtered",
            "unsafe_episode": False,
        }
    )
    assert len(select_cases(rows)) == 20
    with pytest.raises(ValueError, match="selection"):
        select_cases(rows[1:])
    with pytest.raises(ValueError, match="selection"):
        select_cases(rows + [rows[0]])


def synthetic(first_step, *, boundary_only=False):
    frames = []
    for index in range(5):
        samples = []
        for substep in range(5):
            bad = index == first_step and (not boundary_only or substep == 4)
            samples.append({"q": [0.0] * 7, "qd": [2.0 if bad else 0.0] * 7, "contacts": 0})
        frames.append(
            {
                "physical_substeps": samples,
                "filter": {"braking": index >= 1, "reason": "pending_command_violation"},
            }
        )
    frames.append({})
    row = {
        "arm": "bank_filtered",
        "scenario": "fast_clean_d40",
        "seed": 9805801,
        "unsafe_episode": True,
        "filter": {"abort_step": 1},
        "interface_config": {"command_delay_steps": 2},
        "task_config": {"control_dt": 0.01, "timestep": 0.002},
        "joint_position_excess_rad": 0.0,
        "joint_velocity_excess_rad_s": 1.0,
    }
    return row, frames


@pytest.mark.parametrize(
    "step,phase,command",
    [
        (0, "before_abort", "initial_zero"),
        (2, "pending_after_abort", "pre_abort_control"),
        (3, "braking_delivered", "braking"),
    ],
)
def test_first_violation_uses_the_delivered_command_interval(step, phase, command):
    row, frames = synthetic(step)
    report = failure_timing(row, frames, -np.ones(7), np.ones(7), np.ones(7))
    assert report["first_violation_phase"] == phase
    assert report["first_violation_delivered_command_kind"] == command
    assert report["unsafe_substeps_by_phase"][phase] == 5
    assert report["first_violation_velocity_joint_indices"] == list(range(1, 8))


def test_sample_at_braking_arrival_boundary_belongs_to_previous_command():
    row, frames = synthetic(2, boundary_only=True)
    report = failure_timing(row, frames, -np.ones(7), np.ones(7), np.ones(7))
    assert report["first_physical_violation"]["time_s"] == 0.03
    assert report["first_braking_delivery_time_s"] == 0.03
    assert report["first_violation_phase"] == "pending_after_abort"


def test_replay_timing_rejects_nonfinite_substep_data():
    row, frames = synthetic(2)
    frames[0]["physical_substeps"][0]["qd"][0] = float("nan")
    with pytest.raises(ValueError, match="physical substep"):
        failure_timing(row, frames, -np.ones(7), np.ones(7), np.ones(7))


def test_public_evidence_does_not_require_local_cell_shards(tmp_path):
    rows = selected()
    (tmp_path / "complete.json").write_text("{}")
    (tmp_path / "episodes.json").write_text(json.dumps(rows))
    traces = [{"cell": [r["scenario"], r["seed"], r["arm"]], "trace": [{}] * 301} for r in rows]
    (tmp_path / "traces.json.gz").write_bytes(gzip.compress(json.dumps(traces).encode()))
    assert len(read_items(tmp_path, rows)) == 20
    assert not (tmp_path / "cells").exists()
    traces[0]["cell"] = traces[1]["cell"]
    (tmp_path / "traces.json.gz").write_bytes(gzip.compress(json.dumps(traces).encode()))
    with pytest.raises(ValueError, match="selection"):
        read_items(tmp_path, rows)
