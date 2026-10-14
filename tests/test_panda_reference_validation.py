import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from sarrl.evaluation.panda_reference_validation import (
    ARMS,
    CONDITIONS,
    evaluate_gate,
    paired_bootstrap,
)


def synthetic_rows():
    return [
        {
            "scenario": c,
            "seed": seed,
            "arm": arm,
            "success": True,
            "unsafe_episode": False,
            "tracking_rms_rad": [0.0] * 7,
            "joint_error_max_rad": 0.0,
            "delay_estimator": {"final_delay_correct": True},
        }
        for c in CONDITIONS
        for seed in (9805209, 9805210)
        for arm in ARMS
    ]


def test_reference_gate_checks_individual_joints_and_physical_flags():
    rows = synthetic_rows()
    assert evaluate_gate(rows)["reference_passes"]
    target = next(r for r in rows if r["arm"] == "bank_align")
    target["tracking_rms_rad"][2] = 0.00022
    result = evaluate_gate(rows)
    assert not result["reference_passes"]
    assert result["failures"][0]["joint"] == 3
    target["tracking_rms_rad"][2] = 0.0
    target["unsafe_episode"] = True
    assert not evaluate_gate(rows)["reference_passes"]


def test_gate_requires_complete_cells_finite_metrics_and_age_correctness():
    rows = synthetic_rows()
    for invalid in (rows[:-1], rows + rows[:1]):
        with pytest.raises(ValueError, match="factorial"):
            evaluate_gate(invalid)
    target = next(r for r in rows if r["arm"] == "bank_align")
    target["joint_error_max_rad"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_gate(rows)
    target["joint_error_max_rad"] = 0.0
    target["delay_estimator"]["final_delay_correct"] = False
    assert not evaluate_gate(rows)["reference_passes"]


def test_bootstrap_resamples_whole_seed_blocks_across_conditions():
    rows = synthetic_rows()
    for row in rows:
        row["joint_error_max_rad"] = 0.002
        if row["arm"] == "bank_align":
            row["joint_error_max_rad"] += 0.001 if row["scenario"] in CONDITIONS[:10] else -0.001
    result = paired_bootstrap(rows)["bank_align-adaptive/joint_error_max_rad"]
    assert result["paired_seed_count"] == 2
    np.testing.assert_allclose(result["percentile_95_interval_rad"], [0.0, 0.0], atol=1e-15)
    assert paired_bootstrap(list(reversed(rows))) == paired_bootstrap(rows)


def test_validation_preserves_frozen_grid_episode_arithmetic():
    pytest.importorskip("mujoco")
    from sarrl.evaluation.panda_posture_delay import run_episode as original
    from sarrl.evaluation.panda_reference_validation import run_episode

    first = original(9805209, "wide_noise_d30", "bank_align", retain_trace=True)
    assert run_episode(9805209, "wide_noise_d30", "bank_align", retain_trace=True) == first


def test_profile_replay_never_steps_a_physical_plant(monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    from tools.profile_panda_reference import profile_traces, timing_summary

    root = Path("results/panda_posture_delay_pilot")
    runtime = json.loads((root / "manifest.json").read_text())["runtime"]
    if mujoco.__version__ != runtime["mujoco"] or np.__version__ != runtime["numpy"]:
        pytest.skip("retained profile replay uses the recorded runtime")

    def prohibited(*args, **kwargs):
        raise AssertionError("profiling must not advance the plant")

    monkeypatch.setattr(mujoco, "mj_step", prohibited)
    with gzip.open(root / "traces.json.gz", "rt") as file:
        item = next(
            t for t in json.load(file) if t["cell"] == ["wide_noise_d30", 9804900, "bank_align"]
        )
    rows = json.loads((root / "episodes.json").read_text())
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in rows}
    ticks = iter(range(0, 2_000_000_000, 1_000_000))
    timings = profile_traces([item], lookup, repeats=1, clock=lambda: next(ticks))
    assert len(timings) == 300
    assert timing_summary(timings)["p99_ms"] == 1.0
    damaged = copy.deepcopy(timings)
    damaged[0]["duration_ns"] = -1
    with pytest.raises(ValueError, match="positive"):
        timing_summary(damaged)


def test_runner_refuses_alternate_seed_destination_and_invalid_parallelism(tmp_path):
    from tools.run_panda_reference_validation import main

    for arguments in (["--output", str(tmp_path / "other")], ["--workers", "5"]):
        with pytest.raises(SystemExit) as error:
            main(arguments)
        assert error.value.code == 2
