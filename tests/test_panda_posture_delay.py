import copy
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from sarrl.envs.panda_reach import PandaReachEnv
from sarrl.evaluation.panda_delay import run_episode as previous_episode
from sarrl.evaluation.panda_posture_delay import (
    ARMS,
    CONDITIONS,
    MIN_SHIFT_RMS,
    SHIFT_HALF_RANGE,
    condition_config,
    posture_delay_protocol,
    run_episode,
    sample_task,
    summarize,
)
from tools import run_panda_posture_delay as runner


def test_translated_tasks_preserve_the_motion_and_joint_clearance():
    env = PandaReachEnv()
    for seed in range(9804800, 9804810):
        narrow, wide = sample_task(seed, "narrow"), sample_task(seed, "wide")
        shift = np.array(wide.shift)
        assert np.all(np.abs(shift) <= SHIFT_HALF_RANGE)
        assert np.sqrt(np.mean(shift**2)) >= MIN_SHIFT_RMS
        np.testing.assert_allclose(np.array(wide.start) - narrow.start, shift, atol=1e-15)
        np.testing.assert_allclose(np.array(wide.goal) - narrow.goal, shift, atol=1e-15)
        np.testing.assert_allclose(
            np.array(wide.goal) - wide.start, np.array(narrow.goal) - narrow.start, atol=1e-15
        )
        env.reset(start=wide.start, goal=wide.goal)
        assert sample_task(seed, "wide") == wide


def test_grid_protocol_and_bad_conditions():
    protocol = posture_delay_protocol()
    assert len(CONDITIONS) == 20
    assert len(protocol["seeds"]) == 20
    assert protocol["reproduction"]["episodes"] == 560
    for condition in CONDITIONS:
        config = condition_config(condition)
        assert config.command_delay_steps in range(5)
        assert config.position_noise_std_rad == (0.001 if "_noise_" in condition else 0.0)
    with pytest.raises(ValueError, match="condition"):
        condition_config("wide_noise_d50")
    with pytest.raises(ValueError, match="posture"):
        sample_task(9804800, "unknown")
    with pytest.raises(ValueError, match="arm"):
        run_episode(9804800, CONDITIONS[0], "unknown")


@pytest.mark.parametrize("arm", ["fixed", "bank_predict"])
def test_narrow_path_retains_the_previous_delayed_noisy_rollout(arm):
    previous, old_trace = previous_episode(9804820, "noise_delay", arm, retain_trace=True)
    actual, trace = run_episode(9804820, "narrow_noise_d20", arm, retain_trace=True)
    actual.pop("posture")
    actual.pop("task_sampling")
    actual["scenario"] = "noise_delay"
    assert actual == previous
    assert trace == old_trace


@pytest.mark.parametrize("lag", [1, 3, 4])
def test_wide_noisy_grid_uses_frozen_unprivileged_controller(lag):
    row, trace = run_episode(
        9804830 + lag, f"wide_noise_d{lag * 10:02d}", "bank_predict", retain_trace=True
    )
    assert row["delay_estimator"]["selected_delay_steps"] == lag
    assert not row["privileged_delay"] and not row["privileged_parameters"]
    assert len(trace) == 301 and row["physical_steps"] == 1500
    assert row["delay_estimator"]["prediction_virtual_substeps"] > 0
    assert trace[0]["prediction_lookahead_s"] == 0.0


def _paired_summary_fixture():
    previous = json.loads(Path("results/panda_delay_pilot/episodes.json").read_text())
    templates = {
        r["arm"]: r for r in previous if r["scenario"] == "noise_delay" and r["seed"] == 9804700
    }
    records = []
    for condition in CONDITIONS:
        posture = condition.split("_")[0]
        task = sample_task(9804802, posture)
        for arm in ARMS:
            row = copy.deepcopy(templates[arm])
            row.update(
                scenario=condition,
                seed=9804802,
                posture=posture,
                task_sampling=asdict(task),
                start_q=list(task.start),
                goal_q=list(task.goal),
                interface_config=asdict(condition_config(condition)),
                measurement_noise_sha256="noise" if "_noise_" in condition else "clean",
            )
            records.append(row)
    return records


def test_summary_checks_factorial_pairing_and_translation():
    rows = _paired_summary_fixture()
    result = summarize(rows)
    assert result["episodes"] == 140
    assert len(result["cells"]) == 140
    assert len(result["delay_contrasts"]) == 120
    assert len(result["posture_contrasts"]) == 70
    assert all(
        cell["mean_paired_tracking_rms_difference_rad"] == 0.0
        for cell in result["posture_contrasts"].values()
    )
    bad = copy.deepcopy(rows)
    bad[-1]["measurement_noise_sha256"] = "different"
    with pytest.raises(ValueError, match="noise"):
        summarize(bad)
    bad = copy.deepcopy(rows)
    bad[-1]["privileged_delay"] = True
    with pytest.raises(ValueError, match="privileged-delay"):
        summarize(bad)
    bad = copy.deepcopy(rows)
    for row in bad:
        if row["posture"] == "wide":
            row["goal_q"][0] += 0.1
    with pytest.raises(ValueError, match="translated"):
        summarize(bad)
    with pytest.raises(ValueError, match="exactly one"):
        summarize(rows[:-1])


def test_frozen_hash_gate_refuses_a_controller_edit(tmp_path):
    assert runner.verify_frozen_controller()["source_commit"].startswith("b7440fe")
    for name in runner.FROZEN_FILES:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runner.ROOT / name, dest)
    reference = tmp_path / runner.REFERENCE_DIRECTORY
    reference.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        runner.ROOT / runner.REFERENCE_DIRECTORY,
        reference,
        ignore=shutil.ignore_patterns("cells", "reproduction_cells", "campaign.lock"),
    )
    path = tmp_path / runner.FROZEN_FILES[0]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="previous frozen pilot"):
        runner.verify_frozen_controller(tmp_path)


def test_parallel_cells_equal_serial_cells_and_resume_checks_all_saved_data(tmp_path):
    planned = [
        ("narrow_noise_d10", 9804840, "bank_predict"),
        ("wide_noise_d30", 9804841, "bank_predict"),
    ]
    serial = runner.execute_cells(tmp_path / "serial", planned, 9804840, workers=1)
    parallel = runner.execute_cells(tmp_path / "parallel", planned, 9804840, workers=2)
    assert serial == parallel
    assert runner.execute_cells(tmp_path / "parallel", planned, 9804840, workers=2) == serial
    path = runner.cell_path(tmp_path / "parallel", planned[0])
    stored = json.loads(path.read_text())
    stored["payload"]["record"]["success"] = not stored["payload"]["record"]["success"]
    path.write_text(json.dumps(stored))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        runner.execute_cells(tmp_path / "parallel", planned, 9804840, workers=2)


def test_reproduction_mismatch_blocks_new_cells(tmp_path, monkeypatch):
    reference = json.loads(Path("results/panda_delay_pilot/episodes.json").read_text())
    payloads = [{"record": copy.deepcopy(row), "trace": []} for row in reference]
    payloads[0]["record"]["measurement_noise_sha256"] = "changed"
    monkeypatch.setattr(runner, "execute_cells", lambda *args, **kwargs: payloads)
    with pytest.raises(RuntimeError, match="no new seed opened"):
        runner.reproduce_previous(tmp_path)
    report = json.loads((tmp_path / "reproduction.json").read_text())
    assert report["matched"] == 559


def test_runner_refuses_old_seeds_bad_worker_count_and_existing_output(tmp_path):
    for args in (["--seed-start", "9804700"], ["--workers", "5"], ["--output", str(tmp_path)]):
        with pytest.raises(SystemExit) as exc:
            runner.main(args)
        assert exc.value.code == 2
