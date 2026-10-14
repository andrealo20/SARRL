import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from sarrl.evaluation.panda_latency_diagnosis import (
    bounded_fit,
    information_diagnostics,
    replay_bank,
    suffix_selection,
)
from tools.diagnose_panda_latency import main


def test_suffix_reset_discards_prior_score_penalty():
    innovations = np.array([[1, 100], [1, 100], [1, 100], [10, 0], [10, 0]])
    assert np.argmin(innovations.sum(axis=0)) == 0
    np.testing.assert_array_equal(suffix_selection(innovations, 3), [1, 1])
    with pytest.raises(ValueError, match="nonnegative"):
        suffix_selection(np.array([[1, -1]]), 0)


def test_phase_fit_can_hide_alignment_error_in_parameter_shift():
    design = np.vstack([np.eye(8), 2 * np.eye(8)])
    truth = np.full(8, 0.5)
    shift = np.full(8, 0.2)
    correct = design @ truth
    wrong = design @ (truth + shift)
    fit_correct = bounded_fit(design, correct, np.ones(8))
    fit_wrong = bounded_fit(design, wrong, np.ones(8))
    assert fit_correct["mean_squared_nm2"] < 1e-20
    assert fit_wrong["mean_squared_nm2"] < 1e-20
    assert np.mean((wrong - design @ truth) ** 2) > 0.01
    np.testing.assert_allclose(fit_wrong["parameters"], truth + shift)


def test_information_handles_missing_excitation_and_column_units():
    matrix = np.eye(8)
    complete = information_diagnostics(matrix)
    changed_units = information_diagnostics(matrix * np.arange(1, 9))
    assert complete["rank"] == changed_units["rank"] == 8
    assert complete["normalized_condition"] == pytest.approx(changed_units["normalized_condition"])
    matrix[:, -1] = 0
    missing = information_diagnostics(matrix)
    assert missing["rank"] == missing["normalized_rank"] == 7
    assert missing["condition"] is missing["normalized_condition"] is None
    assert missing["column_norms"][-1] == 0


def test_diagnosis_refuses_existing_output(tmp_path):
    protected = tmp_path / "report"
    protected.mkdir()
    marker = protected / "complete.json"
    marker.write_text("retained")
    with pytest.raises(SystemExit):
        main(["--output", str(protected)])
    assert marker.read_text() == "retained"


def test_retained_bank_replay_never_steps_physical_plant(monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    directory = Path("results/panda_score_memory_pilot")
    runtime = json.loads((directory / "manifest.json").read_text())["runtime"]
    if mujoco.__version__ != runtime["mujoco"] or np.__version__ != runtime["numpy"]:
        pytest.skip("exact retained replay uses the recorded numerical runtime")

    def prohibited(*args, **kwargs):
        raise AssertionError("saved-data diagnosis must not advance a plant")

    monkeypatch.setattr(mujoco, "mj_step", prohibited)
    with gzip.open(directory / "traces.json.gz", "rt") as file:
        traces = json.load(file)
    item = next(item for item in traces if item["cell"] == ["cold_clean_up", 9805300, "bank_align"])
    rows = json.loads((directory / "episodes.json").read_text())
    row = next(row for row in rows if [row["scenario"], row["seed"], row["arm"]] == item["cell"])
    report = replay_bank(row, item["trace"])
    assert report["seed"] == 9805300
    assert set(report["phases"]) == {"before", "moving_after", "settling"}
    np.testing.assert_allclose(
        report["candidate_parameters_final"],
        row["delay_estimator"]["candidate_estimates"],
        rtol=1e-9,
        atol=1e-10,
    )


def test_retained_replay_accepts_roundoff_and_rejects_parameter_corruption():
    import copy

    mujoco = pytest.importorskip("mujoco")
    directory = Path("results/panda_score_memory_pilot")
    runtime = json.loads((directory / "manifest.json").read_text())["runtime"]
    if mujoco.__version__ != runtime["mujoco"] or np.__version__ != runtime["numpy"]:
        pytest.skip("retained replay uses the recorded numerical runtime")
    with gzip.open(directory / "traces.json.gz", "rt") as file:
        item = next(
            item
            for item in json.load(file)
            if item["cell"] == ["cold_clean_up", 9805300, "bank_align"]
        )
    rows = json.loads((directory / "episodes.json").read_text())
    row = next(row for row in rows if [row["scenario"], row["seed"], row["arm"]] == item["cell"])
    trace = copy.deepcopy(item["trace"])
    trace[0]["estimated_parameters"][0] += 1e-14
    replay_bank(row, trace)
    trace[0]["estimated_parameters"][0] += 1e-6
    with pytest.raises(AssertionError):
        replay_bank(row, trace)
