import json
from pathlib import Path

import numpy as np
import pytest

from sarrl.controllers.panda_filtered import selected_age_set
from sarrl.safety.panda_joint_filter import PandaJointFilter

LAGS = (0, 1, 2, 3, 4)


def _filter():
    return PandaJointFilter(np.full(7, -2.0), np.full(7, 2.0), np.full(7, 20.0))


def test_age_set_keeps_exact_ties_and_rejects_invalid_scores():
    assert selected_age_set(np.zeros(5), LAGS) == [0, 1, 2, 3, 4]
    assert selected_age_set([3.0, 1.0, 1.0, 1.0, 1.0], LAGS) == [1, 2, 3, 4]
    assert selected_age_set([3.0, 2.0, 1.0, 1.0 + 1e-15, 4.0], LAGS) == [2]
    assert selected_age_set([0.0, np.nan, 1.0, 1.0, 1.0], LAGS) == []
    assert selected_age_set([0.0, np.inf, 1.0, 1.0, 1.0], LAGS) == []
    assert selected_age_set(np.zeros(4), LAGS) == []


def test_all_ages_reproduce_the_default_robust_filter_exactly():
    filter = _filter()
    q, v = np.zeros(7), np.full(7, 0.1)
    history = [np.full(7, index) for index in range(4)]
    for candidate in (np.zeros(7), np.full(7, 100.0)):
        robust = filter.filter(q, v, candidate, np.eye(7), np.zeros(7), history)
        explicit = filter.filter(q, v, candidate, np.eye(7), np.zeros(7), history, list(LAGS))
        np.testing.assert_array_equal(robust.torque, explicit.torque)
        assert robust.diagnostics() == explicit.diagnostics()


def test_any_age_subset_keeps_a_command_the_robust_filter_leaves_unchanged():
    filter = _filter()
    q, v = np.zeros(7), np.full(7, 0.05)
    history = [np.full(7, 0.5) for _ in range(4)]
    candidate = np.full(7, 0.2)
    robust = filter.filter(q, v, candidate, np.eye(7), np.zeros(7), history)
    assert robust.reason == "unchanged"
    for ages in ([0], [4], [1, 2], [2, 3, 4]):
        subset = filter.filter(q, v, candidate, np.eye(7), np.zeros(7), history, ages)
        assert subset.reason == "unchanged"
        np.testing.assert_array_equal(subset.torque, robust.torque)


def test_selected_age_reaches_the_projection_where_all_ages_abort():
    filter = _filter()
    q, v = np.zeros(7), np.zeros(7)
    v[0] = filter.velocity[0] - 0.01
    history = [np.full(7, 20.0) for _ in range(4)]
    robust = filter.filter(q, v, np.full(7, -20.0), np.eye(7), np.zeros(7), history)
    assert not robust.predicted_feasible and robust.reason == "pending_command_violation"
    selected = filter.filter(q, v, np.full(7, -20.0), np.eye(7), np.zeros(7), history, [0])
    assert selected.predicted_feasible


def test_age_argument_rejects_empty_or_out_of_range_sets():
    filter = _filter()
    args = (np.zeros(7), np.zeros(7), np.eye(7), np.zeros(7), [np.zeros(7)] * 4)
    for ages in ([], [5], [-1], [1.0]):
        with pytest.raises(ValueError, match="ages"):
            filter.constraints(*args, ages)


def test_invalid_age_set_latches_braking_with_diagnostics():
    pytest.importorskip("mujoco")
    from sarrl.controllers.panda_filtered import PandaFilteredController
    from sarrl.robots.panda import HOME

    controller = PandaFilteredController(selected_ages=True)
    controller.scores[:] = np.nan
    sent = controller.command(HOME, np.zeros(7), HOME)
    assert controller.aborted and controller.abort_step == 0
    assert controller.last_filter["reason"] == "invalid_age_set"
    assert controller.last_filter["braking"] and not controller.last_filter["predicted_feasible"]
    assert np.all(np.abs(sent) <= controller.dynamics.torque_limit)
    diagnostics = controller.filter_diagnostics()
    assert diagnostics["age_set_at_abort"] == [] and diagnostics["age_sets"] == [[]]
    assert "age_sets" not in PandaFilteredController().filter_diagnostics()


def _records(update):
    from sarrl.evaluation.panda_joint_filter import CONDITIONS
    from sarrl.evaluation.panda_selected_filter import ARMS

    rows = []
    for condition in CONDITIONS:
        for arm in ARMS:
            row = {
                "scenario": condition,
                "seed": 9806200,
                "arm": arm,
                "tracking_rms_rad": [0.0] * 7,
                "success": True,
                "unsafe_episode": False,
            }
            if arm != "cumulative":
                true_age = int(condition[-2:]) // 10
                row["filter"] = {
                    "aborted": False,
                    "abort_step": None,
                    "reasons": {"unchanged": 300},
                    "age_sets": [[true_age]] * 300,
                }
            update(row)
            rows.append(row)
    return rows


def test_mechanism_rules_read_the_predicted_pattern():
    from sarrl.evaluation.panda_selected_filter import decide

    def pattern(row):
        fast, delay = row["scenario"].startswith("fast_"), row["scenario"][-3:]
        if row["arm"] == "robust_filter" and fast and delay in ("d20", "d40"):
            row["filter"]["aborted"] = True
        if row["arm"] == "selected_filter" and fast and delay == "d40":
            row["filter"]["aborted"] = True
            row["unsafe_episode"] = True
        if row["arm"] == "robust_filter" and fast and delay == "d40":
            row["unsafe_episode"] = True

    mechanism = decide(_records(pattern))["mechanism"]
    assert mechanism["prediction_A"] and mechanism["aborts_reduced"]
    assert mechanism["prediction_B"] and not mechanism["risk_C_materialised"]
    assert mechanism["no_new_violations"] and mechanism["no_success_loss"]
    assert mechanism["selected_episodes_excluding_true_age"] == 0


def test_paired_rules_catch_a_new_violation_and_a_lost_success():
    from sarrl.evaluation.panda_selected_filter import decide

    def pattern(row):
        if row["arm"] == "selected_filter" and row["scenario"] == "fast_clean_d00":
            row["unsafe_episode"], row["success"] = True, False
            row["filter"]["age_sets"] = [[1]] * 300

    mechanism = decide(_records(pattern))["mechanism"]
    assert mechanism["risk_C_materialised"] and not mechanism["prediction_A"]
    assert not mechanism["no_new_violations"] and not mechanism["no_success_loss"]
    assert mechanism["paired_selected_unsafe_robust_safe"] == 1
    assert mechanism["selected_episodes_excluding_true_age"] == 1
    with pytest.raises(ValueError, match="factorial"):
        decide(_records(lambda row: None)[:-1])


def test_v28_entry_point_reproduces_a_retained_v26_filtered_episode():
    mujoco = pytest.importorskip("mujoco")
    scipy = pytest.importorskip("scipy")
    from sarrl.evaluation.panda_adaptation import reproduction_mismatches
    from sarrl.evaluation.panda_selected_filter import run_episode

    root = Path(__file__).resolve().parents[1] / "results/panda_joint_filter_pilot"
    runtime = json.loads((root / "manifest.json").read_text())["runtime"]
    if (mujoco.__version__, np.__version__, scipy.__version__) != (
        runtime["mujoco"],
        runtime["numpy"],
        runtime["scipy"],
    ):
        pytest.skip("retained v2.6 records reproduce on the recorded runtime")
    path = root / "episodes.json"
    retained = next(
        row
        for row in json.loads(path.read_text())
        if row["arm"] == "bank_filtered" and row["scenario"] == "fast_clean_d20"
    )
    record, _ = run_episode(retained["seed"], retained["scenario"], "bank_filtered")
    assert not reproduction_mismatches(record, retained)


def test_runner_refuses_alternate_outputs_and_invalid_workers(tmp_path):
    from tools.run_panda_selected_filter import main

    for args in (["--output", str(tmp_path / "alternate")], ["--workers", "5"]):
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code == 2
    assert not (tmp_path / "alternate").exists()
