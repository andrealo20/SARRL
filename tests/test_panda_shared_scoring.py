import copy

import numpy as np
import pytest

from sarrl.controllers.panda_score_memory import PandaScoreMemoryController
from sarrl.controllers.panda_shared_scoring import (
    PandaSharedScoreConfig,
    PandaSharedScoreController,
)

pytest.importorskip("mujoco")


@pytest.mark.parametrize(
    "config",
    [
        PandaSharedScoreConfig(True, 25),
        PandaSharedScoreConfig(1, 25),
        PandaSharedScoreConfig(50, 0),
    ],
)
def test_invalid_shared_configuration(config):
    with pytest.raises(ValueError):
        PandaSharedScoreController(config)


def test_bootstrap_prefix_keeps_original_controller_arithmetic():
    old = PandaScoreMemoryController()
    new = PandaSharedScoreController(PandaSharedScoreConfig(1000, 25))
    rng = np.random.default_rng(9805211)
    for _ in range(60):
        q = np.array([0.0, -0.5, 0.0, -1.7, 0.0, 1.2, 0.7]) + rng.normal(0, 0.01, 7)
        v = rng.normal(0, 0.05, 7)
        nq, nv, sent = q + v * 0.01, v + rng.normal(0, 0.02, 7), rng.uniform(-2, 2, 7)
        for c in (old, new):
            c.observe_transition(q, v, nq, nv, sent, 0.01)
        np.testing.assert_array_equal(old.scores, new.scores)
        assert old.selected_history == new.selected_history
        for a, b in zip(old.estimators, new.estimators, strict=True):
            np.testing.assert_array_equal(a.parameters, b.parameters)
            np.testing.assert_array_equal(a.covariance, b.covariance)
    assert new.shared_parameters is None
    assert new.scoring_parameter_history == [None] * 60


def test_shared_scores_do_not_use_candidate_parameter_compensation(monkeypatch):
    c = PandaSharedScoreController()
    c.shared_parameters = np.full(8, 0.2)
    monkeypatch.setattr(c.regressor, "evaluate", lambda *args: (np.zeros(7), np.eye(7, 8)))
    for index, estimator in enumerate(c.estimators):
        estimator.parameters[:] = index / 4
    c.observe_transition(np.zeros(7), np.zeros(7), np.zeros(7), np.zeros(7), np.zeros(7), 0.01)
    np.testing.assert_allclose(c.innovation_history[0], [0.28] * 5)
    np.testing.assert_array_equal(c.shared_parameters, np.full(8, 0.2))
    assert all(estimator.updates == 1 for estimator in c.estimators)


def test_shared_phase_rejects_invalid_inputs_without_history_mutation():
    c = PandaSharedScoreController()
    c.shared_parameters = np.zeros(8)
    before = copy.deepcopy(c.sent_history)
    with pytest.raises(ValueError):
        c.observe_transition(
            np.zeros(7), np.zeros(7), np.zeros(7), np.zeros(7), np.full(7, 100.0), 0.01
        )
    assert c.sent_history == before
    assert c.score_memory.updates == 0


def test_future_transport_profile_cannot_change_prefix():
    from sarrl.evaluation.panda_variable_delay import run_episode

    pair = []
    for profile in ("warm_noise_down", "warm_noise_fixed40"):
        c = PandaSharedScoreController()
        record, trace = run_episode(
            9805212, profile, "bank_align", retain_trace=True, bank_factory=lambda c=c: c
        )

        assert record["delay_estimator"]["shared_scoring_active"]
        assert record["delay_estimator"]["calibrated_at_update"] >= 50
        assert record["delay_estimator"]["calibration_information_rank"] == 8
        pair.append((record, trace, c))
    assert pair[0][0]["measurement_noise_sha256"] == pair[1][0]["measurement_noise_sha256"]
    for a, b in zip(pair[0][1][:101], pair[1][1][:101], strict=True):
        for field in ("q", "qd", "torque_nm", "estimated_parameters", "delay_scores"):
            assert a[field] == b[field]
    np.testing.assert_array_equal(pair[0][2].shared_parameters, pair[1][2].shared_parameters)


def test_acquisition_is_readonly_and_survives_later_candidate_updates(monkeypatch):
    c = PandaSharedScoreController()
    rng = np.random.default_rng(9805211)
    monkeypatch.setattr(
        c.regressor, "evaluate", lambda *args: (np.zeros(7), rng.normal(size=(7, 8)))
    )
    for update in range(50):
        c.observe_transition(*[np.zeros(7) for _ in range(5)], 0.01)
        assert (c.shared_parameters is None) == (update < 49)
    assert c.calibrated_at_update == 50
    assert c.calibration_information_rank == 8
    copied = c.shared_parameters.copy()
    assert not c.shared_parameters.flags.writeable
    for estimator in c.estimators:
        assert not np.shares_memory(c.shared_parameters, estimator.parameters)
        estimator.parameters[:] = 0.1
    c.observe_transition(*[np.zeros(7) for _ in range(5)], 0.01)
    np.testing.assert_array_equal(c.shared_parameters, copied)
    assert c.scoring_parameter_history[:50] == [None] * 50
    assert c.scoring_parameter_history[50] == copied.tolist()
    assert all(estimator.updates == 51 for estimator in c.estimators)


def test_age_improvement_cannot_hide_one_joint_tracking_regression():
    from sarrl.evaluation.panda_shared_scoring import ARMS, CONDITIONS, evaluate_gate

    rows = [
        {
            "scenario": condition,
            "seed": 9805211,
            "arm": arm,
            "tracking_phase_rms_rad": {"after": [0.0] * 7},
            "joint_error_max_rad": 0.0,
            "unsafe_episode": False,
            "success": True,
            "delay_estimator": {
                "final_delay_correct": not (
                    arm == "bank_exp" and condition.endswith(("_up", "_down"))
                )
            },
        }
        for condition in CONDITIONS
        for arm in ARMS
    ]
    assert evaluate_gate(rows)["candidate_advances"]
    target = next(r for r in rows if r["arm"] == "bank_shared_score")
    target["tracking_phase_rms_rad"]["after"][6] = 0.000101
    decision = evaluate_gate(rows)
    assert not decision["candidate_advances"]
    assert all(f["joint"] == 7 for f in decision["failures"])
    target["arm"] = "bank_joint_memory"
    with pytest.raises(ValueError, match="unexpected arm"):
        evaluate_gate(rows)


def test_reproduction_mismatch_blocks_new_shared_scoring_seeds(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from tools import run_panda_shared_scoring as runner

    rows = json.loads(Path("results/panda_reference_validation/episodes.json").read_text())
    payloads = [{"record": copy.deepcopy(row), "trace": []} for row in rows]
    payloads[0]["record"]["measurement_noise_sha256"] = "changed"
    monkeypatch.setattr(runner, "execute_cells", lambda *args, **kwargs: payloads)
    with pytest.raises(RuntimeError, match="no fresh seed opened"):
        runner.reproduce_previous(tmp_path)
    assert json.loads((tmp_path / "reproduction.json").read_text())["matched"] == 3999


def test_runner_refuses_alternate_outputs_and_invalid_workers(tmp_path):
    from tools.run_panda_shared_scoring import main

    for arguments in (["--output", str(tmp_path / "alternate")], ["--workers", "5"]):
        with pytest.raises(SystemExit) as error:
            main(arguments)
        assert error.value.code == 2
    assert not (tmp_path / "alternate").exists()
