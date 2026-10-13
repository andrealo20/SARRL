import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sarrl.controllers.panda_score_memory import InnovationScoreMemory, PandaScoreMemoryConfig


@pytest.mark.parametrize("rule", ["exponential", "window"])
def test_scores_match_independent_batch_weights_and_causal_prefix(rule):
    cfg = PandaScoreMemoryConfig(rule, 50)
    memory = InnovationScoreMemory(5, cfg)
    values = np.random.default_rng(9805200).uniform(0, 100, (180, 5))
    saved = []
    for k, value in enumerate(values):
        memory.update(value)
        prefix = values[: k + 1]
        expected = (
            prefix[-50:].sum(axis=0)
            if rule == "window"
            else np.sum(prefix * (cfg.decay ** np.arange(k, -1, -1))[:, None], axis=0)
        )
        np.testing.assert_allclose(memory.scores, expected, rtol=1e-13, atol=1e-11)
        saved.append(memory.scores.copy())
    other = InnovationScoreMemory(5, cfg)
    altered = values.copy()
    altered[100:] *= 100
    for k, value in enumerate(altered[:100]):
        other.update(value)
        np.testing.assert_array_equal(other.scores, saved[k])
    assert memory.updates == 180


@pytest.mark.parametrize("rule", ["exponential", "window"])
def test_finite_memory_can_drop_an_old_candidate_advantage(rule):
    memory = InnovationScoreMemory(2, PandaScoreMemoryConfig(rule, 50))
    cumulative = np.zeros(2)
    for _ in range(100):
        memory.update([0, 100])
        cumulative += [0, 100]
    for _ in range(30):
        memory.update([100, 0])
        cumulative += [100, 0]
    assert np.argmin(memory.scores) == 1
    assert np.argmin(cumulative) == 0
    tie = InnovationScoreMemory(2)
    tie.update([1, 1])
    assert np.argmin(tie.scores) == 0


@pytest.mark.parametrize(
    "cfg",
    [
        PandaScoreMemoryConfig("other"),
        PandaScoreMemoryConfig("window", 1),
        PandaScoreMemoryConfig("window", True),
        PandaScoreMemoryConfig("window", 2.5),
    ],
)
def test_memory_configuration_validation(cfg):
    with pytest.raises(ValueError):
        InnovationScoreMemory(5, cfg)


@pytest.mark.parametrize("values", [[-1, 0], [np.nan, 0], [np.inf, 0], [1], [[1, 2]]])
def test_invalid_innovations_leave_memory_unchanged(values):
    memory = InnovationScoreMemory(2, PandaScoreMemoryConfig("window"))
    memory.update([2, 3])
    with pytest.raises(ValueError):
        memory.update(values)
    np.testing.assert_array_equal(memory.scores, [2, 3])
    assert memory.updates == len(memory.recent) == 1


@pytest.mark.parametrize("rule", ["exponential", "window"])
def test_candidate_rls_updates_match_the_frozen_bank_on_identical_transitions(rule):
    pytest.importorskip("mujoco")
    from sarrl.controllers.panda_delay import PandaDelayController
    from sarrl.controllers.panda_score_memory import PandaScoreMemoryController
    from sarrl.robots.panda import HOME

    original = PandaDelayController()
    memory = PandaScoreMemoryController(PandaScoreMemoryConfig(rule))
    rng = np.random.default_rng(9805201)
    for _ in range(80):
        q = HOME + rng.normal(0, 0.05, 7)
        qd = rng.normal(0, 0.1, 7)
        next_q = q + qd * 0.01
        next_qd = qd + rng.normal(0, 0.02, 7)
        sent = rng.uniform(-5, 5, 7)
        for controller in (original, memory):
            controller.observe_transition(q, qd, next_q, next_qd, sent, 0.01)
        for before, after in zip(original.estimators, memory.estimators, strict=True):
            assert before.diagnostics() == after.diagnostics()
    assert not memory.predict and memory.known_delay_steps is None
    assert memory.delay_diagnostics()["score_updates"] == 80
    with pytest.raises(ValueError, match="period"):
        memory.observe_transition(q, qd, next_q, next_qd, sent, 0.02)
    assert memory.delay_diagnostics()["score_updates"] == 80


@pytest.fixture(scope="module")
def factorial():
    pytest.importorskip("mujoco")
    from sarrl.evaluation.panda_score_memory import ARMS, CONDITIONS, run_episode

    return {(c, a): run_episode(9805220, c, a, retain_trace=True) for c in CONDITIONS for a in ARMS}


def test_factorial_pairing_memory_diagnostics_and_physical_counts(factorial):
    from sarrl.evaluation.panda_score_memory import ARMS, CONDITIONS, MEMORY_CONFIGS, summarize

    rows = [row for row, _ in factorial.values()]
    summary = summarize(rows)
    assert summary["episodes"] == 96 and summary["physical_steps"] == 148800
    assert summary["warmup_physical_steps"] == 4800
    for condition in CONDITIONS:
        baseline, _ = factorial[condition, "bank_align"]
        for arm in ARMS:
            row, trace = factorial[condition, arm]
            assert row["initial_q"] == baseline["initial_q"]
            assert row["initial_qd"] == baseline["initial_qd"]
            assert row["initial_command_history"] == baseline["initial_command_history"]
            assert row["measurement_noise_sha256"] == baseline["measurement_noise_sha256"]
            assert row["privileged_delay"] == (arm == "known_align")
            if arm in MEMORY_CONFIGS:
                assert row["delay_estimator"]["score_updates"] == 300
                assert trace[0]["delay_scores"] == [0.0] * 5
                assert trace[0]["estimated_parameters"] == [0.0] * 8
                assert len(trace[0]["score_innovation_squared"]) == 5


def test_future_latency_profile_does_not_change_any_pre_switch_memory_command(factorial):
    for startup in ("cold", "warm"):
        for arm in ("bank_exp", "bank_window"):
            for changing, constant in (("up", "fixed10"), ("down", "fixed40")):
                _, a = factorial[f"{startup}_noise_{changing}", arm]
                _, b = factorial[f"{startup}_noise_{constant}", arm]
                for first, second in zip(a[:101], b[:101], strict=True):
                    for field in ("q", "qd", "torque_nm", "delivered_torque_nm", "delay_scores"):
                        assert first[field] == second[field]


def test_baseline_factory_path_preserves_original_records_and_traces():
    pytest.importorskip("mujoco")
    from sarrl.controllers.panda_delay import PandaDelayController
    from sarrl.evaluation.panda_variable_delay import run_episode

    for condition in ("cold_noise_up", "warm_noise_down"):
        original = run_episode(9805221, condition, "bank_align", retain_trace=True)
        injected = run_episode(
            9805221, condition, "bank_align", retain_trace=True, bank_factory=PandaDelayController
        )
        assert original == injected
    with pytest.raises(ValueError, match="factories"):
        run_episode(9805221, "warm_noise_down", "known_align", bank_factory=PandaDelayController)
    with pytest.raises(ValueError, match="without prediction"):
        run_episode(
            9805221,
            "warm_noise_down",
            "bank_align",
            bank_factory=lambda: PandaDelayController(predict=True),
        )


def test_summary_rejects_wrong_memory_configuration(factorial):
    from sarrl.evaluation.panda_score_memory import summarize

    rows = copy.deepcopy([r for r, _ in factorial.values()])
    rows[-1]["delay_estimator"]["score_memory"]["memory_steps"] = 100
    with pytest.raises(ValueError, match="score-memory"):
        summarize(rows)


def test_serial_parallel_new_variant_records_and_traces_match(tmp_path):
    pytest.importorskip("mujoco")
    from sarrl.evaluation.panda_score_memory import run_episode
    from tools.run_panda_variable_delay import execute_cells

    planned = [("warm_noise_up", 9805240, "bank_exp"), ("warm_noise_down", 9805241, "bank_window")]
    serial = execute_cells(tmp_path / "serial", planned, 9805240, execute=run_episode, workers=1)
    parallel = execute_cells(
        tmp_path / "parallel", planned, 9805240, execute=run_episode, workers=2
    )
    assert serial == parallel


def test_reference_mismatch_blocks_new_score_memory_seeds(tmp_path, monkeypatch):
    pytest.importorskip("mujoco")
    from tools import run_panda_score_memory as runner

    rows = json.loads(Path("results/panda_variable_delay_pilot/episodes.json").read_text())
    payloads = [{"record": copy.deepcopy(r), "trace": []} for r in rows]
    payloads[0]["record"]["measurement_noise_sha256"] = "changed"
    monkeypatch.setattr(runner, "execute_cells", lambda *args, **kwargs: payloads)
    with pytest.raises(RuntimeError, match="no new seed opened"):
        runner.reproduce_previous(tmp_path)
    assert json.loads((tmp_path / "reproduction.json").read_text())["matched"] == 1279


def test_score_memory_runner_guards(tmp_path):
    pytest.importorskip("mujoco")
    from tools import run_panda_score_memory as runner

    for args in (["--seed-start", "9805100"], ["--workers", "5"], ["--output", str(tmp_path)]):
        with pytest.raises(SystemExit) as exc:
            runner.main(args)
        assert exc.value.code == 2
