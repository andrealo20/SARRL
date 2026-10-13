import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sarrl.controllers.panda_delay import PandaDelayConfig, PandaDelayController  # noqa: E402
from sarrl.envs.panda_interface import PandaInterfaceConfig  # noqa: E402
from sarrl.envs.panda_reach import quintic_reference  # noqa: E402
from sarrl.evaluation.panda_adaptation import run_episode as base_episode  # noqa: E402
from sarrl.evaluation.panda_delay import ARMS, run_episode, summarize  # noqa: E402
from sarrl.evaluation.panda_robustness import CONDITIONS, condition_config  # noqa: E402
from sarrl.robots.panda import HOME, PandaDynamics, PandaModelParameters  # noqa: E402


@pytest.mark.parametrize(
    "config",
    [
        PandaDelayConfig(max_delay_steps=-1),
        PandaDelayConfig(max_delay_steps=True),
        PandaDelayConfig(prediction_substeps=0),
        PandaDelayConfig(control_dt=np.nan),
        PandaDelayConfig(max_position_shift_rad=0),
    ],
)
def test_panda_delay_settings_are_validated(config):
    with pytest.raises(ValueError):
        PandaDelayController(config)


@pytest.mark.parametrize("delay", [-1, 5, True, 1.5])
def test_known_delay_requires_a_valid_bank_member(delay):
    with pytest.raises(ValueError):
        PandaDelayController(known_delay_steps=delay)


def test_known_zero_delay_retains_original_adaptive_records_and_trace_fields():
    interface = condition_config("noise")
    first, trace = base_episode(
        9804600, "payload_friction", "adaptive", retain_trace=True, interface_config=interface
    )
    controller = PandaDelayController(known_delay_steps=0)
    second, other = base_episode(
        9804600,
        "payload_friction",
        "adaptive",
        retain_trace=True,
        interface_config=interface,
        controller_override=controller,
    )
    assert first == {k: second[k] for k in first}
    for a, b in zip(trace, other, strict=True):
        assert a == {k: b[k] for k in a}


def test_alignment_uses_prior_commands_and_zero_startup(monkeypatch):
    controller = PandaDelayController(known_delay_steps=2)
    monkeypatch.setattr(
        controller.regressor, "evaluate", lambda *args: (np.zeros(7), np.zeros((7, 8)))
    )
    targets = []
    monkeypatch.setattr(
        controller.estimator, "update", lambda rows, residual: targets.append(residual.copy())
    )
    commands = [np.ones(7) * v for v in (1.0, 2.0, 3.0, 4.0)]
    for command in commands:
        controller.observe_transition(HOME, np.zeros(7), HOME, np.zeros(7), command, 0.01)
    np.testing.assert_array_equal(targets, [np.zeros(7), np.zeros(7), commands[0], commands[1]])
    assert controller.selected_history == [2] * 4


def test_estimated_forward_acceleration_matches_independent_loaded_engine():
    controller = PandaDelayController()
    parameters = PandaModelParameters(1.6, (1.3,) * 7)
    actual = PandaDynamics(parameters=parameters)
    q, qd = HOME + 0.07, np.linspace(-0.2, 0.2, 7)
    torque = actual.inverse_dynamics(q, qd, np.linspace(-0.3, 0.3, 7))
    actual.data.ctrl[:] = torque
    mujoco.mj_forward(actual.model, actual.data)
    np.testing.assert_allclose(
        controller.estimated_acceleration(q, qd, torque, parameters.vector()),
        actual.data.qacc,
        atol=1e-10,
    )


def test_prediction_walks_the_pending_queue_in_execution_order(monkeypatch):
    controller = PandaDelayController(known_delay_steps=2, predict=True)
    controller.sent_history = [np.ones(7) * v for v in (1.0, 2.0, 3.0)]
    applied = []

    def interval(state, torque, initial):
        applied.append(torque.copy())
        return state.copy()

    monkeypatch.setattr(controller, "_integrate_interval", interval)
    q, qd, horizon = controller.prepare_control(HOME, np.zeros(7))
    np.testing.assert_array_equal(applied, [np.ones(7) * 2.0, np.ones(7) * 3.0])
    np.testing.assert_array_equal(q, HOME)
    assert horizon == 0.02
    controller.sent_history = []
    applied.clear()
    controller.prepare_control(HOME, np.zeros(7))
    np.testing.assert_array_equal(applied, np.zeros((2, 7)))


def test_prediction_holds_equilibrium_on_its_private_estimated_model():
    controller = PandaDelayController(known_delay_steps=2, predict=True)
    controller.estimator.parameters[:] = [1.4] + [1.2] * 7
    gravity = controller.dynamics.inverse_dynamics(HOME, np.zeros(7), np.zeros(7))
    controller.sent_history = [gravity.copy(), gravity.copy()]
    q, qd, horizon = controller.prepare_control(HOME, np.zeros(7))
    np.testing.assert_allclose(q, HOME, atol=1e-12)
    np.testing.assert_allclose(qd, 0.0, atol=1e-12)
    assert horizon == 0.02
    assert controller.prediction_substeps == 10
    assert controller.prediction_fallbacks == 0


def test_bad_prediction_falls_back_with_current_time_reference(monkeypatch):
    controller = PandaDelayController(known_delay_steps=2, predict=True)

    def fail(*args):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(controller, "_integrate_interval", fail)
    q, qd, horizon = controller.prepare_control(HOME, np.zeros(7))
    np.testing.assert_array_equal(q, HOME)
    np.testing.assert_array_equal(qd, np.zeros(7))
    assert horizon == 0.0
    assert controller.prediction_fallbacks == 1
    assert controller.last_prediction_fallback


def test_bad_transitions_do_not_enter_command_history():
    controller = PandaDelayController()
    with pytest.raises(ValueError, match="control period"):
        controller.observe_transition(HOME, np.zeros(7), HOME, np.zeros(7), np.zeros(7), 0.02)
    with pytest.raises(ValueError, match="clipped"):
        controller.observe_transition(HOME, np.zeros(7), HOME, np.zeros(7), np.ones(7) * 100, 0.01)
    assert controller.sent_history == []
    assert all(e.updates == 0 for e in controller.estimators)


@pytest.mark.parametrize("lag", range(5))
def test_bank_identifies_delays_across_its_grid_without_true_delay_input(lag):
    controller = PandaDelayController()
    row, _ = base_episode(
        9804610 + lag,
        "payload_friction",
        "adaptive",
        interface_config=PandaInterfaceConfig(command_delay_steps=lag),
        controller_override=controller,
    )
    assert controller.known_delay_steps is None
    assert controller.selected_delay_steps == lag
    assert all(e.updates == 300 for e in controller.estimators)
    assert row["estimator"]["updates"] == 300


@pytest.fixture(scope="module")
def paired_delay_results():
    return {
        (condition, arm): run_episode(9804620, condition, arm, retain_trace=True)
        for condition in CONDITIONS
        for arm in ARMS
    }


def test_delay_comparisons_share_task_parameters_and_sensor_noise(paired_delay_results):
    rows = [value[0] for value in paired_delay_results.values()]
    summary = summarize(rows)
    assert summary["episodes"] == 28
    assert summary["physical_steps"] == 42000
    for row in rows:
        assert row["interface_config"] == asdict(condition_config(row["scenario"]))
        assert row["privileged_delay"] == (row["arm"] in ("known_align", "known_predict"))
        assert row["privileged_parameters"] == (row["arm"] == "oracle")
        if row["estimator"] is not None:
            assert (
                paired_delay_results[row["scenario"], row["arm"]][1][0]["estimated_parameters"]
                == [0.0] * 8
            )


def test_prediction_advances_reference_but_tracking_stays_at_current_time(paired_delay_results):
    row, trace = paired_delay_results["noise_delay", "known_predict"]
    assert trace[0]["prediction_lookahead_s"] == 0.02
    errors = []
    for sample in trace[:-1]:
        current, _, _ = quintic_reference(row["start_q"], row["goal_q"], sample["time_s"], 2.0)
        future, _, _ = quintic_reference(
            row["start_q"], row["goal_q"], sample["time_s"] + sample["prediction_lookahead_s"], 2.0
        )
        np.testing.assert_array_equal(sample["reference"], current)
        np.testing.assert_array_equal(sample["control_reference_q"], future)
        errors.append(np.array(sample["q"]) - current)
    np.testing.assert_allclose(
        row["tracking_rms_rad"], np.sqrt(np.mean(np.square(errors), axis=0)), rtol=1e-12
    )


def test_delay_summary_refuses_privilege_or_pairing_changes(paired_delay_results):
    rows = copy.deepcopy([v[0] for v in paired_delay_results.values()])
    for row in rows:
        if row["scenario"] == "delay":
            row["goal_q"][0] += 0.01
    with pytest.raises(ValueError, match="across delay"):
        summarize(rows)
    rows = copy.deepcopy([v[0] for v in paired_delay_results.values()])
    next(r for r in rows if r["arm"] == "known_align")["privileged_delay"] = False
    with pytest.raises(ValueError, match="privileged-delay"):
        summarize(rows)


def test_bank_prediction_repeats_exactly():
    assert run_episode(9804621, "noise_delay", "bank_predict", retain_trace=True) == run_episode(
        9804621, "noise_delay", "bank_predict", retain_trace=True
    )


def test_delay_runner_refuses_bad_seeds_and_output_replacement(tmp_path):
    from tools.run_panda_delay import main

    with pytest.raises(SystemExit) as exc:
        main(["--seed-start", "9804500", "--output", str(tmp_path / "new")])
    assert exc.value.code == 2
    assert not (tmp_path / "new").exists()
    with pytest.raises(SystemExit) as exc:
        main(["--output", str(tmp_path)])
    assert exc.value.code == 2


def test_delay_reproduction_stops_before_new_cases_after_changed_reference(tmp_path, monkeypatch):
    import tools.run_panda_delay as runner

    reference = json.loads(Path("results/panda_robustness_pilot/episodes.json").read_text())
    payloads = [{"record": copy.deepcopy(r), "trace": []} for r in reference]
    payloads[0]["record"]["measurement_noise_sha256"] = "changed"
    monkeypatch.setattr(runner, "execute_cells", lambda *args, **kwargs: payloads)
    with pytest.raises(RuntimeError, match="no new seed opened"):
        runner.reproduce_previous(tmp_path)
    report = json.loads((tmp_path / "reproduction.json").read_text())
    assert report["matched"] == 239
    assert report["mismatches"] == [
        ["ideal", reference[0]["seed"], "fixed", "measurement_noise_sha256"]
    ]
