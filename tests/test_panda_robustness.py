import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sarrl.envs.panda_interface import (  # noqa: E402
    PandaControlInterface,
    PandaInterfaceConfig,
)
from sarrl.evaluation.panda_adaptation import (  # noqa: E402
    reproduction_mismatches,
)
from sarrl.evaluation.panda_adaptation import (  # noqa: E402
    run_episode as previous_episode,
)
from sarrl.evaluation.panda_adaptation import (  # noqa: E402
    summarize as previous_summary,
)
from sarrl.evaluation.panda_robustness import (  # noqa: E402
    ARMS,
    CONDITIONS,
    condition_config,
    run_episode,
    summarize,
)
from sarrl.robots.panda import HOME  # noqa: E402


@pytest.mark.parametrize(
    "config",
    [
        PandaInterfaceConfig(position_noise_std_rad=np.nan),
        PandaInterfaceConfig(velocity_noise_std_rad_s=-0.1),
        PandaInterfaceConfig(command_delay_steps=-1),
        PandaInterfaceConfig(command_delay_steps=1.5),
        PandaInterfaceConfig(command_delay_steps=True),
    ],
)
def test_panda_interface_rejects_invalid_scales_and_delay(config):
    with pytest.raises(ValueError):
        config.validate()


def test_panda_sensor_sample_is_cached_copied_and_reproducible():
    config = condition_config("noise")
    interface = PandaControlInterface(config, 9804400)
    q, qd = interface.measure(HOME, np.zeros(7), 0)
    expected = (q.copy(), qd.copy())
    q[:] = 0.0
    repeated = interface.measure(HOME, np.zeros(7), 0)
    np.testing.assert_array_equal(repeated, expected)
    assert not np.array_equal(expected[0], HOME)
    assert interface.measurements == 1
    again = PandaControlInterface(config, 9804400)
    np.testing.assert_array_equal(again.measure(HOME, np.zeros(7), 0), expected)
    interface.measure(HOME, np.zeros(7), 1)
    again.measure(HOME, np.zeros(7), 1)
    assert interface.diagnostics() == again.diagnostics()
    with pytest.raises(ValueError, match="successive"):
        interface.measure(HOME, np.zeros(7), 0)
    with pytest.raises(ValueError, match="successive"):
        interface.measure(HOME, np.zeros(7), 3)


def test_panda_fifo_has_two_zero_intervals_then_clipped_sent_commands():
    interface = PandaControlInterface(condition_config("delay"), 9804401)
    limit = np.ones(7) * 2.0
    commands = [np.ones(7) * v for v in (4, -4, 1, -1)]
    delivered = [interface.deliver(c, limit) for c in commands]
    np.testing.assert_array_equal(delivered, [np.zeros(7), np.zeros(7), limit, -limit])
    assert interface.commands == 4
    with pytest.raises(ValueError, match="positive"):
        interface.deliver(commands[0], np.zeros(7))
    assert interface.commands == 4


def test_zero_interface_retains_every_previous_record_and_trace_field():
    previous, old_trace = previous_episode(
        9804402, "payload_friction", "adaptive", retain_trace=True
    )
    actual, trace = run_episode(9804402, "ideal", "adaptive", retain_trace=True)
    assert {k: actual[k] for k in previous if k != "scenario"} == {
        k: v for k, v in previous.items() if k != "scenario"
    }
    for old, new in zip(old_trace, trace, strict=True):
        assert old == {k: new[k] for k in old}
    assert actual["measurement_samples"] == 301
    assert actual["actuation_mismatch_steps"] == 0
    assert actual["measurement_noise_rms"] == [0.0] * 14


def test_adaptive_estimator_gets_cached_measurements_and_current_sent_command(monkeypatch):
    import sarrl.evaluation.panda_adaptation as evaluation

    observed = []
    parent = evaluation.PandaAdaptiveController

    class Spy(parent):
        def observe_transition(self, q, qd, q_next, qd_next, sent_command, dt):
            observed.append([v.copy() for v in (q, qd, q_next, qd_next, sent_command)])
            super().observe_transition(q, qd, q_next, qd_next, sent_command, dt)

    monkeypatch.setattr(evaluation, "PandaAdaptiveController", Spy)
    row, trace = run_episode(9804403, "noise_delay", "adaptive", retain_trace=True)
    assert len(observed) == 300
    for index, values in enumerate(observed):
        for actual, expected in zip(
            values,
            [
                trace[index]["measured_q"],
                trace[index]["measured_qd"],
                trace[index + 1]["measured_q"],
                trace[index + 1]["measured_qd"],
                trace[index]["torque_nm"],
            ],
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
    assert trace[0]["measured_q"] != trace[0]["q"]
    assert trace[0]["delivered_torque_nm"] == [0.0] * 7
    assert trace[0]["torque_nm"] != trace[0]["delivered_torque_nm"]
    assert row["measurement_samples"] == 301
    assert row["estimator"]["updates"] == 300


@pytest.fixture(scope="module")
def paired_records():
    return [run_episode(9804404, c, a)[0] for c in CONDITIONS for a in ARMS]


def test_robustness_records_pair_tasks_parameters_and_sensor_noise(paired_records):
    summary = summarize(paired_records)
    assert summary["episodes"] == 12
    assert summary["physical_steps"] == 18000
    assert (
        len(
            {
                r["measurement_noise_sha256"]
                for r in paired_records
                if r["scenario"] in ("noise", "noise_delay")
            }
        )
        == 1
    )
    for row in paired_records:
        assert row["interface_config"] == asdict(condition_config(row["scenario"]))
        assert row["measurement_samples"] == 301
        if row["scenario"] in ("ideal", "noise"):
            assert row["actuation_mismatch_steps"] == 0
        assert row["privileged_parameters"] == (row["arm"] == "oracle")
        expected = (
            row["joint_error_max_rad"] <= 0.01
            and row["tip_error_m"] <= 0.01
            and row["final_speed_max_rad_s"] <= 0.05
            and not row["unsafe_episode"]
        )
        assert row["success"] == expected


def test_robustness_summary_rejects_changed_cross_condition_pairing(paired_records):
    changed = copy.deepcopy(paired_records)
    for row in changed:
        if row["scenario"] == "noise":
            row["start_q"][0] += 0.01
    with pytest.raises(ValueError, match="across interface"):
        summarize(changed)
    changed = copy.deepcopy(paired_records)
    changed[-1]["measurement_noise_sha256"] = "wrong"
    with pytest.raises(ValueError, match="sensor noise"):
        summarize(changed)


def test_noisy_delayed_episode_repeats_exactly():
    first = run_episode(9804405, "noise_delay", "adaptive", retain_trace=True)
    second = run_episode(9804405, "noise_delay", "adaptive", retain_trace=True)
    assert first == second


def test_nested_reproduction_tolerates_roundoff_and_keeps_counters_exact():
    retained = {"estimator": {"theta": [1.0, 0.0], "updates": 300, "guard": False}}
    actual = copy.deepcopy(retained)
    actual["estimator"]["theta"][0] += 1e-12
    assert reproduction_mismatches(actual, retained) == []
    actual["estimator"]["updates"] = 299
    assert reproduction_mismatches(actual, retained) == ["estimator"]


def test_previous_summary_remains_equal_to_retained_summary():
    root = Path("results/panda_adaptation_pilot")
    assert previous_summary(json.loads((root / "episodes.json").read_text())) == json.loads(
        (root / "summary.json").read_text()
    )


def test_robustness_runner_refuses_unregistered_seeds_and_existing_outputs(tmp_path):
    from tools.run_panda_robustness import main

    with pytest.raises(SystemExit) as exc:
        main(["--seed-start", "9804300", "--output", str(tmp_path / "new")])
    assert exc.value.code == 2
    assert not (tmp_path / "new").exists()
    with pytest.raises(SystemExit) as exc:
        main(["--output", str(tmp_path)])
    assert exc.value.code == 2
    with pytest.raises(ValueError):
        condition_config("missing")


def test_changed_reproduction_stops_with_a_retained_mismatch_report(tmp_path, monkeypatch):
    import tools.run_panda_robustness as runner

    reference = json.loads(Path("results/panda_adaptation_pilot/episodes.json").read_text())
    payloads = [{"record": copy.deepcopy(r), "trace": []} for r in reference]
    payloads[0]["record"]["success"] = not reference[0]["success"]
    monkeypatch.setattr(runner, "execute_cells", lambda *args, **kwargs: payloads)
    with pytest.raises(RuntimeError, match="no new seed opened"):
        runner.reproduce_previous(tmp_path)
    report = json.loads((tmp_path / "reproduction.json").read_text())
    assert report["compared"] == 240
    assert report["matched"] == 239
    assert report["mismatches"] == [["nominal", reference[0]["seed"], "fixed", "success"]]
