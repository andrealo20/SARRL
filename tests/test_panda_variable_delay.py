import copy
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from sarrl.envs.panda_interface import PandaControlInterface, PandaInterfaceConfig
from sarrl.envs.panda_packet_delay import (
    PacketDelayQueue,
    PandaPacketInterface,
    applied_ages,
    send_delay,
)
from sarrl.evaluation.panda_posture_delay import run_episode as previous_episode
from sarrl.evaluation.panda_variable_delay import ARMS, CONDITIONS, run_episode, summarize
from tools import run_panda_variable_delay as runner


@pytest.mark.parametrize("lag", [1, 4])
def test_constant_packet_transport_matches_the_original_zero_start_fifo(lag):
    profile = "fixed10" if lag == 1 else "fixed40"
    config = PandaInterfaceConfig(command_delay_steps=lag)
    original = PandaControlInterface(config, 9805000)
    packet = PandaPacketInterface(config, 9805000, profile, np.zeros((4, 7)))
    for k in range(120):
        command = np.ones(7) * (k % 6)
        np.testing.assert_array_equal(
            packet.deliver(command, np.ones(7) * 4), original.deliver(command, np.ones(7) * 4)
        )
    assert packet.transport.discarded == 0


@pytest.mark.parametrize("profile", ["up", "down"])
def test_arrivals_hold_or_discard_without_replaying_an_older_timestamp(profile):
    history = np.array([np.ones(7) * i for i in range(-4, 0)])
    transport = PacketDelayQueue(profile, history)
    previous = -10
    for k, age in enumerate(applied_ages(profile, 150)):
        torque, metadata = transport.deliver(np.ones(7) * k)
        index = metadata["delivered_command_index"]
        assert index >= previous
        assert metadata["applied_age_steps"] == age
        np.testing.assert_array_equal(torque, np.ones(7) * index)
        previous = index
    if profile == "up":
        assert applied_ages(profile)[100:105] == (1, 2, 3, 4, 4)
        assert transport.discarded == 0
    else:
        assert applied_ages(profile)[100:103] == (4, 1, 1)
        assert transport.discarded == 3


def test_packet_validation():
    with pytest.raises(ValueError, match="profile"):
        send_delay("unknown", 0)
    with pytest.raises(ValueError, match="integer"):
        send_delay("up", True)
    with pytest.raises(ValueError, match="four finite"):
        PacketDelayQueue("up", np.zeros((3, 7)))
    with pytest.raises(ValueError, match="initial delay"):
        PandaPacketInterface(PandaInterfaceConfig(), 9805000, "up", np.zeros((4, 7)))
    packet = PandaPacketInterface(
        PandaInterfaceConfig(command_delay_steps=1), 9805000, "up", np.ones((4, 7)) * 10
    )
    with pytest.raises(ValueError, match="already be clipped"):
        packet.deliver(np.zeros(7), np.ones(7))


@pytest.mark.parametrize("arm", ["adaptive", "bank_align"])
def test_cold_constant_case_preserves_the_previous_wide_noisy_rollout(arm):
    old, old_trace = previous_episode(9805001, "wide_noise_d40", arm, retain_trace=True)
    new, trace = run_episode(9805001, "cold_noise_fixed40", arm, retain_trace=True)
    for field, value in old.items():
        if field == "scenario":
            continue
        if field == "delay_estimator":
            assert all(
                new[field][key] == v
                for key, v in value.items()
                if key != "final_correct_run_start_s"
            )
            assert new[field]["final_regime_acquired_at_s"] == value["final_correct_run_start_s"]
        else:
            assert new[field] == value
    for a, b in zip(old_trace, trace, strict=True):
        assert all(b[key] == value for key, value in a.items())


@pytest.fixture(scope="module")
def paired_results():
    return {(c, a): run_episode(9805020, c, a, retain_trace=True) for c in CONDITIONS for a in ARMS}


def test_full_factorial_handoff_pairing_and_physical_accounting(paired_results):
    rows = [r for r, _ in paired_results.values()]
    summary = summarize(rows)
    assert summary["episodes"] == 64
    assert summary["physical_steps"] == 99200
    assert summary["warmup_physical_steps"] == 3200
    for condition in CONDITIONS:
        baseline, _ = paired_results[condition, "fixed"]
        for arm in ARMS:
            row, trace = paired_results[condition, arm]
            assert row["initial_q"] == baseline["initial_q"]
            assert row["initial_command_history"] == baseline["initial_command_history"]
            assert row["measurement_samples"] == 301
            assert row["privileged_delay"] == (arm == "known_align")
            assert not row["privileged_parameters"]
            if condition.startswith("warm"):
                assert row["physical_steps"] == 1600
                assert len(trace[0]["warmup_trace"]) == 21
                assert trace[0]["warmup_trace"][-1]["q"] == trace[0]["q"]
                assert trace[0]["warmup_trace"][-1]["qd"] == trace[0]["qd"]
                assert np.any(np.array(row["initial_command_history"]) != 0)
                assert np.any(np.abs(row["initial_qd"]) > 1e-6)
            else:
                assert row["physical_steps"] == 1500
            if row["estimator"] is not None:
                assert trace[0]["estimated_parameters"] == [0.0] * 8


def test_known_diagnostic_aligns_to_applied_age_instead_of_requested_packet_delay(paired_results):
    row, trace = paired_results["warm_noise_up", "known_align"]
    ages = applied_ages("up")
    assert row["delay_estimator"]["selected_history"] == list(ages)
    assert row["delay_estimator"]["correct_selection_fraction"] == 1.0
    assert trace[101]["sent_packet_delay_steps"] == 4
    assert trace[101]["applied_age_steps"] == trace[101]["selected_delay_steps"] == 2
    assert row["estimator"]["updates"] == 300


def test_future_switch_does_not_change_pre_switch_states_or_commands(paired_results):
    for startup in ("cold", "warm"):
        for changing, constant in (("up", "fixed10"), ("down", "fixed40")):
            for arm in ARMS:
                _, a = paired_results[f"{startup}_noise_{changing}", arm]
                _, b = paired_results[f"{startup}_noise_{constant}", arm]
                for first, second in zip(a[:101], b[:101], strict=True):
                    for field in ("q", "qd", "torque_nm", "delivered_torque_nm"):
                        assert first[field] == second[field]


def test_prepared_state_and_interface_guards():
    from sarrl.envs.panda_reach import PandaReachEnv
    from sarrl.evaluation.panda_adaptation import run_episode as base_episode

    with pytest.raises(ValueError, match="prepared Panda environment"):
        base_episode(9805000, "payload_friction", "fixed", prepared_env=PandaReachEnv())
    interface = PandaControlInterface(PandaInterfaceConfig(), 9805000)
    with pytest.raises(ValueError, match="prepared Panda interface"):
        base_episode(9805000, "payload_friction", "fixed", prepared_interface=interface)
    with pytest.raises(ValueError, match="phase split"):
        base_episode(9805000, "payload_friction", "fixed", phase_split_step=300)


def test_summary_rejects_handoff_noise_schedule_and_privilege_mismatches(paired_results):
    original = [r for r, _ in paired_results.values()]
    for field, value, match in (
        ("initial_q", [0.0] * 7, "handoff"),
        ("measurement_noise_sha256", "changed", "measurement noise"),
        ("applied_age_sha256", "changed", "age schedule"),
        ("privileged_delay", False, "privilege"),
    ):
        rows = copy.deepcopy(original)
        rows[-1][field] = value
        with pytest.raises(ValueError, match=match):
            summarize(rows)


def test_shared_warmup_violation_is_not_discarded_from_success(monkeypatch):
    import sarrl.evaluation.panda_variable_delay as module

    prepare = module._prepare

    def unsafe(*args):
        values = prepare(*args)
        values[6]["diagnostics"].update(unsafe_episode=True, contact_substeps=1)
        return values

    monkeypatch.setattr(module, "_prepare", unsafe)
    row, _ = module.run_episode(9805021, "warm_clean_fixed10", "known_align")
    assert row["unsafe_episode"] and not row["success"]
    assert row["contact_substeps"] >= 1


def test_serial_parallel_records_and_traces_match(tmp_path):
    planned = [
        ("warm_noise_up", 9805040, "bank_align"),
        ("warm_noise_down", 9805041, "known_align"),
    ]
    serial = runner.execute_cells(tmp_path / "serial", planned, 9805040, workers=1)
    parallel = runner.execute_cells(tmp_path / "parallel", planned, 9805040, workers=2)
    assert serial == parallel


def test_previous_reproduction_mismatch_blocks_new_cells(tmp_path, monkeypatch):
    reference = json.loads(Path("results/panda_posture_delay_pilot/episodes.json").read_text())
    payloads = [{"record": copy.deepcopy(r), "trace": []} for r in reference]
    payloads[0]["record"]["measurement_noise_sha256"] = "changed"
    monkeypatch.setattr(runner, "execute_cells", lambda *args, **kwargs: payloads)
    with pytest.raises(RuntimeError, match="no new seed opened"):
        runner.reproduce_previous(tmp_path)
    assert json.loads((tmp_path / "reproduction.json").read_text())["matched"] == 2799


def test_output_seed_and_worker_guards(tmp_path):
    for args in (["--seed-start", "9804900"], ["--workers", "5"], ["--output", str(tmp_path)]):
        with pytest.raises(SystemExit) as exc:
            runner.main(args)
        assert exc.value.code == 2
