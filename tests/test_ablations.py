from pathlib import Path

import pytest

from sarrl.evaluation import (
    context_data_seed,
    planar_id_randomization_dict,
    validate_context_data_range,
)
from tools.run_planar_ablations import (
    CONDITIONS,
    build_protocol,
    register_a2,
    run_a0,
    run_a1,
)


def _protocol(**overrides):
    kwargs = {
        "seeds": [0, 1, 2, 3, 4],
        "steps": 200_000,
        "validation_seed": 20_000,
        "validation_episodes": 30,
        "heldout_seed": 40_000,
        "heldout_episodes": 100,
    }
    kwargs.update(overrides)
    return build_protocol(**kwargs)


def test_v12_ablation_matrix_is_fixed_and_complete():
    assert [condition.key for condition in CONDITIONS] == [
        "A0",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
    ]
    assert CONDITIONS[0].label == "Computed torque"
    assert CONDITIONS[2].label == "Residual SAC"
    assert CONDITIONS[-1].label == "Full adaptive stack"


def test_v12_protocol_freezes_retained_training_and_evaluation_seeds():
    protocol = _protocol()

    assert protocol["training"]["seeds"] == [0, 1, 2, 3, 4]
    assert protocol["training"]["steps_per_seed"] == 200_000

    assert protocol["validation"]["seed_start"] == 20_000
    assert protocol["validation"]["episodes"] == 30

    assert protocol["heldout"]["seed_start"] == 40_000
    assert protocol["heldout"]["episodes_per_policy"] == 100


def test_v12_protocol_freezes_domain_randomization():
    dr = _protocol()["domain_randomization"]

    assert dr == {
        "mass_fraction": 0.15,
        "friction_fraction": 0.30,
        "motor_gain_fraction": 0.15,
        "payload_range": [0.0, 1.0],
        "action_delay_max": 2,
    }


def test_v12_protocol_rejects_validation_heldout_leakage():
    with pytest.raises(
        ValueError,
        match="validation and held-out seed ranges must not overlap",
    ):
        _protocol(
            validation_seed=20_000,
            validation_episodes=100,
            heldout_seed=20_050,
        )


def test_v12_protocol_rejects_duplicate_training_seeds():
    with pytest.raises(ValueError, match="training seeds must be unique"):
        _protocol(seeds=[0, 1, 1, 2])


def test_v12_statistics_require_sample_sd_and_paired_comparison():
    statistics = _protocol()["statistics"]

    assert statistics["multi_seed_spread"] == "sample_sd_ddof_1"
    assert statistics["episode_success_interval"] == "wilson_95"
    assert statistics["paired_comparison"] == "paired_bootstrap_95"


def test_a0_smoke_writes_auditable_outputs(tmp_path):
    run_a0(tmp_path, heldout_seed=40_000, heldout_episodes=3)

    condition = tmp_path / "A0_computed_torque"
    assert (condition / "heldout_episodes.csv").is_file()
    assert (condition / "summary.json").is_file()

    rows = (condition / "heldout_episodes.csv").read_text().splitlines()
    assert len(rows) == 4  # header + 3 episodes
    assert "40000" in rows[1]
    assert "40001" in rows[2]
    assert "40002" in rows[3]


def test_a1_requires_explicit_training_confirmation(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("training must not start without explicit confirmation")

    monkeypatch.setattr("tools.run_planar_ablations.subprocess.run", forbidden)

    run_a1(
        root=Path("."),
        out=tmp_path,
        seeds=[0],
        steps=100,
        validation_seed=20_000,
        validation_episodes=2,
        heldout_seed=40_000,
        heldout_episodes=3,
        confirm_training=False,
    )


def test_a1_confirmed_command_uses_direct_sac_and_randomization(tmp_path, monkeypatch):
    calls = []

    def capture(cmd, cwd, check):
        calls.append((cmd, cwd, check))

    monkeypatch.setattr("tools.run_planar_ablations.subprocess.run", capture)

    run_a1(
        root=Path("."),
        out=tmp_path,
        seeds=[0, 1],
        steps=123,
        validation_seed=20_000,
        validation_episodes=2,
        heldout_seed=40_000,
        heldout_episodes=3,
        confirm_training=True,
    )

    assert len(calls) == 1
    cmd, _, check = calls[0]

    assert check is True
    assert "--mode" in cmd
    assert cmd[cmd.index("--mode") + 1] == "torque"
    assert "--randomize" in cmd
    assert "--resume-existing" in cmd
    assert "--seeds" in cmd
    assert "0" in cmd
    assert "1" in cmd


def test_a2_registers_retained_v11_evidence(tmp_path):
    root = Path(".").resolve()

    register_a2(root, tmp_path)

    record = tmp_path / "A2_residual_sac" / "retained_source.json"
    assert record.is_file()

    text = record.read_text()
    assert '"condition": "A2"' in text
    assert '"source_release": "v1.1.0"' in text
    assert '"reused_without_retraining": true' in text
    assert "9f832614ce8b51c207873ff4861986ab72903115" in text


def test_v12_context_pretraining_protocol_is_frozen():
    context = _protocol()["context_pretraining"]

    assert context == {
        "per_training_seed": True,
        "samples_per_seed": 2000,
        "history": 16,
        "optimization_steps": 1500,
        "data_seed_base": 100000,
        "data_seed_stride": 10000,
        "device": "cpu",
        "encoder": {
            "context_dim": 8,
            "latent_dim": 16,
            "hidden_dim": 64,
        },
        "excitation_action_range": [-0.7, 0.7],
        "supervision_target": "raw_dynamics_context",
        "target_normalization": "none",
        "runtime_ground_truth_access": False,
    }


def test_v12_context_data_seed_ranges_are_independent_and_disjoint_from_evaluation():
    assert context_data_seed(0) == 100000
    assert context_data_seed(1) == 110000
    assert context_data_seed(4) == 140000

    ranges = [validate_context_data_range(seed, 2000) for seed in range(5)]

    for index, (start, end) in enumerate(ranges):
        assert start == 100000 + 10000 * index
        assert end == start + 1999

        # Validation is 20000..20029 and held-out is 40000..40099.
        assert start > 40099

    for left, right in zip(ranges, ranges[1:], strict=False):
        assert left[1] < right[0]


def test_v12_randomization_has_single_canonical_representation():
    assert _protocol()["domain_randomization"] == planar_id_randomization_dict()
