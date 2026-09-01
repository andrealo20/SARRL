from pathlib import Path

import pytest

from sarrl.evaluation import (
    V12_ENSEMBLE_BATCH_SIZE,
    V12_ENSEMBLE_DATA_SEED_BASE,
    V12_ENSEMBLE_DATA_SEED_STRIDE,
    V12_ENSEMBLE_SAMPLES,
    V12_ENSEMBLE_TRAINING_STEPS,
    context_data_seed,
    ensemble_data_seed,
    planar_ensemble_randomization_dict,
    planar_id_randomization_dict,
    validate_context_data_range,
    validate_ensemble_data_range,
)
from tools.run_planar_ablations import (
    CONDITIONS,
    _sha256,
    _validate_a3_context_artifact,
    _validate_a4_ensemble_artifact,
    build_protocol,
    prepare_a4_ensembles,
    register_a2,
    run_a0,
    run_a1,
    run_a3,
    run_a4,
    run_a5,
    run_a6,
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
    assert CONDITIONS[3].label == "Residual SAC + context"
    assert CONDITIONS[3].status == "ready"
    assert CONDITIONS[4].label == "Residual SAC + uncertainty gate"
    assert CONDITIONS[4].status == "ready"
    assert CONDITIONS[5].status == "ready"
    assert CONDITIONS[-1].label == "Full adaptive stack"
    assert CONDITIONS[-1].status == "ready"


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


def test_v12_uncertainty_gate_protocol_is_frozen():
    gate = _protocol()["uncertainty_gate"]

    assert gate == {
        "gain": 4.0,
        "min_scale": 0.1,
        "safety_certificate": False,
        "policy_source": "A2_retained_residual_sac",
        "ensemble_pairing": "one_per_training_seed",
    }


def test_v12_hocbf_and_full_stack_protocols_are_frozen():
    protocol = _protocol()

    assert protocol["hocbf_safety"] == {
        "torque_limit": [40.0, 40.0],
        "joint_lower": [-3.05, -3.05],
        "joint_upper": [3.05, 3.05],
        "velocity_limit": [7.0, 7.0],
        "joint_gamma1": 5.0,
        "joint_gamma2": 5.0,
        "velocity_dt": 0.02,
        "feasibility_tol": 2e-8,
        "obstacles": [],
        "require_safety": True,
        "infeasible_semantics": "abort_episode_unsuccessful",
        "intervention_tolerance": 1e-9,
        "guarantee_scope": "nominal_model_relative",
    }
    assert protocol["full_stack"]["context_action"] == (
        "normalized_raw_policy_residual"
    )
    assert protocol["full_stack"]["physical_command"] == (
        "baseline_plus_gated_residual_then_HOCBF"
    )
    assert protocol["full_stack"]["hocbf_required"] is True


def test_v12_ensemble_pretraining_protocol_is_frozen():
    ensemble = _protocol()["ensemble_pretraining"]

    assert ensemble == {
        "per_training_seed": True,
        "samples_per_seed": 10_000,
        "optimization_steps": 2_000,
        "batch_size": 128,
        "data_seed_base": 200_000,
        "data_seed_stride": 20_000,
        "device": "cpu",
        "ensemble_config": {
            "state_dim": 4,
            "action_dim": 2,
            "output_dim": 2,
            "hidden": [128, 128],
            "ensemble_size": 5,
            "learning_rate": 1e-3,
            "weight_decay": 1e-6,
        },
        "domain_randomization": {
            "mass_fraction": 0.25,
            "friction_fraction": 0.4,
            "motor_gain_fraction": 0.2,
            "payload_range": [0.0, 1.5],
            "action_delay_max": 0,
        },
        "excitation": {
            "distribution": "uniform",
            "low": -30.0,
            "high": 30.0,
            "space": "commanded_torque_nm",
        },
        "supervision": {
            "target": "observed_minus_nominal_acceleration",
            "torque_input": "commanded_torque",
        },
    }

    assert V12_ENSEMBLE_SAMPLES == 10_000
    assert V12_ENSEMBLE_TRAINING_STEPS == 2_000
    assert V12_ENSEMBLE_BATCH_SIZE == 128
    assert V12_ENSEMBLE_DATA_SEED_BASE == 200_000
    assert V12_ENSEMBLE_DATA_SEED_STRIDE == 20_000
    assert planar_ensemble_randomization_dict() == ensemble["domain_randomization"]


def test_v12_ensemble_data_ranges_are_independent_and_disjoint():
    assert ensemble_data_seed(0) == 200_000
    assert ensemble_data_seed(4) == 280_000

    ranges = [validate_ensemble_data_range(seed, 10_000) for seed in range(5)]
    for index, (start, end) in enumerate(ranges):
        assert start == 200_000 + 20_000 * index
        assert end == start + 9_999
        assert start > 40_099

    for left, right in zip(ranges, ranges[1:], strict=False):
        assert left[1] < right[0]


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


def test_a3_requires_explicit_training_confirmation(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A3 training must not start without explicit confirmation")

    monkeypatch.setattr(
        "tools.run_planar_ablations.subprocess.run",
        forbidden,
    )

    run_a3(
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


def test_a3_confirmed_sweep_uses_residual_context_pipeline(
    tmp_path,
    monkeypatch,
):
    calls = []

    context_root = tmp_path / "A3_residual_sac_context" / "contexts"

    def fake_prepare(root, out, seeds):
        assert seeds == [0, 1]
        context_root.mkdir(parents=True, exist_ok=True)
        return context_root

    def capture(cmd, cwd, check):
        calls.append((cmd, cwd, check))

    monkeypatch.setattr(
        "tools.run_planar_ablations.prepare_a3_contexts",
        fake_prepare,
    )
    monkeypatch.setattr(
        "tools.run_planar_ablations.subprocess.run",
        capture,
    )

    run_a3(
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
    assert cmd[cmd.index("--mode") + 1] == "residual"
    assert "--randomize" in cmd
    assert "--resume-existing" in cmd
    assert "--context-root" in cmd
    assert cmd[cmd.index("--context-root") + 1] == str(context_root)

    assert cmd[cmd.index("--start-steps") + 1] == "5000"
    assert cmd[cmd.index("--batch-size") + 1] == "256"
    assert cmd[cmd.index("--replay-capacity") + 1] == "200000"
    assert cmd[cmd.index("--validate-every") + 1] == "25000"

    seed_index = cmd.index("--seeds")
    assert cmd[seed_index + 1 : seed_index + 3] == ["0", "1"]


def test_a3_context_artifact_rejects_different_git_commit(tmp_path):
    context_dir = tmp_path / "context_seed_0"
    context_dir.mkdir()

    (context_dir / "context.pt").write_bytes(b"checkpoint")
    (context_dir / "context.npz").write_bytes(b"dataset")

    (context_dir / "context_manifest.json").write_text(
        """{
  "config": {},
  "runtime": {"git_commit": "old-commit"},
  "extra": {}
}
"""
    )

    with pytest.raises(
        ValueError,
        match="git commit mismatch",
    ):
        _validate_a3_context_artifact(
            context_dir,
            training_seed=0,
            expected_commit="new-commit",
        )


def test_a4_requires_one_policy_and_ensemble_per_seed(tmp_path):
    with pytest.raises(ValueError, match="one policy checkpoint"):
        run_a4(
            root=Path("."),
            out=tmp_path,
            seeds=[0],
            heldout_seed=40_000,
            heldout_episodes=2,
            policy_checkpoints=[],
            ensemble_checkpoints=[tmp_path / "ensemble.pt"],
        )

    with pytest.raises(ValueError, match="one ensemble checkpoint"):
        run_a4(
            root=Path("."),
            out=tmp_path,
            seeds=[0],
            heldout_seed=40_000,
            heldout_episodes=2,
            policy_checkpoints=[tmp_path / "best.pt"],
            ensemble_checkpoints=[],
        )


def test_a4_rejects_noncanonical_gate_parameters(tmp_path):
    with pytest.raises(ValueError, match="parameters are frozen"):
        run_a4(
            root=Path("."),
            out=tmp_path,
            seeds=[0],
            heldout_seed=40_000,
            heldout_episodes=2,
            policy_checkpoints=[tmp_path / "best.pt"],
            ensemble_checkpoints=[tmp_path / "ensemble.pt"],
            gate_gain=3.0,
        )


def test_a4_smoke_writes_auditable_outputs(tmp_path, monkeypatch):
    from sarrl.models import ResidualDynamicsConfig, ResidualDynamicsEnsemble
    from sarrl.rl import SACAgent

    policy = tmp_path / "best.pt"
    ensemble = tmp_path / "ensemble.pt"
    SACAgent(8, 2, seed=3).save(policy)
    ResidualDynamicsEnsemble(
        ResidualDynamicsConfig(hidden=(8,), ensemble_size=2),
        seed=4,
    ).save(ensemble)

    monkeypatch.setattr(
        "tools.run_planar_ablations._retained_a2_checkpoint_hashes",
        lambda root: {0: _sha256(policy)},
    )
    monkeypatch.setattr(
        "tools.run_planar_ablations._validate_a4_ensemble_artifact",
        lambda checkpoint, training_seed, expected_commit=None: True,
    )

    run_a4(
        root=Path(".").resolve(),
        out=tmp_path,
        seeds=[0],
        heldout_seed=40_000,
        heldout_episodes=2,
        policy_checkpoints=[policy],
        ensemble_checkpoints=[ensemble],
    )

    condition = tmp_path / "A4_residual_sac_uncertainty_gate"
    assert (condition / "evaluation_manifest.json").is_file()
    assert (condition / "heldout_episodes.csv").is_file()
    assert (condition / "gate_diagnostics.csv").is_file()
    assert (condition / "summary.csv").is_file()
    assert (condition / "paired_comparison.csv").is_file()
    assert (condition / "aggregate.json").is_file()

    assert len((condition / "heldout_episodes.csv").read_text().splitlines()) == 3
    assert len((condition / "gate_diagnostics.csv").read_text().splitlines()) == 3
    assert '"condition": "A4"' in (condition / "aggregate.json").read_text()


def test_a4_ensemble_artifact_rejects_partial_outputs(tmp_path):
    checkpoint = tmp_path / "ensemble.pt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="partial A4 ensemble artifact"):
        _validate_a4_ensemble_artifact(
            checkpoint,
            training_seed=0,
            expected_commit="commit",
        )


def test_a4_ensemble_preparation_uses_frozen_protocol(tmp_path, monkeypatch):
    calls = []
    validation_calls = {}

    def fake_validate(checkpoint, training_seed, expected_commit=None):
        key = str(checkpoint)
        validation_calls[key] = validation_calls.get(key, 0) + 1
        return validation_calls[key] > 1

    def capture(cmd, cwd, check):
        calls.append((cmd, cwd, check))

    monkeypatch.setattr(
        "tools.run_planar_ablations._validate_a4_ensemble_artifact",
        fake_validate,
    )
    monkeypatch.setattr("tools.run_planar_ablations.subprocess.run", capture)
    monkeypatch.setattr(
        "tools.run_planar_ablations.repository_commit",
        lambda root: "frozen-commit",
    )

    checkpoints = prepare_a4_ensembles(
        root=Path(".").resolve(),
        out=tmp_path,
        seeds=[0, 1],
    )

    assert len(checkpoints) == 2
    assert len(calls) == 2
    for seed, (cmd, _, check) in enumerate(calls):
        assert check is True
        assert cmd[cmd.index("--samples") + 1] == "10000"
        assert cmd[cmd.index("--steps") + 1] == "2000"
        assert cmd[cmd.index("--batch-size") + 1] == "128"
        assert cmd[cmd.index("--seed") + 1] == str(seed)
        assert cmd[cmd.index("--device") + 1] == "cpu"


def test_a5_requires_one_retained_policy_per_seed(tmp_path):
    with pytest.raises(ValueError, match="one policy checkpoint"):
        run_a5(
            root=Path("."),
            out=tmp_path,
            seeds=[0],
            heldout_seed=40_000,
            heldout_episodes=2,
            policy_checkpoints=[],
        )


def test_a6_requires_all_three_per_seed_artifact_families(tmp_path):
    common = {
        "root": Path("."),
        "out": tmp_path,
        "seeds": [0],
        "heldout_seed": 40_000,
        "heldout_episodes": 2,
    }
    with pytest.raises(ValueError, match="one policy checkpoint"):
        run_a6(
            **common,
            policy_checkpoints=[],
            context_checkpoints=[tmp_path / "context.pt"],
            ensemble_checkpoints=[tmp_path / "ensemble.pt"],
        )
    with pytest.raises(ValueError, match="one context checkpoint"):
        run_a6(
            **common,
            policy_checkpoints=[tmp_path / "best.pt"],
            context_checkpoints=[],
            ensemble_checkpoints=[tmp_path / "ensemble.pt"],
        )
    with pytest.raises(ValueError, match="one ensemble checkpoint"):
        run_a6(
            **common,
            policy_checkpoints=[tmp_path / "best.pt"],
            context_checkpoints=[tmp_path / "context.pt"],
            ensemble_checkpoints=[],
        )
