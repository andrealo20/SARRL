import subprocess
import sys
from pathlib import Path

from sarrl.evaluation import (
    V14_CONDITIONS,
    V14_EPISODES,
    V14_EVALUATION_SEED,
    V14_PAIRINGS,
    V14_TRAINING_SEEDS,
    v14_protocol_dict,
)
from tools.run_planar_v14 import inputs_from_v13_manifest


def test_v14_cli_entrypoint_loads_from_repository_root():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "tools" / "run_planar_v14.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "paired quantified-safety campaign" in result.stdout


def test_v14_protocol_freezes_paired_filter_comparisons():
    protocol = v14_protocol_dict()

    assert V14_EVALUATION_SEED == 50_000
    assert V14_EPISODES == 100
    assert V14_TRAINING_SEEDS == (0, 1, 2, 3, 4)
    assert V14_CONDITIONS == (
        "A2_unfiltered",
        "A5_hocbf",
        "A6_prefilter",
        "A6_hocbf",
    )
    assert V14_PAIRINGS == (
        ("A5_hocbf", "A2_unfiltered", "A2_to_A5_hocbf"),
        ("A6_hocbf", "A6_prefilter", "A6_hocbf_effect"),
    )
    assert protocol["evaluation"]["paired_across_filter_variants"] is True
    assert len(protocol["scenarios"]) == 3
    assert len(V14_CONDITIONS) * len(V14_TRAINING_SEEDS) * 3 * V14_EPISODES == 6_000


def test_v14_protocol_quantifies_frequency_severity_and_task_tradeoff():
    metrics = v14_protocol_dict()["metrics"]

    assert "unsafe_episode_rate" in metrics["state"]
    assert "normalized_violation_integral" in metrics["state"]
    assert "safety_infeasible_rate" in metrics["command"]
    assert metrics["task"] == ["success_rate"]
    scope = v14_protocol_dict()["guarantee_scope"]
    assert scope["certificate"] == "nominal_instantaneous_command_model_only"
    assert "actuator_delay" in scope["excluded"]
    assert "injected_faults" in scope["excluded"]


def test_v14_manifest_loader_rejects_non_v13_campaign(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"config": {"campaign": "wrong", "inputs": []}}')

    try:
        inputs_from_v13_manifest(tmp_path, manifest)
    except ValueError as exc:
        assert "not the v1.3 robustness campaign" in str(exc)
    else:
        raise AssertionError("wrong campaign manifest must be rejected")
