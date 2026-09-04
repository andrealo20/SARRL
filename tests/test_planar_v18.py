import copy
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm
from sarrl.envs import PlanarReachEnv, SafetyProjectedEnv
from sarrl.evaluation import evaluate_safety_episodes
from sarrl.evaluation.planar_v18 import CONDITIONS, Protocol, classify_result, crossed_intervals
from sarrl.rl import ReplayBuffer
from sarrl.runtime import ControlStackConfig, SARRLControlStack
from sarrl.safety import HOCBFSafetyFilter, SafetyResult
from tools.run_planar_v18 import expected_config, validate_evaluation, validate_manifest, write_csv
from tools.train_sac import _validation_env, _validation_reward


def test_protocol_ranges_and_budget():
    spec, smoke = Protocol(), Protocol(True)
    assert spec.steps * len(spec.seeds) * len(CONDITIONS) == 2000000
    assert sum(n for _, n in spec.evaluation.values()) * 10 == 7000
    assert not set(spec.seeds) & set(smoke.seeds)
    used = set(
        range(spec.validation["seed"], spec.validation["seed"] + spec.validation["episodes"])
    )
    for start, count in spec.evaluation.values():
        values = set(range(start, start + count))
        assert not used & values
        used |= values


def test_crossed_bootstrap_matches_shared_column_reference():
    matrix = np.arange(3 * 9 * 2, dtype=float).reshape(3, 9, 2)
    got = crossed_intervals(matrix, seed=71, samples=64)
    rng = np.random.Generator(np.random.PCG64(71))
    rows = rng.integers(3, size=(64, 3))
    cols = rng.integers(9, size=(64, 9))
    distribution = np.array(
        [matrix[np.ix_(r, c)].mean(axis=(0, 1)) for r, c in zip(rows, cols, strict=True)]
    )
    for metric in range(2):
        assert got[metric]["difference"] == matrix[:, :, metric].mean()
        assert got[metric]["ci95_low"] == np.quantile(distribution[:, metric], 0.025)
        assert got[metric]["upper_one_sided_95"] == np.quantile(distribution[:, metric], 0.95)
    # Identical columns across all rows must not be independently resampled per row.
    identical = np.tile(np.arange(100)[None, :, None], (5, 1, 1))
    assert crossed_intervals(identical, seed=1)[0]["ci95_high"] > 54


def intervals():
    def interval(point, low, upper):
        return {
            "difference": point,
            "ci95_low": low,
            "ci95_high": upper,
            "upper_one_sided_95": upper,
        }

    return {
        scenario: {
            "success": interval(0.05, 0.01, 0.09),
            "unsafe_episode": interval(0, -0.01, 0.02),
            "safety_infeasible": interval(0, -0.01, 0.01),
        }
        for scenario in Protocol().evaluation
    }


def test_decision_exhaustive_cases():
    values = intervals()
    assert classify_result(values, [0.05] * 5, 0) == "advance"
    assert classify_result(values, [], 0, valid=False) == "incomplete"
    assert classify_result(values, [0.05] * 5, 0.011) == "no_go"
    assert classify_result(values, [0.05, 0.05, 0.05, 0, 0], 0) == "inconclusive"
    values["id_reference"]["success"]["difference"] = 0.029
    assert classify_result(values, [0.05] * 5, 0) == "inconclusive"
    values["id_reference"]["success"]["difference"] = 0
    assert classify_result(values, [0.05] * 5, 0) == "no_go"
    for scenario in Protocol().evaluation:
        for metric, bad in (("unsafe_episode", 0.021), ("safety_infeasible", 0.011)):
            values = intervals()
            values[scenario][metric]["difference"] = bad
            assert classify_result(values, [0.05] * 5, 0) == "no_go"


@pytest.mark.parametrize("condition", CONDITIONS)
def test_manifest_rejects_penalty_flags_and_nonexperimental_changes(condition):
    spec = Protocol()
    payload = {
        "runtime": {"git_commit": "frozen"},
        "config": {**expected_config(spec, condition, 30), "resume": None},
    }
    validate_manifest(payload, spec, condition, 30, "frozen")
    changes = [
        ("trainer", "infeasible_reward", -100.0),
        ("trainer", "validation_infeasible_reward", -250.0),
        ("trainer", "training_hocbf", False),
        ("trainer", "validation_hocbf", False),
        ("trainer", "batch_size", 128),
        ("safety", "validation_infeasible_reward", -250.0),
        ("agent_config", "gamma", 0.98),
    ]
    for section, key, value in changes:
        bad = copy.deepcopy(payload)
        bad["config"][section][key] = value
        with pytest.raises(ValueError, match="configuration"):
            validate_manifest(bad, spec, condition, 30, "frozen")
    with pytest.raises(ValueError, match="configuration"):
        validate_manifest(
            payload, spec, next(c for c in CONDITIONS if c != condition), 30, "frozen"
        )


def test_historical_validation_default_and_override():
    assert _validation_reward(-250, None) == -250
    assert _validation_reward(-250, -500) == -500
    for invalid in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            _validation_reward(-250, invalid)
    env = SafetyProjectedEnv(PlanarReachEnv(mode="residual"), infeasible_reward=-250)
    validation = _validation_env(env, safety_projected=True, infeasible_reward=-500)
    assert env.infeasible_reward == -250 and validation.infeasible_reward == -500


def fail_filter(state, candidate, obstacles=()):
    return SafetyResult(
        torque=np.asarray(candidate),
        success=False,
        correction_norm=1.5,
        min_margin=-1.0,
        active_constraints=(),
        current_safe=True,
    )


@pytest.mark.parametrize("infeasible", [False, True])
def test_same_physics_raw_replay_only_abort_reward_differs(monkeypatch, infeasible):
    environments = [
        SafetyProjectedEnv(PlanarReachEnv(mode="residual"), infeasible_reward=r)
        for r in CONDITIONS.values()
    ]
    action = np.array([0.9, -0.85], dtype=np.float32)
    results = []
    for env in environments:
        obs, _ = env.reset(seed=99018)
        if infeasible:
            monkeypatch.setattr(env.safety_filter, "filter", fail_filter)
        result = env.step(action)
        replay = ReplayBuffer(8, 2, 10, seed=99018)
        replay.add(obs, action, result[1], result[0], result[2])
        np.testing.assert_array_equal(replay.actions[0], action)
        if infeasible:
            np.testing.assert_array_equal(obs, result[0])
            assert env.steps == 0 and result[2] and not result[3]
        results.append(result)
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(environments[0].state, environments[1].state)
    assert results[0][2:4] == results[1][2:4]
    assert results[1][1] - results[0][1] == (250 if infeasible else 0)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_heldout_first_command_abort_has_zero_physical_reward(monkeypatch, condition, tmp_path):
    class Policy:
        def act(self, obs, deterministic=True):
            return np.zeros(2, dtype=np.float32)

    nominal = PlanarArm()
    safety = HOCBFSafetyFilter(nominal)
    monkeypatch.setattr(safety, "filter", fail_filter)
    stack = SARRLControlStack(
        ComputedTorqueController(nominal),
        Policy(),
        ControlStackConfig(require_safety=True),
        safety_filter=safety,
    )
    episodes, diagnostics = evaluate_safety_episodes(
        stack,
        HOCBFSafetyFilter(nominal),
        PlanarReachEnv(mode="torque"),
        episodes=1,
        seed=9901810,
        controller=condition,
    )
    assert episodes[0].reward == 0 and episodes[0].steps == 0 and not episodes[0].success
    assert diagnostics[0].safety_infeasible and diagnostics[0].command_attempts == 1
    assert diagnostics[0].safety_intervention_steps == 1
    spec = Protocol(True)
    outcome_rows, safety_rows = [], []
    for arm in CONDITIONS:
        for scenario, (start, count) in spec.evaluation.items():
            for episode_seed in range(start, start + count):
                fields = {
                    "controller": f"{arm}_train_seed_99018",
                    "scenario": scenario,
                    "seed": episode_seed,
                }
                outcome_rows.append(asdict(replace(episodes[0], **fields)))
                safety_rows.append(asdict(replace(diagnostics[0], **fields)))
    write_csv(tmp_path / "episodes.csv", outcome_rows)
    write_csv(tmp_path / "safety_diagnostics.csv", safety_rows)
    validate_evaluation(tmp_path, spec, 99018)
    write_csv(tmp_path / "episodes.csv", outcome_rows + [outcome_rows[0]])
    with pytest.raises(ValueError, match="keys, counts"):
        validate_evaluation(tmp_path, spec, 99018)


def test_csv_writer_uses_row_schema(tmp_path):
    path = tmp_path / "safety.csv"
    write_csv(path, [{"custom_diagnostic": 3, "success": True}])
    assert path.read_text().splitlines() == ["custom_diagnostic,success", "3,True"]
    assert not path.with_suffix(".csv.tmp").exists()


@pytest.mark.parametrize("reward", [-500.0, -250.0])
def test_real_cli_resume_preserves_selection_and_rejects_both_reward_mismatches(tmp_path, reward):
    root = Path(__file__).resolve().parents[1]
    base = [
        sys.executable,
        str(root / "tools/train_sac.py"),
        "--steps",
        "8",
        "--seed",
        "99018",
        "--hidden",
        "16",
        "16",
        "--start-steps",
        "0",
        "--batch-size",
        "4",
        "--replay-capacity",
        "30",
        "--training-hocbf",
        "--validation-hocbf",
        "--infeasible-reward",
        str(reward),
        "--validation-infeasible-reward",
        "-500",
        "--checkpoint-every",
        "4",
        "--validate-every",
        "4",
        "--validation-episodes",
        "1",
        "--validation-seed",
        "9901800",
        "--output",
        str(tmp_path / "original"),
    ]
    subprocess.run(base, cwd=root, check=True, capture_output=True)
    checkpoint = tmp_path / "original/train_step4.pt"
    command = base.copy()
    command[command.index("--output") + 1] = str(tmp_path / "resumed")
    command += ["--resume", str(checkpoint)]
    subprocess.run(command, cwd=root, check=True, capture_output=True)
    first = torch.load(
        tmp_path / "original/training_final.pt", map_location="cpu", weights_only=False
    )
    second = torch.load(
        tmp_path / "resumed/training_final.pt", map_location="cpu", weights_only=False
    )
    for network in ("actor", "q1", "q2", "q1_target", "q2_target"):
        for name, tensor in first["agent"][network].items():
            torch.testing.assert_close(tensor, second["agent"][network][name], rtol=0, atol=0)
    for filename in ("validation.csv", "episodes.csv", "training_safety.csv"):
        assert (tmp_path / "original" / filename).read_bytes() == (
            tmp_path / "resumed" / filename
        ).read_bytes()
    for option in ("--infeasible-reward", "--validation-infeasible-reward", "--seed"):
        wrong = command.copy()
        wrong[wrong.index(option) + 1] = "-123"
        result = subprocess.run(wrong, cwd=root, capture_output=True, text=True)
        assert result.returncode != 0 and "does not match" in result.stderr
    manifest = json.loads((tmp_path / "resumed/run_manifest.json").read_text())
    assert manifest["config"]["trainer"]["validation_infeasible_reward"] == -500
