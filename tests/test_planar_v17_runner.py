import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from sarrl.envs import DomainRandomization, PlanarReachEnv, SafetyProjectedEnv
from tools import run_planar_v17


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _training_manifest(condition: str) -> dict:
    base = PlanarReachEnv(
        mode="residual",
        randomization=DomainRandomization(
            mass_fraction=0.15,
            friction_fraction=0.30,
            motor_gain_fraction=0.15,
            payload_range=(0.0, 1.0),
            action_delay_max=2,
        ),
    )
    projected = SafetyProjectedEnv(
        base,
        infeasible_reward=run_planar_v17.V17_INFEASIBLE_REWARD,
    ).constructor_config()
    training_hocbf = condition == "C1_inloop_hocbf"
    manifest = {
        "runtime": {"git_commit": "frozen"},
        "config": {
            "seed": run_planar_v17.V17_TRAINING_SEEDS[0],
            "requested_steps": run_planar_v17.V17_TRAINING_STEPS,
            "replay_capacity": run_planar_v17.TRAINING_REPLAY_CAPACITY,
            "agent_config": {
                "gamma": 0.99,
                "tau": 0.005,
                "actor_lr": 3e-4,
                "critic_lr": 3e-4,
                "alpha_lr": 3e-4,
                "init_alpha": 0.2,
                "hidden": list(run_planar_v17.TRAINING_HIDDEN),
            },
            "trainer": {
                "batch_size": run_planar_v17.TRAINING_BATCH_SIZE,
                "start_steps": run_planar_v17.V17_START_STEPS,
                "update_every": run_planar_v17.TRAINING_UPDATE_EVERY,
                "context_checkpoint": None,
                "context_checkpoint_sha256": None,
                "training_hocbf": training_hocbf,
                "validation_hocbf": True,
                "infeasible_reward": run_planar_v17.V17_INFEASIBLE_REWARD,
            },
            "context": {
                "enabled": False,
                "checkpoint": None,
                "checkpoint_sha256": None,
                "runtime_device": None,
                "latent_dim": None,
            },
            "validation": {
                "every": run_planar_v17.TRAINING_VALIDATE_EVERY,
                "episodes": run_planar_v17.V17_VALIDATION_EPISODES,
                "seed": run_planar_v17.V17_VALIDATION_SEED,
            },
            "environment": projected if training_hocbf else base.constructor_config(),
            "safety": {
                "training_hocbf": training_hocbf,
                "validation_hocbf": True,
                "infeasible_reward": run_planar_v17.V17_INFEASIBLE_REWARD,
                "training_environment": projected if training_hocbf else None,
                "validation_environment": projected,
            },
        },
    }
    return json.loads(json.dumps(manifest))


@pytest.mark.parametrize("condition", run_planar_v17.V17_CONDITIONS)
def test_training_manifest_accepts_only_the_frozen_protocol(condition: str):
    manifest = _training_manifest(condition)
    run_planar_v17._validate_training_manifest(
        manifest,
        condition=condition,
        training_seed=run_planar_v17.V17_TRAINING_SEEDS[0],
        expected_commit="frozen",
    )

    altered = deepcopy(manifest)
    altered["config"]["trainer"]["batch_size"] += 1
    with pytest.raises(ValueError, match="trainer configuration"):
        run_planar_v17._validate_training_manifest(
            altered,
            condition=condition,
            training_seed=run_planar_v17.V17_TRAINING_SEEDS[0],
            expected_commit="frozen",
        )


def test_selected_validation_row_applies_earliest_tie_break():
    rows = []
    for step in range(
        run_planar_v17.TRAINING_VALIDATE_EVERY,
        run_planar_v17.V17_TRAINING_STEPS + 1,
        run_planar_v17.TRAINING_VALIDATE_EVERY,
    ):
        rows.append(
            {
                "step": str(step),
                "successes": "15",
                "episodes": str(run_planar_v17.V17_VALIDATION_EPISODES),
                "success_rate": "0.5",
                "reward_mean": "-10.0",
                "reward_std": "1.0",
                "final_distance_mean": "0.2",
            }
        )
    rows[2]["successes"] = "18"
    rows[2]["success_rate"] = "0.6"
    rows[4]["successes"] = "18"
    rows[4]["success_rate"] = "0.6"

    selected = run_planar_v17._selected_validation_row(rows)

    assert int(selected["step"]) == 75_000


def test_final_checkpoint_is_resumable_until_outputs_are_complete(
    tmp_path: Path,
    monkeypatch,
):
    run_dir = tmp_path / "training"
    run_dir.mkdir()
    final = run_dir / "training_final.pt"
    final.write_bytes(b"checkpoint")
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"runtime": {}, "config": {}}) + "\n"
    )
    monkeypatch.setattr(
        run_planar_v17,
        "_validate_training_manifest",
        lambda *args, **kwargs: None,
    )

    resume, complete = run_planar_v17._existing_training_state(
        run_dir,
        expected_commit="frozen",
        expected_condition="C1_inloop_hocbf",
        expected_training_seed=run_planar_v17.V17_TRAINING_SEEDS[0],
    )

    assert resume == final
    assert complete is False


def test_training_safety_rows_must_align_with_episode_rows(tmp_path: Path):
    episodes = [
        {
            "episode": 1,
            "step": 100,
            "reward": -20.0,
            "success": 0,
            "final_distance": 0.2,
        },
        {
            "episode": 2,
            "step": 180,
            "reward": 5.0,
            "success": 1,
            "final_distance": 0.04,
        },
    ]
    safety = [
        {
            "episode": 1,
            "step": 100,
            "safety_infeasible": 0,
            "command_attempts": 100,
            "safety_certified_steps": 100,
            "safety_intervention_steps": 10,
            "safety_correction_sum": 4.0,
            "safety_correction_max": 1.0,
        },
        {
            "episode": 2,
            "step": 180,
            "safety_infeasible": 1,
            "command_attempts": 81,
            "safety_certified_steps": 80,
            "safety_intervention_steps": 5,
            "safety_correction_sum": 3.0,
            "safety_correction_max": 0.8,
        },
    ]
    _write_csv(tmp_path / "episodes.csv", episodes)
    _write_csv(tmp_path / "training_safety.csv", safety)

    run_planar_v17._validate_training_episode_artifacts(
        tmp_path, "C1_inloop_hocbf"
    )

    safety[1]["safety_certified_steps"] = 79
    _write_csv(tmp_path / "training_safety.csv", safety)
    with pytest.raises(ValueError, match="counters are inconsistent"):
        run_planar_v17._validate_training_episode_artifacts(
            tmp_path, "C1_inloop_hocbf"
        )


@pytest.mark.parametrize(
    ("condition", "expects_training_hocbf"),
    [
        ("C0_posthoc_hocbf", False),
        ("C1_inloop_hocbf", True),
    ],
)
def test_train_shard_changes_only_the_training_projection(
    tmp_path: Path,
    monkeypatch,
    condition: str,
    expects_training_hocbf: bool,
):
    captured = {}
    monkeypatch.setattr(run_planar_v17, "assert_source_tree_clean", lambda root: None)
    monkeypatch.setattr(run_planar_v17, "repository_commit", lambda root: "frozen")

    def capture(command, cwd, check):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["check"] = check

    monkeypatch.setattr(run_planar_v17.subprocess, "run", capture)
    root = tmp_path / "repo"
    root.mkdir()
    output = root / "results"
    run_planar_v17.train_shard(
        root, output, condition, run_planar_v17.V17_TRAINING_SEEDS[0]
    )

    command = captured["command"]
    assert "--validation-hocbf" in command
    assert ("--training-hocbf" in command) is expects_training_hocbf
    assert command[command.index("--validation-seed") + 1] == "630000"
    assert command[command.index("--infeasible-reward") + 1] == "-500.0"
    assert captured["cwd"] == root
    assert captured["check"] is True


def test_train_shard_rejects_nonprotocol_seed(tmp_path: Path):
    with pytest.raises(ValueError, match="outside the frozen protocol"):
        run_planar_v17.train_shard(
            tmp_path,
            tmp_path / "results",
            "C0_posthoc_hocbf",
            19,
        )


def test_controller_label_round_trip():
    assert run_planar_v17._condition_and_seed(
        "C1_inloop_hocbf_train_seed_24"
    ) == ("C1_inloop_hocbf", 24)
    with pytest.raises(ValueError, match="invalid v1.7 controller"):
        run_planar_v17._condition_and_seed("unknown_seed_24")


def test_aggregate_evaluation_accepts_complete_paired_shards(
    tmp_path: Path,
    monkeypatch,
):
    root = tmp_path / "repo"
    output = root / "results" / "safety_aware_training"
    root.mkdir()
    output.mkdir(parents=True)
    inventory_path = output / "checkpoint_inventory.json"
    inventory_path.write_text("{}\n")
    inventory_hash = run_planar_v17._sha256(inventory_path)
    monkeypatch.setattr(run_planar_v17, "assert_source_tree_clean", lambda root: None)
    monkeypatch.setattr(
        run_planar_v17,
        "_load_inventory",
        lambda root, output: (inventory_path, {}),
    )
    monkeypatch.setattr(
        run_planar_v17,
        "V17_EVALUATION",
        {
            "id_reference": {"seed": 100, "episodes": 2},
            "ood_compound": {"seed": 200, "episodes": 2},
            "motor_fault": {"seed": 300, "episodes": 2},
        },
    )
    monkeypatch.setattr(run_planar_v17, "V17_BOOTSTRAP_SAMPLES", 50)

    def write_manifest(path, config, root):
        del root
        path.write_text(json.dumps({"config": config, "runtime": {}}) + "\n")

    monkeypatch.setattr(run_planar_v17, "write_run_manifest", write_manifest)

    for training_seed in run_planar_v17.V17_TRAINING_SEEDS:
        episodes = []
        diagnostics = []
        for condition in run_planar_v17.V17_CONDITIONS:
            controller = f"{condition}_train_seed_{training_seed}"
            treatment = condition == "C1_inloop_hocbf"
            for scenario, config in run_planar_v17.V17_EVALUATION.items():
                for offset in range(config["episodes"]):
                    episode_seed = config["seed"] + offset
                    success = treatment or offset == 0
                    episodes.append(
                        {
                            "scenario": scenario,
                            "controller": controller,
                            "seed": episode_seed,
                            "reward": -10.0,
                        }
                    )
                    diagnostics.append(
                        {
                            "scenario": scenario,
                            "controller": controller,
                            "seed": episode_seed,
                            "success": str(success),
                            "unsafe_episode": "False",
                            "safety_infeasible": "False",
                            "unsafe_state_fraction": 0.0,
                            "normalized_violation_integral": 0.0,
                            "safety_intervention_fraction": 0.2,
                            "safety_correction_mean": 1.0,
                            "command_attempts": 10,
                            "state_observations": 11,
                            "safety_intervention_steps": 2,
                            "unsafe_state_observations": 0,
                        }
                    )

        shard = output / "evaluation_shards" / f"seed_{training_seed}"
        episode_path = shard / "episodes.csv"
        safety_path = shard / "safety_diagnostics.csv"
        manifest_path = shard / "evaluation_manifest.json"
        _write_csv(episode_path, episodes)
        _write_csv(safety_path, diagnostics)
        manifest_path.write_text(
            json.dumps(
                {
                    "config": {
                        "training_seed": training_seed,
                        "protocol": run_planar_v17.v17_protocol_dict(),
                        "checkpoint_inventory_sha256": inventory_hash,
                        "episodes_sha256": run_planar_v17._sha256(episode_path),
                        "safety_diagnostics_sha256": run_planar_v17._sha256(safety_path),
                    },
                    "runtime": {"git_commit": None},
                }
            )
            + "\n"
        )

    run_planar_v17.aggregate_evaluation(root, output)
    retained = output / "evaluation"
    with (retained / "episodes.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 60
    decision = json.loads((retained / "decision.json").read_text())
    assert decision["decision"] == "advance"
