"""Private exploratory diagnosis of frozen checkpoints, no training."""

import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import torch

from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm
from sarrl.envs import PlanarReachEnv
from sarrl.evaluation import evaluate_safety_episodes, planar_safety_config, v13_scenarios
from sarrl.rl import SACAgent
from sarrl.runtime import ControlStackConfig, SARRLControlStack
from sarrl.safety import HOCBFSafetyFilter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
OFFICIAL = OUT.parent
RANGES = {"id_reference": 9801800, "ood_compound": 9801900, "motor_fault": 9802000}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def truth(value):
    return value is True or value == "True"


def key(row):
    return row["controller"], row["scenario"], int(row["seed"])


def integrity():
    checked = []

    def check(path, expected):
        actual = sha(path)
        if actual != expected:
            raise ValueError(f"Hash mismatch: {path}")
        checked.append({"path": str(path.relative_to(ROOT)), "sha256": actual,
                        "bytes": path.stat().st_size})

    inventory = read(OFFICIAL / "checkpoint_inventory.json")
    check(OFFICIAL / "campaign.json", inventory["campaign_sha256"])
    for record in inventory["records"]:
        directory = OFFICIAL / "training" / record["condition"] / f"seed_{record['training_seed']}"
        for name, expected in record["hashes"].items():
            check(directory / name, expected)
    completion = read(OFFICIAL / "complete.json")
    check(OFFICIAL / "checkpoint_inventory.json", completion["inventory_sha256"])
    for name, expected in completion["hashes"].items():
        check(OFFICIAL / name, expected)
    for record in completion["shards"]:
        directory = OFFICIAL / "evaluation_shards" / f"seed_{record['seed']}"
        check(directory / "complete.json", record["completion_sha256"])
        for name, expected in read(directory / "complete.json")["hashes"].items():
            check(directory / name, expected)
    workflow = read(OFFICIAL / "workflow_complete.json")
    check(OFFICIAL / "complete.json", workflow["evaluation_completion_sha256"])
    check(OFFICIAL / "critic_diagnostics.json", workflow["critic_diagnostics_sha256"])
    old = read(ROOT / "results/safety_aware_training/checkpoint_inventory.json")
    for record in old["config"]["records"]:
        for name, expected in record.items():
            if name.endswith("_sha256"):
                if expected is None and record[name[:-7]] is None:
                    continue
                check(ROOT / record[name[:-7]], expected)
    save("integrity.json", {"verified_entries": len(checked), "files": checked})
    return inventory


def official_summary():
    episodes = rows(OFFICIAL / "episodes.csv")
    diagnostics = {key(row): row for row in rows(OFFICIAL / "safety_diagnostics.csv")}
    assert len(episodes) == len(diagnostics) == len({key(row) for row in episodes}) == 7000
    groups = {}
    for row in episodes:
        condition = row["controller"].split("_train_seed_")[0]
        group = groups.setdefault((condition, row["scenario"]), [])
        diagnostic = diagnostics[key(row)]
        kind = "success" if truth(row["success"]) else (
            "abort" if truth(diagnostic["safety_infeasible"]) else "timeout")
        if kind == "timeout":
            assert int(row["steps"]) == 250
        assert int(diagnostic["state_observations"]) == int(row["steps"]) + 1
        group.append((row, diagnostic, kind))
    result = []
    for (condition, scenario), group in groups.items():
        record = {"condition": condition, "scenario": scenario, "episodes": len(group)}
        for kind in ("success", "abort", "timeout"):
            selected = [(e, d) for e, d, k in group if k == kind]
            record[kind] = {
                "count": len(selected),
                "median_final_distance_m": float(np.median([float(e["final_distance"]) for e, d in selected])) if selected else None,
                "final_within_radius_count": sum(float(e["final_distance"]) <= .05 for e, d in selected),
                "mean_intervention_fraction": float(np.mean([float(d["safety_intervention_fraction"]) for e, d in selected])) if selected else None,
            }
        result.append(record)
    save("official_failure_summary.json", result)


def evaluate(record):
    torch.set_num_threads(1)
    condition, training_seed = record["condition"], record["training_seed"]
    checkpoint = OFFICIAL / "training" / condition / f"seed_{training_seed}" / "best.pt"
    assert sha(checkpoint) == record["hashes"]["best.pt"]
    policy = SACAgent.from_checkpoint(checkpoint, seed=0, load_optimizers=False)
    nominal = PlanarArm()
    observer = HOCBFSafetyFilter(nominal, planar_safety_config())
    stack = SARRLControlStack(ComputedTorqueController(nominal), policy,
                             ControlStackConfig(require_safety=True), safety_filter=observer)
    specifications = {s.key: s for s in v13_scenarios()}
    output = []
    for scenario, start in RANGES.items():
        specification = specifications[scenario]
        env = PlanarReachEnv(mode="torque", randomization=specification.randomization,
                             fault=specification.fault)
        traces = {}

        def observe(event):
            trace = traces.setdefault(event["episode_seed"], [])
            state = env.state
            command = event["command"]
            trace.append({"attempt": event["step"], "executed": event["info"] is not None,
                          "distance_m": float(np.linalg.norm(env.target - env.arm.forward_kinematics(state[:2]))),
                          "speed_rad_s": float(np.linalg.norm(state[2:])),
                          "raw_residual": command.raw_residual.tolist(),
                          "correction": float(command.safety_correction)})

        outcomes, diagnostics = evaluate_safety_episodes(
            stack, observer, env, episodes=2, seed=start, scenario=scenario,
            controller=f"{condition}_train_seed_{training_seed}", transition_callback=observe)
        for outcome, diagnostic in zip(outcomes, diagnostics, strict=True):
            trace = traces[outcome.seed]
            physical = [step for step in trace if step["executed"]]
            assert len(physical) == outcome.steps
            assert abs(trace[-1]["distance_m"] - outcome.final_distance) < 1e-10
            assert outcome.success == (trace[-1]["distance_m"] <= .05 and trace[-1]["speed_rad_s"] <= .35 and not diagnostic.safety_infeasible)
            output.append({"episode": asdict(outcome), "safety": asdict(diagnostic),
                           "condition": condition, "training_seed": training_seed,
                           "minimum_observed_distance_m": min(step["distance_m"] for step in trace),
                           "final_speed_rad_s": trace[-1]["speed_rad_s"],
                           "near_target_physical_steps": sum(step["distance_m"] <= .05 for step in physical),
                           "tail50_mean_distance_m": float(np.mean([s["distance_m"] for s in physical[-50:]])) if physical else None,
                           "tail50_mean_speed_rad_s": float(np.mean([s["speed_rad_s"] for s in physical[-50:]])) if physical else None,
                           "trace": trace})
    print(f"diagnostic complete: {condition} seed={training_seed}", flush=True)
    return output


if __name__ == "__main__":
    assert not (OUT / "complete.json").exists(), "Do not overwrite a completed audit"
    inventory = integrity()
    official_summary()
    save("manifest.json", {"script_sha256": sha(Path(__file__)), "protocol_sha256": sha(OUT / "PROTOCOL.md"),
                           "inventory_sha256": sha(OFFICIAL / "checkpoint_inventory.json"),
                           "evaluation_seed_starts": RANGES, "episodes_per_cell": 2,
                           "workers": 5, "device": "cpu", "exploratory": True})
    with ProcessPoolExecutor(max_workers=5) as pool:
        results = [item for block in pool.map(evaluate, inventory["records"]) for item in block]
    assert len(results) == 60
    save("trajectories.json", results)
    save("episode_summary.json", [{k: v for k, v in r.items() if k != "trace"} for r in results])
    save("complete.json", {"episodes": 60, "hashes": {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix in (".json", ".py", ".md")}})
    print("Audit complete: 60 exploratory episodes, no training.", flush=True)
