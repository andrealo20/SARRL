"""Reproduce the covariance baseline, train five policies and evaluate held-out tasks."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import scipy
import torch

try:
    import mujoco
except ImportError:
    mujoco = None

import sarrl
from sarrl.evaluation.adaptive_pilot import run_case
from sarrl.evaluation.margin_campaign import V22_MARGINS, V22_PLANT_OPTIONS, v22_config
from sarrl.evaluation.planar_residual import (
    ARMS,
    EVALUATION_SEEDS,
    SCENARIOS,
    TRAINING_SEEDS,
    analyze,
    historical_reproduction_report,
    run_episode,
    study_protocol,
)
from sarrl.rl.adaptive_residual_training import train
from sarrl.rl.sac import SACAgent
from tools.campaign_guards import seed_range_is_unopened, valid_completion_marker
from tools.run_adaptive_campaign import CampaignLock, sha, write_json
from tools.run_panda_adaptation import canonical, cell_key
from tools.run_panda_variable_delay import execute_cells

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/planar_residual_v26"
REFERENCE = ROOT / "results/uncertainty_margin_v22/episodes.jsonl"
OUTPUT_FILES = (
    "manifest.json",
    "reproduction.json",
    "reproduction_records.json",
    "models.json",
    "episodes.json",
    "summary.json",
    "decision.json",
    "traces.json.gz",
)
_POLICIES = {}


def source_fingerprint():
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    ).splitlines()
    dirty = [line for line in dirty if not line[3:].startswith("results/")]
    if dirty:
        raise RuntimeError("commit source changes before the planar study")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = subprocess.check_output(
        ["git", "ls-files", "sarrl", "tools", "tests"], cwd=ROOT, text=True
    ).splitlines()
    sources = [name for name in files if name.endswith(".py")]
    sources += [
        "docs/adaptive-residual-comparison.md",
        "docs/seed_registry.json",
        "pyproject.toml",
        "results/panda_joint_filter_pilot/decision.json",
        "results/panda_joint_filter_pilot/complete.json",
        "results/uncertainty_margin_v22/manifest.json",
        "results/uncertainty_margin_v22/episodes.jsonl",
    ]
    return {
        "source_commit": commit,
        "source_sha256": {name: sha(ROOT / name) for name in sources},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "mujoco": mujoco.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "numerical_threads": 1,
        },
    }


def reproduce_case(seed, scenario, arm, *, retain_trace=False):
    from dataclasses import asdict

    row = asdict(
        run_case(
            arm,
            scenario,
            seed,
            "official",
            v22_config(),
            True,
            "mujoco",
            V22_PLANT_OPTIONS,
            certificate_margin=V22_MARGINS[arm],
        )
    )
    return row, []


def reproduce_previous(output, workers):
    reference = [json.loads(line) for line in REFERENCE.read_text().splitlines()]
    reference = [
        r for r in reference if r["arm"] == "adaptive_hocbf_margin" and 56200 <= r["seed"] < 56300
    ]
    if len(reference) != 300 or len({cell_key(r) for r in reference}) != 300:
        raise RuntimeError("the historical reproduction requires 300 unique baseline records")
    payloads = execute_cells(
        output / "reproduction_cells",
        [cell_key(r) for r in reference],
        -1,
        execute=reproduce_case,
        workers=min(workers, 4),
    )
    records = [p["record"] for p in payloads]
    report = historical_reproduction_report(records, reference, sha(REFERENCE))
    record_path = output / "reproduction_records.json"
    if record_path.exists() and json.loads(record_path.read_text()) != records:
        raise RuntimeError("reproduction records differ from their checked cells")
    if not record_path.exists():
        write_json(record_path, records)
    existing = output / "reproduction.json"
    if existing.exists() and json.loads(existing.read_text()) != report:
        raise RuntimeError("reproduction report differs from its checked cells")
    if not existing.exists():
        write_json(existing, report)
    if report["mismatches"]:
        raise RuntimeError(
            "historical reproduction failed; no new training seed opened: "
            f"{report['mismatches'][:6]}"
        )
    return report


def evaluate_case(seed, scenario, arm, *, retain_trace=False):
    policy = None
    if arm.startswith("residual_"):
        training_seed = int(arm.split("_")[-1])
        path = OUTPUT / "training" / f"seed_{training_seed}" / "final.pt"
        if training_seed not in _POLICIES:
            models = json.loads((OUTPUT / "models.json").read_text())
            if sha(path) != models[str(training_seed)]["policy_sha256"]:
                raise RuntimeError("policy identity changed before evaluation")
            _POLICIES[training_seed] = SACAgent.from_checkpoint(path, load_optimizers=False)
        policy = _POLICIES[training_seed]
    row, trace = run_episode(seed, scenario, arm, policy=policy, retain_trace=retain_trace)
    return row, trace


def train_models(output, signature, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [
            pool.submit(train, seed, output / "training" / f"seed_{seed}", signature)
            for seed in TRAINING_SEEDS
        ]
        reports = [future.result() for future in futures]
    models = {}
    for seed, report in zip(TRAINING_SEEDS, reports, strict=True):
        path = output / "training" / f"seed_{seed}" / "final.pt"
        agent = SACAgent.from_checkpoint(path, load_optimizers=False)
        networks = (agent.actor, agent.q1, agent.q2, agent.q1_target, agent.q2_target)
        if (
            not all(torch.isfinite(p).all() for net in networks for p in net.parameters())
            or not torch.isfinite(agent.log_alpha).all()
        ):
            raise RuntimeError(
                "a trained policy has nonfinite parameters; held-out block stays closed"
            )
        models[str(seed)] = {
            "policy_sha256": sha(path),
            "steps": report["steps"],
            "updates": report["updates"],
            "episodes": report["episodes"],
            "executed_control_steps": report["executed_control_steps"],
            "abort_calls": report["abort_calls"],
            "completion_sha256": sha(path.parent / "complete.json"),
        }
    existing = output / "models.json"
    if existing.exists() and json.loads(existing.read_text()) != models:
        raise RuntimeError("completed training model identities changed")
    if not existing.exists():
        write_json(existing, models)
    return models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--resume", action="store_true")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--reproduce-only", action="store_true")
    phase.add_argument("--train-only", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output != OUTPUT.resolve():
        parser.error("the frozen study requires its canonical directory")
    if Path(sarrl.__file__).resolve().parent.parent != ROOT:
        parser.error("imported package differs from this checkout")
    if (output / "complete.json").exists():
        parser.error("completed study is retained and never replaced")
    if output.exists() != args.resume:
        parser.error("existing interrupted output requires --resume")
    if mujoco is None:
        parser.error("install the recorded MuJoCo runtime before execution")
    fingerprint = source_fingerprint()
    fingerprint["runtime"]["workers"] = args.workers
    fingerprint["runtime"]["reproduction_workers"] = min(args.workers, 4)
    fingerprint["runtime"]["evaluation_workers"] = min(args.workers, 4)
    for name, version in study_protocol()["runtime"].items():
        if (
            name in ("python", "numpy", "mujoco", "torch", "scipy")
            and fingerprint["runtime"][name] != version
        ):
            parser.error(f"this study requires {name}=={version}")
    # The independent Panda study must have completed and its decision evaluated.
    from tools.run_panda_joint_filter import OUTPUT_FILES as PANDA_FILES

    if not valid_completion_marker(
        ROOT / "results/panda_joint_filter_pilot/complete.json", PANDA_FILES
    ):
        parser.error("the preceding Panda study has invalid completion hashes")
    prerequisite = json.loads((ROOT / "results/panda_joint_filter_pilot/decision.json").read_text())
    if type(prerequisite.get("filter_advances")) is not bool:
        parser.error("the preceding Panda decision has not been evaluated")
    protocol = json.loads(json.dumps(study_protocol()))
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["protocol"] != protocol or any(
            manifest[k] != v for k, v in fingerprint.items()
        ):
            parser.error("resume requires the original source, protocol and runtime")
    else:
        scans = [
            seed_range_is_unopened(ROOT, low, high, "docs/seed_registry.json")
            for low, high in ((9806000, 9806099), (9806100, 9806199))
        ]
        manifest = {
            **fingerprint,
            "protocol": protocol,
            "seed_scans": scans,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        output.mkdir(parents=True)
    with CampaignLock(output):
        if (output / "complete.json").exists():
            parser.error("study completed before acquiring the lock")
        if not args.resume:
            write_json(output / "manifest.json", manifest)
        report = reproduce_previous(output, args.workers)
        if args.reproduce_only:
            print(json.dumps({"reproduction": report, "fresh_training_seeds": 0}, indent=2))
            return 0
        models = train_models(output, fingerprint, args.workers)
        if args.train_only:
            print(
                "All five fixed-budget policies completed; held-out seeds stay closed", flush=True
            )
            return 0
        planned = [(c, s, a) for c in SCENARIOS for s in EVALUATION_SEEDS for a in ARMS]
        payloads = execute_cells(
            output / "cells",
            planned,
            EVALUATION_SEEDS[0],
            execute=evaluate_case,
            workers=min(args.workers, 4),
        )
        rows = [p["record"] for p in payloads]
        if any(
            sha(output / "training" / f"seed_{seed}" / "final.pt") != model["policy_sha256"]
            for seed, model in models.items()
        ):
            raise RuntimeError("policy identity changed during held-out evaluation")
        summary = analyze(rows)
        write_json(output / "episodes.json", rows)
        write_json(output / "summary.json", summary)
        write_json(output / "decision.json", summary["decision"])
        traces = [
            {"cell": list(cell_key(p["record"])), "trace": p["trace"]}
            for p in payloads
            if p["trace"]
        ]
        temporary = output / "traces.json.gz.tmp"
        temporary.write_bytes(gzip.compress(canonical(traces), mtime=0))
        temporary.replace(output / "traces.json.gz")
        write_json(
            output / "complete.json",
            {
                "episodes": len(rows),
                "models": len(TRAINING_SEEDS),
                "hashes": {name: sha(output / name) for name in OUTPUT_FILES},
            },
        )
    print(json.dumps(summary["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
