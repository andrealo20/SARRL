"""Diagnose retained Panda delay-bank traces without new physical episodes."""

# Numerical settings precede imports.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import scipy

from sarrl.evaluation.panda_latency_diagnosis import BANK_ARMS, replay_bank
from sarrl.robots.panda import model_provenance, mujoco
from tools.campaign_guards import valid_completion_marker
from tools.run_panda_adaptation import OUTPUT_FILES

ROOT = Path(__file__).resolve().parents[1]
FROZEN = (
    "sarrl/robots/panda.py",
    "sarrl/controllers/panda_nominal.py",
    "sarrl/controllers/panda_adaptive.py",
    "sarrl/controllers/panda_delay.py",
    "sarrl/controllers/panda_score_memory.py",
)
ANALYSIS = (
    "sarrl/evaluation/panda_latency_diagnosis.py",
    "tools/diagnose_panda_latency.py",
    "docs/panda-latency-diagnosis.md",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_evidence(directory):
    if not valid_completion_marker(directory / "complete.json", OUTPUT_FILES):
        raise ValueError("invalid retained completion hashes")
    manifest = json.loads((directory / "manifest.json").read_text())
    if mujoco is None:
        raise RuntimeError("install the retained MuJoCo runtime")
    for package, version in (("numpy", np.__version__), ("mujoco", mujoco.__version__)):
        if version != manifest["runtime"][package]:
            raise RuntimeError(
                f"retained replay requires {package}=={manifest['runtime'][package]}"
            )
    if model_provenance() != manifest["model"]:
        raise ValueError("packaged model differs from the retained model")
    for name in FROZEN:
        expected = manifest["source_sha256"][name]
        blob = subprocess.check_output(
            ["git", "show", f"{manifest['source_commit']}:{name}"], cwd=ROOT
        )
        if hashlib.sha256(blob).hexdigest() != expected or digest(ROOT / name) != expected:
            raise ValueError(f"retained source differs: {name}")
    rows = json.loads((directory / "episodes.json").read_text())
    with gzip.open(directory / "traces.json.gz", "rt") as file:
        traces = json.load(file)
    lookup = {(row["scenario"], row["seed"], row["arm"]): row for row in rows}
    protocol = manifest["protocol"]
    expected = {
        (scenario, protocol["seeds"][0], arm)
        for scenario in protocol["scenarios"]
        for arm in protocol["arms"]
    }
    keys = [tuple(item["cell"]) for item in traces]
    if len(lookup) != len(rows) or len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("retained episode or first-seed trace identities differ from the protocol")
    if not expected <= lookup.keys():
        raise ValueError("trace has no matching episode record")
    return manifest, rows, traces, lookup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/panda_score_memory_pilot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; retained diagnostics are never overwritten")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode()
    if status.strip():
        parser.error("commit the analysis implementation before running retained diagnostics")
    source, rows, traces, lookup = load_evidence(args.input)
    provenance = {
        "kind": "saved_trace_latency_diagnosis",
        "analysis_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip(),
        "source_sha256": {name: digest(ROOT / name) for name in ANALYSIS},
        "evidence_commit": source["source_commit"],
        "evidence_hashes": json.loads((args.input / "complete.json").read_text())["hashes"],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "mujoco": mujoco.__version__,
        },
        "selection": "every retained first-seed trace for each of the three bank arms",
        "seed": source["protocol"]["seeds"][0],
        "physical_simulation_steps": 0,
        "new_episode_seeds": [],
        "scope": "descriptive diagnosis; suffix scores and batch fits are offline diagnostics",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for item in traces:
        cell = tuple(item["cell"])
        if cell[2] not in BANK_ARMS:
            continue
        report = replay_bank(lookup[cell], item["trace"])
        reports.append(report)
        print(f"Replayed {len(reports)}/48: {cell[0]} {cell[2]}", flush=True)
    totals = {}
    for arm in BANK_ARMS:
        for changing in (False, True):
            retained = [
                row
                for row in rows
                if row["arm"] == arm and row["scenario"].endswith(("_up", "_down")) == changing
            ]
            totals[f"{arm}_{'changing' if changing else 'constant'}"] = {
                "episodes": len(retained),
                "successes": sum(row["success"] for row in retained),
                "correct_final_age": sum(
                    row["delay_estimator"]["final_delay_correct"] for row in retained
                ),
            }
    data = {"replayed_traces": len(reports), "retained_record_totals": totals, "traces": reports}
    for name, value in (("manifest.json", provenance), ("diagnostics.json", data)):
        (args.output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    complete = {
        "hashes": {
            name: digest(args.output / name) for name in ("manifest.json", "diagnostics.json")
        }
    }
    (args.output / "complete.json").write_text(json.dumps(complete, indent=2) + "\n")
    print(
        json.dumps(
            {
                "replayed_traces": len(reports),
                "physical_simulation_steps": 0,
                "new_episode_seeds": [],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
