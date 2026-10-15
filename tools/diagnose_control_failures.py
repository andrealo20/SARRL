"""Diagnose the retained v2.6 studies without training or physical simulation."""

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

from sarrl.evaluation.control_diagnosis import (
    panda_prefix_reasons,
    panda_rows,
    panda_trace,
    planar_pairs,
    planar_trace,
)
from tools.campaign_guards import valid_completion_marker

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/control_failure_diagnosis_v26"
INPUTS = {
    "planar_residual_v26": "bc29d1f1b7b3ad0a69bc21a044bcfb8683a732994ffce5435738bb8920850889",
    "panda_joint_filter_pilot": "e7f62d3732b20c9ef49dbdaa6777ebd3ee50672ae616d5ab18ebce411c0318c8",
}
SOURCES = (
    "sarrl/evaluation/control_diagnosis.py",
    "tools/diagnose_control_failures.py",
    "tests/test_control_diagnosis.py",
    "docs/control-failure-methods.md",
    "sarrl/controllers/panda_adaptive.py",
    "sarrl/controllers/panda_nominal.py",
    "sarrl/robots/panda.py",
    "sarrl/safety/panda_joint_filter.py",
    "sarrl/envs/planar_reach.py",
    "sarrl/envs/mujoco_planar.py",
    "sarrl/envs/adaptive_projected.py",
    "sarrl/evaluation/safety_audit.py",
    "sarrl/dynamics/planar_arm.py",
    "sarrl/runtime/control_stack.py",
    "sarrl/safety/filter.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def evidence(name):
    directory = ROOT / "results" / name
    marker = directory / "complete.json"
    if sha(marker) != INPUTS[name]:
        raise ValueError("v2.6 input identity changed")
    hashes = json.loads(marker.read_text())["hashes"]
    if not valid_completion_marker(marker, tuple(hashes)):
        raise ValueError("retained input hashes differ")
    rows = json.loads((directory / "episodes.json").read_text())
    traces = json.loads(gzip.decompress((directory / "traces.json.gz").read_bytes()))
    manifest = json.loads((directory / "manifest.json").read_text())
    planned = {
        (c, s, a)
        for c in manifest["protocol"]["scenarios"]
        for s in manifest["protocol"].get("evaluation_seeds", manifest["protocol"].get("seeds"))
        for a in manifest["protocol"]["arms"]
    }
    keys = [(r["scenario"], r["seed"], r["arm"]) for r in rows]
    first = min(key[1] for key in planned)
    trace_keys = [tuple(t["cell"]) for t in traces]
    if (
        len(keys) != len(set(keys))
        or set(keys) != planned
        or len(trace_keys) != len(set(trace_keys))
        or set(trace_keys) != {key for key in planned if key[1] == first}
    ):
        raise ValueError("input pairing or complete first-seed trace selection differs")
    return manifest, rows, traces


def calculate():
    import numpy as np
    import scipy

    from sarrl.controllers.panda_adaptive import PandaRegressor
    from sarrl.evaluation.panda_adaptation import reproduction_mismatches
    from sarrl.robots.panda import VELOCITY_LIMIT, model_provenance, mujoco
    from sarrl.safety.panda_joint_filter import PandaJointFilter, PandaJointFilterConfig

    planar_manifest, rows, traces = evidence("planar_residual_v26")
    panda_manifest, panda_records, panda_traces = evidence("panda_joint_filter_pilot")
    if mujoco is None:
        raise RuntimeError("recorded Panda MuJoCo runtime required")
    runtime = {"numpy": np.__version__, "scipy": scipy.__version__, "mujoco": mujoco.__version__}
    if any(v != panda_manifest["runtime"][k] for k, v in runtime.items()):
        raise RuntimeError("use the recorded Panda numerical runtime")
    if model_provenance() != panda_manifest["model"]:
        raise ValueError("Panda model differs from retained evidence")

    def prohibited(*args, **kwargs):
        raise RuntimeError("physical simulation is prohibited during saved-data diagnosis")

    original_step = mujoco.mj_step
    mujoco.mj_step = prohibited
    try:
        lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in rows}
        planar = [planar_trace(lookup[tuple(t["cell"])], t["trace"]) for t in traces]
        regressor = PandaRegressor()
        bounds = regressor.nominal.model.jnt_range
        panda_lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in panda_records}
        panda = []
        for item in panda_traces:
            row = panda_lookup[tuple(item["cell"])]
            report = panda_trace(row, item["trace"], bounds[:, 0], bounds[:, 1], VELOCITY_LIMIT)
            if row["arm"] == "bank_filtered":
                report["delay_prefix_diagnostic"] = panda_prefix_reasons(
                    row, item["trace"], regressor, PandaJointFilter, PandaJointFilterConfig
                )
            panda.append(report)
    finally:
        mujoco.mj_step = original_step
    data = {
        "scope": "retrospective saved-data diagnosis; no confirmation or candidate selection",
        "new_episode_seeds": [],
        "physical_simulation_steps": 0,
        "planar_paired_outcomes": planar_pairs(rows),
        "planar_first_seed_traces": planar,
        "panda_filter_outcomes": panda_rows(panda_records),
        "panda_first_seed_traces": panda,
        "planar_evidence_commit": planar_manifest["source_commit"],
        "panda_evidence_commit": panda_manifest["source_commit"],
    }
    return data, {**runtime, "python": platform.python_version()}, reproduction_mismatches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if args.audit:
        if not valid_completion_marker(
            out / "complete.json", ("manifest.json", "diagnostics.json")
        ):
            parser.error("invalid diagnosis completion hashes")
        manifest = json.loads((out / "manifest.json").read_text())
        if manifest["input_completion_sha256"] != INPUTS:
            parser.error("diagnosis input identity differs")
        for name, digest in manifest["source_sha256"].items():
            blob = subprocess.check_output(
                ["git", "show", f"{manifest['analysis_commit']}:{name}"], cwd=ROOT
            )
            if hashlib.sha256(blob).hexdigest() != digest or sha(ROOT / name) != digest:
                parser.error(f"check out the frozen diagnosis source: {name}")
        data, _, mismatches = calculate()
        if mismatches(data, json.loads((out / "diagnostics.json").read_text())):
            parser.error("retained diagnosis does not reproduce")
        print(
            "Verified 2100 planar records, 480 Panda records and all 45 saved traces; no new steps"
        )
        return 0
    if out != OUTPUT.resolve() or out.exists():
        parser.error("use the unopened canonical diagnosis directory; existing data are retained")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        parser.error("commit the diagnostic implementation before execution")
    data, runtime, _ = calculate()
    manifest = {
        "analysis_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip(),
        "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
        "input_completion_sha256": INPUTS,
        "runtime": runtime,
        "selection": "all 2580 retained episodes and all 45 first-seed traces",
        "physical_simulation_steps": 0,
        "new_episode_seeds": [],
    }
    out.mkdir()
    write(out / "manifest.json", manifest)
    write(out / "diagnostics.json", data)
    write(
        out / "complete.json",
        {"hashes": {name: sha(out / name) for name in ("manifest.json", "diagnostics.json")}},
    )
    print("Retained diagnosis: 2580 episodes, 45 traces, zero new episodes or simulation steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
