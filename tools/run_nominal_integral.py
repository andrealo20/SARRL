#!/usr/bin/env python3
"""Prepare or explicitly execute the fixed twelve-episode nominal integral study."""

# Numerical thread limits precede all numerical imports.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import torch

from sarrl.envs import PlanarReachEnv
from sarrl.evaluation import assert_repository_import_root, v13_scenarios
from sarrl.evaluation.nominal_integral import (
    compare_reference,
    compare_tree_exact,
    decision,
    evaluate_nominal,
)
from sarrl.evaluation.provenance import runtime_metadata
from sarrl.evaluation.residual_diagnosis import CASES, invalid_data
from tools.run_residual_diagnosis import exclusive_run, read, sha, write_new

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "results/residual_diagnosis_20260905"
HISTORY_HASH = "76961ce34cc81aff4df6d8db227409f12140003cef2eac730345a3f1dede8c18"
PROTOCOL_HASH = "3af217b1ef351a6f02a02545dd0bfa4742fc7d50efde55b7eab98c2a1084090c"
OUTPUT_NAME = "nominal_integral_v19a_20260907"


def cell_specs():
    # Never depend on sorted JSON dictionary order for the scientific sequence.
    for scenario in ("id_reference", "ood_compound", "motor_fault"):
        for seed in CASES[scenario]:
            for controller in ("R0", "I1"):
                yield controller, scenario, seed


def cell_name(controller, scenario, seed):
    return f"{controller}__{scenario}__{seed}"


def execution_metadata():
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ValueError("one numerical thread is required")
    return {
        "device": "cpu",
        "controller_dtype": "float64",
        "observation_dtype": "float32",
        "workers": 1,
        "intraop_threads": 1,
        "interop_threads": 1,
        "thread_environment": {
            key: os.environ[key]
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def build_plan(protocol):
    assert_repository_import_root(ROOT)
    if sha(protocol) != PROTOCOL_HASH:
        raise ValueError("reviewed protocol changed")
    if sha(HISTORY / "complete.json") != HISTORY_HASH:
        raise ValueError("historical completion changed")
    hashes = read(HISTORY / "complete.json")["hashes"]
    if sha(HISTORY / "manifest.json") != hashes["manifest.json"]:
        raise ValueError("historical manifest changed")
    retained = read(HISTORY / "manifest.json")
    # Every old source stays byte-identical, including its evaluator and recorder.
    for name, digest in retained["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"legacy source changed: {name}")
    history_hashes = {}
    for _, scenario, seed in cell_specs():
        name = cell_name("Z", scenario, seed) + ".json"
        if sha(HISTORY / name) != hashes[name]:
            raise ValueError(f"historical reference changed: {name}")
        history_hashes[name] = hashes[name]
    sources = sorted((ROOT / "sarrl").rglob("*.py")) + [
        Path(__file__),
        ROOT / "tools/run_residual_diagnosis.py",
        ROOT / "tests/test_nominal_integral.py",
        ROOT / "pyproject.toml",
    ]
    return {
        "schema_version": 1,
        "experiment": "v1.9-A",
        "exploratory": True,
        "training": False,
        "episodes": 12,
        "max_physical_steps": 3000,
        "technical_physical_steps": 0,
        "max_episode_steps": 250,
        "cells": [list(key) for key in cell_specs()],
        "parameters": {
            "kp": 36.0,
            "kd": 12.0,
            "ki": 36.0,
            "kaw": 4.0,
            "z_limit": 36.0,
            "dt": 0.02,
            "torque_limit": 40.0,
        },
        "protocol_sha256": PROTOCOL_HASH,
        "history_complete_sha256": HISTORY_HASH,
        "history_hashes": history_hashes,
        "source_hashes": {str(path.relative_to(ROOT)): sha(path) for path in sources},
        "execution": execution_metadata(),
        "runtime": runtime_metadata(ROOT),
    }


def validate_cell(result, key, reference=None):
    if (
        result["status"] != "valid"
        or tuple(result["episode"][field] for field in ("controller", "scenario", "seed")) != key
    ):
        raise ValueError("cell status or identity mismatch")
    steps = result["episode"]["steps"]
    if type(steps) is not int or not 0 <= steps <= 250:
        raise ValueError("episode physical budget exceeded")
    if key[0] == "R0":
        compare_reference(result, reference)


def comparisons(records):
    pairs = []
    for r0, i1 in zip(records[::2], records[1::2], strict=True):
        compare_tree_exact(r0["initial"], i1["initial"], "paired initial state")
        compare_tree_exact(
            r0["initial_observation"], i1["initial_observation"], "paired observation"
        )
        common_time = r0["episode"]["steps"] == i1["episode"]["steps"]
        scenario, seed = (r0["episode"][field] for field in ("scenario", "seed"))
        pairs.append(
            {
                "scenario": scenario,
                "seed": seed,
                "motivating_case": (scenario, seed)
                in {
                    ("id_reference", 9801800),
                    ("motor_fault", 9802001),
                },
                "R0": {field: r0[field] for field in ("episode", "safety", "summary")},
                "I1": {
                    field: i1[field]
                    for field in ("episode", "safety", "summary", "integral_summary")
                },
                "common_terminal_time": common_time,
                "final_distance_difference_m": (
                    i1["episode"]["final_distance"] - r0["episode"]["final_distance"]
                    if common_time
                    else None
                ),
            }
        )
    return pairs


def execute(plan, output, protocol):
    if build_plan(protocol) != plan:
        raise ValueError("sources, protocol or runtime changed after preflight")
    output = output.resolve()
    if output != (ROOT / "results" / OUTPUT_NAME).resolve():
        raise ValueError("output must be the dedicated nominal integral study directory")
    with exclusive_run(output):
        if (output / "complete.json").exists():
            raise FileExistsError("integral study already complete")
        if (output / "incomplete.json").exists() or list(output.glob("*.tmp")):
            raise ValueError("incomplete study requires inspection, not retry")
        manifest = output / "manifest.json"
        if manifest.exists():
            if read(manifest) != plan:
                raise ValueError("resume manifest differs")
        else:
            if {p.name for p in output.iterdir()} - {".run.lock"}:
                raise ValueError("nonempty output has no manifest")
            write_new(manifest, plan)
        frozen = output / "protocol.md"
        if not frozen.exists():
            with frozen.open("xb") as stream:
                stream.write(protocol.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
        if sha(frozen) != plan["protocol_sha256"]:
            raise ValueError("frozen protocol differs")
        manifest_hash = sha(manifest)
        specs = {s.key: s for s in v13_scenarios()}
        completed = []
        try:
            # Check ALL interrupted cells before any new environment is constructed.
            gap = False
            retained_records = {}
            for key in cell_specs():
                name = cell_name(*key)
                marker = output / f"{name}.complete.json"
                if (output / f"{name}.invalid.json").exists():
                    raise ValueError(f"invalid cell requires inspection: {name}")
                if not marker.exists() and any(
                    (output / f"{name}.{suffix}").exists() for suffix in ("started.json", "json")
                ):
                    raise ValueError(f"interrupted cell requires inspection: {name}")
                if not marker.exists():
                    gap = True
                    continue
                if gap:
                    raise ValueError("completed cells are not a prefix of the fixed order")
                data = output / f"{name}.json"
                if read(marker) != {"sha256": sha(data), "manifest_sha256": manifest_hash}:
                    raise ValueError(f"completed cell hash mismatch: {name}")
                if read(output / f"{name}.started.json") != {
                    "key": list(key),
                    "status": "started",
                    "manifest_sha256": manifest_hash,
                    "reserved_max_physical_steps": 250,
                }:
                    raise ValueError(f"completed cell reservation mismatch: {name}")
                result = read(data)
                history = read(HISTORY / (cell_name("Z", key[1], key[2]) + ".json"))
                validate_cell(result, key, history)
                if key[0] == "I1":
                    r0 = retained_records[("R0", key[1], key[2])]
                    compare_tree_exact(r0["initial"], result["initial"], "paired initial")
                    compare_tree_exact(
                        r0["initial_observation"],
                        result["initial_observation"],
                        "paired observation",
                    )
                retained_records[key] = result
            for key in cell_specs():
                controller, scenario, seed = key
                name = cell_name(*key)
                data = output / f"{name}.json"
                marker = output / f"{name}.complete.json"
                history = read(HISTORY / (cell_name("Z", scenario, seed) + ".json"))
                if marker.exists():
                    result = retained_records[key]
                else:
                    write_new(
                        output / f"{name}.started.json",
                        {
                            "key": list(key),
                            "status": "started",
                            "manifest_sha256": manifest_hash,
                            "reserved_max_physical_steps": 250,
                        },
                    )
                    result = None
                    try:
                        spec = specs[scenario]
                        env = PlanarReachEnv(
                            mode="torque",
                            dt=0.02,
                            max_steps=250,
                            torque_limit=40.0,
                            randomization=spec.randomization,
                            fault=spec.fault,
                        )
                        result = evaluate_nominal(
                            env, seed=seed, scenario=scenario, controller=controller
                        )
                        validate_cell(result, key, history)
                        if controller == "I1":
                            compare_tree_exact(
                                completed[-1]["initial"], result["initial"], "paired initial"
                            )
                            compare_tree_exact(
                                completed[-1]["initial_observation"],
                                result["initial_observation"],
                                "paired observation",
                            )
                        write_new(data, result)
                        write_new(marker, {"sha256": sha(data), "manifest_sha256": manifest_hash})
                    except Exception as exc:
                        write_new(
                            output / f"{name}.invalid.json",
                            invalid_data(
                                {
                                    "status": "invalid",
                                    "key": list(key),
                                    "error": str(exc),
                                    "result": result,
                                    "partial": getattr(exc, "partial", None),
                                }
                            ),
                        )
                        raise
                completed.append(result)
            pairs = comparisons(completed)
            outcome = decision(completed)
            if outcome == "incomplete" or sum(r["episode"]["steps"] for r in completed) > 3000:
                raise ValueError("incomplete matrix or exceeded physical budget")
            report = {
                "decision": outcome,
                "exploratory": True,
                "pairs": pairs,
                "other_four_cases": [p for p in pairs if not p["motivating_case"]],
                "physical_steps": sum(r["episode"]["steps"] for r in completed),
                "promotion_authorized": False,
            }
            report_path = output / "report.json"
            if report_path.exists():
                if read(report_path) != report:
                    raise ValueError("retained report differs")
            else:
                write_new(report_path, report)
            write_new(
                output / "complete.json",
                {
                    "episodes": 12,
                    "decision": outcome,
                    "hashes": {
                        p.name: sha(p)
                        for p in sorted(output.iterdir())
                        if p.is_file() and p.name != ".run.lock"
                    },
                },
            )
        except Exception as exc:
            write_new(output / "incomplete.json", {"decision": "incomplete", "error": str(exc)})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--execute", action="store_true", help="execute the twelve authorized episodes"
    )
    args = parser.parse_args(argv)
    if args.command == "run" and (not args.execute or args.output is None):
        parser.error("run requires --execute and --output")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    plan = build_plan(args.protocol)
    if args.command == "plan":
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        execute(plan, args.output, args.protocol)


if __name__ == "__main__":
    main()
