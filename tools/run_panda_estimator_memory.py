"""Reproduce retained Panda episodes before the frozen parameter-memory comparison."""

# Numerical settings precede imports.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import gzip
import json
import os
from datetime import datetime, timezone
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import sarrl
from sarrl.evaluation.panda_adaptation import reproduction_mismatches
from sarrl.evaluation.panda_estimator_memory import (
    ARMS,
    CONDITIONS,
    estimator_memory_protocol,
    evaluate_gate,
    run_episode,
    summarize,
)
from sarrl.evaluation.panda_score_memory import run_episode as previous_episode
from sarrl.robots.panda import model_provenance, mujoco
from tools.campaign_guards import seed_range_is_unopened, valid_completion_marker
from tools.run_adaptive_campaign import CampaignLock, sha, write_json
from tools.run_panda_adaptation import OUTPUT_FILES as REFERENCE_FILES
from tools.run_panda_adaptation import canonical, cell_key, source_fingerprint
from tools.run_panda_score_memory import EXTRA_SOURCES as PREVIOUS_SOURCES
from tools.run_panda_score_memory import FROZEN_FILES as PREVIOUS_FROZEN
from tools.run_panda_variable_delay import execute_cells

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIRECTORY = "results/panda_score_memory_pilot"
OUTPUT_FILES = (*REFERENCE_FILES, "decision.json")
FROZEN_FILES = (*PREVIOUS_FROZEN, "sarrl/controllers/panda_score_memory.py")
EXTRA_SOURCES = (
    *PREVIOUS_SOURCES,
    "sarrl/__init__.py",
    "sarrl/controllers/panda_estimator_memory.py",
    "sarrl/evaluation/panda_estimator_memory.py",
    "tools/run_panda_estimator_memory.py",
    "tools/audit_panda_estimator_memory.py",
    "tests/test_panda_estimator_memory.py",
    "docs/panda-estimator-memory.md",
    f"{REFERENCE_DIRECTORY}/manifest.json",
    f"{REFERENCE_DIRECTORY}/episodes.json",
    f"{REFERENCE_DIRECTORY}/complete.json",
)


def verify_frozen_controller(root=ROOT):
    directory = root / REFERENCE_DIRECTORY
    if not valid_completion_marker(directory / "complete.json", REFERENCE_FILES):
        raise RuntimeError("retained score-memory reference has invalid completion hashes")
    reference = json.loads((directory / "manifest.json").read_text())
    hashes = {name: reference["source_sha256"][name] for name in FROZEN_FILES}
    if any(sha(root / name) != digest for name, digest in hashes.items()):
        raise RuntimeError("reference controllers, plant and interfaces must remain frozen")
    if model_provenance() != reference["model"]:
        raise RuntimeError("packaged model differs from the retained reference")
    return {"source_commit": reference["source_commit"], "source_sha256": hashes}


def reproduce_previous(output, *, execute=previous_episode, workers=1):
    directory = ROOT / REFERENCE_DIRECTORY
    if not valid_completion_marker(directory / "complete.json", REFERENCE_FILES):
        raise RuntimeError("invalid retained reference completion hashes")
    reference = json.loads((directory / "episodes.json").read_text())
    if len(reference) != 1920 or len({cell_key(row) for row in reference}) != 1920:
        raise RuntimeError("reproduction requires 1,920 unique retained episodes")
    payloads = execute_cells(
        output / "reproduction_cells",
        [cell_key(row) for row in reference],
        -1,
        execute=execute,
        workers=workers,
    )
    mismatches = [
        [*cell_key(retained), field]
        for payload, retained in zip(payloads, reference, strict=True)
        for field in reproduction_mismatches(payload["record"], retained)
    ]
    report = {
        "compared": 1920,
        "matched": 1920 - len({tuple(item[:3]) for item in mismatches}),
        "mismatches": mismatches,
        "reference_sha256": sha(directory / "episodes.json"),
        "physical_steps": sum(p["record"]["physical_steps"] for p in payloads),
        "prediction_virtual_substeps": 0,
        "fields": sorted({name for row in reference for name in row}),
    }
    path = output / "reproduction.json"
    if path.exists() and json.loads(path.read_text()) != report:
        raise RuntimeError("reproduction report differs from its checked cells")
    if not path.exists():
        write_json(path, report)
    if mismatches:
        raise RuntimeError(f"retained reproduction failed; no fresh seed opened: {mismatches[:5]}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/panda_estimator_memory_pilot"))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reproduce-only",
        action="store_true",
        help="stop after checking retained episodes; use --resume for the new comparison",
    )
    args = parser.parse_args(argv)
    if Path(sarrl.__file__).resolve().parent.parent != ROOT:
        parser.error("imported package does not match this checkout")
    output = args.output.resolve()
    if output != (ROOT / "results/panda_estimator_memory_pilot").resolve():
        parser.error("this frozen comparison requires its canonical output directory")
    if (output / "complete.json").exists():
        parser.error("completed output is retained and never replaced")
    if output.exists() and not args.resume:
        parser.error("output exists; --resume is only for an interrupted comparison")
    if args.resume and not output.is_dir():
        parser.error("no interrupted comparison exists")
    if mujoco is None:
        parser.error("install the retained numerical runtime before execution")
    fingerprint = source_fingerprint()
    fingerprint["runtime"]["workers"] = args.workers
    fingerprint["frozen_controller"] = verify_frozen_controller()
    fingerprint["source_sha256"].update({name: sha(ROOT / name) for name in EXTRA_SOURCES})
    reference = json.loads((ROOT / REFERENCE_DIRECTORY / "manifest.json").read_text())
    for package in ("numpy", "mujoco"):
        if fingerprint["runtime"][package] != reference["runtime"][package]:
            parser.error(f"reproduction requires {package}=={reference['runtime'][package]}")
    protocol = estimator_memory_protocol()
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["protocol"] != protocol or any(
            manifest[k] != v for k, v in fingerprint.items()
        ):
            parser.error("resume requires the original source commit, protocol and runtime")
    else:
        scan = seed_range_is_unopened(ROOT, 9805400, 9805499, "docs/seed_registry.json")
        manifest = {
            **fingerprint,
            "protocol": protocol,
            "model": model_provenance(),
            "seed_scan": scan,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        output.mkdir(parents=True)
    planned = [
        (condition, seed, arm)
        for condition in CONDITIONS
        for seed in protocol["seeds"]
        for arm in ARMS
    ]
    with CampaignLock(output):
        if (output / "complete.json").exists():
            parser.error("comparison completed before acquiring the lock")
        if not args.resume:
            write_json(output / "manifest.json", manifest)
        reproduction = reproduce_previous(output, workers=args.workers)
        if args.reproduce_only:
            print(json.dumps({"reproduction": reproduction, "fresh_episodes": 0}, indent=2))
            return 0
        payloads = execute_cells(
            output / "cells",
            planned,
            protocol["seeds"][0],
            execute=run_episode,
            workers=args.workers,
        )
        records = [payload["record"] for payload in payloads]
        summary = summarize(records)
        write_json(output / "episodes.json", records)
        write_json(output / "summary.json", summary)
        write_json(output / "decision.json", evaluate_gate(records))
        traces = [
            {"cell": list(cell_key(p["record"])), "trace": p["trace"]}
            for p in payloads
            if p["trace"]
        ]
        temporary = output / "traces.json.gz.tmp"
        with temporary.open("wb") as handle:
            handle.write(gzip.compress(canonical(traces), mtime=0))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output / "traces.json.gz")
        write_json(
            output / "complete.json",
            {
                "episodes": len(records),
                "reproduction_episodes": 1920,
                "hashes": {name: sha(output / name) for name in OUTPUT_FILES},
            },
        )
    print(json.dumps(summary["exploratory_gate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
