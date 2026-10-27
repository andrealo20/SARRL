"""Reproduce the v2.6 joint-filter records, then run the selected-delay mechanism test."""

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

import scipy

import sarrl
from sarrl.evaluation.panda_adaptation import reproduction_mismatches
from sarrl.evaluation.panda_selected_filter import (
    ARMS,
    decide,
    run_episode,
    selected_filter_protocol,
    summarize,
)
from sarrl.robots.panda import model_provenance, mujoco
from tools.campaign_guards import seed_range_is_unopened, valid_completion_marker
from tools.run_adaptive_campaign import CampaignLock, sha, write_json
from tools.run_panda_adaptation import canonical, cell_key, source_fingerprint
from tools.run_panda_joint_filter import EXTRA_SOURCES as PREVIOUS_SOURCES
from tools.run_panda_joint_filter import OUTPUT_FILES, verify_frozen_controller
from tools.run_panda_variable_delay import execute_cells

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = "results/panda_selected_filter_pilot"
PREVIOUS = "results/panda_joint_filter_pilot"
SEED_BLOCK = (9806200, 9806299)
EXTRA_SOURCES = (
    *PREVIOUS_SOURCES,
    "sarrl/evaluation/panda_selected_filter.py",
    "tools/run_panda_selected_filter.py",
    "tools/audit_panda_selected_filter.py",
    "tests/test_panda_selected_filter.py",
    "docs/panda-selected-filter.md",
    f"{PREVIOUS}/manifest.json",
    f"{PREVIOUS}/episodes.json",
    f"{PREVIOUS}/complete.json",
    f"{PREVIOUS}/decision.json",
)


def reproduce_previous(output, *, execute=run_episode, workers=1):
    """Every v2.6 record through the v2.8 entry point, with its original arm name."""
    directory = ROOT / PREVIOUS
    if not valid_completion_marker(directory / "complete.json", OUTPUT_FILES):
        raise RuntimeError("invalid retained v2.6 completion hashes")
    reference = json.loads((directory / "episodes.json").read_text())
    if len(reference) != 480 or len({cell_key(row) for row in reference}) != 480:
        raise RuntimeError("reproduction requires 480 unique retained v2.6 episodes")
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
        "compared": 480,
        "matched": 480 - len({tuple(item[:3]) for item in mismatches}),
        "mismatches": mismatches,
        "reference_sha256": sha(directory / "episodes.json"),
        "physical_steps": sum(p["record"]["physical_steps"] for p in payloads),
        "fields": sorted({name for row in reference for name in row}),
    }
    path = output / "reproduction.json"
    if path.exists() and json.loads(path.read_text()) != report:
        raise RuntimeError("reproduction report differs from its checked cells")
    if not path.exists():
        write_json(path, report)
    if mismatches:
        raise RuntimeError(f"v2.6 reproduction failed; no fresh seed opened: {mismatches[:5]}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(CANONICAL))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reproduce-only",
        action="store_true",
        help="stop after checking the v2.6 records; use --resume for the new comparison",
    )
    args = parser.parse_args(argv)
    if Path(sarrl.__file__).resolve().parent.parent != ROOT:
        parser.error("imported package does not match this checkout")
    output = args.output.resolve()
    if output != (ROOT / CANONICAL).resolve():
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
    fingerprint["runtime"]["scipy"] = scipy.__version__
    if scipy.__version__ != "1.18.1":
        parser.error("the filter comparison requires scipy==1.18.1")
    fingerprint["frozen_controller"] = verify_frozen_controller()
    fingerprint["source_sha256"].update({name: sha(ROOT / name) for name in EXTRA_SOURCES})
    previous = json.loads((ROOT / PREVIOUS / "manifest.json").read_text())
    for package in ("numpy", "mujoco", "scipy"):
        if fingerprint["runtime"][package] != previous["runtime"][package]:
            parser.error(f"reproduction requires {package}=={previous['runtime'][package]}")
    protocol = selected_filter_protocol()
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if (
            manifest["protocol"] != protocol
            or any(manifest[k] != v for k, v in fingerprint.items() if k != "runtime")
            or {k: v for k, v in manifest["runtime"].items() if k != "workers"}
            != {k: v for k, v in fingerprint["runtime"].items() if k != "workers"}
        ):
            parser.error("resume requires the original source commit, protocol and runtime")
    else:
        scan = seed_range_is_unopened(ROOT, *SEED_BLOCK, "docs/seed_registry.json")
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
        for condition in protocol["scenarios"]
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
        write_json(output / "decision.json", decide(records))
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
                "reproduction_episodes": 480,
                "hashes": {name: sha(output / name) for name in OUTPUT_FILES},
            },
        )
    print(json.dumps(summary["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
