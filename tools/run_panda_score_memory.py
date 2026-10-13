"""Freeze, reproduce and compare three Panda delay-score memory rules."""

# Numerical thread settings precede imports of numerical libraries.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import gzip
import json
import os
from datetime import datetime, timezone
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import sarrl
from sarrl.evaluation.panda_adaptation import reproduction_mismatches
from sarrl.evaluation.panda_score_memory import (
    ARMS,
    CONDITIONS,
    run_episode,
    score_memory_protocol,
    summarize,
)
from sarrl.evaluation.panda_variable_delay import run_episode as previous_episode
from sarrl.robots.panda import model_provenance, mujoco
from tools.campaign_guards import seed_range_is_unopened, valid_completion_marker
from tools.run_adaptive_campaign import CampaignLock, sha, write_json
from tools.run_panda_adaptation import OUTPUT_FILES, canonical, cell_key, source_fingerprint
from tools.run_panda_variable_delay import EXTRA_SOURCES as PREVIOUS_SOURCES
from tools.run_panda_variable_delay import FROZEN_FILES as PREVIOUS_FROZEN
from tools.run_panda_variable_delay import execute_cells

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIRECTORY = "results/panda_variable_delay_pilot"
FROZEN_FILES = (*PREVIOUS_FROZEN, "sarrl/envs/panda_packet_delay.py")
EXTRA_SOURCES = (
    *PREVIOUS_SOURCES,
    "sarrl/controllers/panda_score_memory.py",
    "sarrl/evaluation/panda_score_memory.py",
    "tools/run_panda_score_memory.py",
    "tools/audit_panda_score_memory.py",
    "tools/make_panda_score_memory_figure.py",
    "docs/panda-score-memory.md",
    f"{REFERENCE_DIRECTORY}/manifest.json",
    f"{REFERENCE_DIRECTORY}/episodes.json",
    f"{REFERENCE_DIRECTORY}/complete.json",
)


def verify_frozen_controller(root=ROOT):
    reference = root / REFERENCE_DIRECTORY
    if not valid_completion_marker(reference / "complete.json", OUTPUT_FILES):
        raise RuntimeError("the score-memory reference has invalid completion hashes")
    manifest = json.loads((reference / "manifest.json").read_text())
    frozen = {name: manifest["source_sha256"][name] for name in FROZEN_FILES}
    if any(sha(root / name) != digest for name, digest in frozen.items()):
        raise RuntimeError("baseline controller, plant and transport must match the previous pilot")
    if model_provenance() != manifest["model"]:
        raise RuntimeError("Panda model provenance must match the previous pilot")
    return {"source_commit": manifest["source_commit"], "source_sha256": frozen}


def reproduce_previous(output, *, execute=previous_episode, workers=1):
    directory = ROOT / REFERENCE_DIRECTORY
    if not valid_completion_marker(directory / "complete.json", OUTPUT_FILES):
        raise RuntimeError("the retained variable-delay reference has invalid completion hashes")
    reference = json.loads((directory / "episodes.json").read_text())
    if len(reference) != 1280 or len({cell_key(r) for r in reference}) != 1280:
        raise RuntimeError("the variable-delay reference must have 1280 unique episodes")
    payloads = execute_cells(
        output / "reproduction_cells",
        [cell_key(r) for r in reference],
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
        "compared": 1280,
        "matched": 1280 - len({tuple(m[:3]) for m in mismatches}),
        "mismatches": mismatches,
        "reference_sha256": sha(directory / "episodes.json"),
        "physical_steps": sum(p["record"]["physical_steps"] for p in payloads),
        "prediction_virtual_substeps": 0,
        "fields": sorted({key for r in reference for key in r}),
    }
    path = output / "reproduction.json"
    if path.exists() and json.loads(path.read_text()) != report:
        raise RuntimeError("saved reproduction report differs from its checked cells")
    if not path.exists():
        write_json(path, report)
    if mismatches:
        raise RuntimeError(
            f"variable-delay reproduction failed; no new seed opened: {mismatches[:5]}"
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=9805300)
    parser.add_argument("--output", type=Path, default=Path("results/panda_score_memory_pilot"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("use 1..4 independent workers")
    if not 1 <= args.episodes <= 100 or not 9805300 <= args.seed_start <= 9805400 - args.episodes:
        parser.error("use 1..100 episodes within the score-memory block 9805300..9805399")
    if Path(sarrl.__file__).resolve().parent.parent != ROOT:
        parser.error("the imported SARRL package does not match this checkout")
    output = args.output.resolve()
    if output.exists() and not args.resume:
        parser.error("output already exists; --resume is for an interrupted pilot")
    if args.resume and not output.is_dir():
        parser.error("no pilot directory exists to resume")
    if (output / "complete.json").exists():
        if not valid_completion_marker(output / "complete.json", OUTPUT_FILES):
            parser.error("invalid completion hashes; retain and inspect the outputs")
        parser.error("this pilot is complete; its outputs will not be replaced")
    if mujoco is None:
        parser.error("install the mujoco extra before running the Panda pilot")
    fingerprint = source_fingerprint()
    fingerprint["runtime"]["workers"] = args.workers
    fingerprint["frozen_controller"] = verify_frozen_controller()
    fingerprint["source_sha256"].update({p: sha(ROOT / p) for p in EXTRA_SOURCES})
    reference_runtime = json.loads((ROOT / REFERENCE_DIRECTORY / "manifest.json").read_text())[
        "runtime"
    ]
    for package in ("numpy", "mujoco"):
        if fingerprint["runtime"][package] != reference_runtime[package]:
            parser.error(f"reference reproduction requires {package}=={reference_runtime[package]}")
    protocol = score_memory_protocol(args.episodes, args.seed_start)
    planned = [(s, seed, a) for s in CONDITIONS for seed in protocol["seeds"] for a in ARMS]
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["protocol"] != protocol or any(
            manifest[k] != v for k, v in fingerprint.items()
        ):
            parser.error("resume requires the original protocol, commit, sources and runtime")
    else:
        scan = seed_range_is_unopened(ROOT, 9805300, 9805399, "docs/seed_registry.json")
        manifest = {
            **fingerprint,
            "protocol": protocol,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_provenance(),
            "seed_scan": scan,
        }
        output.mkdir(parents=True)
    with CampaignLock(output):
        if (output / "complete.json").exists():
            parser.error("pilot completed before the lock was acquired; retain its outputs")
        if not args.resume:
            write_json(output / "manifest.json", manifest)
        reproduce_previous(output, workers=args.workers)
        payloads = execute_cells(
            output / "cells",
            planned,
            protocol["seeds"][0],
            execute=run_episode,
            workers=args.workers,
        )
        records = [p["record"] for p in payloads]
        summary = summarize(records)
        write_json(output / "episodes.json", records)
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
        write_json(output / "summary.json", summary)
        write_json(
            output / "complete.json",
            {
                "episodes": len(records),
                "reproduction_episodes": 1280,
                "hashes": {name: sha(output / name) for name in OUTPUT_FILES},
            },
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
