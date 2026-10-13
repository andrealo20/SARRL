"""Freeze, reproduce and run the paired Panda noise/delay engineering pilot."""

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
from sarrl.evaluation.panda_adaptation import run_episode as previous_episode
from sarrl.evaluation.panda_robustness import (
    ARMS,
    CONDITIONS,
    robustness_protocol,
    run_episode,
    summarize,
)
from sarrl.robots.panda import model_provenance, mujoco
from tools.campaign_guards import seed_range_is_unopened, valid_completion_marker
from tools.run_adaptive_campaign import CampaignLock, sha, write_json
from tools.run_panda_adaptation import (
    OUTPUT_FILES,
    canonical,
    cell_key,
    execute_cells,
    source_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIRECTORY = "results/panda_adaptation_pilot"
EXTRA_SOURCES = (
    "sarrl/envs/panda_interface.py",
    "sarrl/evaluation/panda_robustness.py",
    "tools/run_panda_robustness.py",
    "docs/panda-robustness.md",
    f"{REFERENCE_DIRECTORY}/manifest.json",
    f"{REFERENCE_DIRECTORY}/episodes.json",
    f"{REFERENCE_DIRECTORY}/complete.json",
)


def reproduce_previous(output: Path, *, execute=previous_episode):
    reference_directory = ROOT / REFERENCE_DIRECTORY
    if not valid_completion_marker(reference_directory / "complete.json", OUTPUT_FILES):
        raise RuntimeError("the retained adaptation reference has invalid completion hashes")
    reference = json.loads((reference_directory / "episodes.json").read_text())
    if len(reference) != 240 or len({cell_key(r) for r in reference}) != 240:
        raise RuntimeError("the adaptation reference must have 240 unique episodes")
    planned = [cell_key(r) for r in reference]
    payloads = execute_cells(output / "reproduction_cells", planned, -1, execute=execute)
    mismatches = []
    for payload, retained in zip(payloads, reference, strict=True):
        mismatches.extend(
            [*cell_key(retained), field]
            for field in reproduction_mismatches(payload["record"], retained)
        )
    report = {
        "compared": 240,
        "matched": 240 - len({tuple(m[:3]) for m in mismatches}),
        "mismatches": mismatches,
        "reference_sha256": sha(reference_directory / "episodes.json"),
        "physical_steps": sum(p["record"]["physical_steps"] for p in payloads),
        "fields": sorted(reference[0]),
    }
    path = output / "reproduction.json"
    if path.exists() and json.loads(path.read_text()) != report:
        raise RuntimeError("the saved reproduction report differs from its checked cells")
    if not path.exists():
        write_json(path, report)
    if mismatches:
        raise RuntimeError(f"adaptation reproduction failed; no new seed opened: {mismatches[:5]}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=9804500)
    parser.add_argument("--output", type=Path, default=Path("results/panda_robustness_pilot"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.episodes <= 100 or not 9804500 <= args.seed_start <= 9804600 - args.episodes:
        parser.error("use 1..100 episodes within the noise/delay block 9804500..9804599")
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
    fingerprint["source_sha256"].update({p: sha(ROOT / p) for p in EXTRA_SOURCES})
    reference_runtime = json.loads((ROOT / REFERENCE_DIRECTORY / "manifest.json").read_text())[
        "runtime"
    ]
    for package in ("numpy", "mujoco"):
        if fingerprint["runtime"][package] != reference_runtime[package]:
            parser.error(f"reference reproduction requires {package}=={reference_runtime[package]}")
    protocol = robustness_protocol(args.episodes, args.seed_start)
    planned = [(s, seed, a) for s in CONDITIONS for seed in protocol["seeds"] for a in ARMS]
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["protocol"] != protocol or any(
            manifest[k] != v for k, v in fingerprint.items()
        ):
            parser.error("resume requires the original protocol, commit, sources and runtime")
    else:
        scan = seed_range_is_unopened(ROOT, 9804500, 9804599, "docs/seed_registry.json")
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
        reproduce_previous(output)
        payloads = execute_cells(
            output / "cells", planned, protocol["seeds"][0], execute=run_episode
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
                "reproduction_episodes": 240,
                "hashes": {name: sha(output / name) for name in OUTPUT_FILES},
            },
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
