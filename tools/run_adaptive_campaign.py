#!/usr/bin/env python3
"""Execute the preregistered v1.9 adaptive-nominal campaign once and apply its decision rule.

The protocol is frozen in ``sarrl.evaluation.adaptive_campaign`` and in
``docs/experiments.md``. The runner refuses to start unless the frozen source
paths of HEAD are identical to the sealed commit and carry no local change,
unless the official seed range is absent from every retained artifact, and
unless the canonical output directory does not exist. It writes the manifest
before the first episode, appends every episode to a JSON-lines log, reloads
that log for the analysis, and closes with a completion marker hashing every
output. The output path is fixed so that the official seeds are opened once.
"""

# Numerical thread limits precede all numerical imports.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields
from importlib import metadata
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

from sarrl.evaluation import assert_repository_import_root
from sarrl.evaluation.adaptive_campaign import (
    V19_COMPENSATE_DELAY,
    V19_FROZEN_PATHS,
    V19_FROZEN_SOURCE_COMMIT,
    V19_UNOPENED_SEED_RANGE,
    analyze,
    load_episodes,
    v19_cells,
    v19_config,
    v19_protocol_dict,
)
from sarrl.evaluation.adaptive_pilot import PilotEpisode, run_case
from sarrl.evaluation.provenance import runtime_metadata
from tools.scan_seed_usage import scan_tracked

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "adaptive_nominal_v19"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def verify_frozen_sources() -> dict:
    """HEAD's frozen paths must equal the sealed commit and carry no local change."""
    if V19_FROZEN_SOURCE_COMMIT is None:
        raise RuntimeError("the protocol has not been sealed: V19_FROZEN_SOURCE_COMMIT is unset")
    head = git("rev-parse", "HEAD")
    diff = subprocess.run(
        ["git", "diff", "--quiet", V19_FROZEN_SOURCE_COMMIT, "HEAD", "--", *V19_FROZEN_PATHS],
        cwd=ROOT,
        check=False,
    )
    if diff.returncode != 0:
        raise RuntimeError(
            f"frozen paths differ between HEAD {head[:12]} and the sealed commit "
            f"{V19_FROZEN_SOURCE_COMMIT[:12]}"
        )
    status = git("status", "--porcelain", "--untracked-files=all", "--", *V19_FROZEN_PATHS)
    if status:
        raise RuntimeError("frozen paths carry local modifications or untracked files:\n" + status)
    return {
        "head": head,
        "sealed_commit": V19_FROZEN_SOURCE_COMMIT,
        "frozen_paths": list(V19_FROZEN_PATHS),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "path_trees": {path: git("rev-parse", f"HEAD:{path}") for path in V19_FROZEN_PATHS},
    }


def installed_distributions() -> dict:
    return {
        dist.metadata["Name"]: dist.version
        for dist in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower())
    }


def run_cell(cell):
    arm, scenario, seed = cell
    return asdict(run_case(arm, scenario, seed, "official", v19_config(), V19_COMPENSATE_DELAY))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    assert_repository_import_root(ROOT)
    if OUTPUT.exists():
        raise FileExistsError(f"{OUTPUT} exists; the official campaign runs once")
    frozen = verify_frozen_sources()
    scan = scan_tracked(ROOT, *V19_UNOPENED_SEED_RANGE)
    if scan["hits"]:
        raise RuntimeError(f"official seed range already used: {scan['files_with_hits']}")

    OUTPUT.mkdir(parents=True)
    manifest = {
        "protocol": v19_protocol_dict(),
        "frozen_sources": frozen,
        "seed_scan": scan,
        "runtime": runtime_metadata(ROOT),
        "python_executable": sys.executable,
        "installed_distributions": installed_distributions(),
        "workers": args.workers,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(OUTPUT / "manifest.json", manifest)

    cells = list(v19_cells())
    started = time.time()
    log_path = OUTPUT / "episodes.jsonl"
    with log_path.open("w") as log:
        if args.workers == 1:
            results = map(run_cell, cells)
        else:
            pool = ProcessPoolExecutor(max_workers=args.workers)
            results = pool.map(run_cell, cells, chunksize=8)
        for index, record in enumerate(results, start=1):
            log.write(json.dumps(record) + "\n")
            log.flush()
            if index % 200 == 0 or index == len(cells):
                print(f"[{index}/{len(cells)}]", flush=True)
        if args.workers > 1:
            pool.shutdown()

    # The decision is computed from the serialised log, not from in-memory objects.
    episodes = load_episodes(log_path)
    if [(e.arm, e.scenario, e.seed) for e in episodes] != cells:
        raise RuntimeError("reloaded episode log does not match the planned cells")

    columns = [
        f.name
        for f in fields(PilotEpisode)
        if f.name not in ("estimated_parameters", "true_parameters", "prediction_error_rms")
    ]
    with (OUTPUT / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for episode in episodes:
            row = asdict(episode)
            writer.writerow({name: row[name] for name in columns})

    report = analyze(episodes)
    report["elapsed_seconds"] = time.time() - started
    report["physical_steps"] = int(sum(e.steps for e in episodes))
    write_json(OUTPUT / "decision.json", report)
    hashes = {
        name: sha(OUTPUT / name)
        for name in ("manifest.json", "episodes.jsonl", "episodes.csv", "decision.json")
    }
    write_json(
        OUTPUT / "complete.json",
        {"episodes": len(episodes), "decision": report["decision"], "hashes": hashes},
    )
    print(f"v1.9 decision: {report['decision']}")
    for reason in report["inconclusive_reasons"]:
        print(f"  {reason}")
    for scenario, contrast in report["contrasts"].items():
        success = contrast["success"]
        unsafe = contrast["unsafe_episode"]
        print(
            f"  {scenario}: success {success['difference']:+.3f} "
            f"[{success['ci95_low']:+.3f}, {success['ci95_high']:+.3f}]; "
            f"unsafe {unsafe['difference']:+.3f} "
            f"[{unsafe['ci95_low']:+.3f}, {unsafe['ci95_high']:+.3f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
