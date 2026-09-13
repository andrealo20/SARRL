#!/usr/bin/env python3
"""Execute the preregistered v1.9 adaptive-nominal campaign once and apply its decision rule.

The protocol is frozen in ``sarrl.evaluation.adaptive_campaign`` and in
``docs/experiments.md``; a seal file outside the frozen paths records the
freeze commit. The runner refuses to start unless the frozen paths of HEAD are
identical to the sealed commit and carry no local change, and unless the
decision seed range is absent from every committed CSV/JSON artifact. It
writes the manifest before the first episode, appends every episode to a
JSON-lines log, reloads that log for the analysis, and closes with a
completion marker hashing every output. The output path is fixed. An
interrupted run resumes from its validated log prefix; a completed run is
never rerun.
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
    V19_PRIMARY_EPISODES,
    V19_PRIMARY_SEED_START,
    V19_REPRODUCTION_REFERENCE,
    V19_SEAL_FILE,
    analyze,
    load_episodes,
    v19_cells,
    v19_config,
    v19_protocol_dict,
)
from sarrl.evaluation.adaptive_pilot import PilotEpisode, run_case
from sarrl.evaluation.provenance import runtime_metadata
from tools.scan_seed_usage import scan_revision

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "adaptive_nominal_v19"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def verify_frozen_sources(root: Path = ROOT, seal_file: str = V19_SEAL_FILE) -> dict:
    """HEAD's frozen paths must equal the sealed commit and carry no local change.

    The seal lives outside the frozen paths, so the sealing commit can follow
    the freeze commit without changing what is compared. The seal file itself
    must be committed and unmodified.
    """
    seal_path = root / seal_file
    if not seal_path.exists():
        raise RuntimeError(f"the protocol has not been sealed: {seal_file} is missing")
    seal = json.loads(seal_path.read_text())
    frozen_commit = seal["frozen_source_commit"]
    if git(root, "status", "--porcelain", "--untracked-files=all", "--", seal_file):
        raise RuntimeError(f"{seal_file} is modified or not committed")
    head = git(root, "rev-parse", "HEAD")
    diff = subprocess.run(
        ["git", "diff", "--quiet", frozen_commit, "HEAD", "--", *V19_FROZEN_PATHS],
        cwd=root,
        check=False,
    )
    if diff.returncode != 0:
        raise RuntimeError(
            f"frozen paths differ between HEAD {head[:12]} and the sealed commit "
            f"{frozen_commit[:12]}"
        )
    status = git(root, "status", "--porcelain", "--untracked-files=all", "--", *V19_FROZEN_PATHS)
    if status:
        raise RuntimeError("frozen paths carry local modifications or untracked files:\n" + status)
    return {
        "head": head,
        "sealed_commit": frozen_commit,
        "seal": seal,
        "frozen_paths": list(V19_FROZEN_PATHS),
        "tree": git(root, "rev-parse", "HEAD^{tree}"),
        "path_trees": {path: git(root, "rev-parse", f"HEAD:{path}") for path in V19_FROZEN_PATHS},
    }


def installed_distributions() -> dict:
    return {
        dist.metadata["Name"]: dist.version
        for dist in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower())
    }


def run_cell(cell):
    arm, scenario, seed = cell
    return asdict(run_case(arm, scenario, seed, "official", v19_config(), V19_COMPENSATE_DELAY))


def build_manifest(frozen: dict, scan: dict, workers: int) -> dict:
    return {
        "protocol": v19_protocol_dict(),
        "frozen_sources": frozen,
        "seed_scan": scan,
        "runtime": runtime_metadata(ROOT),
        "python_executable": sys.executable,
        "installed_distributions": installed_distributions(),
        "workers": workers,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def validated_prefix(log_path: Path, cells: list) -> int:
    """Number of planned cells already present, in order, in an existing log."""
    if not log_path.exists():
        return 0
    done = load_episodes(log_path)
    keys = [(e.arm, e.scenario, e.seed) for e in done]
    if keys != cells[: len(keys)]:
        raise RuntimeError("existing episode log does not match the planned cell order")
    return len(keys)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    assert_repository_import_root(ROOT)
    if (OUTPUT / "complete.json").exists():
        raise FileExistsError(f"{OUTPUT} is complete; the official campaign runs once")
    frozen = verify_frozen_sources()
    scan = scan_revision(
        ROOT, V19_PRIMARY_SEED_START, V19_PRIMARY_SEED_START + V19_PRIMARY_EPISODES - 1
    )
    if scan["hits"]:
        raise RuntimeError(f"decision seed range already used: {scan['files_with_hits']}")

    cells = list(v19_cells())
    log_path = OUTPUT / "episodes.jsonl"
    manifest_path = OUTPUT / "manifest.json"
    if OUTPUT.exists():
        # Resume: the manifest must describe this protocol and this source state.
        if not manifest_path.exists():
            raise RuntimeError(f"{OUTPUT} exists without a manifest; inspect it before continuing")
        previous = json.loads(manifest_path.read_text())
        if previous["protocol"] != v19_protocol_dict():
            raise RuntimeError("existing manifest was written under a different protocol")
        if previous["frozen_sources"]["tree"] != frozen["tree"]:
            raise RuntimeError("existing manifest was written from a different source tree")
        start = validated_prefix(log_path, cells)
        print(f"resuming after {start} validated episodes", flush=True)
    else:
        OUTPUT.mkdir(parents=True)
        write_json(manifest_path, build_manifest(frozen, scan, args.workers))
        start = 0

    started = time.time()
    pending = cells[start:]
    with log_path.open("a") as log:
        if args.workers == 1:
            results = map(run_cell, pending)
        else:
            pool = ProcessPoolExecutor(max_workers=args.workers)
            results = pool.map(run_cell, pending, chunksize=8)
        for index, record in enumerate(results, start=start + 1):
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

    report = analyze(episodes, ROOT / V19_REPRODUCTION_REFERENCE)
    report["elapsed_seconds_last_session"] = time.time() - started
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
    check = report.get("reproduction_check")
    if check:
        print(f"  reproduction of retained A0 rows: {check['matched']}/{check['compared']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
