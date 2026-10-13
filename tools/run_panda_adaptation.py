"""Run the paired Panda adaptation engineering pilot with checked cell-level resume."""

# Numerical thread settings precede imports of numerical libraries.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

import sarrl
from sarrl.evaluation.panda_adaptation import (
    ARMS,
    SCENARIOS,
    pilot_protocol,
    reproduction_mismatches,
    run_episode,
    summarize,
)
from sarrl.evaluation.panda_nominal import run_episode as run_nominal_episode
from sarrl.robots.panda import model_provenance, mujoco
from tools.campaign_guards import seed_range_is_unopened, valid_completion_marker
from tools.run_adaptive_campaign import CampaignLock, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "results/panda_nominal_pilot/episodes.json"
OUTPUT_FILES = (
    "manifest.json",
    "reproduction.json",
    "episodes.json",
    "traces.json.gz",
    "summary.json",
)
SOURCES = (
    "sarrl/robots/panda.py",
    "sarrl/controllers/panda_nominal.py",
    "sarrl/controllers/panda_adaptive.py",
    "sarrl/envs/panda_reach.py",
    "sarrl/evaluation/panda_nominal.py",
    "sarrl/evaluation/panda_adaptation.py",
    "tools/run_panda_adaptation.py",
    "tools/run_adaptive_campaign.py",
    "tools/campaign_guards.py",
    "tools/scan_seed_usage.py",
    "docs/seed_registry.json",
    "docs/panda-adaptation.md",
    "pyproject.toml",
    "sarrl/assets/panda/source.json",
    REFERENCE,
)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def cell_key(record: dict) -> tuple[str, int, str]:
    return record["scenario"], record["seed"], record["arm"]


def cell_path(directory: Path, key: tuple) -> Path:
    scenario, seed, arm = key
    return directory / f"{scenario}__{seed}__{arm}.json"


def load_cell(path: Path, key: tuple) -> dict:
    stored = json.loads(path.read_text())
    if set(stored) != {"payload", "sha256"}:
        raise RuntimeError(f"unexpected saved-cell fields: {path.name}")
    if hashlib.sha256(canonical(stored["payload"])).hexdigest() != stored["sha256"]:
        raise RuntimeError(f"saved-cell hash mismatch: {path.name}")
    if cell_key(stored["payload"]["record"]) != key:
        raise RuntimeError(f"saved-cell identity mismatch: {path.name}")
    return stored["payload"]


def execute_cells(directory: Path, planned: list[tuple], trace_seed: int, *, execute=run_episode):
    directory.mkdir(exist_ok=True)
    expected = {cell_path(directory, key).name for key in planned}
    if any(path.name not in expected for path in directory.glob("*.json")):
        raise RuntimeError("the saved directory contains an unplanned cell")
    # Validate all existing evidence before opening any pending episode.
    for key in planned:
        path = cell_path(directory, key)
        if path.exists():
            load_cell(path, key)
    for index, key in enumerate(planned):
        path = cell_path(directory, key)
        if path.exists():
            load_cell(path, key)
            continue
        scenario, seed, arm = key
        record, trace = execute(seed, scenario, arm, retain_trace=seed == trace_seed)
        if cell_key(record) != key:
            raise RuntimeError("a pilot episode returned the wrong cell identity")
        payload = {"record": record, "trace": trace}
        write_json(
            path, {"payload": payload, "sha256": hashlib.sha256(canonical(payload)).hexdigest()}
        )
        print(
            f"[{index + 1}/{len(planned)}] {scenario} {seed} {arm}: success={record['success']}",
            flush=True,
        )
    # Analyze the saved canonical records, including cells loaded on resume.
    return [load_cell(cell_path(directory, key), key) for key in planned]


def source_fingerprint() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    ).splitlines()
    dirty = [line for line in dirty if not line[3:].startswith("results/")]
    if dirty:
        raise RuntimeError(f"commit source changes before running the Panda pilot: {dirty[:8]}")
    return {
        "source_commit": commit,
        "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "platform": platform.platform(),
            "workers": 1,
            "numerical_threads": 1,
        },
    }


def reproduce_nominal(path: Path) -> dict:
    reference_hash = sha(ROOT / REFERENCE)
    if path.exists():
        report = json.loads(path.read_text())
        if report.get("reference_sha256") != reference_hash or report.get("mismatches"):
            raise RuntimeError("saved reproduction result is invalid")
        if report.get("matched") != 20 or report.get("compared") != 20:
            raise RuntimeError("saved reproduction result is incomplete")
        return report
    reference = json.loads((ROOT / REFERENCE).read_text())
    if len(reference) != 20:
        raise RuntimeError("nominal reproduction reference must contain twenty rows")
    mismatches = []
    for retained in reference:
        actual, _ = run_nominal_episode(retained["seed"])
        for name in reproduction_mismatches(actual, retained):
            mismatches.append([retained["seed"], name])
    report = {
        "compared": 20,
        "matched": 20 - len({m[0] for m in mismatches}),
        "fields": sorted(reference[0]),
        "mismatches": mismatches,
        "reference_sha256": reference_hash,
    }
    write_json(path, report)
    if mismatches:
        raise RuntimeError(
            f"nominal reproduction failed; no new pilot seed opened: {mismatches[:5]}"
        )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=9804300)
    parser.add_argument("--output", type=Path, default=Path("results/panda_adaptation_pilot"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.episodes <= 100 or not 9804300 <= args.seed_start <= 9804400 - args.episodes:
        parser.error("use 1..100 episodes inside the Panda adaptation pilot block 9804300..9804399")
    if Path(sarrl.__file__).resolve().parent.parent != ROOT:
        parser.error("the imported SARRL package does not match this checkout")
    output = args.output.resolve()
    if output.exists() and not args.resume:
        parser.error("output already exists; use --resume only for an interrupted pilot")
    if args.resume and not output.is_dir():
        parser.error("no pilot directory exists to resume")
    if (output / "complete.json").exists():
        if not valid_completion_marker(output / "complete.json", OUTPUT_FILES):
            parser.error(
                "existing completion marker has invalid hashes; retain and inspect the files"
            )
        parser.error("this pilot is complete; its outputs will not be replaced")
    if mujoco is None:
        parser.error("install the mujoco extra before running the Panda pilot")
    fingerprint = source_fingerprint()
    protocol = pilot_protocol(args.episodes, args.seed_start)
    planned = [(s, seed, arm) for s in SCENARIOS for seed in protocol["seeds"] for arm in ARMS]
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["protocol"] != protocol or any(
            manifest[k] != v for k, v in fingerprint.items()
        ):
            parser.error("resume requires the original protocol, commit, sources and runtime")
    else:
        scan = seed_range_is_unopened(ROOT, 9804300, 9804399, "docs/seed_registry.json")
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
            parser.error("this pilot completed before the lock was acquired; retain its outputs")
        if not args.resume:
            write_json(output / "manifest.json", manifest)
        reproduce_nominal(output / "reproduction.json")
        payloads = execute_cells(output / "cells", planned, protocol["seeds"][0])
        records = [payload["record"] for payload in payloads]
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
                "reproduction_episodes": 20,
                "hashes": {name: sha(output / name) for name in OUTPUT_FILES},
            },
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
