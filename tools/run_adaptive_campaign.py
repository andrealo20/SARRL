#!/usr/bin/env python3
"""Execute the preregistered v1.9 adaptive-nominal campaign and apply its decision rule.

The protocol is frozen in ``sarrl.evaluation.adaptive_campaign`` and in
``docs/experiments.md`` at the source commit recorded in the manifest. The
runner writes the manifest before the first episode, appends every episode
to a JSON-lines log as it completes, and closes the campaign with the
analysis and a completion marker that hashes every output.
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
import time
from dataclasses import asdict, fields
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

from sarrl.evaluation import assert_repository_import_root
from sarrl.evaluation.adaptive_campaign import (
    V19_COMPENSATE_DELAY,
    analyze,
    v19_cells,
    v19_config,
    v19_protocol_dict,
)
from sarrl.evaluation.adaptive_pilot import PilotEpisode, run_case
from sarrl.evaluation.provenance import runtime_metadata

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "sarrl/controllers/adaptive_nominal.py",
    "sarrl/controllers/computed_torque.py",
    "sarrl/dynamics/planar_arm.py",
    "sarrl/envs/planar_reach.py",
    "sarrl/evaluation/adaptive_campaign.py",
    "sarrl/evaluation/adaptive_pilot.py",
    "sarrl/evaluation/safety_audit.py",
    "sarrl/evaluation/planar_v12.py",
    "sarrl/evaluation/planar_v13.py",
    "sarrl/safety/filter.py",
    "tools/run_adaptive_campaign.py",
    "docs/experiments.md",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def working_tree_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/adaptive_nominal_v19"))
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run even when tracked sources differ from the recorded commit",
    )
    args = parser.parse_args()
    assert_repository_import_root(ROOT)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if not output.resolve().is_relative_to(ROOT / "results"):
        raise ValueError("campaign output must be inside repository results")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; the campaign runs once into a new directory")
    output.mkdir(parents=True, exist_ok=True)

    dirty = [line for line in working_tree_status() if not line.startswith("??")]
    if dirty and not args.allow_dirty:
        raise RuntimeError("tracked files are modified; commit them so the manifest is exact")

    manifest = {
        "protocol": v19_protocol_dict(),
        "source_hashes": {name: sha(ROOT / name) for name in SOURCES},
        "runtime": runtime_metadata(ROOT),
        "tracked_modifications": dirty,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(output / "manifest.json", manifest)

    config = v19_config()
    cells = list(v19_cells())
    episodes: list[PilotEpisode] = []
    started = time.time()
    with (output / "episodes.jsonl").open("w") as log:
        for index, (arm, scenario, seed) in enumerate(cells, start=1):
            episode = run_case(arm, scenario, seed, "official", config, V19_COMPENSATE_DELAY)
            episodes.append(episode)
            log.write(json.dumps(asdict(episode)) + "\n")
            log.flush()
            if index % 40 == 0 or index == len(cells):
                print(f"[{index}/{len(cells)}] {arm} {scenario} {seed}", flush=True)

    columns = [
        f.name
        for f in fields(PilotEpisode)
        if f.name not in ("estimated_parameters", "true_parameters", "prediction_error_rms")
    ]
    with (output / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for episode in episodes:
            row = asdict(episode)
            writer.writerow({name: row[name] for name in columns})

    report = analyze(episodes)
    report["elapsed_seconds"] = time.time() - started
    report["physical_steps"] = int(sum(e.steps for e in episodes))
    write_json(output / "decision.json", report)
    hashes = {
        name: sha(output / name)
        for name in ("manifest.json", "episodes.jsonl", "episodes.csv", "decision.json")
    }
    write_json(
        output / "complete.json",
        {"episodes": len(episodes), "decision": report["decision"], "hashes": hashes},
    )
    print(f"v1.9 decision: {report['decision']}")
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
