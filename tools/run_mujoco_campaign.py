#!/usr/bin/env python3
"""Execute the preregistered v2.0 MuJoCo campaign once and apply its decision rule.

Same guarantees as the v1.9 runner, whose checks it reuses: sealed source
state, committed-artifact seed scan, exclusive lock, session-tagged
at-least-once journal, canonical log reloaded for the analysis, completion
marker hashing every output. The decision block runs on the MuJoCo plant with
randomised actuator parameters and a measured state; the transfer block runs
the unfiltered fixed nominal on both plants without noise or actuator
options, so its analytical rows must reproduce the retained A0 rows.
"""

# Numerical thread limits precede all numerical imports.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import mujoco

from sarrl.evaluation import assert_repository_import_root
from sarrl.evaluation.adaptive_campaign import V19_REPRODUCTION_REFERENCE
from sarrl.evaluation.adaptive_pilot import PilotEpisode, PlantOptions, run_case
from sarrl.evaluation.mujoco_campaign import (
    V20_COMPENSATE_DELAY,
    V20_FROZEN_PATHS,
    V20_OUTPUT,
    V20_PLANT_OPTIONS,
    V20_PRIMARY_EPISODES,
    V20_PRIMARY_SEED_START,
    V20_SEAL_FILE,
    analyze,
    load_episodes,
    v20_cells,
    v20_config,
    v20_protocol_dict,
)
from sarrl.evaluation.provenance import runtime_metadata
from tools.run_adaptive_campaign import (
    CampaignLock,
    execution_fingerprint,
    sha,
    verify_frozen_sources,
    write_json,
)
from tools.scan_seed_usage import scan_revision

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / V20_OUTPUT
CSV_EXCLUDED = ("estimated_parameters", "true_parameters", "prediction_error_rms")


def run_cell(cell):
    arm, scenario, seed, plant = cell
    # Transfer cells carry no noise and no actuator options on either plant.
    options = V20_PLANT_OPTIONS if seed >= V20_PRIMARY_SEED_START else PlantOptions()
    episode = run_case(
        arm, scenario, seed, "official", v20_config(), V20_COMPENSATE_DELAY, plant, options
    )
    return asdict(episode)


def cell_key(record: dict) -> tuple:
    return (record["arm"], record["scenario"], record["seed"], record["plant"])


JOURNAL_FIELDS = set(PilotEpisode.__dataclass_fields__) | {"session"}


def journal_records(journal: Path, planned: set) -> dict:
    done = {}
    if not journal.exists():
        return done
    for line_number, line in enumerate(journal.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if set(record) != JOURNAL_FIELDS:
            raise RuntimeError(f"journal record {line_number} has unexpected fields")
        record.pop("session")
        key = cell_key(record)
        if key not in planned or key in done:
            raise RuntimeError(f"journal record {line_number} is not a planned, unique cell")
        done[key] = record
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    assert_repository_import_root(ROOT)
    frozen = verify_frozen_sources(ROOT, V20_SEAL_FILE, V20_FROZEN_PATHS)
    scan = scan_revision(
        ROOT, V20_PRIMARY_SEED_START, V20_PRIMARY_SEED_START + V20_PRIMARY_EPISODES - 1
    )
    if scan["hits"]:
        raise RuntimeError(f"decision seed range already used: {scan['files_with_hits']}")
    reference = ROOT / V19_REPRODUCTION_REFERENCE
    fingerprint = execution_fingerprint(args.workers)
    fingerprint["mujoco"] = mujoco.__version__

    cells = list(v20_cells())
    planned = set(cells)
    journal = OUTPUT / "journal.jsonl"
    manifest_path = OUTPUT / "manifest.json"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with CampaignLock(OUTPUT):
        if (OUTPUT / "complete.json").exists():
            raise FileExistsError(f"{OUTPUT} is complete; the official campaign runs once")
        session = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["protocol"] != v20_protocol_dict():
                raise RuntimeError("existing manifest was written under a different protocol")
            if previous["frozen_sources"]["tree"] != frozen["tree"]:
                raise RuntimeError("existing manifest was written from a different source tree")
            if previous["execution"] != fingerprint:
                raise RuntimeError("execution fingerprint differs from the manifest; refused")
            done = journal_records(journal, planned)
            previous["sessions"] = previous.get("sessions", []) + [session]
            write_json(manifest_path, previous)
            print(f"resuming with {len(done)} completed cells", flush=True)
        else:
            if any(p.name != "campaign.lock" for p in OUTPUT.iterdir()):
                raise RuntimeError(f"{OUTPUT} holds files but no manifest; inspect it first")
            write_json(
                manifest_path,
                {
                    "protocol": v20_protocol_dict(),
                    "frozen_sources": frozen,
                    "seed_scan": scan,
                    "reproduction_reference_sha256": sha(reference),
                    "runtime": runtime_metadata(ROOT),
                    "execution": fingerprint,
                    "started_utc": session,
                    "sessions": [session],
                },
            )
            done = {}

        started = time.time()
        pending = [cell for cell in cells if cell not in done]
        with journal.open("a") as log:
            if args.workers == 1:
                iterator = (run_cell(cell) for cell in pending)
            else:
                pool = ProcessPoolExecutor(max_workers=args.workers)
                futures = [pool.submit(run_cell, cell) for cell in pending]
                iterator = (future.result() for future in as_completed(futures))
            for record in iterator:
                log.write(json.dumps({**record, "session": session}) + "\n")
                log.flush()
                os.fsync(log.fileno())
                done[cell_key(record)] = record
                if len(done) % 200 == 0 or len(done) == len(cells):
                    print(f"[{len(done)}/{len(cells)}]", flush=True)
            if args.workers > 1:
                pool.shutdown()

        done = journal_records(journal, planned)
        if set(done) != planned:
            raise RuntimeError("journal does not cover the planned cells")
        log_path = OUTPUT / "episodes.jsonl"
        log_path.write_text("".join(json.dumps(done[cell]) + "\n" for cell in cells))
        episodes = load_episodes(log_path)
        if [(e.arm, e.scenario, e.seed, e.plant) for e in episodes] != cells:
            raise RuntimeError("reloaded episode log does not match the planned cells")

        columns = [f.name for f in fields(PilotEpisode) if f.name not in CSV_EXCLUDED]
        columns += ["prediction_error_rms_joint1", "prediction_error_rms_joint2"]
        with (OUTPUT / "episodes.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for episode in episodes:
                row = asdict(episode)
                rms = row.pop("prediction_error_rms") or (None, None)
                row["prediction_error_rms_joint1"], row["prediction_error_rms_joint2"] = rms
                writer.writerow({name: row[name] for name in columns})

        report = analyze(episodes, reference)
        report["elapsed_seconds_last_session"] = time.time() - started
        report["physical_steps"] = int(sum(e.steps for e in episodes))
        write_json(OUTPUT / "decision.json", report)
        outputs = (
            "manifest.json",
            "journal.jsonl",
            "episodes.jsonl",
            "episodes.csv",
            "decision.json",
        )
        write_json(
            OUTPUT / "complete.json",
            {
                "episodes": len(episodes),
                "decision": report["decision"],
                "hashes": {name: sha(OUTPUT / name) for name in outputs},
            },
        )
    print(f"v2.0 decision: {report['decision']}")
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
    check = report["transfer_check"]["reproduction_of_retained_a0"]
    print(f"  reproduction of retained A0 rows: {check['matched']}/{check['compared']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
