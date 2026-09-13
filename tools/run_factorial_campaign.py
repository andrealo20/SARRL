#!/usr/bin/env python3
"""Execute the preregistered v2.1 factorial campaign once and apply its decision rule.

Same guarantees as the v1.9 and v2.0 runners, whose checks it reuses: sealed
source state, committed-artifact seed scan (HEAD and every release tag) plus
the seed registry, exclusive lock, session-tagged at-least-once journal,
canonical log reloaded for the analysis, atomic output files, completion
marker hashing every output. The campaign runs in two phases: the
reproduction block (the two v2.0 arms on the first hundred v2.0 decision
seeds per scenario) is executed, reloaded and checked field by field
against the retained v2.0 log before the first decision seed is opened;
the decision block and the descriptive arm follow.
"""

# Numerical thread limits precede all numerical imports.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import mujoco

from sarrl.evaluation import assert_repository_import_root
from sarrl.evaluation.adaptive_pilot import PilotEpisode, run_case
from sarrl.evaluation.factorial_campaign import (
    V21_DECISION_EPISODES,
    V21_DECISION_SEED_START,
    V21_FROZEN_PATHS,
    V21_OUTPUT,
    V21_PLANT_OPTIONS,
    V21_REPRODUCTION_REFERENCE,
    V21_SEAL_FILE,
    V21_SEED_REGISTRY,
    analyze,
    arm_stack_and_compensation,
    load_episodes,
    reproduction_check,
    v21_cells,
    v21_config,
    v21_decision_cells,
    v21_protocol_dict,
    v21_reproduction_cells,
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
OUTPUT = ROOT / V21_OUTPUT
CSV_EXCLUDED = ("estimated_parameters", "true_parameters", "prediction_error_rms")
OUTPUT_FILES = ("manifest.json", "journal.jsonl", "episodes.jsonl", "episodes.csv", "decision.json")


def run_cell(cell):
    arm, scenario, seed, plant = cell
    stack, compensate = arm_stack_and_compensation(arm)
    episode = run_case(
        stack, scenario, seed, "official", v21_config(), compensate, plant, V21_PLANT_OPTIONS
    )
    record = asdict(episode)
    record["arm"] = arm  # the campaign label, which may differ from the stack name
    return record


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


def seed_range_is_unopened(root: Path, low: int, high: int) -> dict:
    """Scan HEAD and every release tag, then the registry; raise on any collision."""
    revisions = ["HEAD"] + subprocess.run(
        ["git", "tag", "--list", "v*"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split()
    scans = {}
    seen: dict = {}
    for revision in revisions:
        scan = scan_revision(root, low, high, revision, seen)
        scans[revision] = {"files_scanned": scan["files_scanned"], "hits": scan["hits"]}
        if scan["hits"]:
            raise RuntimeError(
                f"decision seed range already used in {revision}: {scan['files_with_hits']}"
            )
    registry = json.loads((root / V21_SEED_REGISTRY).read_text())
    overlaps = [
        entry
        for entry in registry["ranges"]
        if entry["low"] <= high and low <= entry["high"] and entry["status"] != "reserved"
    ]
    if overlaps:
        raise RuntimeError(f"decision seed range overlaps registered ranges: {overlaps}")
    reserved = [
        entry
        for entry in registry["ranges"]
        if entry["status"] == "reserved" and entry["low"] == low and entry["high"] == high
    ]
    if len(reserved) != 1:
        raise RuntimeError("decision seed range is not reserved exactly once in the registry")
    return {"revisions": scans, "registry": V21_SEED_REGISTRY, "reserved_entry": reserved[0]}


def valid_completion_marker(path: Path) -> bool:
    """A completion marker counts only if it parses and its hashes match the outputs."""
    try:
        record = json.loads(path.read_text())
        hashes = record["hashes"]
    except (ValueError, KeyError, OSError):
        return False
    return set(hashes) == set(OUTPUT_FILES) and all(
        (path.parent / name).exists() and sha(path.parent / name) == digest
        for name, digest in hashes.items()
    )


def execute(cells, done, journal, session, workers, total):
    pending = [cell for cell in cells if cell not in done]
    with journal.open("a") as log:
        if workers == 1:
            iterator = (run_cell(cell) for cell in pending)
        else:
            pool = ProcessPoolExecutor(max_workers=workers)
            futures = [pool.submit(run_cell, cell) for cell in pending]
            iterator = (future.result() for future in as_completed(futures))
        for record in iterator:
            log.write(json.dumps({**record, "session": session}) + "\n")
            log.flush()
            os.fsync(log.fileno())
            done[cell_key(record)] = record
            if len(done) % 200 == 0 or len(done) == total:
                print(f"[{len(done)}/{total}]", flush=True)
        if workers > 1:
            pool.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    assert_repository_import_root(ROOT)
    frozen = verify_frozen_sources(ROOT, V21_SEAL_FILE, V21_FROZEN_PATHS)
    scan = seed_range_is_unopened(
        ROOT, V21_DECISION_SEED_START, V21_DECISION_SEED_START + V21_DECISION_EPISODES - 1
    )
    reference = ROOT / V21_REPRODUCTION_REFERENCE
    fingerprint = execution_fingerprint(args.workers)
    fingerprint["mujoco"] = mujoco.__version__

    reproduction_cells = list(v21_reproduction_cells())
    decision_cells = list(v21_decision_cells())
    cells = list(v21_cells())
    planned = set(cells)
    journal = OUTPUT / "journal.jsonl"
    manifest_path = OUTPUT / "manifest.json"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with CampaignLock(OUTPUT):
        marker = OUTPUT / "complete.json"
        if marker.exists():
            if valid_completion_marker(marker):
                raise FileExistsError(f"{OUTPUT} is complete; the official campaign runs once")
            raise RuntimeError(f"{marker} exists but is invalid; inspect it before anything else")
        session = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["protocol"] != v21_protocol_dict():
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
                    "protocol": v21_protocol_dict(),
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
        # Phase 1: the reproduction block, checked before any decision seed is opened.
        execute(reproduction_cells, done, journal, session, args.workers, len(cells))
        done = journal_records(journal, planned)
        reproduced = load_episodes_from_records([done[cell] for cell in reproduction_cells])
        check = reproduction_check(reproduced, reference)
        if check["mismatches"]:
            raise RuntimeError(
                "reproduction block does not match the retained v2.0 rows; "
                f"no decision seed opened: {check['mismatches'][:3]}"
            )
        print(f"reproduction block: {check['matched']}/{check['compared']} rows match", flush=True)

        # Phase 2: the decision block and the descriptive arm.
        execute(decision_cells, done, journal, session, args.workers, len(cells))
        done = journal_records(journal, planned)
        if set(done) != planned:
            raise RuntimeError("journal does not cover the planned cells")
        log_path = OUTPUT / "episodes.jsonl"
        temporary = log_path.with_name(log_path.name + ".tmp")
        with temporary.open("w") as handle:
            handle.write("".join(json.dumps(done[cell]) + "\n" for cell in cells))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(log_path)
        episodes = load_episodes(log_path)
        if [(e.arm, e.scenario, e.seed, e.plant) for e in episodes] != cells:
            raise RuntimeError("reloaded episode log does not match the planned cells")

        columns = [f.name for f in fields(PilotEpisode) if f.name not in CSV_EXCLUDED]
        columns += ["prediction_error_rms_joint1", "prediction_error_rms_joint2"]
        csv_temporary = OUTPUT / "episodes.csv.tmp"
        with csv_temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for episode in episodes:
                row = asdict(episode)
                rms = row.pop("prediction_error_rms") or (None, None)
                row["prediction_error_rms_joint1"], row["prediction_error_rms_joint2"] = rms
                writer.writerow({name: row[name] for name in columns})
            handle.flush()
            os.fsync(handle.fileno())
        csv_temporary.replace(OUTPUT / "episodes.csv")

        report = analyze(episodes, reference)
        report["elapsed_seconds_last_session"] = time.time() - started
        report["physical_steps"] = int(sum(e.steps for e in episodes))
        write_json(OUTPUT / "decision.json", report)
        write_json(
            marker,
            {
                "episodes": len(episodes),
                "decision": report["decision"],
                "hashes": {name: sha(OUTPUT / name) for name in OUTPUT_FILES},
            },
        )
    print(f"v2.1 decision: {report['decision']}")
    for name in report["failed_statements"]:
        print(f"  failed: {name}")
    for name in report["invalid_estimator_cells"]:
        print(f"  estimator invalid: {name}")
    for scenario, effects in report["success"].items():
        controller = effects["controller_at_fixed_certificate"]
        certificate = effects["certificate_at_identified_controller"]
        print(
            f"  {scenario}: controller {controller['difference']:+.3f} "
            f"[{controller['ci95_low']:+.3f}, {controller['ci95_high']:+.3f}]; "
            f"certificate {certificate['difference']:+.3f} "
            f"[{certificate['ci95_low']:+.3f}, {certificate['ci95_high']:+.3f}]"
        )
    return 0


def load_episodes_from_records(records: list[dict]) -> list[PilotEpisode]:
    """Validate journal records through the same loader the analysis uses."""
    temporary = OUTPUT / "reproduction_block.jsonl.tmp"
    temporary.write_text("".join(json.dumps(record) + "\n" for record in records))
    try:
        return load_episodes(temporary)
    finally:
        temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
