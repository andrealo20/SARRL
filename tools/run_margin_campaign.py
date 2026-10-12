#!/usr/bin/env python3
"""Execute the preregistered v2.2 uncertainty-margin campaign once and apply its decision rule.

Same guarantees as the v2.1 runner, whose checks it reuses: sealed source
state, committed-artifact seed scan over every reachable commit plus the
seed registry, exclusive lock, session-tagged at-least-once journal that
survives a truncated final record, canonical log reloaded for the analysis,
atomic output files, completion marker hashing every output. The campaign
runs in two phases: the reproduction block (the no-margin arm on the first
hundred v2.1 decision seeds per scenario) is executed, reloaded and checked
field by field against the retained v2.1 log before the first decision seed
is opened; the decision block follows.
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
from sarrl.evaluation.adaptive_pilot import PilotEpisode, run_case
from sarrl.evaluation.margin_campaign import (
    V22_COMPENSATE_DELAY,
    V22_DECISION_EPISODES,
    V22_DECISION_SEED_START,
    V22_FROZEN_PATHS,
    V22_MARGINS,
    V22_OUTPUT,
    V22_PLANT_OPTIONS,
    V22_REPRODUCTION_REFERENCE,
    V22_SEAL_FILE,
    V22_SEED_REGISTRY,
    analyze,
    load_episodes,
    reproduction_check,
    v22_cells,
    v22_config,
    v22_decision_cells,
    v22_protocol_dict,
    v22_reproduction_cells,
)
from sarrl.evaluation.provenance import runtime_metadata
from tools.campaign_guards import (
    journal_records,
    run_two_phases,
    seed_range_is_unopened,
    valid_completion_marker,
)
from tools.run_adaptive_campaign import (
    CampaignLock,
    execution_fingerprint,
    sha,
    verify_frozen_sources,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / V22_OUTPUT
PAIRED_COLUMNS = ("prediction_error_rms", "margin_mean_command")
CSV_EXCLUDED = ("estimated_parameters", "true_parameters", *PAIRED_COLUMNS)
OUTPUT_FILES = ("manifest.json", "journal.jsonl", "episodes.jsonl", "episodes.csv", "decision.json")


def run_cell(cell):
    arm, scenario, seed, plant = cell
    episode = run_case(
        arm,
        scenario,
        seed,
        "official",
        v22_config(),
        V22_COMPENSATE_DELAY,
        plant,
        V22_PLANT_OPTIONS,
        certificate_margin=V22_MARGINS[arm],
    )
    return asdict(episode)


def cell_key(record: dict) -> tuple:
    return (record["arm"], record["scenario"], record["seed"], record["plant"])


JOURNAL_FIELDS = set(PilotEpisode.__dataclass_fields__) | {"session"}


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


def write_csv(path: Path, episodes: list[PilotEpisode]) -> None:
    """One row per episode; paired fields are split into one column per joint."""
    columns = [f.name for f in fields(PilotEpisode) if f.name not in CSV_EXCLUDED]
    for name in PAIRED_COLUMNS:
        columns += [f"{name}_joint1", f"{name}_joint2"]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for episode in episodes:
            row = asdict(episode)
            for name in PAIRED_COLUMNS:
                row[f"{name}_joint1"], row[f"{name}_joint2"] = row.pop(name) or (None, None)
            writer.writerow({name: row[name] for name in columns})
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    assert_repository_import_root(ROOT)
    frozen = verify_frozen_sources(ROOT, V22_SEAL_FILE, V22_FROZEN_PATHS)
    scan = seed_range_is_unopened(
        ROOT,
        V22_DECISION_SEED_START,
        V22_DECISION_SEED_START + V22_DECISION_EPISODES - 1,
        V22_SEED_REGISTRY,
    )
    reference = ROOT / V22_REPRODUCTION_REFERENCE
    fingerprint = execution_fingerprint(args.workers)
    fingerprint["mujoco"] = mujoco.__version__

    reproduction_cells = list(v22_reproduction_cells())
    decision_cells = list(v22_decision_cells())
    cells = list(v22_cells())
    planned = set(cells)
    journal = OUTPUT / "journal.jsonl"
    manifest_path = OUTPUT / "manifest.json"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with CampaignLock(OUTPUT):
        marker = OUTPUT / "complete.json"
        if marker.exists():
            if valid_completion_marker(marker, OUTPUT_FILES):
                raise FileExistsError(f"{OUTPUT} is complete; the official campaign runs once")
            raise RuntimeError(f"{marker} exists but is invalid; inspect it before anything else")
        session = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["protocol"] != v22_protocol_dict():
                raise RuntimeError("existing manifest was written under a different protocol")
            if previous["frozen_sources"]["tree"] != frozen["tree"]:
                raise RuntimeError("existing manifest was written from a different source tree")
            if previous["execution"] != fingerprint:
                raise RuntimeError("execution fingerprint differs from the manifest; refused")
            done = journal_records(journal, planned, JOURNAL_FIELDS, cell_key)
            previous["sessions"] = previous.get("sessions", []) + [session]
            write_json(manifest_path, previous)
            print(f"resuming with {len(done)} completed cells", flush=True)
        else:
            if any(p.name != "campaign.lock" for p in OUTPUT.iterdir()):
                raise RuntimeError(f"{OUTPUT} holds files but no manifest; inspect it first")
            write_json(
                manifest_path,
                {
                    "protocol": v22_protocol_dict(),
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

        def run_block(block):
            execute(block, done, journal, session, args.workers, len(cells))

        def check_reproduction():
            reloaded = journal_records(journal, planned, JOURNAL_FIELDS, cell_key)
            reproduced = load_episodes_from_records([reloaded[c] for c in reproduction_cells])
            report = reproduction_check(reproduced, reference)
            print(
                f"reproduction block: {report['matched']}/{report['compared']} rows match",
                flush=True,
            )
            return report

        # Phase 1 runs and is checked before phase 2 opens a decision seed.
        run_two_phases(reproduction_cells, decision_cells, run_block, check_reproduction)
        done = journal_records(journal, planned, JOURNAL_FIELDS, cell_key)
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
        write_csv(OUTPUT / "episodes.csv", episodes)

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
    print(f"v2.2 decision: {report['decision']}")
    for name in report["failed_statements"]:
        print(f"  failed: {name}")
    for name in report["invalid_estimator_cells"]:
        print(f"  estimator invalid: {name}")
    for scenario, contrasts in report["contrasts"].items():
        margin = contrasts["covariance_minus_none"]
        shape = contrasts["covariance_minus_constant"]
        print(
            f"  {scenario}: unsafe {margin['unsafe_episode']['difference']:+.3f} "
            f"[{margin['unsafe_episode']['ci95_low']:+.3f}, "
            f"{margin['unsafe_episode']['ci95_high']:+.3f}]; "
            f"success {margin['success']['difference']:+.3f} "
            f"[{margin['success']['ci95_low']:+.3f}, {margin['success']['ci95_high']:+.3f}]; "
            f"shape {shape['unsafe_episode']['difference']:+.3f} "
            f"[{shape['unsafe_episode']['ci95_low']:+.3f}, "
            f"{shape['unsafe_episode']['ci95_high']:+.3f}]"
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
