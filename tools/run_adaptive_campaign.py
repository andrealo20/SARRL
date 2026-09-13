#!/usr/bin/env python3
"""Execute the preregistered v1.9 adaptive-nominal campaign once and apply its decision rule.

The protocol is frozen in ``sarrl.evaluation.adaptive_campaign`` and in
``docs/experiments.md``; a seal file outside the frozen paths records the
freeze commit. The runner refuses to start unless the frozen paths of HEAD are
identical to the sealed commit and carry no local change, and unless the
decision seed range is absent from every committed CSV/JSON artifact. It
writes the manifest before the first episode, journals every completed cell
as an atomic record, assembles the canonical ordered log from the journal,
reloads that log for the analysis, and closes with a completion marker
hashing every output. The output path is fixed. An interrupted run resumes
from the validated journal under the same protocol, source tree and
execution fingerprint; a completed run is never rerun. Resume is at least
once: a cell that finished after the last journal write is executed again,
and every record carries the session that produced it. An exclusive lock on
the output directory keeps a second runner out.
"""

# Numerical thread limits precede all numerical imports.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
CSV_EXCLUDED = ("estimated_parameters", "true_parameters", "prediction_error_rms")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def verify_frozen_sources(
    root: Path = ROOT, seal_file: str = V19_SEAL_FILE, frozen_paths=V19_FROZEN_PATHS
) -> dict:
    """HEAD's frozen paths must equal the sealed commit and carry no local change.

    The seal lives outside the frozen paths, so the sealing commit can follow
    the freeze commit without changing what is compared. The seal must name a
    full commit id that is an ancestor of HEAD, and must itself be committed
    and unmodified.
    """
    seal_path = root / seal_file
    if not seal_path.exists():
        raise RuntimeError(f"the protocol has not been sealed: {seal_file} is missing")
    seal = json.loads(seal_path.read_text())
    frozen_commit = str(seal.get("frozen_source_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", frozen_commit):
        raise RuntimeError("the seal must name a full 40-character commit id")
    kind = subprocess.run(
        ["git", "cat-file", "-t", frozen_commit], cwd=root, capture_output=True, text=True
    )
    if kind.returncode != 0 or kind.stdout.strip() != "commit":
        raise RuntimeError("the sealed commit id does not name a commit in this repository")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", frozen_commit, "HEAD"], cwd=root, check=False
    )
    if ancestor.returncode != 0:
        raise RuntimeError("the sealed commit is not an ancestor of HEAD")
    if git(root, "status", "--porcelain", "--untracked-files=all", "--", seal_file):
        raise RuntimeError(f"{seal_file} is modified or not committed")
    head = git(root, "rev-parse", "HEAD")
    diff = subprocess.run(
        ["git", "diff", "--quiet", frozen_commit, "HEAD", "--", *frozen_paths],
        cwd=root,
        check=False,
    )
    if diff.returncode != 0:
        raise RuntimeError(
            f"frozen paths differ between HEAD {head[:12]} and the sealed commit "
            f"{frozen_commit[:12]}"
        )
    status = git(root, "status", "--porcelain", "--untracked-files=all", "--", *frozen_paths)
    if status:
        raise RuntimeError("frozen paths carry local modifications or untracked files:\n" + status)
    return {
        "head": head,
        "sealed_commit": frozen_commit,
        "seal": seal,
        "frozen_paths": list(frozen_paths),
        "tree": git(root, "rev-parse", "HEAD^{tree}"),
        "path_trees": {path: git(root, "rev-parse", f"HEAD:{path}") for path in frozen_paths},
    }


def execution_fingerprint(workers: int) -> dict:
    """Everything a resumed session must share with the first one."""
    return {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "workers": workers,
        "installed_distributions": {
            dist.metadata["Name"]: dist.version
            for dist in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower())
        },
    }


def run_cell(cell):
    arm, scenario, seed = cell
    return asdict(run_case(arm, scenario, seed, "official", v19_config(), V19_COMPENSATE_DELAY))


def cell_key(record: dict) -> tuple:
    return (record["arm"], record["scenario"], record["seed"])


EPISODE_FIELDS = set(PilotEpisode.__dataclass_fields__)
JOURNAL_FIELDS = EPISODE_FIELDS | {"session"}


def journal_records(journal: Path, planned: set) -> dict:
    """Completed cells from the unordered journal, validated against the plan.

    Each line is an episode record plus the session that wrote it. The
    session tag is stripped from the returned records.
    """
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


class CampaignLock:
    """Exclusive lock on the output directory for the whole run, released by the OS on exit.

    POSIX uses flock; Windows uses msvcrt byte-range locking. Both are
    non-blocking and both are dropped automatically if the process dies.
    """

    def __init__(self, output: Path):
        self.path = output / "campaign.lock"
        self.handle = None

    def __enter__(self):
        self.handle = self.path.open("a+")
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise RuntimeError("another runner holds the campaign lock") from exc
        return self

    def __exit__(self, *exc_info):
        if os.name == "nt":
            import msvcrt

            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    assert_repository_import_root(ROOT)
    frozen = verify_frozen_sources()
    scan = scan_revision(
        ROOT, V19_PRIMARY_SEED_START, V19_PRIMARY_SEED_START + V19_PRIMARY_EPISODES - 1
    )
    if scan["hits"]:
        raise RuntimeError(f"decision seed range already used: {scan['files_with_hits']}")
    reference = ROOT / V19_REPRODUCTION_REFERENCE
    fingerprint = execution_fingerprint(args.workers)

    cells = list(v19_cells())
    planned = set(cells)
    journal = OUTPUT / "journal.jsonl"
    manifest_path = OUTPUT / "manifest.json"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with CampaignLock(OUTPUT):
        return _run_locked(
            args, frozen, scan, reference, fingerprint, cells, planned, journal, manifest_path
        )


def _run_locked(args, frozen, scan, reference, fingerprint, cells, planned, journal, manifest_path):
    # Checked under the lock: a runner that waited for the lock must not rewrite
    # the outputs of a run that completed in the meantime.
    if (OUTPUT / "complete.json").exists():
        raise FileExistsError(f"{OUTPUT} is complete; the official campaign runs once")
    session = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if manifest_path.exists():
        # Resume: same protocol, same source tree, same execution fingerprint.
        previous = json.loads(manifest_path.read_text())
        if previous["protocol"] != v19_protocol_dict():
            raise RuntimeError("existing manifest was written under a different protocol")
        if previous["frozen_sources"]["tree"] != frozen["tree"]:
            raise RuntimeError("existing manifest was written from a different source tree")
        if previous["execution"] != fingerprint:
            raise RuntimeError("execution fingerprint differs from the manifest; resume refused")
        done = journal_records(journal, planned)
        previous["sessions"] = previous.get("sessions", []) + [session]
        write_json(manifest_path, previous)
        print(f"resuming with {len(done)} completed cells", flush=True)
    else:
        if any(p.name != "campaign.lock" for p in OUTPUT.iterdir()):
            raise RuntimeError(f"{OUTPUT} holds files but no manifest; inspect before continuing")
        started_utc = session
        write_json(
            manifest_path,
            {
                "protocol": v19_protocol_dict(),
                "frozen_sources": frozen,
                "seed_scan": scan,
                "reproduction_reference_sha256": sha(reference),
                "runtime": runtime_metadata(ROOT),
                "execution": fingerprint,
                "started_utc": started_utc,
                "sessions": [started_utc],
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
            # One line per completed cell, flushed and synced at once. A crash
            # between a worker finishing and this write loses only that cell,
            # which a resumed session executes again (at least once, never lost).
            log.write(json.dumps({**record, "session": session}) + "\n")
            log.flush()
            os.fsync(log.fileno())
            done[cell_key(record)] = record
            if len(done) % 200 == 0 or len(done) == len(cells):
                print(f"[{len(done)}/{len(cells)}]", flush=True)
        if args.workers > 1:
            pool.shutdown()

    # Canonical ordered log assembled from the journal as it stands on disk,
    # re-validated in full, then reloaded for the analysis.
    done = journal_records(journal, planned)
    if set(done) != planned:
        raise RuntimeError("journal does not cover the planned cells")
    log_path = OUTPUT / "episodes.jsonl"
    log_path.write_text("".join(json.dumps(done[cell]) + "\n" for cell in cells))
    episodes = load_episodes(log_path)
    if [(e.arm, e.scenario, e.seed) for e in episodes] != cells:
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
    outputs = ("manifest.json", "journal.jsonl", "episodes.jsonl", "episodes.csv", "decision.json")
    hashes = {name: sha(OUTPUT / name) for name in outputs}
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
    check = report["reproduction_check"]
    print(f"  reproduction of retained A0 rows: {check['matched']}/{check['compared']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
