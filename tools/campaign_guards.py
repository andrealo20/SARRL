"""Launch and resume guards shared by the campaign runners, free of any simulator import.

Kept apart from the runners so that the guards are tested where the
simulator is not installed: seed-range hygiene over the whole repository
history and the seed registry, validation of a completion marker, journal
reloading that survives a truncated final record, and the two-phase
schedule that opens no decision cell before the reproduction check passes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

from tools.scan_seed_usage import committed_blobs, scan_blobs


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reachable_trees(root: Path) -> tuple[list[tuple[str, str]], int]:
    """One (commit, tree) per distinct tree over every reachable commit, and the commit count."""
    output = subprocess.run(
        ["git", "rev-list", "--all", "--format=%H %T", "--no-commit-header"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")
    seen = set()
    trees = []
    commits = 0
    for line in output:
        if not line.strip():
            continue
        commit, tree = line.split()
        commits += 1
        if tree not in seen:
            seen.add(tree)
            trees.append((commit, tree))
    return trees, commits


def seed_range_is_unopened(root: Path, low: int, high: int, registry_path: str) -> dict:
    """Scan every reachable commit's tree and the registry; raise on any collision."""
    trees, commits = reachable_trees(root)
    seen: dict = {}
    scanned = []
    for commit, tree in trees:
        entries = committed_blobs(root, tree)
        hits = scan_blobs(root, entries, low, high, seen)
        scanned.append({"commit": commit, "tree": tree, "files": len(entries)})
        if hits:
            raise RuntimeError(
                f"decision seed range already used in commit {commit}: "
                f"{sorted({source for source, _, _ in hits})}"
            )
    registry = json.loads((root / registry_path).read_text())
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
    return {
        "range": [low, high],
        "reachable_commits": commits,
        "distinct_trees_scanned": len(scanned),
        "trees": scanned,
        "blobs_read": len(seen),
        "registry": registry_path,
        "reserved_entry": reserved[0],
    }


def valid_completion_marker(path: Path, output_files: tuple[str, ...]) -> bool:
    """A completion marker counts only if it parses and its hashes match the outputs."""
    try:
        record = json.loads(path.read_text())
        hashes = record["hashes"]
    except (ValueError, KeyError, OSError):
        return False
    return set(hashes) == set(output_files) and all(
        (path.parent / name).exists() and sha256(path.parent / name) == digest
        for name, digest in hashes.items()
    )


def journal_records(journal: Path, planned: set, fields: set, cell_key: Callable) -> dict:
    """Reload complete journal records and repair a torn tail before any append.

    An interrupted append can leave a partial last line, or a complete last
    record without its newline. Every earlier line must parse and validate.
    A partial last line is discarded and physically truncated from the file
    (its cell runs again); a complete last record missing its newline gets
    the newline written back. A malformed line anywhere else, an unplanned
    cell or a duplicate cell stops the resume. Callers hold the campaign
    lock, so the repair races with nothing.
    """
    done = {}
    if not journal.exists():
        return done
    data = journal.read_bytes()
    lines = data.split(b"\n")
    terminated = data.endswith(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    for index, raw in enumerate(lines):
        line = raw.decode("utf-8", errors="replace")
        if not line.strip():
            continue
        last = index == len(lines) - 1
        try:
            record = json.loads(line)
        except ValueError:
            if last and not terminated:
                with journal.open("r+b") as handle:
                    handle.truncate(len(data) - len(raw))
                    handle.flush()
                break
            raise RuntimeError(f"journal record {index + 1} is malformed") from None
        if set(record) != fields:
            raise RuntimeError(f"journal record {index + 1} has unexpected fields")
        record.pop("session")
        key = cell_key(record)
        if key not in planned or key in done:
            raise RuntimeError(f"journal record {index + 1} is not a planned, unique cell")
        done[key] = record
        if last and not terminated:
            with journal.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
    return done


def run_two_phases(
    reproduction_cells: list,
    decision_cells: list,
    execute: Callable[[list], None],
    check: Callable[[], dict],
) -> dict:
    """Execute the reproduction block, check it, and only then execute the decision block."""
    execute(reproduction_cells)
    report = check()
    if report.get("mismatches"):
        raise RuntimeError(
            "reproduction block does not match the retained rows; no decision seed opened: "
            f"{report['mismatches'][:3]}"
        )
    execute(decision_cells)
    return report
