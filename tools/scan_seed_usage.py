#!/usr/bin/env python3
"""Report every retained artifact that uses an episode seed inside a given range.

The scan reads the committed blobs of a revision (default ``HEAD``), not the
working tree, so a local edit cannot hide a collision. Seed fields are
recognised conservatively: any CSV column or JSON key whose name contains
``seed`` (case-insensitive) counts, which covers ``seed``, ``episode_seed``,
``training_seeds``, ``common_episode_seeds`` and also a few non-seed fields
such as ``steps_per_seed``; the latter only matter if their values fall inside
the range. Keys ending in ``start`` are expanded into a range when a sibling
key gives the matching ``end`` or a count. Step counters and rewards are never
seeds. Exit status 1 when any hit exists.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COUNT_KEYS = ("episodes", "count", "n", "num", "size", "samples", "length")


def is_seed_key(name: str) -> bool:
    return "seed" in name.lower()


def numbers(value):
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield value
    elif isinstance(value, float) and value.is_integer():
        yield int(value)
    elif isinstance(value, str):
        try:
            yield int(value)
        except ValueError:
            return
    elif isinstance(value, list):
        for item in value:
            yield from numbers(item)


def expand_ranges(node: dict):
    """Yield (key, start, end) for seed keys ending in start with a sibling end or count."""
    for key, value in node.items():
        lower = key.lower()
        if not is_seed_key(key) or not lower.endswith("start"):
            continue
        starts = list(numbers(value))
        if len(starts) != 1:
            continue
        prefix = key[: -len("start")]
        end_key = next((k for k in node if k.lower() == (prefix + "end").lower()), None)
        if end_key is not None:
            ends = list(numbers(node[end_key]))
            if len(ends) == 1:
                yield key, starts[0], ends[0]
                continue
        count_key = next(
            (k for k in node if k.lower() in COUNT_KEYS or k.lower().endswith("episodes")), None
        )
        if count_key is not None:
            counts = list(numbers(node[count_key]))
            if len(counts) == 1 and 0 < counts[0] <= 1_000_000:
                yield key, starts[0], starts[0] + counts[0] - 1


def walk_json(node, path, low, high, hits, source):
    if isinstance(node, dict):
        for key, start, end in expand_ranges(node):
            if start <= high and end >= low:
                hits.append((source, f"{path}/{key} range {start}..{end}", max(start, low)))
        for key, value in node.items():
            if is_seed_key(str(key)):
                for number in numbers(value):
                    if low <= number <= high:
                        hits.append((source, f"{path}/{key}", number))
            walk_json(value, f"{path}/{key}", low, high, hits, source)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            walk_json(item, f"{path}[{index}]", low, high, hits, source)


def scan_csv_text(text: str, source: str, low, high, hits):
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if not header:
        return
    columns = [i for i, name in enumerate(header) if is_seed_key(name)]
    if not columns:
        return
    for row_index, row in enumerate(reader, start=2):
        for column in columns:
            if column < len(row):
                for number in numbers(row[column]):
                    if low <= number <= high:
                        hits.append((source, f"row {row_index}", number))


def scan_json_text(text: str, source: str, low, high, hits):
    if source.endswith(".jsonl"):
        for line_number, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                walk_json(json.loads(line), f"line {line_number}", low, high, hits, source)
    else:
        walk_json(json.loads(text), "", low, high, hits, source)


def committed_blobs(root: Path, revision: str) -> list[tuple[str, str]]:
    """(path, blob id) of every CSV/JSON/JSONL file in the revision's tree."""
    output = subprocess.run(
        ["git", "ls-tree", "-r", "-z", revision], cwd=root, capture_output=True, check=True
    ).stdout.decode()
    entries = []
    for record in output.split("\0"):
        if not record:
            continue
        meta, path = record.split("\t", 1)
        _, kind, blob = meta.split(" ")
        if kind == "blob" and path.endswith((".csv", ".json", ".jsonl")):
            entries.append((path, blob))
    return entries


def scan_blobs(root: Path, entries, low: int, high: int, seen: dict) -> list:
    """Scan (path, blob) entries; `seen` maps blob ids to their hits so shared blobs scan once."""
    hits: list[tuple[str, str, int]] = []
    for path, blob in entries:
        if blob in seen:
            hits.extend((path, where, number) for _, where, number in seen[blob])
            continue
        text = subprocess.run(
            ["git", "cat-file", "-p", blob], cwd=root, capture_output=True, check=True
        ).stdout.decode("utf-8", errors="replace")
        found: list[tuple[str, str, int]] = []
        if path.endswith(".csv"):
            scan_csv_text(text, path, low, high, found)
        else:
            scan_json_text(text, path, low, high, found)
        seen[blob] = found
        hits.extend(found)
    return hits


def scan_revision(
    root: Path, low: int, high: int, revision: str = "HEAD", seen: dict | None = None
) -> dict:
    """Scan the committed CSV/JSON/JSONL blobs of a revision for seeds in [low, high].

    Pass one `seen` dictionary across calls to scan several revisions without
    reading a blob they share more than once.
    """
    entries = committed_blobs(root, revision)
    hits = scan_blobs(root, entries, low, high, {} if seen is None else seen)
    tree = subprocess.run(
        ["git", "rev-parse", f"{revision}^{{tree}}"], cwd=root, capture_output=True, check=True
    ).stdout.decode().strip()
    return {
        "revision": revision,
        "tree": tree,
        "range": [low, high],
        "files_scanned": len(entries),
        "hits": len(hits),
        "distinct_seeds": sorted({number for _, _, number in hits}),
        "files_with_hits": sorted({source for source, _, _ in hits}),
        "examples": [list(hit) for hit in hits[:20]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=int, required=True)
    parser.add_argument("--high", type=int, required=True)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args()
    summary = scan_revision(ROOT, args.low, args.high, args.revision)
    if args.json:
        print(json.dumps(summary, indent=1))
    else:
        for source, where, number in summary["examples"]:
            print(f"{source}: {where}: {number}")
        print(
            f"scanned {summary['files_scanned']} committed CSV/JSON files at {args.revision}; "
            f"{summary['hits']} seed hits in {args.low}..{args.high} across "
            f"{len(summary['files_with_hits'])} files"
        )
    return 1 if summary["hits"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
