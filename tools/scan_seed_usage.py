#!/usr/bin/env python3
"""Report every retained artifact that uses an episode seed inside a given range.

Seed columns are recognised by name (`seed`, `episode_seed`, `*_seed`,
`seed_*`), in CSV headers and JSON keys, so step counters and rewards that
happen to fall in the range are not counted. Prints one line per hit and a
final summary; exit status 1 when any hit exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def is_seed_key(name: str) -> bool:
    lower = name.lower()
    return lower == "seed" or lower.endswith("_seed") or lower.startswith("seed")


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


def walk_json(node, path, low, high, hits, source):
    if isinstance(node, dict):
        for key, value in node.items():
            if is_seed_key(str(key)):
                for number in numbers(value):
                    if low <= number <= high:
                        hits.append((source, f"{path}/{key}", number))
            walk_json(value, f"{path}/{key}", low, high, hits, source)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            walk_json(item, f"{path}[{index}]", low, high, hits, source)


def scan_csv(path: Path, low, high, hits):
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
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
                            hits.append((str(path.relative_to(ROOT)), f"row {row_index}", number))


def scan_json(path: Path, low, high, hits):
    text = path.read_text()
    source = str(path.relative_to(ROOT))
    if path.suffix == ".jsonl":
        for line_number, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                walk_json(json.loads(line), f"line {line_number}", low, high, hits, source)
    else:
        walk_json(json.loads(text), "", low, high, hits, source)


def tracked_files(root: Path) -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
    ).stdout
    return [root / name for name in output.decode().split("\0") if name]


def scan_tracked(root: Path, low: int, high: int) -> dict:
    """Scan every tracked CSV/JSON/JSONL file for seed values in [low, high]."""
    global ROOT
    ROOT = root
    hits: list[tuple[str, str, int]] = []
    scanned = 0
    for path in tracked_files(root):
        if path.suffix == ".csv":
            scan_csv(path, low, high, hits)
            scanned += 1
        elif path.suffix in (".json", ".jsonl"):
            scan_json(path, low, high, hits)
            scanned += 1
    return {
        "range": [low, high],
        "files_scanned": scanned,
        "hits": len(hits),
        "distinct_seeds": sorted({number for _, _, number in hits}),
        "files_with_hits": sorted({source for source, _, _ in hits}),
        "examples": [list(hit) for hit in hits[:20]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=int, required=True)
    parser.add_argument("--high", type=int, required=True)
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args()
    summary = scan_tracked(ROOT, args.low, args.high)
    if args.json:
        print(json.dumps(summary, indent=1))
    else:
        for source, where, number in summary["examples"]:
            print(f"{source}: {where}: {number}")
        print(
            f"scanned {summary['files_scanned']} tracked CSV/JSON files; {summary['hits']} "
            f"seed hits in {args.low}..{args.high} across {len(summary['files_with_hits'])} files"
        )
    return 1 if summary["hits"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
