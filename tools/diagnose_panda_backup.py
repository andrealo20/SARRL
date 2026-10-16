"""Describe paired outcomes and event ordering from completed backup integration traces."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from sarrl.evaluation.panda_failure_timing import failure_timing
from sarrl.robots.panda import VELOCITY_LIMIT, load_panda_model
from tools.campaign_guards import valid_completion_marker
from tools.recover_panda_failures import INPUT_HASH, sha
from tools.run_panda_backup_integration import FILES

ROOT = Path(__file__).resolve().parents[1]


def diagnose(out):
    if not valid_completion_marker(out / "complete.json", FILES):
        raise ValueError("completed integration hashes are required")
    original = ROOT / "results/panda_joint_filter_pilot"
    if sha(original / "complete.json") != INPUT_HASH:
        raise ValueError("original evidence identity differs")
    marker = json.loads((original / "complete.json").read_text())
    if any(sha(original / name) != digest for name, digest in marker["hashes"].items()):
        raise ValueError("original evidence content differs")
    rows = json.loads((out / "episodes.json").read_text())
    traces = json.loads(gzip.decompress((out / "traces.json.gz").read_bytes()))
    original_rows = json.loads((original / "episodes.json").read_text())
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in original_rows}
    model, reports = load_panda_model(), []
    for row, item in zip(rows, traces, strict=True):
        if item["cell"] != [row["scenario"], row["seed"], row["arm"]]:
            raise ValueError("trace identity differs")
        if row["unsafe_episode"]:
            # Adapt labels for the existing interval/FIFO timing calculation, without changing data.
            adapted = dict(row, arm="bank_filtered", filter=row["backup"])
            frames = [
                dict(
                    frame,
                    filter={
                        "braking": frame["supervisor"]["aborted"],
                        "reason": frame["supervisor"]["abort_reason"],
                    },
                )
                for frame in item["trace"][:-1]
            ] + [item["trace"][-1]]
            report = failure_timing(
                adapted, frames, model.jnt_range[:, 0], model.jnt_range[:, 1], VELOCITY_LIMIT
            )
            report["cell"] = item["cell"]
            report["first_violation_phase"] = report["first_violation_phase"].replace(
                "braking_delivered", "fallback_delivered"
            )
            for name in list(report):
                if "braking_delivery" in name:
                    report[name.replace("braking_delivery", "fallback_delivery")] = report.pop(name)
            report["unsafe_substeps_by_phase"] = {
                name.replace("braking_delivered", "fallback_delivered"): value
                for name, value in report["unsafe_substeps_by_phase"].items()
            }
            if report["first_violation_delivered_command_kind"] == "braking":
                report["first_violation_delivered_command_kind"] = "fallback"
            reports.append(report)
    paired = {}
    for arm in ("bank_align", "bank_filtered"):
        group = [lookup[r["scenario"], r["seed"], arm] for r in rows]
        paired[arm] = {
            "successes": sum(r["success"] for r in group),
            "unsafe": sum(r["unsafe_episode"] for r in group),
            "aborts": sum(r.get("filter", {}).get("aborted", False) for r in group),
        }
    cycles = np.array([f["control_cycle_ms"] for item in traces for f in item["trace"][:-1]])
    return {
        "input_completion_sha256": sha(out / "complete.json"),
        "scope": (
            "post-hoc saved-data description of selected used tasks; "
            "no physical steps or confirmation"
        ),
        "paired_original_arms": paired,
        "newly_unsafe_against_original_filter": [
            [r["scenario"], r["seed"]]
            for r in rows
            if r["unsafe_episode"]
            and not lookup[r["scenario"], r["seed"], "bank_filtered"]["unsafe_episode"]
        ],
        "no_longer_unsafe_against_original_filter": [
            [r["scenario"], r["seed"]]
            for r in rows
            if not r["unsafe_episode"]
            and lookup[r["scenario"], r["seed"], "bank_filtered"]["unsafe_episode"]
        ],
        "first_violation_phases": {
            phase: sum(r["first_violation_phase"] == phase for r in reports)
            for phase in ("before_abort", "pending_after_abort", "fallback_delivered")
        },
        "timing": {
            "cycles": len(cycles),
            "median_ms": float(np.median(cycles)),
            "p99_ms": float(np.quantile(cycles, 0.99)),
            "maximum_ms": float(np.max(cycles)),
            "over_10_ms": int(np.sum(cycles > 10)),
        },
        "cases": reports,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/panda_backup_integration_v26")
    args = parser.parse_args(argv)
    print(json.dumps(diagnose(args.input.resolve()), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
