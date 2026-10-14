"""Profile saved Panda control cycles without new physical episodes."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from sarrl.controllers.panda_delay import PandaDelayController
from sarrl.envs.panda_reach import quintic_reference
from sarrl.evaluation.panda_adaptation import reproduction_mismatches
from tools.audit_panda_reference_validation import check_replay_numeric, ensure, verify_sources
from tools.campaign_guards import valid_completion_marker
from tools.run_adaptive_campaign import CampaignLock, sha, write_json
from tools.run_panda_reference_validation import OUTPUT_FILES

ROOT = Path(__file__).resolve().parents[1]
PROFILE_FILES = ("manifest.json", "summary.json", "timings.json.gz")


def timing_summary(timings):
    if any(type(r["duration_ns"]) is not int for r in timings):
        raise ValueError("profile durations must be integer nanoseconds")
    values = np.array([r["duration_ns"] for r in timings], dtype=np.int64)
    if not len(values) or np.any(values <= 0):
        raise ValueError("profile requires positive cycle durations")
    quantiles = np.quantile(values / 1e6, [0.50, 0.95, 0.99]).tolist()
    cells = {}
    for condition in sorted({r["condition"] for r in timings}):
        selected = np.array([r["duration_ns"] for r in timings if r["condition"] == condition])
        cells[condition] = {
            "cycles": len(selected),
            "p99_ms": float(np.quantile(selected / 1e6, 0.99)),
            "max_ms": float(np.max(selected) / 1e6),
            "over_period_fraction": float(np.mean(selected > 10_000_000)),
        }
    return {
        "cycles": len(values),
        "p50_ms": quantiles[0],
        "p95_ms": quantiles[1],
        "p99_ms": quantiles[2],
        "max_ms": float(np.max(values) / 1e6),
        "over_period_fraction": float(np.mean(values > 10_000_000)),
        "control_period_ms": 10.0,
        "p99_within_period": quantiles[2] <= 10.0,
        "cells": cells,
    }


def profile_traces(items, lookup, *, repeats=5, clock=perf_counter_ns):
    timings = []
    for repetition in range(repeats + 1):
        for item in items:
            row = lookup[tuple(item["cell"])]
            controller = PandaDelayController(predict=False)
            trace = item["trace"]
            for step, (current, following) in enumerate(zip(trace[:-1], trace[1:], strict=True)):
                q, qd = current["measured_q"], current["measured_qd"]
                begin = clock()
                cq, cv, lookahead = controller.prepare_control(q, qd)
                reference = quintic_reference(
                    row["start_q"], row["goal_q"], current["time_s"] + lookahead, 2.0
                )
                sent = controller.command(cq, cv, *reference)
                controller.observe_transition(
                    q,
                    qd,
                    following["measured_q"],
                    following["measured_qd"],
                    current["torque_nm"],
                    0.01,
                )
                duration = clock() - begin
                check_replay_numeric(sent, current["torque_nm"])
                if repetition:
                    timings.append(
                        {
                            "condition": row["scenario"],
                            "repeat": repetition,
                            "step": step,
                            "duration_ns": duration,
                        }
                    )
            ensure(
                controller.selected_history == row["delay_estimator"]["selected_history"],
                "profiling replay selection differs",
            )
    return timings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/panda_reference_validation")
    parser.add_argument("--output", type=Path, default=ROOT / "results/panda_reference_profile")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args(argv)
    out = args.output
    if args.audit:
        ensure(
            valid_completion_marker(out / "complete.json", PROFILE_FILES), "invalid profile hashes"
        )
        timings = json.loads(gzip.decompress((out / "timings.json.gz").read_bytes()))
        ensure(
            not reproduction_mismatches(
                timing_summary(timings), json.loads((out / "summary.json").read_text())
            ),
            "profile summary mismatch",
        )
        identities = {(r["condition"], r["repeat"], r["step"]) for r in timings}
        manifest = json.loads((out / "manifest.json").read_text())
        verify_sources(manifest)
        planned = {
            (c, repeat, step)
            for c in manifest["conditions"]
            for repeat in range(1, 6)
            for step in range(300)
        }
        ensure(
            len(timings) == len(identities) == 30000 and identities == planned,
            "incomplete profile cycles",
        )
        ensure(
            manifest["physical_steps"] == 0 and manifest["new_episode_seeds"] == [],
            "profile opened physical data",
        )
        ensure(
            sha(args.input / "complete.json") == manifest["input_complete_sha256"],
            "profile input changed",
        )
        print(json.dumps({"verified_profile_cycles": len(timings)}, indent=2))
        return 0
    if out.resolve() != (ROOT / "results/panda_reference_profile").resolve():
        parser.error("profiling requires its canonical output directory")
    if out.exists():
        parser.error("profile output exists; retained timings are never replaced")
    ensure(
        valid_completion_marker(args.input / "complete.json", OUTPUT_FILES),
        "invalid validation hashes",
    )
    source = json.loads((args.input / "manifest.json").read_text())
    verify_sources(source)
    rows = json.loads((args.input / "episodes.json").read_text())
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in rows}
    traces = json.loads(gzip.decompress((args.input / "traces.json.gz").read_bytes()))
    selected = [item for item in traces if item["cell"][2] == "bank_align"]
    expected = {
        (c, source["protocol"]["seeds"][0], "bank_align") for c in source["protocol"]["scenarios"]
    }
    ensure(
        len(selected) == 20
        and {tuple(t["cell"]) for t in selected} == expected
        and all(len(t["trace"]) == 301 for t in selected),
        "wrong reference traces",
    )
    cpu = platform.processor()
    info = Path("/proc/cpuinfo")
    if info.exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in info.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    manifest = {
        "source_commit": source["source_commit"],
        "source_sha256": source["source_sha256"],
        "input_complete_sha256": sha(args.input / "complete.json"),
        "runtime": {**source["runtime"], "workers": 1},
        "model": source["model"],
        "profile_workers": 1,
        "clock": "perf_counter_ns",
        "cpu": cpu,
        "platform": platform.platform(),
        "warmup_passes": 1,
        "measured_passes": 5,
        "conditions": sorted({t["cell"][0] for t in selected}),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "physical_steps": 0,
        "new_episode_seeds": [],
        "scope": "saved observed inputs; construction, plant, I/O and logging excluded",
    }
    out.mkdir(parents=True)
    with CampaignLock(out):
        write_json(out / "manifest.json", manifest)
        timings = profile_traces(selected, lookup)
        summary = timing_summary(timings)
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "manifest.json", manifest)
        write_json(out / "summary.json", summary)
        (out / "timings.json.gz").write_bytes(
            gzip.compress(json.dumps(timings, allow_nan=False, sort_keys=True).encode(), mtime=0)
        )
        write_json(
            out / "complete.json", {"hashes": {name: sha(out / name) for name in PROFILE_FILES}}
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
