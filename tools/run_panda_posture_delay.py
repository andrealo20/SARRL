"""Freeze, reproduce and run the paired Panda posture and delay-grid engineering pilot."""

# Numerical thread settings precede imports of numerical libraries.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import sarrl
from sarrl.evaluation.panda_adaptation import reproduction_mismatches
from sarrl.evaluation.panda_delay import run_episode as previous_episode
from sarrl.evaluation.panda_posture_delay import (
    ARMS,
    CONDITIONS,
    posture_delay_protocol,
    run_episode,
    summarize,
)
from sarrl.robots.panda import model_provenance, mujoco
from tools.campaign_guards import seed_range_is_unopened, valid_completion_marker
from tools.run_adaptive_campaign import CampaignLock, sha, write_json
from tools.run_panda_adaptation import (
    OUTPUT_FILES,
    canonical,
    cell_key,
    cell_path,
    load_cell,
    source_fingerprint,
)
from tools.run_panda_adaptation import execute_cells as serial_cells

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIRECTORY = "results/panda_delay_pilot"
EXTRA_SOURCES = (
    "sarrl/envs/panda_interface.py",
    "sarrl/controllers/panda_delay.py",
    "sarrl/evaluation/panda_robustness.py",
    "sarrl/evaluation/panda_delay.py",
    "sarrl/evaluation/panda_posture_delay.py",
    "tools/run_panda_robustness.py",
    "tools/run_panda_delay.py",
    "tools/run_panda_posture_delay.py",
    "tools/make_panda_posture_delay_figure.py",
    "docs/panda-posture-delay.md",
    f"{REFERENCE_DIRECTORY}/manifest.json",
    f"{REFERENCE_DIRECTORY}/episodes.json",
    f"{REFERENCE_DIRECTORY}/complete.json",
)
FROZEN_FILES = (
    "sarrl/controllers/panda_nominal.py",
    "sarrl/controllers/panda_adaptive.py",
    "sarrl/controllers/panda_delay.py",
    "sarrl/robots/panda.py",
    "sarrl/envs/panda_reach.py",
    "sarrl/envs/panda_interface.py",
)


def verify_frozen_controller(root=ROOT):
    reference = root / REFERENCE_DIRECTORY
    if not valid_completion_marker(reference / "complete.json", OUTPUT_FILES):
        raise RuntimeError("the frozen-controller reference has invalid completion hashes")
    manifest = json.loads((reference / "manifest.json").read_text())
    frozen = {name: manifest["source_sha256"][name] for name in FROZEN_FILES}
    if any(sha(root / name) != digest for name, digest in frozen.items()):
        raise RuntimeError("controller, plant and interface must match the previous frozen pilot")
    return {"source_commit": manifest["source_commit"], "source_sha256": frozen}


def _execute_one(execute, key, trace_seed):
    scenario, seed, arm = key
    record, trace = execute(seed, scenario, arm, retain_trace=seed == trace_seed)
    if cell_key(record) != key:
        raise RuntimeError("a pilot episode returned the wrong cell identity")
    return {"record": record, "trace": trace}


def execute_cells(directory, planned, trace_seed, *, execute=run_episode, workers=1):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("workers must be an integer in 1..4")
    if workers == 1:
        return serial_cells(directory, planned, trace_seed, execute=execute)
    directory.mkdir(exist_ok=True)
    expected = {cell_path(directory, key).name for key in planned}
    if any(path.name not in expected for path in directory.glob("*.json")):
        raise RuntimeError("the saved directory contains an unplanned cell")
    pending = []
    for key in planned:
        path = cell_path(directory, key)
        if path.exists():
            load_cell(path, key)
        else:
            pending.append(key)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(_execute_one, execute, key, trace_seed): key for key in pending}
        completed = len(planned) - len(pending)
        try:
            for future in as_completed(futures):
                key = futures[future]
                payload = future.result()
                write_json(
                    cell_path(directory, key),
                    {
                        "payload": payload,
                        "sha256": hashlib.sha256(canonical(payload)).hexdigest(),
                    },
                )
                completed += 1
                print(
                    f"[{completed}/{len(planned)}] {key}: success={payload['record']['success']}",
                    flush=True,
                )
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return [load_cell(cell_path(directory, key), key) for key in planned]


def reproduce_previous(output: Path, *, execute=previous_episode, workers=1):
    reference_directory = ROOT / REFERENCE_DIRECTORY
    if not valid_completion_marker(reference_directory / "complete.json", OUTPUT_FILES):
        raise RuntimeError(
            "the retained delay-identification reference has invalid completion hashes"
        )
    reference = json.loads((reference_directory / "episodes.json").read_text())
    if len(reference) != 560 or len({cell_key(r) for r in reference}) != 560:
        raise RuntimeError("the delay-identification reference must have 560 unique episodes")
    planned = [cell_key(r) for r in reference]
    payloads = execute_cells(
        output / "reproduction_cells", planned, -1, execute=execute, workers=workers
    )
    mismatches = []
    for payload, retained in zip(payloads, reference, strict=True):
        mismatches.extend(
            [*cell_key(retained), field]
            for field in reproduction_mismatches(payload["record"], retained)
        )
    report = {
        "compared": 560,
        "matched": 560 - len({tuple(m[:3]) for m in mismatches}),
        "mismatches": mismatches,
        "reference_sha256": sha(reference_directory / "episodes.json"),
        "physical_steps": sum(p["record"]["physical_steps"] for p in payloads),
        "fields": sorted(reference[0]),
    }
    path = output / "reproduction.json"
    if path.exists() and json.loads(path.read_text()) != report:
        raise RuntimeError("the saved reproduction report differs from its checked cells")
    if not path.exists():
        write_json(path, report)
    if mismatches:
        raise RuntimeError(
            f"delay-identification reproduction failed; no new seed opened: {mismatches[:5]}"
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=9804900)
    parser.add_argument("--output", type=Path, default=Path("results/panda_posture_delay_pilot"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("use 1..4 independent workers")
    if not 1 <= args.episodes <= 100 or not 9804900 <= args.seed_start <= 9805000 - args.episodes:
        parser.error("use 1..100 episodes within the posture/delay block 9804900..9804999")
    if Path(sarrl.__file__).resolve().parent.parent != ROOT:
        parser.error("the imported SARRL package does not match this checkout")
    output = args.output.resolve()
    if output.exists() and not args.resume:
        parser.error("output already exists; --resume is for an interrupted pilot")
    if args.resume and not output.is_dir():
        parser.error("no pilot directory exists to resume")
    if (output / "complete.json").exists():
        if not valid_completion_marker(output / "complete.json", OUTPUT_FILES):
            parser.error("invalid completion hashes; retain and inspect the outputs")
        parser.error("this pilot is complete; its outputs will not be replaced")
    if mujoco is None:
        parser.error("install the mujoco extra before running the Panda pilot")
    fingerprint = source_fingerprint()
    fingerprint["runtime"]["workers"] = args.workers
    fingerprint["frozen_controller"] = verify_frozen_controller()
    fingerprint["source_sha256"].update({p: sha(ROOT / p) for p in EXTRA_SOURCES})
    reference_runtime = json.loads((ROOT / REFERENCE_DIRECTORY / "manifest.json").read_text())[
        "runtime"
    ]
    for package in ("numpy", "mujoco"):
        if fingerprint["runtime"][package] != reference_runtime[package]:
            parser.error(f"reference reproduction requires {package}=={reference_runtime[package]}")
    protocol = posture_delay_protocol(args.episodes, args.seed_start)
    planned = [(s, seed, a) for s in CONDITIONS for seed in protocol["seeds"] for a in ARMS]
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["protocol"] != protocol or any(
            manifest[k] != v for k, v in fingerprint.items()
        ):
            parser.error("resume requires the original protocol, commit, sources and runtime")
    else:
        scan = seed_range_is_unopened(ROOT, 9804900, 9804999, "docs/seed_registry.json")
        manifest = {
            **fingerprint,
            "protocol": protocol,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_provenance(),
            "seed_scan": scan,
        }
        output.mkdir(parents=True)
    with CampaignLock(output):
        if (output / "complete.json").exists():
            parser.error("pilot completed before the lock was acquired; retain its outputs")
        if not args.resume:
            write_json(output / "manifest.json", manifest)
        reproduce_previous(output, workers=args.workers)
        payloads = execute_cells(
            output / "cells",
            planned,
            protocol["seeds"][0],
            execute=run_episode,
            workers=args.workers,
        )
        records = [p["record"] for p in payloads]
        summary = summarize(records)
        write_json(output / "episodes.json", records)
        traces = [
            {"cell": list(cell_key(p["record"])), "trace": p["trace"]}
            for p in payloads
            if p["trace"]
        ]
        temporary = output / "traces.json.gz.tmp"
        with temporary.open("wb") as handle:
            handle.write(gzip.compress(canonical(traces), mtime=0))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output / "traces.json.gz")
        write_json(output / "summary.json", summary)
        write_json(
            output / "complete.json",
            {
                "episodes": len(records),
                "reproduction_episodes": 560,
                "hashes": {name: sha(output / name) for name in OUTPUT_FILES},
            },
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
