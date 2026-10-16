"""Frozen used-task integration checks for the queue-aware Panda backup controller."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/panda_backup_integration_v26"
ORIGINAL = ROOT / "results/panda_joint_filter_pilot"
FILES = (
    "manifest.json",
    "reference_reproduction.json",
    "episodes.json",
    "traces.json.gz",
    "summary.json",
    "audit.json",
)


def helpers():
    sys.path.insert(0, str(ROOT))
    from tools import recover_panda_failures

    return recover_panda_failures


def selected(records):
    tasks = {
        (r["scenario"], r["seed"])
        for r in records
        if r["arm"] == "bank_filtered" and r["unsafe_episode"]
    }
    conditions = {r["scenario"] for r in records}
    tasks |= {(condition, 9805800) for condition in conditions}
    if len(tasks) != 32 or len(conditions) != 12:
        raise ValueError(
            "the twelve first-seed tasks and twenty retained failures must remain fixed"
        )
    return sorted(tasks)


def sources():
    paths = [
        *ROOT.glob("sarrl/**/*.py"),
        *ROOT.glob("tools/*.py"),
        *ROOT.glob("tests/test_panda*backup*.py"),
        ROOT / "tests/test_panda_supervisor.py",
    ]
    paths += [
        ROOT / name
        for name in (
            "tools/run_panda_backup_integration.py",
            "tools/recover_panda_failures.py",
            "tools/campaign_guards.py",
            "tools/run_adaptive_campaign.py",
            "tools/audit_panda_joint_filter_portable.py",
            "tools/audit_panda_estimator_memory.py",
            "tools/audit_panda_reference_validation.py",
            "docs/panda-backup-integration-methods.md",
        )
    ]
    return sorted({str(path.relative_to(ROOT)).replace("\\", "/") for path in paths})


def records(source):
    helper = helpers()
    original, _ = helper.load_original(source)
    rows = json.loads((ORIGINAL / "episodes.json").read_text())
    return original, rows, selected(rows)


def cell(out, kind, index):
    return out / "cells" / f"{kind}-{index:02d}.json.gz"


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def retain(path, value):
    helper = helpers()
    temporary = Path(str(path) + ".tmp")
    temporary.write_bytes(gzip.compress(helper.canonical(value), mtime=0))
    temporary.replace(path)


def worker(source, out, reference):
    helper = helpers()
    original, rows, tasks = records(source)
    mujoco, _ = helper.runtime(original, source if reference else ROOT)
    if reference:
        from sarrl.evaluation.panda_joint_filter import run_episode

        planned = sorted(
            (r for r in rows if (r["scenario"], r["seed"]) in tasks),
            key=lambda r: (r["scenario"], r["seed"], r["arm"]),
        )
    else:
        from sarrl.evaluation.panda_backup_integration import run_episode

        planned = [{"scenario": c, "seed": s} for c, s in tasks]
    for index, expected in enumerate(planned):
        path = cell(out, "reference" if reference else "candidate", index)
        if path.exists():
            item = read(path)
        else:
            calls, original_step = 0, mujoco.mj_step

            def counted(*args, _step=original_step, **kwargs):
                nonlocal calls
                calls += 1
                return _step(*args, **kwargs)

            mujoco.mj_step = counted
            try:
                if reference:
                    actual, trace = run_episode(
                        expected["seed"], expected["scenario"], expected["arm"]
                    )
                else:
                    actual, trace = run_episode(expected["seed"], expected["scenario"])
            finally:
                mujoco.mj_step = original_step
            item = {
                "record": json.loads(helper.canonical(actual)),
                "trace": trace,
                "physical_steps": calls,
            }
            retain(path, item)
        if item["physical_steps"] != 1500:
            raise ValueError("integration physical-step count differs")
        if reference and helper.mismatches(item["record"], expected):
            raise ValueError(f"reference mismatch retained at {index}")
        if not reference and (item["record"]["scenario"], item["record"]["seed"]) != (
            expected["scenario"],
            expected["seed"],
        ):
            raise ValueError("resumed candidate identity differs")
        print(
            f"{'Reproduced' if reference else 'Retained'} {index + 1}/{len(planned)}: "
            f"{expected['scenario']} {expected['seed']}",
            flush=True,
        )


def projection_auditor():
    argv = sys.argv
    try:
        sys.argv = [
            str(ROOT / "tools/audit_panda_joint_filter_portable.py"),
            "--source-checkout",
            str(ROOT),
        ]
        spec = importlib.util.spec_from_file_location("integration_projection_audit", sys.argv[0])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
    return module.independent_projection_check


def audit_worker(source, out):
    helper = helpers()
    original, rows, tasks = records(source)
    mujoco, _ = helper.runtime(original, ROOT)
    from sarrl.evaluation.panda_backup_integration import selected_tasks, summarize
    from sarrl.evaluation.panda_supervisor_audit import audit_episode

    if selected_tasks(rows) != tasks:
        raise ValueError("integration task selection differs")
    reproduction = json.loads((out / "reference_reproduction.json").read_text())
    planned = [r for r in rows if (r["scenario"], r["seed"]) in tasks]
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in planned}
    actual = reproduction["records"]
    if len(actual) != 64 or len({(r["scenario"], r["seed"], r["arm"]) for r in actual}) != 64:
        raise ValueError("reference reproduction must retain all sixty-four records")
    for row in actual:
        if helper.mismatches(row, lookup[row["scenario"], row["seed"], row["arm"]]):
            raise ValueError("retained reference reproduction differs")
    if reproduction["matched"] != 64 or reproduction["physical_steps"] != 96000:
        raise ValueError("reference reproduction counts differ")
    episodes = json.loads((out / "episodes.json").read_text())
    traces = read(out / "traces.json.gz")
    if [(r["scenario"], r["seed"]) for r in episodes] != tasks or len(traces) != 32:
        raise ValueError("completed integration selection differs")
    if helper.mismatches(summarize(episodes), json.loads((out / "summary.json").read_text())):
        raise ValueError("integration summary differs")
    projection = projection_auditor()
    original_step = mujoco.mj_step

    def prohibited(*args, **kwargs):
        raise RuntimeError("integration audit must not step the physical plant")

    mujoco.mj_step = prohibited
    totals = {}
    try:
        for row, item in zip(episodes, traces, strict=True):
            if item["cell"] != [row["scenario"], row["seed"], row["arm"]]:
                raise ValueError("integration trace identity differs")
            for key, value in audit_episode(row, item["trace"], projection).items():
                totals[key] = totals.get(key, 0) + value
    finally:
        mujoco.mj_step = original_step
    print(json.dumps(totals | {"audit_physical_steps": 0, "reference_records": 64}))


def child(source, out, flag, capture=False):
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--source-checkout",
            str(source),
            "--output",
            str(out),
            flag,
        ],
        cwd=source if flag == "--reference-worker" else ROOT,
        check=True,
        capture_output=capture,
        text=capture,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit", action="store_true")
    for flag in ("reference-worker", "candidate-worker", "audit-worker"):
        parser.add_argument("--" + flag, action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    source, out, helper = args.source_checkout.resolve(), args.output.resolve(), helpers()
    if args.reference_worker or args.candidate_worker:
        worker(source, out, args.reference_worker)
        return 0
    if args.audit_worker:
        audit_worker(source, out)
        return 0
    original, _, tasks = records(source)
    from tools.campaign_guards import valid_completion_marker

    if args.audit:
        if not valid_completion_marker(out / "complete.json", FILES):
            parser.error("invalid integration completion hashes")
        manifest = json.loads((out / "manifest.json").read_text())
        marker = json.loads((out / "complete.json").read_text())
        if (
            manifest["production_commit"] != helper.FROZEN
            or manifest["input_completion_sha256"] != helper.INPUT_HASH
            or manifest["tasks"] != [list(task) for task in tasks]
            or manifest["new_task_seeds"] != []
            or marker["episodes"] != 32
            or marker["physical_steps"] != 144000
        ):
            parser.error("integration provenance or counts differ")
        if set(manifest["source_sha256"]) != set(sources()):
            parser.error("integration source coverage differs")
        for name, digest in manifest["source_sha256"].items():
            blob = subprocess.check_output(
                ["git", "show", f"{manifest['source_commit']}:{name}"], cwd=ROOT
            )
            if helper.sha(ROOT / name) != digest or hashlib.sha256(blob).hexdigest() != digest:
                parser.error("check out the frozen integration implementation")
        checks = json.loads(child(source, out, "--audit-worker", True).stdout)
        if checks != json.loads((out / "audit.json").read_text()):
            parser.error("independent integration audit differs")
        print(json.dumps(checks, indent=2))
        return 0
    if out != OUTPUT.resolve() or (out / "complete.json").exists() or out.exists() != args.resume:
        parser.error("use the canonical unfinished output and --resume for an interruption")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode()
    if any(
        line
        for line in status.splitlines()
        if not line.startswith("?? results/panda_backup_integration_v26/")
    ):
        parser.error("commit the implementation and protocol before integration execution")
    helper.runtime(original, ROOT)
    manifest = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip(),
        "source_sha256": {name: helper.sha(ROOT / name) for name in sources()},
        "production_commit": helper.FROZEN,
        "input_completion_sha256": helper.INPUT_HASH,
        "tasks": [list(task) for task in tasks],
        "new_task_seeds": [],
        "runtime": {
            "python": platform.python_version(),
            "numpy": original["runtime"]["numpy"],
            "scipy": original["runtime"]["scipy"],
            "mujoco": original["runtime"]["mujoco"],
            "workers": 1,
            "numerical_threads": 1,
        },
        "retained_physical_steps": {"reference": 96000, "candidate": 48000, "total": 144000},
        "scope": "used-task integration; no fresh confirmation or candidate advancement",
    }
    if args.resume and json.loads((out / "manifest.json").read_text()) != manifest:
        parser.error("resume requires identical frozen implementation and protocol")
    from tools.run_adaptive_campaign import CampaignLock

    out.mkdir(exist_ok=True)
    with CampaignLock(out):
        if not args.resume:
            helper.write(out / "manifest.json", manifest)
        (out / "cells").mkdir(exist_ok=True)
        child(source, out, "--reference-worker")
        references = [read(cell(out, "reference", index))["record"] for index in range(64)]
        helper.write(
            out / "reference_reproduction.json",
            {"records": references, "matched": 64, "physical_steps": 96000},
        )
        child(source, out, "--candidate-worker")
        items = [read(cell(out, "candidate", index)) for index in range(32)]
        episodes = [item["record"] for item in items]
        from sarrl.evaluation.panda_backup_integration import summarize

        helper.write(out / "episodes.json", episodes)
        retain(
            out / "traces.json.gz",
            [
                {"cell": [r["scenario"], r["seed"], r["arm"]], "trace": item["trace"]}
                for r, item in zip(episodes, items, strict=True)
            ],
        )
        helper.write(out / "summary.json", summarize(episodes))
        helper.write(
            out / "audit.json", json.loads(child(source, out, "--audit-worker", True).stdout)
        )
        helper.write(
            out / "complete.json",
            {
                "episodes": 32,
                "physical_steps": 144000,
                "hashes": {name: helper.sha(out / name) for name in FILES},
            },
        )
    print(json.dumps(summarize(episodes), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
