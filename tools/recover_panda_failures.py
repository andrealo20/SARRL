"""Recover complete traces for the twenty retained unsafe Panda filter cases."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/panda_failure_replay_v26"
ORIGINAL = ROOT / "results/panda_joint_filter_pilot"
FROZEN = "6891dd27edf8b28514160d1ade350e3213826058"
INPUT_HASH = "e7f62d3732b20c9ef49dbdaa6777ebd3ee50672ae616d5ab18ebce411c0318c8"
SOURCES = (
    "tools/recover_panda_failures.py",
    "sarrl/evaluation/panda_failure_timing.py",
    "sarrl/evaluation/control_diagnosis.py",
    "tools/audit_panda_joint_filter_portable.py",
    "tests/test_panda_failure_replay.py",
    "docs/panda-failure-replay-methods.md",
    "sarrl/robots/panda.py",
    "tools/campaign_guards.py",
    "tools/run_adaptive_campaign.py",
)
FILES = ("manifest.json", "episodes.json", "traces.json.gz", "diagnostics.json", "audit.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write(path, value):
    temp = Path(str(path) + ".tmp")
    temp.write_bytes(json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
    temp.replace(path)


def mismatches(actual, expected, path=""):
    if type(actual) is not type(expected):
        return [path]
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            return [path]
        return [p for k in expected for p in mismatches(actual[k], expected[k], f"{path}.{k}")]
    if isinstance(expected, list):
        if len(actual) != len(expected):
            return [path]
        return [
            p
            for i, (a, b) in enumerate(zip(actual, expected, strict=True))
            for p in mismatches(a, b, f"{path}[{i}]")
        ]
    if isinstance(expected, float):
        equal = (
            math.isfinite(actual)
            and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10)
        )
    else:
        equal = actual == expected
    return [] if equal else [path]


def select_cases(rows):
    chosen = sorted(
        (r for r in rows if r["arm"] == "bank_filtered" and r["unsafe_episode"]),
        key=lambda r: (r["scenario"], r["seed"]),
    )
    keys = [(r["scenario"], r["seed"], r["arm"]) for r in chosen]
    if (
        len(chosen) != 20
        or len(set(keys)) != 20
        or any(not 9805800 <= r["seed"] < 9805820 for r in chosen)
        or any(
            sum(r["scenario"] == c for r in chosen) != 10
            for c in ("fast_clean_d40", "fast_noise_d40")
        )
    ):
        raise ValueError("the twenty retained unsafe cells must remain the selection")
    return chosen


def load_original(source):
    if sha(ORIGINAL / "complete.json") != INPUT_HASH:
        raise ValueError("original evidence identity changed")
    for name, digest in json.loads((ORIGINAL / "complete.json").read_text())["hashes"].items():
        if sha(ORIGINAL / name) != digest:
            raise ValueError(f"original evidence changed: {name}")
    manifest = json.loads((ORIGINAL / "manifest.json").read_text())
    if manifest["source_commit"] != FROZEN:
        raise ValueError("unexpected production source")
    for name, digest in manifest["source_sha256"].items():
        if name.endswith(".py") and sha(source / name) != digest:
            raise ValueError(f"original Python source differs: {name}")
    return manifest, select_cases(json.loads((ORIGINAL / "episodes.json").read_text()))


def runtime(manifest, source):
    sys.path.insert(0, str(source))
    import numpy as np
    import scipy

    import sarrl
    from sarrl.robots.panda import model_provenance, mujoco

    if Path(sarrl.__file__).resolve().parent != source / "sarrl":
        raise RuntimeError("worker must import the frozen production checkout")
    actual = {"numpy": np.__version__, "scipy": scipy.__version__, "mujoco": mujoco.__version__}
    if any(v != manifest["runtime"][k] for k, v in actual.items()):
        raise RuntimeError("recorded Panda numerical runtime required")
    if model_provenance() != manifest["model"]:
        raise RuntimeError("Panda model differs from original")
    return mujoco, actual


def cell_path(out, index):
    return out / "cells" / f"{index:02d}.json.gz"


def read_cell(out, index, row):
    item = json.loads(gzip.decompress(cell_path(out, index).read_bytes()))
    if item["cell"] != [row["scenario"], row["seed"], row["arm"]]:
        raise ValueError("replay cell identity differs")
    if item["mismatches"] or mismatches(item["record"], row):
        raise ValueError("retained replay mismatch; preserve it for investigation")
    if item["physical_step_calls"] != 1500 or len(item["trace"]) != 301:
        raise ValueError("replay horizon differs")
    return item


def read_items(out, rows):
    if not (out / "complete.json").exists():
        return [read_cell(out, i, row) for i, row in enumerate(rows)]
    records = json.loads((out / "episodes.json").read_text())
    traces = json.loads(gzip.decompress((out / "traces.json.gz").read_bytes()))

    def key(row):
        return row["scenario"], row["seed"], row["arm"]

    lookup = {key(r): r for r in records}
    trace_lookup = {tuple(t["cell"]): t["trace"] for t in traces}
    planned = {key(r) for r in rows}
    if (
        len(records) != len(traces)
        or len(records) != 20
        or set(lookup) != planned
        or set(trace_lookup) != planned
    ):
        raise ValueError("public replay selection differs")
    items = []
    for row in rows:
        record, trace = lookup[key(row)], trace_lookup[key(row)]
        if mismatches(record, row) or len(trace) != 301:
            raise ValueError("public replay record or horizon differs")
        items.append({"cell": list(key(row)), "record": record, "trace": trace})
    return items


def replay_worker(source, out):
    manifest, rows = load_original(source)
    mujoco, versions = runtime(manifest, source)
    from sarrl.evaluation.panda_joint_filter import run_episode

    for index, row in enumerate(rows):
        if cell_path(out, index).exists():
            read_cell(out, index, row)
            continue
        calls = 0
        original_step = mujoco.mj_step

        def counted(*args, _step=original_step, **kwargs):
            nonlocal calls
            calls += 1
            return _step(*args, **kwargs)

        mujoco.mj_step = counted
        try:
            actual, trace = run_episode(row["seed"], row["scenario"], row["arm"], retain_trace=True)
        finally:
            mujoco.mj_step = original_step
        actual = json.loads(canonical(actual))
        difference = mismatches(actual, row)
        item = {
            "cell": [row["scenario"], row["seed"], row["arm"]],
            "record": actual,
            "trace": trace,
            "mismatches": difference,
            "physical_step_calls": calls,
            "runtime": versions,
        }
        target = cell_path(out, index)
        temporary = Path(str(target) + ".tmp")
        temporary.write_bytes(gzip.compress(canonical(item), mtime=0))
        temporary.replace(target)
        if difference or calls != 1500:
            raise RuntimeError(f"reproduction stopped at {item['cell']}: {difference[:8]}")
        print(f"Reproduced {index + 1}/20: {row['scenario']} {row['seed']}", flush=True)


def audit_worker(source, out):
    manifest, rows = load_original(source)
    mujoco, _ = runtime(manifest, source)
    import numpy as np

    from sarrl.controllers.panda_filtered import PandaFilteredController
    from sarrl.envs.panda_reach import quintic_reference
    from sarrl.evaluation.panda_joint_filter import CONFIG

    spec = importlib.util.spec_from_file_location(
        "retained_projection_audit", ROOT / "tools/audit_panda_joint_filter_portable.py"
    )
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    original_step = mujoco.mj_step

    def prohibited(*args, **kwargs):
        raise RuntimeError("audit must not step the physical plant")

    mujoco.mj_step = prohibited
    projections = 0
    try:
        for row, item in zip(rows, read_items(out, rows), strict=True):
            trace = item["trace"]
            audit.check_noise(trace, row["seed"], row["interface_config"], row)
            controller, sent = PandaFilteredController(CONFIG), []
            lag = row["interface_config"]["command_delay_steps"]
            for step, frame in enumerate(trace[:-1]):
                reference = quintic_reference(
                    row["start_q"],
                    row["goal_q"],
                    frame["time_s"],
                    row["task_config"]["move_seconds"],
                )
                audit.check_replay_numeric(reference[0], frame["reference"])
                audit.check_replay_numeric(
                    frame["estimated_parameters"], controller.estimator.parameters
                )
                audit.check_replay_numeric(frame["delay_scores"], controller.scores)
                command = controller.command(frame["measured_q"], frame["measured_qd"], *reference)
                audit.check_replay_numeric(command, frame["torque_nm"])
                if mismatches(json.loads(canonical(controller.last_filter)), frame["filter"]):
                    raise ValueError("causal filter diagnostics differ")
                audit.check_replay_numeric(
                    controller.last_nominal_command, frame["nominal_command_nm"]
                )
                if frame["filter"]["predicted_feasible"]:
                    audit.independent_projection_check(
                        controller,
                        frame["measured_q"],
                        frame["measured_qd"],
                        command,
                        frame["nominal_command_nm"],
                    )
                    projections += 1
                sent.append(np.asarray(frame["torque_nm"]))
                if np.any(np.abs(command) > controller.dynamics.torque_limit + 1e-12):
                    raise ValueError("sent command exceeds physical torque bounds")
                np.testing.assert_array_equal(
                    frame["delivered_torque_nm"], sent[step - lag] if step >= lag else np.zeros(7)
                )
                if len(frame["physical_substeps"]) != 5:
                    raise ValueError("missing physical substeps")
                for key in ("q", "qd"):
                    np.testing.assert_array_equal(
                        frame["physical_substeps"][-1][key], trace[step + 1][key]
                    )
                controller.observe_transition(
                    frame["measured_q"],
                    frame["measured_qd"],
                    trace[step + 1]["measured_q"],
                    trace[step + 1]["measured_qd"],
                    command,
                    0.01,
                )
            if mismatches(
                json.loads(canonical(controller.filter_diagnostics())), row["filter"]
            ) or mismatches(
                json.loads(canonical(controller.estimator.diagnostics())), row["estimator"]
            ):
                raise ValueError("final causal controller diagnostics differ")
            for field in (
                "scores",
                "selected_history",
                "candidate_estimates",
                "candidate_bound_hits",
            ):
                if mismatches(controller.delay_diagnostics()[field], row["delay_estimator"][field]):
                    raise ValueError("causal delay-bank diagnostics differ")
    finally:
        mujoco.mj_step = original_step
    print(
        json.dumps(
            {
                "verified_cases": 20,
                "causal_controller_replays": 20,
                "causal_control_cycles": 6000,
                "physical_trace_substeps": 30000,
                "independent_feasible_projection_checks": projections,
                "audit_physical_steps": 0,
            }
        )
    )


def child(source, out, flag, *, capture=False):
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            flag,
            "--source-checkout",
            str(source),
            "--output",
            str(out),
        ],
        cwd=source,
        check=True,
        capture_output=capture,
        text=capture,
    )


def analyze(out, rows):
    from sarrl.evaluation.panda_failure_timing import failure_timing, summarize_failures
    from sarrl.robots.panda import VELOCITY_LIMIT, load_panda_model

    model = load_panda_model()
    items = read_items(out, rows)
    reports = [
        failure_timing(
            row, item["trace"], model.jnt_range[:, 0], model.jnt_range[:, 1], VELOCITY_LIMIT
        )
        for row, item in zip(rows, items, strict=True)
    ]
    return items, {"summary": summarize_failures(reports), "cases": reports}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--audit-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    source, out = args.source_checkout.resolve(), args.output.resolve()
    if args.worker or args.audit_worker:
        (audit_worker if args.audit_worker else replay_worker)(source, out)
        return 0
    from tools.campaign_guards import valid_completion_marker
    from tools.run_adaptive_campaign import CampaignLock

    manifest, rows = load_original(source)
    if args.audit:
        if not valid_completion_marker(out / "complete.json", FILES):
            parser.error("invalid recovered evidence hashes")
        retained = json.loads((out / "manifest.json").read_text())
        if (
            retained["input_completion_sha256"] != INPUT_HASH
            or retained["production_commit"] != FROZEN
        ):
            parser.error("recovery provenance differs")
        if set(retained["source_sha256"]) != set(SOURCES):
            parser.error("recovery source coverage differs")
        for name, digest in retained["source_sha256"].items():
            blob = subprocess.check_output(
                ["git", "show", f"{retained['recovery_commit']}:{name}"], cwd=ROOT
            )
            if hashlib.sha256(blob).hexdigest() != digest or sha(ROOT / name) != digest:
                parser.error(f"check out the frozen recovery source: {name}")
        _, data = analyze(out, rows)
        if mismatches(data, json.loads((out / "diagnostics.json").read_text())):
            parser.error("recovered event diagnosis differs")
        checks = json.loads(child(source, out, "--audit-worker", capture=True).stdout)
        if checks != json.loads((out / "audit.json").read_text()):
            parser.error("causal audit totals differ")
        print(json.dumps(checks, indent=2))
        return 0
    if out != OUTPUT.resolve() or (out / "complete.json").exists() or out.exists() != args.resume:
        parser.error("use the canonical unfinished output; interrupted runs require --resume")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        parser.error("commit the recovery implementation before execution")
    provenance = {
        "recovery_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip(),
        "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
        "production_commit": FROZEN,
        "input_completion_sha256": INPUT_HASH,
        "selected_cells": [[r["scenario"], r["seed"], r["arm"]] for r in rows],
        "runtime": manifest["runtime"],
        "workers": 1,
        "record_tolerance": {"rtol": 1e-9, "atol": 1e-10, "discrete_types": "exact"},
        "new_task_seeds": [],
        "retained_replay_physical_steps": 30000,
        "scope": "unchanged-source reproduction on used seeds; no new candidate",
    }
    if args.resume and json.loads((out / "manifest.json").read_text()) != provenance:
        parser.error("resume requires the original recovery source and protocol")
    out.mkdir(exist_ok=True)
    with CampaignLock(out):
        if not args.resume:
            write(out / "manifest.json", provenance)
        (out / "cells").mkdir(exist_ok=True)
        child(source, out, "--worker")
        items, data = analyze(out, rows)
        checks = json.loads(child(source, out, "--audit-worker", capture=True).stdout)
        write(out / "episodes.json", [item["record"] for item in items])
        (out / "traces.json.gz").write_bytes(
            gzip.compress(
                canonical([{"cell": item["cell"], "trace": item["trace"]} for item in items]),
                mtime=0,
            )
        )
        write(out / "diagnostics.json", data)
        write(out / "audit.json", checks)
        write(
            out / "complete.json",
            {
                "episodes": 20,
                "physical_steps": 30000,
                "hashes": {name: sha(out / name) for name in FILES},
            },
        )
    print(json.dumps(data["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
