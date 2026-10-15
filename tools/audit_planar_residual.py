"""Audit retained residual statistics; optionally verify policies and causal stack traces."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch

try:
    import mujoco
except ImportError:
    mujoco = None

from sarrl.evaluation.adaptive_pilot import build_arm
from sarrl.evaluation.margin_campaign import v22_config
from sarrl.evaluation.panda_adaptation import reproduction_mismatches
from sarrl.evaluation.planar_residual import (
    RecordingStack,
    analyze,
    historical_reproduction_report,
    study_protocol,
)
from sarrl.rl.sac import SACAgent
from tools.campaign_guards import valid_completion_marker
from tools.run_adaptive_campaign import sha
from tools.run_planar_residual import OUTPUT_FILES

ROOT = Path(__file__).resolve().parents[1]


def ensure(condition, message):
    if not condition:
        raise RuntimeError(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/planar_residual_v26")
    parser.add_argument(
        "--models",
        type=Path,
        help="local ignored training directory for policy and causal inference checks",
    )
    args = parser.parse_args(argv)
    out = args.input
    ensure(
        valid_completion_marker(out / "complete.json", OUTPUT_FILES), "invalid completion hashes"
    )
    manifest = json.loads((out / "manifest.json").read_text())
    for name, digest in manifest["source_sha256"].items():
        blob = subprocess.check_output(
            ["git", "show", f"{manifest['source_commit']}:{name}"], cwd=ROOT
        )
        ensure(hashlib.sha256(blob).hexdigest() == digest, f"frozen source mismatch: {name}")
        if name.endswith(".py"):
            ensure(sha(ROOT / name) == digest, f"check out recorded Python sources: {name}")
    ensure(np.__version__ == manifest["runtime"]["numpy"], "recorded NumPy required for analysis")
    ensure(manifest["protocol"] == json.loads(json.dumps(study_protocol())), "protocol mismatch")
    rows = json.loads((out / "episodes.json").read_text())
    protocol = manifest["protocol"]
    planned = {
        (c, s, a)
        for c in protocol["scenarios"]
        for s in protocol["evaluation_seeds"]
        for a in protocol["arms"]
    }
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in rows}
    ensure(len(rows) == len(lookup) == 2100 and set(lookup) == planned, "planned cells mismatch")
    summary = analyze(rows)
    ensure(
        not reproduction_mismatches(summary, json.loads((out / "summary.json").read_text())),
        "summary mismatch",
    )
    ensure(
        not reproduction_mismatches(
            summary["decision"], json.loads((out / "decision.json").read_text())
        ),
        "decision mismatch",
    )
    report = json.loads((out / "reproduction.json").read_text())
    reference_path = ROOT / "results/uncertainty_margin_v22/episodes.jsonl"
    retained = [json.loads(line) for line in reference_path.read_text().splitlines()]
    retained = [
        r for r in retained if r["arm"] == "adaptive_hocbf_margin" and 56200 <= r["seed"] < 56300
    ]
    replayed = json.loads((out / "reproduction_records.json").read_text())
    ensure(
        report == historical_reproduction_report(replayed, retained, sha(reference_path)),
        "historical reproduction report differs from retained records",
    )
    ensure(
        report["compared"] == report["matched"] == 300 and not report["mismatches"],
        "historical reproduction incomplete",
    )
    ensure(
        report["reference_sha256"] == sha(ROOT / "results/uncertainty_margin_v22/episodes.jsonl"),
        "historical reference changed",
    )
    models = json.loads((out / "models.json").read_text())
    ensure(
        set(models) == {str(s) for s in protocol["training_seeds"]},
        "missing or extra trained models",
    )
    ensure(
        all(m["steps"] == 200000 and m["updates"] == 195000 for m in models.values()),
        "training budget differs",
    )
    policies = {}
    if args.models is not None:
        ensure(
            mujoco is not None
            and mujoco.__version__ == manifest["runtime"]["mujoco"]
            and torch.__version__ == manifest["runtime"]["torch"],
            "recorded model runtime required",
        )
        for seed, identity in models.items():
            directory = args.models / f"seed_{seed}"
            ensure(
                sha(directory / "final.pt") == identity["policy_sha256"],
                "local policy hash differs",
            )
            ensure(
                sha(directory / "complete.json") == identity["completion_sha256"],
                "training completion changed",
            )
            marker = json.loads((directory / "complete.json").read_text())
            ensure(
                all(sha(directory / name) == digest for name, digest in marker["hashes"].items()),
                "training artifact hashes differ",
            )
            policies[seed] = SACAgent.from_checkpoint(directory / "final.pt", load_optimizers=False)
    for row in rows:
        ensure(row["outcome"] in ("success", "timeout", "abort"), "unknown outcome")
        ensure(row["safety_infeasible"] == (row["outcome"] == "abort"), "abort flag mismatch")
        ensure(row["success"] == (row["outcome"] == "success"), "success flag mismatch")
        ensure(
            row["plant"] == "mujoco" and row["origin"] == "heldout", "unexpected evaluation scope"
        )
        ensure(
            row["certificate_margin"] == (0.0 if row["arm"] == "fixed" else 0.25),
            "certificate margin mismatch",
        )
        for field in (
            "final_distance",
            "max_speed",
            "max_command_torque",
            "normalized_violation_max",
            "joint_position_violation_max_rad",
            "joint_velocity_violation_max_rad_s",
        ):
            ensure(np.isfinite(row[field]) and row[field] >= 0, f"invalid endpoint: {field}")
        ensure(type(row["steps"]) is int and 0 <= row["steps"] <= 250, "invalid physical horizon")
    traces = json.loads(gzip.decompress((out / "traces.json.gz").read_bytes()))
    expected = {key for key in planned if key[1] == protocol["evaluation_seeds"][0]}
    ensure(
        len(traces) == len(expected) and {tuple(t["cell"]) for t in traces} == expected,
        "trace identities mismatch",
    )
    inferred = causal = 0
    for item in traces:
        row, trace = lookup[tuple(item["cell"])], item["trace"]
        ensure(
            len(trace) == row["steps"] + int(row["safety_infeasible"]),
            "trace attempt count mismatch",
        )
        ensure(bool(trace), "missing first-seed trace")
        policy = (
            policies.get(row["arm"].split("_")[-1]) if row["arm"].startswith("residual_") else None
        )
        stack = controller = None
        if args.models is not None:
            controller, certificate, estimator = build_arm(
                "fixed_hocbf" if row["arm"] == "fixed" else "adaptive_hocbf_margin",
                v22_config(),
                row["certificate_margin"],
            )
            if hasattr(controller, "reset"):
                controller.reset()
            stack = RecordingStack(
                controller,
                certificate,
                compensate_delay=row["arm"] != "fixed",
                policy=policy,
                estimator=estimator,
            )
        residual_squares = []
        for frame in trace:
            state = np.array(frame["physical_state"])
            measured = np.array(frame["measured_state"])
            ensure(
                state.shape == measured.shape == (4,)
                and np.all(np.isfinite(state))
                and np.all(np.isfinite(measured)),
                "invalid recorded state",
            )
            residual = np.array(frame["raw_residual"])
            ensure(
                residual.shape == (2,) and np.all(np.abs(residual) <= 8.0),
                "residual exceeds frozen bound",
            )
            ensure(
                np.max(np.abs(frame["torque"])) <= 40.0 + 1e-8, "sent command exceeds torque bounds"
            )
            if row["arm"] in ("fixed", "adaptive"):
                ensure(np.all(residual == 0), "baseline received residual input")
            else:
                residual_squares.append(float(residual @ residual))
                if policy is not None:
                    action = policy.act(frame["observation"], deterministic=True)
                    np.testing.assert_allclose(
                        np.clip(action, -1, 1) * 8, residual, rtol=1e-7, atol=1e-7
                    )
                    inferred += 1
            if stack is not None:
                command = stack.command(
                    frame["observation"],
                    frame["measured_state"],
                    frame["q_des"],
                    deterministic=True,
                )
                np.testing.assert_allclose(command.torque, frame["torque"], rtol=1e-8, atol=1e-8)
                ensure(command.executable == frame["executed"], "causal executability mismatch")
                if row["arm"] != "fixed" and frame["executed"]:
                    controller.observe(
                        frame["measured_state"], frame["measured_next_state"], command.torque
                    )
            if not frame["executed"]:
                np.testing.assert_array_equal(frame["physical_state"], frame["physical_next_state"])
        if residual_squares:
            np.testing.assert_allclose(
                row["residual_rms"], np.sqrt(np.mean(residual_squares)), rtol=1e-12
            )
        if stack is not None:
            if row["arm"] != "fixed":
                np.testing.assert_allclose(
                    row["estimated_parameters"], controller.parameters, rtol=1e-9, atol=1e-10
                )
            causal += 1
    print(
        json.dumps(
            {
                "verified_episodes": len(rows),
                "verified_historical_records": report["compared"],
                "historical_strict_matches": report["strict_matched"],
                "verified_traces": len(traces),
                "policy_action_checks": inferred,
                "causal_stack_replays": causal,
                "policy_inference_verified": args.models is not None,
                "source_commit": manifest["source_commit"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
