"""Post-hoc saved-state review of frozen Panda backup prediction discrepancies."""

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

from tools.campaign_guards import valid_completion_marker
from tools.recover_panda_failures import canonical, mismatches, sha, write

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/panda_backup_integration_v26"
OUTPUT = ROOT / "results/panda_prediction_gap_position_v26"
INPUT_HASH = "351674a90849d64e7359fa101f72aee86ea1536ff2713b1b79d558b1d2f9831a"
FILES = ("manifest.json", "diagnostics.json", "curves.json.gz")
FOCUS = {("fast_clean_d00", 9805800), ("fast_noise_d00", 9805800)}
SOURCES = (
    "sarrl/evaluation/panda_prediction_gap.py",
    "tools/diagnose_panda_prediction_gap.py",
    "tests/test_panda_prediction_gap.py",
    "docs/panda-prediction-gap-methods.md",
    "docs/panda-prediction-gap-position-methods.md",
    "sarrl/controllers/panda_adaptive.py",
    "sarrl/controllers/panda_nominal.py",
    "sarrl/robots/panda.py",
    "tools/campaign_guards.py",
    "tools/recover_panda_failures.py",
)


def calculate():
    import numpy as np
    import scipy

    from sarrl.controllers.panda_adaptive import PandaRegressor
    from sarrl.evaluation.panda_prediction_gap import acceleration_replacements, prediction_gap
    from sarrl.robots.panda import VELOCITY_LIMIT, model_provenance, mujoco

    marker = INPUT / "complete.json"
    if sha(marker) != INPUT_HASH:
        raise ValueError("integration input identity changed")
    hashes = json.loads(marker.read_text())["hashes"]
    if not valid_completion_marker(marker, tuple(hashes)):
        raise ValueError("integration input content differs")
    manifest = json.loads((INPUT / "manifest.json").read_text())
    runtime = {"numpy": np.__version__, "scipy": scipy.__version__, "mujoco": mujoco.__version__}
    if any(value != manifest["runtime"][key] for key, value in runtime.items()):
        raise ValueError("use the retained Panda numerical runtime")
    original = json.loads((ROOT / "results/panda_joint_filter_pilot/manifest.json").read_text())
    if model_provenance() != original["model"]:
        raise ValueError("Panda model provenance differs")
    rows = json.loads((INPUT / "episodes.json").read_text())
    traces = json.loads(gzip.decompress((INPUT / "traces.json.gz").read_bytes()))
    if (
        len(rows) != 32
        or len(traces) != 32
        or len({(r["scenario"], r["seed"]) for r in rows}) != 32
    ):
        raise ValueError("all thirty-two retained task identities are required")
    original_step = mujoco.mj_step

    def prohibited(*args, **kwargs):
        raise RuntimeError("physical stepping is prohibited in prediction-gap diagnosis")

    mujoco.mj_step = prohibited
    try:
        regressor, reports, curves = PandaRegressor(), [], []
        bounds = regressor.nominal.model.jnt_range
        for row, item in zip(rows, traces, strict=True):
            if item["cell"] != [row["scenario"], row["seed"], row["arm"]]:
                raise ValueError("trace identity differs")
            if not row["backup"]["aborted"]:
                continue
            report, curve = prediction_gap(
                row, item["trace"], bounds[:, 0], bounds[:, 1], VELOCITY_LIMIT
            )
            report["acceleration_replacements"] = acceleration_replacements(
                report, row, item["trace"], regressor
            )
            reports.append(report)
            if (row["scenario"], row["seed"]) in FOCUS:
                curves.append({"cell": item["cell"], "samples": curve})
    finally:
        mujoco.mj_step = original_step
    if len(reports) != 26 or len(curves) != 2:
        raise ValueError("all aborted episodes and both zero-delay regressions are required")
    return (
        {
            "scope": (
                "post-hoc used-task diagnosis; descriptive replacements without causal attribution"
            ),
            "reviewed_episodes": 26,
            "unsafe_episodes": sum(r["first_violation"] is not None for r in reports),
            "new_physical_steps": 0,
            "new_task_seeds": [],
            "maximum_velocity_identity_residual_rad_s": max(
                r["maximum_identity_residual_rad_s"] for r in reports
            ),
            "maximum_position_identity_residual_rad": max(
                r["maximum_position_identity_residual_rad"] for r in reports
            ),
            "cases": reports,
        },
        curves,
        runtime,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if args.audit:
        if not valid_completion_marker(out / "complete.json", FILES):
            parser.error("invalid prediction-gap completion hashes")
        manifest = json.loads((out / "manifest.json").read_text())
        if manifest["input_completion_sha256"] != INPUT_HASH or set(
            manifest["source_sha256"]
        ) != set(SOURCES):
            parser.error("prediction-gap provenance differs")
        for name, digest in manifest["source_sha256"].items():
            blob = subprocess.check_output(
                ["git", "show", f"{manifest['analysis_commit']}:{name}"], cwd=ROOT
            )
            if sha(ROOT / name) != digest or hashlib.sha256(blob).hexdigest() != digest:
                parser.error("check out the frozen prediction-gap source")
        data, curves, _ = calculate()
        if mismatches(data, json.loads((out / "diagnostics.json").read_text())) or mismatches(
            curves, json.loads(gzip.decompress((out / "curves.json.gz").read_bytes()))
        ):
            parser.error("prediction-gap diagnosis differs")
        print(
            "Verified 26 saved episodes, two focus curves and both velocity identities; zero steps"
        )
        return 0
    if out != OUTPUT.resolve() or out.exists():
        parser.error("use the unopened canonical prediction-gap directory")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        parser.error("commit diagnostic implementation and methods before retaining output")
    data, curves, runtime = calculate()
    manifest = {
        "analysis_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip(),
        "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
        "input_completion_sha256": INPUT_HASH,
        "runtime": runtime,
        "selection": "All 26 aborted episodes; clean/noisy zero-delay regressions for curves",
        "physical_steps": 0,
        "new_task_seeds": [],
    }
    out.mkdir()
    write(out / "manifest.json", manifest)
    write(out / "diagnostics.json", data)
    (out / "curves.json.gz").write_bytes(gzip.compress(canonical(curves), mtime=0))
    write(out / "complete.json", {"hashes": {name: sha(out / name) for name in FILES}})
    print("Retained 26 saved episode diagnoses and two focus curves; zero physical steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
