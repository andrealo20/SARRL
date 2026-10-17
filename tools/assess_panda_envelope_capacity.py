"""Static necessary-error screen for retained Panda backup terminal capacity."""

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
from tools.recover_panda_failures import mismatches, sha, write

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/panda_backup_integration_v26"
OUTPUT = ROOT / "results/panda_envelope_capacity_v26"
INPUT_HASH = "351674a90849d64e7359fa101f72aee86ea1536ff2713b1b79d558b1d2f9831a"
FILES = ("manifest.json", "capacity.json")
SOURCES = (
    "tools/assess_panda_envelope_capacity.py",
    "sarrl/evaluation/panda_envelope_capacity.py",
    "sarrl/evaluation/panda_prediction_gap.py",
    "sarrl/safety/panda_backup_tube.py",
    "sarrl/safety/panda_backup.py",
    "sarrl/safety/panda_joint_filter.py",
    "sarrl/robots/panda.py",
    "sarrl/evaluation/panda_adaptation.py",
    "tests/test_panda_envelope_capacity.py",
    "docs/panda-envelope-capacity-methods.md",
    "tools/campaign_guards.py",
    "tools/recover_panda_failures.py",
)


def calculate():
    import numpy as np
    import scipy

    from sarrl.evaluation.panda_envelope_capacity import affine_box_error, optimistic_capacity
    from sarrl.evaluation.panda_prediction_gap import retained_witness
    from sarrl.robots.panda import PandaDynamics, PandaModelParameters, model_provenance, mujoco

    if sha(INPUT / "complete.json") != INPUT_HASH:
        raise ValueError("integration completion identity differs")
    hashes = json.loads((INPUT / "complete.json").read_text())["hashes"]
    if not valid_completion_marker(INPUT / "complete.json", tuple(hashes)):
        raise ValueError("integration content hashes differ")
    original = json.loads((INPUT / "manifest.json").read_text())
    runtime = {"numpy": np.__version__, "scipy": scipy.__version__, "mujoco": mujoco.__version__}
    if any(original["runtime"][key] != value for key, value in runtime.items()):
        raise ValueError("use the retained Panda numerical runtime")
    model_record = json.loads((ROOT / "results/panda_joint_filter_pilot/manifest.json").read_text())
    if model_provenance() != model_record["model"]:
        raise ValueError("packaged model provenance differs")
    rows = json.loads((INPUT / "episodes.json").read_text())
    traces = json.loads(gzip.decompress((INPUT / "traces.json.gz").read_bytes()))
    if len(rows) != 32 or len(traces) != 32:
        raise ValueError("all retained integration records are required")

    def prohibited(*args, **kwargs):
        raise RuntimeError("time integration is prohibited in the static capacity screen")

    names = ("mj_step", "mj_step1", "mj_step2", "mj_Euler", "mj_implicit", "mj_RungeKutta")
    saved = {name: getattr(mujoco, name) for name in names}
    for name in names:
        setattr(mujoco, name, prohibited)
    try:
        models = [
            PandaDynamics(parameters=PandaModelParameters(mass, (0.5,) * 7)) for mass in (0.5, 2.5)
        ]
        lower, upper = models[0].model.jnt_range.T
        reports = []
        for row, trace in zip(rows, traces, strict=True):
            cell = [row["scenario"], row["seed"], row["arm"]]
            if trace["cell"] != cell:
                raise ValueError("retained trace identity differs")
            if not row["backup"]["aborted"]:
                continue
            start, witness = retained_witness(row, trace["trace"])
            policy, points = witness["policy"], []
            q = np.asarray(policy["anchor"])
            if np.any(q <= lower + 0.03) or np.any(q >= upper - 0.03):
                raise ValueError("anchor is not interior to reserved joint limits")
            np.testing.assert_array_equal(policy["torque_limit"], models[0].torque_limit)
            for payload, dynamics in zip((0.5, 2.5), models, strict=True):
                mass = dynamics.mass_matrix(q)
                np.linalg.cholesky(mass)
                matrix = np.linalg.solve(mass, np.eye(7))
                drift = np.linalg.solve(mass, dynamics.data.qfrc_passive - dynamics.data.qfrc_bias)
                floor, corners = affine_box_error(
                    matrix, drift, policy["matrix"], policy["drift"], policy["torque_limit"]
                )
                accelerations = []
                for torque in [np.zeros(7), *corners]:
                    dynamics.data.ctrl[:] = torque
                    dynamics.data.qacc_warmstart[:] = 0
                    mujoco.mj_forward(dynamics.model, dynamics.data)
                    if dynamics.data.ncon or dynamics.data.nefc or dynamics.data.time != 0:
                        raise ValueError("static point has constraints or advanced time")
                    np.testing.assert_array_equal(dynamics.data.qpos, q)
                    np.testing.assert_array_equal(dynamics.data.qvel, np.zeros(7))
                    np.testing.assert_allclose(
                        dynamics.data.qacc, matrix @ torque + drift, rtol=1e-9, atol=1e-9
                    )
                    accelerations.append(dynamics.data.qacc.copy())
                attained = np.abs(
                    np.diag(
                        np.asarray(accelerations[1:])
                        - corners @ np.asarray(policy["matrix"]).T
                        - np.asarray(policy["drift"])
                    )
                )
                np.testing.assert_allclose(attained, floor, rtol=1e-9, atol=1e-9)
                points.append(
                    {
                        "payload_kg": payload,
                        "extra_viscous": [0.5] * 7,
                        "matrix": matrix.tolist(),
                        "drift": drift.tolist(),
                        "mass_condition_number": float(np.linalg.cond(mass)),
                        "error_floor_rad_s2": floor.tolist(),
                        "attaining_torques_nm": corners.tolist(),
                        "direct_zero_torque_acceleration": accelerations[0].tolist(),
                        "direct_corner_accelerations": np.asarray(accelerations[1:]).tolist(),
                    }
                )
            floor = np.max([point["error_floor_rad_s2"] for point in points], axis=0)
            cases = optimistic_capacity(policy, floor, lower, upper)
            reports.append(
                {
                    "cell": cell,
                    "policy_step": start,
                    "policy_sha256": witness["policy_sha256"],
                    "anchor": q.tolist(),
                    "static_points": points,
                    "necessary_error_floor_rad_s2": floor.tolist(),
                    "lags": cases,
                    "screen_rejected": any(not case["feasible"] for case in cases),
                }
            )
    finally:
        for name, function in saved.items():
            setattr(mujoco, name, function)
    if len(reports) != 26 or len({tuple(report["cell"]) for report in reports}) != 26:
        raise ValueError("all 26 distinct aborted task identities are required")
    return {
        "scope": "selected used-policy necessary screen, not a full-domain enclosure",
        "physical_steps": 0,
        "new_task_seeds": [],
        "policies": 26,
        "static_parameter_points": 52,
        "direct_forward_checks": 416,
        "rejected_policies": sum(report["screen_rejected"] for report in reports),
        "cases": reports,
    }, runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.audit:
        if not valid_completion_marker(output / "complete.json", FILES):
            parser.error("capacity output hashes differ")
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["input_completion_sha256"] != INPUT_HASH or set(
            manifest["source_sha256"]
        ) != set(SOURCES):
            parser.error("capacity provenance differs")
        for name, digest in manifest["source_sha256"].items():
            blob = subprocess.check_output(
                ["git", "show", f"{manifest['analysis_commit']}:{name}"], cwd=ROOT
            )
            if sha(ROOT / name) != digest or hashlib.sha256(blob).hexdigest() != digest:
                parser.error("check out the frozen capacity source")
        data, _ = calculate()
        if mismatches(data, json.loads((output / "capacity.json").read_text())):
            parser.error("static capacity calculation differs")
        print("Verified 26 retained policies, 52 static points and 416 direct checks; zero steps")
        return 0
    if output != OUTPUT.resolve() or output.exists():
        parser.error("use the unopened canonical capacity directory")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        parser.error("freeze implementation and methods before static point evaluation")
    data, runtime = calculate()
    manifest = {
        "analysis_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip(),
        "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
        "input_completion_sha256": INPUT_HASH,
        "runtime": runtime,
        "selection": "All 26 final retained backup contexts from aborted integration episodes",
        "physical_steps": 0,
        "new_task_seeds": [],
    }
    output.mkdir()
    write(output / "manifest.json", manifest)
    write(output / "capacity.json", data)
    write(output / "complete.json", {"hashes": {name: sha(output / name) for name in FILES}})
    print(f"Retained 26 policy screens: {data['rejected_policies']} rejected; zero physical steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
