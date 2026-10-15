"""Audit frozen joint-filter sources with numerical replay and exact recorded FIFO checks."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

bootstrap = argparse.ArgumentParser(description=__doc__)
bootstrap.add_argument("--source-checkout", type=Path, required=True)
bootstrap.add_argument(
    "--input", type=Path, help="retained results; defaults to the frozen checkout"
)
source_checkout, _ = bootstrap.parse_known_args()
ROOT = source_checkout.source_checkout.resolve()
if not (ROOT / "tools/audit_panda_joint_filter.py").is_file():
    bootstrap.error("a frozen source checkout is required")
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scipy.optimize import nnls

from sarrl.controllers.panda_delay import PandaDelayController
from sarrl.controllers.panda_filtered import PandaFilteredController
from sarrl.envs.panda_reach import quintic_reference
from sarrl.evaluation.panda_adaptation import reproduction_mismatches, sample_parameters
from sarrl.evaluation.panda_joint_filter import (
    CONFIG,
    configurations,
    joint_filter_protocol,
    summarize,
)
from sarrl.evaluation.panda_posture_delay import sample_task
from sarrl.robots.panda import VELOCITY_LIMIT, load_panda_model, mujoco
from tools.audit_panda_estimator_memory import check_noise, check_replay_numeric, ensure
from tools.audit_panda_reference_validation import verify_sources
from tools.campaign_guards import valid_completion_marker
from tools.run_panda_joint_filter import OUTPUT_FILES


def independent_projection_check(controller, q, v, sent, candidate):
    """Rebuild every inequality and dual check from the recorded causal inputs."""
    cfg = controller.joint_filter.config
    theta = controller.estimator.parameters
    bias = controller.regressor.inverse_dynamics(q, v, np.zeros(7), theta)
    M = controller.regressor.mass_matrix(q, theta)
    B = np.linalg.solve(M, np.eye(7))
    drift = -B @ bias
    bounds = controller.regressor.nominal.model.jnt_range
    lo, hi = bounds[:, 0] + cfg.position_reserve_rad, bounds[:, 1] - cfg.position_reserve_rad
    vmax, limit = VELOCITY_LIMIT - cfg.velocity_reserve_rad_s, controller.dynamics.torque_limit
    dt, gamma = cfg.control_dt, cfg.gamma
    rows, rhs = [], []
    history = controller.sent_history

    def interval_check(position, velocity, acceleration):
        for j in range(7):
            times = [0.0, dt]
            if acceleration[j] != 0 and 0 < -velocity[j] / acceleration[j] < dt:
                times.append(-velocity[j] / acceleration[j])
            for time in times:
                value = position[j] + velocity[j] * time + 0.5 * acceleration[j] * time * time
                ensure(lo[j] - 1e-8 <= value <= hi[j] + 1e-8, "polynomial position check failed")
            ensure(
                max(abs(velocity[j]), abs(velocity[j] + dt * acceleration[j])) <= vmax[j] + 1e-8,
                "polynomial velocity check failed",
            )

    for lag in range(cfg.max_delay_steps + 1):
        pq, pv = np.array(q).copy(), np.array(v).copy()
        for index in range(len(history) - lag, len(history)):
            prior = history[index] if index >= 0 else np.zeros(7)
            a = B @ prior + drift
            interval_check(pq, pv, a)
            pq += pv * dt + 0.5 * a * dt * dt
            pv += a * dt
        ensure(
            np.all(pq >= lo - 1e-8) and np.all(pq <= hi + 1e-8),
            "arrival position outside model domain",
        )
        ensure(np.all(np.abs(pv) <= vmax + 1e-8), "arrival velocity outside model domain")
        ensure(
            np.all(pv + gamma * (pq - lo) >= -1e-8) and np.all(-pv + gamma * (hi - pq) >= -1e-8),
            "arrival barrier outside model domain",
        )
        interval_check(pq, pv, B @ sent + drift)
        for j in range(7):
            rows.extend((B[j], -B[j], B[j], -B[j]))
            rhs.extend(
                (
                    -drift[j] - 2 * gamma * pv[j] - gamma * gamma * (pq[j] - lo[j]),
                    drift[j] + 2 * gamma * pv[j] - gamma * gamma * (hi[j] - pq[j]),
                    (-vmax[j] - pv[j]) / dt - drift[j],
                    -(vmax[j] - pv[j]) / dt + drift[j],
                )
            )
    A = np.vstack((np.array(rows) * limit, np.eye(7), -np.eye(7)))
    b = np.concatenate((rhs, -np.ones(14)))
    norm = np.linalg.norm(A, axis=1)
    A, b = A / norm[:, None], b / norm
    x, center = np.array(sent) / limit, np.array(candidate) / limit
    margin = A @ x - b
    ensure(min(margin) >= -cfg.feasibility_tol - 1e-10, "independent primal check failed")
    active = np.flatnonzero(margin <= 1e-7)
    if len(active):
        dual, residual = nnls(A[active].T, x - center, maxiter=1000)
        complementarity = max(np.abs(dual * margin[active]))
    else:
        residual, complementarity = np.linalg.norm(x - center), 0.0
    ensure(
        residual <= cfg.stationarity_tol and complementarity <= cfg.stationarity_tol,
        "independent optimality check failed",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "results/panda_joint_filter_pilot")
    out = parser.parse_args(argv).input
    ensure(
        valid_completion_marker(out / "complete.json", OUTPUT_FILES), "invalid completion hashes"
    )
    manifest = json.loads((out / "manifest.json").read_text())
    verify_sources(manifest)
    ensure(scipy.__version__ == manifest["runtime"]["scipy"], "recorded SciPy required")
    protocol = joint_filter_protocol()
    ensure(manifest["protocol"] == protocol, "protocol mismatch")
    rows = json.loads((out / "episodes.json").read_text())
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in rows}
    expected = {
        (c, s, a)
        for c in protocol["scenarios"]
        for s in protocol["seeds"]
        for a in protocol["arms"]
    }
    ensure(
        len(rows) == len(lookup) == len(expected) and set(lookup) == expected,
        "planned cells mismatch",
    )
    summary = summarize(rows)
    ensure(
        not reproduction_mismatches(summary, json.loads((out / "summary.json").read_text())),
        "summary mismatch",
    )
    ensure(
        not reproduction_mismatches(
            summary["exploratory_gate"], json.loads((out / "decision.json").read_text())
        ),
        "decision mismatch",
    )
    report = json.loads((out / "reproduction.json").read_text())
    ensure(
        report["compared"] == report["matched"] == 4000 and not report["mismatches"],
        "reproduction incomplete",
    )
    ensure(
        report["physical_steps"] == 6000000 and report["prediction_virtual_substeps"] == 0,
        "reproduction counts mismatch",
    )
    reference = ROOT / "results/panda_reference_validation/episodes.json"
    ensure(
        report["reference_sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest(),
        "reference hash mismatch",
    )
    model = load_panda_model()
    probe = mujoco.MjData(model)
    for row in rows:
        task_cfg, interface_cfg = configurations(row["scenario"])
        ensure(
            not reproduction_mismatches(
                asdict(sample_task(row["seed"], "wide")), row["task_sampling"]
            ),
            "task mismatch",
        )
        ensure(
            not reproduction_mismatches(
                asdict(sample_parameters(row["seed"], "payload_friction")), row["true_parameters"]
            ),
            "parameter mismatch",
        )
        ensure(
            row["task_config"] == asdict(task_cfg)
            and row["interface_config"] == asdict(interface_cfg),
            "configuration mismatch",
        )
        ensure(
            row["physical_steps"] == 1500
            and row["steps"] == 300
            and row["measurement_samples"] == 301,
            "step counts mismatch",
        )
        unsafe = (
            row["joint_position_excess_rad"] > 0
            or row["joint_velocity_excess_rad_s"] > 0
            or row["contact_substeps"] > 0
        )
        ensure(row["unsafe_episode"] == unsafe, "unsafe flag mismatch")
        aborted = row.get("filter", {}).get("aborted", False)
        task_success = (
            row["joint_error_max_rad"] <= 0.01
            and row["tip_error_m"] <= 0.01
            and row["final_speed_max_rad_s"] <= 0.05
            and not unsafe
        )
        ensure(
            row["task_success"] == task_success
            and row["success"] == (task_success and not aborted),
            "success mismatch",
        )
        check_replay_numeric(
            row["joint_error_max_rad"], np.max(np.abs(np.array(row["goal_q"]) - row["final_q"]))
        )
        check_replay_numeric(row["final_speed_max_rad_s"], np.max(np.abs(row["final_qd"])))
        positions = []
        for value in (row["goal_q"], row["final_q"]):
            probe.qpos[:] = value
            mujoco.mj_forward(model, probe)
            positions.append(probe.site_xpos[model.site("attachment_site").id].copy())
        check_replay_numeric(row["tip_error_m"], np.linalg.norm(positions[0] - positions[1]))
        if row["arm"] == "bank_filtered":
            ensure(row["filter"]["config"] == asdict(CONFIG), "filter settings mismatch")
    traces = json.loads(gzip.decompress((out / "traces.json.gz").read_bytes()))
    planned_traces = {key for key in expected if key[1] == protocol["seeds"][0]}
    ensure(
        len(traces) == len(planned_traces) and {tuple(t["cell"]) for t in traces} == planned_traces,
        "trace cells mismatch",
    )
    projections = substeps = 0
    for item in traces:
        row, trace = lookup[tuple(item["cell"])], item["trace"]
        ensure(len(trace) == 301, "trace length mismatch")
        check_noise(trace, row["seed"], row["interface_config"], row)
        controller = (
            PandaFilteredController(CONFIG)
            if row["arm"] == "bank_filtered"
            else PandaDelayController(predict=False)
        )
        lag = row["interface_config"]["command_delay_steps"]
        physical, errors, sent = [], [], []
        for step, t in enumerate(trace[:-1]):
            reference = quintic_reference(
                row["start_q"], row["goal_q"], t["time_s"], row["task_config"]["move_seconds"]
            )
            check_replay_numeric(reference[0], t["reference"])
            errors.append(np.array(t["q"]) - reference[0])
            check_replay_numeric(t["estimated_parameters"], controller.estimator.parameters)
            check_replay_numeric(t["delay_scores"], controller.scores)
            command = controller.command(t["measured_q"], t["measured_qd"], *reference)
            check_replay_numeric(command, t["torque_nm"])
            ensure(
                np.all(np.abs(command) <= controller.dynamics.torque_limit),
                "sent torque exceeds bounds",
            )
            if row["arm"] == "bank_filtered":
                ensure(
                    not reproduction_mismatches(controller.last_filter, t["filter"]),
                    "filter replay mismatch",
                )
                check_replay_numeric(controller.last_nominal_command, t["nominal_command_nm"])
                if t["filter"]["predicted_feasible"]:
                    independent_projection_check(
                        controller,
                        t["measured_q"],
                        t["measured_qd"],
                        command,
                        t["nominal_command_nm"],
                    )
                    projections += 1
                if t["filter"].get("braking"):
                    ensure(not t["filter"]["predicted_feasible"], "braking labelled feasible")
            sent.append(np.array(t["torque_nm"]))
            np.testing.assert_array_equal(
                t["delivered_torque_nm"], sent[step - lag] if step >= lag else np.zeros(7)
            )
            ensure(len(t["physical_substeps"]) == 5, "substep trace length mismatch")
            physical.extend(t["physical_substeps"])
            np.testing.assert_array_equal(t["physical_substeps"][-1]["q"], trace[step + 1]["q"])
            np.testing.assert_array_equal(t["physical_substeps"][-1]["qd"], trace[step + 1]["qd"])
            controller.observe_transition(
                t["measured_q"],
                t["measured_qd"],
                trace[step + 1]["measured_q"],
                trace[step + 1]["measured_qd"],
                command,
                0.01,
            )
        q, v = np.array([s["q"] for s in physical]), np.array([s["qd"] for s in physical])
        check_replay_numeric(
            row["joint_position_excess_rad"],
            max(
                0.0, float(np.max(np.maximum(model.jnt_range[:, 0] - q, q - model.jnt_range[:, 1])))
            ),
        )
        check_replay_numeric(
            row["joint_velocity_excess_rad_s"], max(0.0, float(np.max(np.abs(v) - VELOCITY_LIMIT)))
        )
        ensure(
            row["contact_substeps"] == sum(s["contacts"] > 0 for s in physical),
            "contact substeps mismatch",
        )
        check_replay_numeric(row["tracking_rms_rad"], np.sqrt(np.mean(np.square(errors), axis=0)))
        check_replay_numeric(row["max_speed_rad_s"], np.max(np.abs(v), axis=0))
        ensure(
            not reproduction_mismatches(row["estimator"], controller.estimator.diagnostics()),
            "estimator diagnostics mismatch",
        )
        if row["arm"] == "bank_filtered":
            ensure(
                not reproduction_mismatches(row["filter"], controller.filter_diagnostics()),
                "filter final diagnostics mismatch",
            )
        for field in ("scores", "selected_history", "candidate_estimates", "candidate_bound_hits"):
            ensure(
                not reproduction_mismatches(
                    {field: row["delay_estimator"][field]},
                    {field: controller.delay_diagnostics()[field]},
                ),
                "bank replay mismatch",
            )
        substeps += len(physical)
    print(
        json.dumps(
            {
                "verified_episodes": len(rows),
                "verified_traces": len(traces),
                "causal_controller_replays": len(traces),
                "verified_physical_substeps": substeps,
                "independent_feasible_projection_checks": projections,
                "physical_steps_including_reproduction": summary["physical_steps"]
                + report["physical_steps"],
                "source_commit": manifest["source_commit"],
                "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "delivered_torque_check": "exact retained-command FIFO",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
