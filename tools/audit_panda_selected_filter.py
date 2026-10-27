"""Audit the selected-delay Panda joint-filter mechanism test from its retained files."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import scipy

from sarrl.controllers.panda_filtered import selected_age_set
from sarrl.envs.panda_reach import quintic_reference
from sarrl.evaluation.panda_adaptation import reproduction_mismatches, sample_parameters
from sarrl.evaluation.panda_joint_filter import CONFIG, configurations
from sarrl.evaluation.panda_posture_delay import sample_task
from sarrl.evaluation.panda_selected_filter import (
    _FACTORIES,
    decide,
    selected_filter_protocol,
    summarize,
)
from sarrl.robots.panda import VELOCITY_LIMIT, load_panda_model, mujoco
from tools.audit_panda_estimator_memory import check_noise, check_replay_numeric, ensure
from tools.audit_panda_joint_filter import independent_projection_check
from tools.audit_panda_reference_validation import verify_sources
from tools.campaign_guards import valid_completion_marker
from tools.run_panda_joint_filter import OUTPUT_FILES

ROOT = Path(__file__).resolve().parents[1]


def check_row(row, model, probe):
    task_cfg, interface_cfg = configurations(row["scenario"])
    ensure(
        not reproduction_mismatches(asdict(sample_task(row["seed"], "wide")), row["task_sampling"]),
        "task mismatch",
    )
    ensure(
        not reproduction_mismatches(
            asdict(sample_parameters(row["seed"], "payload_friction")), row["true_parameters"]
        ),
        "parameter mismatch",
    )
    ensure(
        row["task_config"] == asdict(task_cfg) and row["interface_config"] == asdict(interface_cfg),
        "configuration mismatch",
    )
    ensure(
        row["physical_steps"] == 1500 and row["steps"] == 300 and row["measurement_samples"] == 301,
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
        row["task_success"] == task_success and row["success"] == (task_success and not aborted),
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
    if row["arm"] != "cumulative":
        ensure(row["filter"]["config"] == asdict(CONFIG), "filter settings mismatch")
    if row["arm"] == "selected_filter":
        ensure(len(row["filter"]["age_sets"]) == 300, "age-set history length mismatch")


def replay_trace(row, trace, model):
    """Causal controller replay on recorded measurements; returns substeps and projections."""
    ensure(len(trace) == 301, "trace length mismatch")
    check_noise(trace, row["seed"], row["interface_config"], row)
    factory, filtered = _FACTORIES[row["arm"]]
    controller = factory()
    selected = row["arm"] == "selected_filter"
    lag = row["interface_config"]["command_delay_steps"]
    physical, errors, sent, projections = [], [], [], 0
    for step, t in enumerate(trace[:-1]):
        reference = quintic_reference(
            row["start_q"], row["goal_q"], t["time_s"], row["task_config"]["move_seconds"]
        )
        np.testing.assert_array_equal(reference[0], t["reference"])
        errors.append(np.array(t["q"]) - reference[0])
        check_replay_numeric(t["estimated_parameters"], controller.estimator.parameters)
        check_replay_numeric(t["delay_scores"], controller.scores)
        command = controller.command(t["measured_q"], t["measured_qd"], *reference)
        check_replay_numeric(command, t["torque_nm"])
        ensure(
            np.all(np.abs(command) <= controller.dynamics.torque_limit),
            "sent torque exceeds bounds",
        )
        if filtered:
            ensure(
                not reproduction_mismatches(controller.last_filter, t["filter"]),
                "filter replay mismatch",
            )
            check_replay_numeric(controller.last_nominal_command, t["nominal_command_nm"])
            ages = None
            if selected:
                ages = controller.age_sets[-1]
                ensure(
                    selected_age_set(t["delay_scores"], controller.lags) == ages,
                    "age set does not follow the recorded scores",
                )
            if t["filter"]["predicted_feasible"]:
                independent_projection_check(
                    controller,
                    t["measured_q"],
                    t["measured_qd"],
                    command,
                    t["nominal_command_nm"],
                    ages,
                )
                projections += 1
            if t["filter"].get("braking"):
                ensure(not t["filter"]["predicted_feasible"], "braking labelled feasible")
        sent.append(command)
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
        max(0.0, float(np.max(np.maximum(model.jnt_range[:, 0] - q, q - model.jnt_range[:, 1])))),
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
    if filtered:
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
    return len(physical), projections


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/panda_selected_filter_pilot")
    out = parser.parse_args(argv).input
    ensure(
        valid_completion_marker(out / "complete.json", OUTPUT_FILES), "invalid completion hashes"
    )
    manifest = json.loads((out / "manifest.json").read_text())
    verify_sources(manifest)
    ensure(scipy.__version__ == manifest["runtime"]["scipy"], "recorded SciPy required")
    protocol = selected_filter_protocol()
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
        not reproduction_mismatches(decide(rows), json.loads((out / "decision.json").read_text())),
        "decision mismatch",
    )
    report = json.loads((out / "reproduction.json").read_text())
    ensure(
        report["compared"] == report["matched"] == 480 and not report["mismatches"],
        "reproduction incomplete",
    )
    ensure(report["physical_steps"] == 720000, "reproduction counts mismatch")
    reference = ROOT / "results/panda_joint_filter_pilot/episodes.json"
    ensure(
        report["reference_sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest(),
        "reference hash mismatch",
    )
    model = load_panda_model()
    probe = mujoco.MjData(model)
    for row in rows:
        check_row(row, model, probe)
    traces = json.loads(gzip.decompress((out / "traces.json.gz").read_bytes()))
    planned_traces = {key for key in expected if key[1] == protocol["seeds"][0]}
    ensure(
        len(traces) == len(planned_traces) and {tuple(t["cell"]) for t in traces} == planned_traces,
        "trace cells mismatch",
    )
    substeps = projections = 0
    for item in traces:
        counted, checked = replay_trace(lookup[tuple(item["cell"])], item["trace"], model)
        substeps, projections = substeps + counted, projections + checked
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
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
