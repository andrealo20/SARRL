"""Audit constant-delay validation records and causal first-seed controller replays."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from sarrl.controllers.panda_adaptive import PandaAdaptiveController
from sarrl.controllers.panda_delay import PandaDelayController
from sarrl.controllers.panda_nominal import PandaComputedTorqueController
from sarrl.envs.panda_reach import quintic_reference
from sarrl.evaluation.panda_adaptation import reproduction_mismatches, sample_parameters
from sarrl.evaluation.panda_posture_delay import sample_task
from sarrl.evaluation.panda_reference_validation import SEEDS, reference_protocol, summarize
from sarrl.robots.panda import load_panda_model, model_provenance, mujoco
from tools.audit_panda_estimator_memory import check_noise, check_replay_numeric, ensure
from tools.campaign_guards import valid_completion_marker
from tools.run_panda_reference_validation import OUTPUT_FILES

ROOT = Path(__file__).resolve().parents[1]


def verify_sources(manifest):
    for name, digest in manifest["source_sha256"].items():
        blob = subprocess.check_output(
            ["git", "show", f"{manifest['source_commit']}:{name}"], cwd=ROOT
        )
        ensure(hashlib.sha256(blob).hexdigest() == digest, f"frozen source mismatch: {name}")
        if name.endswith(".py"):
            ensure(
                hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest,
                f"check out the recorded sources before auditing: {name}",
            )
    ensure(
        mujoco is not None
        and np.__version__ == manifest["runtime"]["numpy"]
        and mujoco.__version__ == manifest["runtime"]["mujoco"],
        "recorded runtime required",
    )
    ensure(model_provenance() == manifest["model"], "model provenance mismatch")


def controller_for(row):
    if row["arm"] == "fixed":
        return PandaComputedTorqueController()
    if row["arm"] == "adaptive":
        return PandaAdaptiveController()
    return PandaDelayController(
        known_delay_steps=row["interface_config"]["command_delay_steps"]
        if row["arm"] == "known_align"
        else None,
        predict=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/panda_reference_validation")
    args = parser.parse_args(argv)
    out = args.input
    ensure(
        valid_completion_marker(out / "complete.json", OUTPUT_FILES), "invalid completion hashes"
    )
    manifest = json.loads((out / "manifest.json").read_text())
    verify_sources(manifest)
    protocol = reference_protocol()
    ensure(manifest["protocol"] == protocol, "protocol mismatch")
    records = json.loads((out / "episodes.json").read_text())
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in records}
    planned = {
        (c, seed, arm) for c in protocol["scenarios"] for seed in SEEDS for arm in protocol["arms"]
    }
    ensure(len(records) == len(lookup) == 4000 and set(lookup) == planned, "wrong factorial cells")
    summary = summarize(records)
    ensure(
        not reproduction_mismatches(summary, json.loads((out / "summary.json").read_text())),
        "summary mismatch",
    )
    ensure(
        summary["reference_gate"] == json.loads((out / "decision.json").read_text()),
        "gate mismatch",
    )
    reproduction = json.loads((out / "reproduction.json").read_text())
    ensure(
        reproduction["matched"] == reproduction["compared"] == 1920
        and not reproduction["mismatches"]
        and reproduction["physical_steps"] == 2976000,
        "reproduction mismatch",
    )
    reference = ROOT / "results/panda_estimator_memory_pilot/episodes.json"
    ensure(
        reproduction["reference_sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest(),
        "reference hash mismatch",
    )
    model = load_panda_model()
    probe = mujoco.MjData(model)
    site = model.site("attachment_site").id
    for row in records:
        task = sample_task(row["seed"], row["posture"])
        ensure(
            row["task_sampling"] == json.loads(json.dumps(asdict(task))), "task sampling mismatch"
        )
        ensure(
            row["true_parameters"]
            == json.loads(json.dumps(asdict(sample_parameters(row["seed"], "payload_friction")))),
            "parameter draw mismatch",
        )
        check_replay_numeric(row["start_q"], task.start)
        check_replay_numeric(row["goal_q"], task.goal)
        ensure(
            row["physical_steps"] == 1500 and row["measurement_samples"] == 301,
            "step count mismatch",
        )
        unsafe = (
            row["joint_position_excess_rad"] > 0
            or row["joint_velocity_excess_rad_s"] > 0
            or row["contact_substeps"] > 0
        )
        ensure(row["unsafe_episode"] == unsafe, "unsafe flag mismatch")
        success = (
            row["joint_error_max_rad"] <= 0.01
            and row["tip_error_m"] <= 0.01
            and row["final_speed_max_rad_s"] <= 0.05
            and not unsafe
        )
        ensure(row["success"] == success, "success flag mismatch")
        check_replay_numeric(
            row["joint_error_max_rad"], np.max(np.abs(np.array(row["goal_q"]) - row["final_q"]))
        )
        probe.qpos[:] = row["goal_q"]
        mujoco.mj_forward(model, probe)
        check_replay_numeric(row["target_xyz_m"], probe.site_xpos[site])
        probe.qpos[:] = row["final_q"]
        mujoco.mj_forward(model, probe)
        check_replay_numeric(
            row["tip_error_m"], np.linalg.norm(probe.site_xpos[site] - row["target_xyz_m"])
        )
        if "delay_estimator" in row:
            d = row["delay_estimator"]
            selected, lag = d["selected_history"], row["interface_config"]["command_delay_steps"]
            ensure(
                len(selected) == 300 and d["final_delay_correct"] == (selected[-1] == lag),
                "age selection flag mismatch",
            )
            ensure(
                d["correct_selection_fraction"] == sum(v == lag for v in selected) / 300,
                "selection fraction mismatch",
            )
            bad = [i for i, value in enumerate(selected) if value != lag]
            acquired = ((bad[-1] + 2) * 0.01 if bad else 0.01) if selected[-1] == lag else None
            ensure(d["final_correct_run_start_s"] == acquired, "acquisition time mismatch")
            ensure(d["prediction_virtual_substeps"] == 0, "unexpected prediction")
    traces = json.loads(gzip.decompress((out / "traces.json.gz").read_bytes()))
    ensure(
        len(traces) == 80
        and {tuple(t["cell"]) for t in traces} == {k for k in planned if k[1] == SEEDS[0]},
        "wrong or duplicate trace cells",
    )
    replays = 0
    for item in traces:
        row, trace = lookup[tuple(item["cell"])], item["trace"]
        ensure(len(trace) == 301, "trace length mismatch")
        check_noise(trace, row["seed"], row["interface_config"], row)
        controller = controller_for(row)
        sent = np.array([t["torque_nm"] for t in trace[:-1]])
        lag = row["interface_config"]["command_delay_steps"]
        delivered = np.concatenate([np.zeros((lag, 7)), sent])[:300]
        check_replay_numeric(delivered, [t["delivered_torque_nm"] for t in trace[:-1]])
        errors, innovations, saturation = [], [], 0
        for index, (current, following) in enumerate(zip(trace[:-1], trace[1:], strict=True)):
            reference = quintic_reference(row["start_q"], row["goal_q"], current["time_s"], 2.0)
            check_replay_numeric(current["reference"], reference[0])
            errors.append(np.array(current["q"]) - reference[0])
            q, qd = current["measured_q"], current["measured_qd"]
            if hasattr(controller, "prepare_control"):
                cq, cv, horizon = controller.prepare_control(q, qd)
                ensure(horizon == 0, "unexpected lookahead")
                check_replay_numeric(current["control_state"], np.concatenate([cq, cv]))
            command = controller.command(q, qd, *reference)
            check_replay_numeric(sent[index], command)
            saturation += int(np.any(np.abs(command - controller.last_raw_torque) > 1e-12))
            if row["arm"] != "fixed":
                check_replay_numeric(
                    current["estimated_parameters"], controller.estimator.parameters
                )
                if isinstance(controller, PandaDelayController):
                    nominal, design = controller.regressor.evaluate(
                        (np.array(q) + following["measured_q"]) / 2,
                        (np.array(qd) + following["measured_qd"]) / 2,
                        (np.array(following["measured_qd"]) - qd) / 0.01,
                    )
                    values = []
                    for delay, estimator in zip(
                        controller.lags, controller.estimators, strict=True
                    ):
                        aligned = sent[index - delay] if index >= delay else np.zeros(7)
                        innovation = aligned - nominal - design @ estimator.parameters
                        values.append(float(innovation @ innovation))
                    innovations.append(values)
                controller.observe_transition(
                    q, qd, following["measured_q"], following["measured_qd"], sent[index], 0.01
                )
                if isinstance(controller, PandaDelayController):
                    check_replay_numeric(controller.scores, np.sum(innovations, axis=0))
        ensure(saturation == row["controller_saturated_steps"], "clipping count mismatch")
        check_replay_numeric(row["tracking_rms_rad"], np.sqrt(np.mean(np.square(errors), axis=0)))
        check_replay_numeric(row["tracking_error_max_rad"], np.max(np.abs(errors), axis=0))
        check_replay_numeric(trace[0]["q"], row["start_q"])
        check_replay_numeric(trace[-1]["q"], row["final_q"])
        check_replay_numeric(np.max(np.abs(trace[-1]["qd"])), row["final_speed_max_rad_s"])
        if row["arm"] != "fixed":
            check_replay_numeric(
                row["estimator"]["estimated_parameters"], controller.estimator.parameters
            )
            check_replay_numeric(
                row["estimator"]["covariance_diagonal"], np.diag(controller.estimator.covariance)
            )
            if isinstance(controller, PandaDelayController):
                final = controller.delay_diagnostics()
                for field in ("scores", "candidate_estimates"):
                    check_replay_numeric(row["delay_estimator"][field], final[field])
                for field in ("selected_history", "candidate_bound_hits", "selection_switches"):
                    ensure(
                        row["delay_estimator"][field] == final[field], f"replay mismatch: {field}"
                    )
            replays += 1
    print(
        json.dumps(
            {
                "verified_episodes": len(records),
                "verified_traces": len(traces),
                "causal_estimator_replays": replays,
                "reference_passes": summary["reference_gate"]["reference_passes"],
                "physical_steps_including_reproduction": summary["physical_steps"]
                + reproduction["physical_steps"],
                "source_commit": manifest["source_commit"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
