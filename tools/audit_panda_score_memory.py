"""Verify retained score-memory evidence without new physical episodes."""

# Numerical thread settings precede imports of numerical libraries.
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

from sarrl.controllers.panda_adaptive import PandaAdaptiveController, PandaRegressor
from sarrl.controllers.panda_delay import PandaDelayController
from sarrl.controllers.panda_score_memory import PandaScoreMemoryController
from sarrl.envs.panda_packet_delay import applied_ages, send_delay
from sarrl.envs.panda_reach import quintic_reference
from sarrl.evaluation.panda_adaptation import sample_parameters
from sarrl.evaluation.panda_posture_delay import sample_task
from sarrl.evaluation.panda_score_memory import MEMORY_CONFIGS, summarize
from sarrl.evaluation.panda_variable_delay import KnownAgeDiagnostic
from sarrl.robots.panda import load_panda_model, mujoco
from tools.campaign_guards import valid_completion_marker
from tools.run_panda_adaptation import OUTPUT_FILES

ROOT = Path(__file__).resolve().parents[1]
LIMIT = np.array([87.0] * 4 + [12.0] * 3)


def ensure(condition, message):
    if not condition:
        raise RuntimeError(message)


def noise_values(seed, config, length):
    rng = np.random.default_rng(seed ^ 0x50A07)
    return rng.normal(size=(length, 14)) * (
        [config["position_noise_std_rad"]] * 7 + [config["velocity_noise_std_rad_s"]] * 7
    )


def check_noise(trace, seed, config, diagnostic):
    noise = noise_values(seed, config, len(trace))
    q = np.array([t["q"] for t in trace])
    qd = np.array([t["qd"] for t in trace])
    measured = np.array([t["measured_q"] + t["measured_qd"] for t in trace])
    np.testing.assert_array_equal(measured, np.concatenate([q, qd], axis=1) + noise)
    ensure(
        hashlib.sha256(noise.astype("<f8").tobytes()).hexdigest()
        == diagnostic["measurement_noise_sha256"],
        "sensor-noise hash mismatch",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/panda_score_memory_pilot")
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
        ensure(hashlib.sha256(blob).hexdigest() == digest, f"frozen Git source mismatch: {name}")
        if name.endswith(".py"):
            ensure(
                hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest,
                f"check out the recorded Python sources before auditing: {name}",
            )
    ensure(mujoco is not None, "install the recorded numerical runtime")
    ensure(
        np.__version__ == manifest["runtime"]["numpy"]
        and mujoco.__version__ == manifest["runtime"]["mujoco"],
        "audit requires the recorded NumPy/MuJoCo versions",
    )
    reference = json.loads((ROOT / "results/panda_variable_delay_pilot/manifest.json").read_text())
    ensure(manifest["model"] == reference["model"], "model provenance differs from the reference")
    for name, digest in manifest["frozen_controller"]["source_sha256"].items():
        ensure(
            digest == reference["source_sha256"][name] == manifest["source_sha256"][name],
            "controller freeze mismatch",
        )
    protocol = manifest["protocol"]
    rows = json.loads((out / "episodes.json").read_text())
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in rows}
    planned = {
        (c, s, a)
        for c in protocol["scenarios"]
        for s in protocol["seeds"]
        for a in protocol["arms"]
    }
    ensure(
        len(rows) == len(lookup) == len(planned) and set(lookup) == planned,
        "wrong planned cell identities",
    )
    summary = summarize(rows)
    ensure(summary == json.loads((out / "summary.json").read_text()), "summary mismatch")
    report = json.loads((out / "reproduction.json").read_text())
    ensure(
        report["compared"] == report["matched"] == 1280 and not report["mismatches"],
        "incomplete reproduction",
    )
    ensure(
        report["physical_steps"] == 1984000 and report["prediction_virtual_substeps"] == 0,
        "incorrect reproduction step counts",
    )
    model = load_panda_model()
    probe = mujoco.MjData(model)
    site = model.site("attachment_site").id
    for row in rows:
        task = sample_task(row["seed"], "wide")
        ensure(row["task_sampling"] == json.loads(json.dumps(asdict(task))), "task sample mismatch")
        ensure(
            row["true_parameters"]
            == json.loads(json.dumps(asdict(sample_parameters(row["seed"], "payload_friction")))),
            "parameter draw mismatch",
        )
        np.testing.assert_array_equal(row["start_q"], task.start)
        np.testing.assert_array_equal(row["goal_q"], task.goal)
        warm = row["warmup"]
        ensure(
            row["scored_physical_steps"] == 1500
            and row["physical_steps"] == 1500 + (100 if warm else 0),
            "physical-step count mismatch",
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
        np.testing.assert_allclose(
            row["joint_error_max_rad"],
            np.max(np.abs(np.array(row["goal_q"]) - row["final_q"])),
            rtol=1e-12,
        )
        probe.qpos[:] = row["goal_q"]
        mujoco.mj_forward(model, probe)
        np.testing.assert_allclose(
            probe.site_xpos[site], row["target_xyz_m"], rtol=1e-12, atol=1e-12
        )
        probe.qpos[:] = row["final_q"]
        mujoco.mj_forward(model, probe)
        np.testing.assert_allclose(
            np.linalg.norm(probe.site_xpos[site] - row["target_xyz_m"]),
            row["tip_error_m"],
            rtol=1e-10,
            atol=1e-12,
        )
        profile = row["scenario"].split("_")[-1]
        ages = applied_ages(profile)
        ensure(
            row["discarded_packets"] == (3 if profile == "down" else 0),
            "stale-packet count mismatch",
        )
        if "delay_estimator" in row:
            d = row["delay_estimator"]
            selected = d["selected_history"]
            ensure(len(selected) == 300, "wrong selection-history length")
            correct = np.array(selected) == ages
            ensure(d["final_delay_correct"] == bool(correct[-1]), "wrong final-delay flag")
            ensure(
                d["correct_selection_fraction"] == float(np.mean(correct)),
                "wrong selected-age fraction",
            )
            ensure(
                d["before_switch_correct_fraction"] == float(np.mean(correct[:100]))
                and d["after_switch_correct_fraction"] == float(np.mean(correct[100:])),
                "wrong phase selection fractions",
            )
            stable = next(i for i in range(300) if all(a == ages[-1] for a in ages[i:]))
            bad = [i for i in range(stable, 300) if selected[i] != ages[-1]]
            acquired = (
                ((bad[-1] + 2) * 0.01 if bad else (stable + 1) * 0.01) if correct[-1] else None
            )
            ensure(
                d["final_regime_acquired_at_s"] == acquired, "wrong final-regime acquisition time"
            )
            if row["arm"] == "known_align":
                ensure(selected == list(ages), "privileged diagnostic did not use applied ages")
    traces = json.loads(gzip.decompress((out / "traces.json.gz").read_bytes()))
    ensure(
        len(traces) == len(protocol["scenarios"]) * len(protocol["arms"]),
        "wrong first-seed trace count",
    )
    regressor = PandaRegressor()
    replayed = 0
    for item in traces:
        row = lookup[tuple(item["cell"])]
        trace = item["trace"]
        ensure(row["seed"] == protocol["seeds"][0] and len(trace) == 301, "wrong scored trace")
        check_noise(trace, row["seed"], row["interface_config"], row)
        history = np.array(row["initial_command_history"])
        warm_trace = trace[0]["warmup_trace"]
        if row["warmup"]:
            ensure(len(warm_trace) == 21, "wrong warmup trace length")
            check_noise(
                warm_trace,
                row["seed"] ^ 0xA117,
                row["interface_config"],
                row["warmup"]["interface"],
            )
            sent_warm = np.array([t["torque_nm"] for t in warm_trace[:-1]])
            delivered_warm = np.array([t["delivered_torque_nm"] for t in warm_trace[:-1]])
            lag = row["interface_config"]["command_delay_steps"]
            np.testing.assert_array_equal(
                delivered_warm, np.concatenate([np.zeros((lag, 7)), sent_warm])[:20]
            )
            np.testing.assert_array_equal(history, sent_warm[-4:])
            np.testing.assert_array_equal(row["initial_held_command"], delivered_warm[-1])
            np.testing.assert_array_equal(warm_trace[0]["q"], row["start_q"])
            np.testing.assert_array_equal(warm_trace[0]["qd"], np.zeros(7))
            np.testing.assert_array_equal(warm_trace[-1]["q"], trace[0]["q"])
            np.testing.assert_array_equal(warm_trace[-1]["qd"], trace[0]["qd"])
            saturation = 0
            for t in warm_trace[:-1]:
                q, qd = np.array(t["measured_q"]), np.array(t["measured_qd"])
                acceleration = 100 * (np.array(row["start_q"]) - q) - 20 * qd
                raw = regressor.inverse_dynamics(q, qd, acceleration, np.zeros(8))
                command = np.clip(raw, -LIMIT, LIMIT)
                np.testing.assert_allclose(command, t["torque_nm"], rtol=1e-11, atol=1e-10)
                saturation += int(np.any(np.abs(command - raw) > 1e-12))
            ensure(
                saturation == row["warmup"]["controller_saturated_steps"],
                "warmup clipping mismatch",
            )
        else:
            ensure(not warm_trace, "unexpected cold warmup trace")
            np.testing.assert_array_equal(history, np.zeros((4, 7)))
            np.testing.assert_array_equal(row["initial_q"], row["start_q"])
            np.testing.assert_array_equal(row["initial_qd"], np.zeros(7))
        np.testing.assert_array_equal(trace[0]["q"], row["initial_q"])
        np.testing.assert_array_equal(trace[0]["qd"], row["initial_qd"])
        np.testing.assert_array_equal(trace[-1]["q"], row["final_q"])
        np.testing.assert_allclose(
            np.max(np.abs(trace[-1]["qd"])), row["final_speed_max_rad_s"], rtol=1e-12
        )
        sent = np.array([t["torque_nm"] for t in trace[:-1]])
        errors = []
        innovations = []
        profile = row["scenario"].split("_")[-1]
        ages = applied_ages(profile)
        replay = None
        if row["arm"] == "bank_align":
            replay = PandaDelayController(predict=False)
            replay.sent_history = [u.copy() for u in history]
        elif row["arm"] in MEMORY_CONFIGS:
            replay = PandaScoreMemoryController(MEMORY_CONFIGS[row["arm"]])
            replay.sent_history = [u.copy() for u in history]
        elif row["arm"] == "known_align":
            replay = KnownAgeDiagnostic(ages, history)
        elif row["arm"] == "adaptive":
            replay = PandaAdaptiveController()
        saturation = 0
        for i, t in enumerate(trace[:-1]):
            index = i - ages[i]
            delivered = history[index + 4] if index < 0 else sent[index]
            np.testing.assert_array_equal(t["delivered_torque_nm"], delivered)
            ensure(
                t["applied_age_steps"] == ages[i] and t["delivered_command_index"] == index,
                "wrong applied packet age/index",
            )
            ensure(
                t["sent_packet_delay_steps"] == send_delay(profile, i), "wrong sent-packet delay"
            )
            reference = quintic_reference(row["start_q"], row["goal_q"], t["time_s"], 2.0)
            np.testing.assert_array_equal(t["reference"], reference[0])
            errors.append(np.array(t["q"]) - reference[0])
            state = np.array(t["measured_q"] + t["measured_qd"])
            if "control_state" in t:
                np.testing.assert_array_equal(state, t["control_state"])
                np.testing.assert_array_equal(reference[0], t["control_reference_q"])
                ensure(t["prediction_lookahead_s"] == 0, "unexpected prediction")
            theta = (
                np.zeros(8)
                if t["estimated_parameters"] is None
                else np.array(t["estimated_parameters"])
            )
            acceleration = (
                reference[2] + 100 * (reference[0] - state[:7]) + 20 * (reference[1] - state[7:])
            )
            raw = regressor.inverse_dynamics(state[:7], state[7:], acceleration, theta)
            command = np.clip(raw, -LIMIT, LIMIT)
            np.testing.assert_allclose(command, sent[i], rtol=1e-11, atol=1e-10)
            saturation += int(np.any(np.abs(command - raw) > 1e-12))
            if replay is not None:
                np.testing.assert_array_equal(theta, replay.estimator.parameters)
                following = trace[i + 1]
                if row["arm"] in MEMORY_CONFIGS:
                    np.testing.assert_array_equal(t["delay_scores"], replay.scores)
                    mq, mv = np.array(t["measured_q"]), np.array(t["measured_qd"])
                    nq, nv = np.array(following["measured_q"]), np.array(following["measured_qd"])
                    nominal, design = regressor.evaluate(
                        (mq + nq) / 2, (mv + nv) / 2, (nv - mv) / 0.01
                    )
                    squared = []
                    for lag, estimator in zip(replay.lags, replay.estimators, strict=True):
                        j = i - lag
                        aligned = history[j + 4] if j < 0 else sent[j]
                        innovation = aligned - nominal - design @ estimator.parameters
                        squared.append(float(innovation @ innovation))
                    np.testing.assert_allclose(
                        squared, t["score_innovation_squared"], rtol=1e-12, atol=1e-10
                    )
                    innovations.append(squared)
                replay.observe_transition(
                    t["measured_q"],
                    t["measured_qd"],
                    following["measured_q"],
                    following["measured_qd"],
                    sent[i],
                    0.01,
                )
                if row["arm"] in MEMORY_CONFIGS:
                    cfg = MEMORY_CONFIGS[row["arm"]]
                    values = np.array(innovations)
                    if cfg.rule == "window":
                        expected = values[-cfg.memory_steps :].sum(axis=0)
                    else:
                        weights = cfg.decay ** np.arange(len(values) - 1, -1, -1)
                        expected = np.sum(weights[:, None] * values, axis=0)
                    np.testing.assert_allclose(replay.scores, expected, rtol=1e-12, atol=1e-10)
                    ensure(
                        replay.selected_delay_steps == int(np.argmin(expected)),
                        "score ranking mismatch",
                    )
        ensure(saturation == row["scored_controller_saturated_steps"], "scored clipping mismatch")
        np.testing.assert_allclose(
            row["tracking_rms_rad"], np.sqrt(np.mean(np.square(errors), axis=0)), rtol=1e-12
        )
        for phase, part in (("before", errors[:100]), ("after", errors[100:])):
            np.testing.assert_allclose(
                row["tracking_phase_rms_rad"][phase],
                np.sqrt(np.mean(np.square(part), axis=0)),
                rtol=1e-12,
            )
        if replay is not None:
            np.testing.assert_array_equal(
                row["estimator"]["estimated_parameters"], replay.estimator.parameters
            )
            if hasattr(replay, "selected_history"):
                ensure(
                    row["delay_estimator"]["selected_history"] == replay.selected_history,
                    "causal selection replay differs",
                )
            if isinstance(replay, PandaDelayController):
                final = replay.delay_diagnostics()
                for field in (
                    "scores",
                    "candidate_estimates",
                    "candidate_bound_hits",
                    "selection_switches",
                ):
                    ensure(
                        row["delay_estimator"][field] == final[field],
                        f"bank replay differs: {field}",
                    )
                if row["arm"] in MEMORY_CONFIGS:
                    for field in ("score_memory", "score_decay", "score_updates"):
                        ensure(
                            row["delay_estimator"][field] == final[field],
                            f"memory replay differs: {field}",
                        )
                    np.testing.assert_array_equal(trace[-1]["delay_scores"], replay.scores)
            replayed += 1
    print(
        json.dumps(
            {
                "verified_episodes": len(rows),
                "verified_traces": len(traces),
                "causal_estimator_replays": replayed,
                "physical_steps_including_reproduction": summary["physical_steps"]
                + report["physical_steps"],
                "reproduction_virtual_substeps": report["prediction_virtual_substeps"],
                "source_commit": manifest["source_commit"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
