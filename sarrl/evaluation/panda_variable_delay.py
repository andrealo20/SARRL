"""Frozen alignment under a populated queue and changing packet latency."""

from __future__ import annotations

import hashlib
from dataclasses import asdict

import numpy as np

from sarrl.controllers.panda_adaptive import PandaAdaptiveController
from sarrl.controllers.panda_delay import PandaDelayController
from sarrl.controllers.panda_nominal import PandaComputedTorqueController
from sarrl.envs.panda_interface import PandaControlInterface, PandaInterfaceConfig
from sarrl.envs.panda_packet_delay import PROFILES, SWITCH_STEP, PandaPacketInterface, applied_ages
from sarrl.envs.panda_reach import PandaReachEnv
from sarrl.evaluation.panda_adaptation import run_episode as adaptation_episode
from sarrl.evaluation.panda_adaptation import sample_parameters
from sarrl.evaluation.panda_adaptation import summarize as adaptation_summary
from sarrl.evaluation.panda_posture_delay import posture_delay_protocol, sample_task
from sarrl.robots.panda import vector7

ARMS = ("fixed", "adaptive", "bank_align", "known_align")
WARM_STEPS = 20
WARM_NOISE_MASK = 0xA117
CONDITION_PARTS = {
    f"{startup}_{noise}_{profile}": (startup, noise, profile)
    for startup in ("cold", "warm")
    for noise in ("clean", "noise")
    for profile in PROFILES
}
CONDITIONS = tuple(CONDITION_PARTS)


def condition_config(condition):
    if condition not in CONDITION_PARTS:
        raise ValueError(f"unknown variable-delay condition: {condition}")
    _, noise, profile = CONDITION_PARTS[condition]
    sd = 0.001 if noise == "noise" else 0.0
    return PandaInterfaceConfig(sd, sd, PROFILES[profile][0])


class KnownAgeDiagnostic(PandaAdaptiveController):
    """Privileged current applied-command age; no delivered torque or true parameters."""

    def __init__(self, ages, history):
        super().__init__()
        self.ages, self.sent_history = tuple(ages), [np.array(u).copy() for u in history]
        self.selected_history = []
        self.last_control_state = None

    def prepare_control(self, q, qd):
        self.last_control_state = np.concatenate([vector7(q, "q"), vector7(qd, "qd")])
        return self.last_control_state[:7].copy(), self.last_control_state[7:].copy(), 0.0

    def observe_transition(self, q, qd, q_next, qd_next, sent_command, dt):
        lag = self.ages[len(self.selected_history)]
        self.sent_history.append(vector7(sent_command, "sent_command").copy())
        aligned = self.sent_history[len(self.sent_history) - 1 - lag]
        super().observe_transition(q, qd, q_next, qd_next, aligned, dt)
        self.selected_history.append(lag)

    def control_diagnostics(self):
        index = min(len(self.selected_history), len(self.ages) - 1)
        return {
            "selected_delay_steps": self.ages[index],
            "prediction_lookahead_s": 0.0,
            "prediction_fallback": False,
            "control_state": self.last_control_state.tolist(),
        }

    def delay_diagnostics(self):
        return {
            "selected_delay_steps": self.selected_history[-1],
            "selected_history": self.selected_history.copy(),
            "prediction_virtual_substeps": 0,
            "prediction_fallbacks": 0,
        }


def variable_delay_protocol(episodes=20, seed_start=9805100):
    protocol = posture_delay_protocol(episodes, seed_start)
    protocol.update(
        kind="exploratory_panda_warm_variable_delay_pilot",
        arms=list(ARMS),
        scenarios=list(CONDITIONS),
        interfaces={c: asdict(condition_config(c)) for c in CONDITIONS},
        posture={
            **protocol["posture"],
            "ranges": ["wide"],
            "pairing": "same wide planned trajectory across startup, profile, noise and arm",
        },
        actuation="clipped commands, scheduled packet arrivals, newest timestamp wins",
        state_interface="cached measured q and qd; paired scored noise after each handoff",
        profiles={p: {"before_steps": a, "after_steps": b} for p, (a, b) in PROFILES.items()},
        transport="arrival k + send_delay(k); latest timestamp wins; hold between arrivals",
        switch_step=SWITCH_STEP,
        startup={
            "cold": "original wide state at rest, four known zero prior commands",
            "warm": "20 shared nominal-hold intervals; fresh estimators with four known sends",
            "warmup_steps": WARM_STEPS,
            "warmup_noise_rng": "default_rng((episode_seed ^ 0xA117) ^ 0x50A07)",
            "scored_noise_rng": "original episode-seed stream, reset at handoff for pairing",
            "contrast": "warm handoff changes state and queue together",
        },
        privileged_delay_arms=["known_align"],
        oracle="known_align receives applied-command ages, never true physical parameters",
        prediction="disabled in every arm",
        frozen_controller_reference="results/panda_posture_delay_pilot/manifest.json",
        reproduction={
            **protocol["reproduction"],
            "episodes": 2800,
            "reference": "results/panda_posture_delay_pilot/episodes.json",
        },
    )
    return protocol


def _prepare(seed, condition, retain_trace):
    startup, _, profile = CONDITION_PARTS[condition]
    task = sample_task(seed, "wide")
    env = PandaReachEnv(parameters=sample_parameters(seed, "payload_friction"))
    env.reset(seed=seed, start=task.start, goal=task.goal)
    config = condition_config(condition)
    history = np.zeros((4, 7))
    held, warm_trace, warmup = np.zeros(7), [], None
    if startup == "warm":
        controller = PandaComputedTorqueController()
        interface = PandaControlInterface(config, seed ^ WARM_NOISE_MASK)
        sent_history, saturated = [], 0
        for k in range(WARM_STEPS):
            q, qd = env.data.qpos.copy(), env.data.qvel.copy()
            measured_q, measured_qd = interface.measure(q, qd, k)
            sent = controller.command(measured_q, measured_qd, task.start, np.zeros(7), np.zeros(7))
            saturated += int(np.any(np.abs(sent - controller.last_raw_torque) > 1e-12))
            held = interface.deliver(sent, env.torque_limit)
            sent_history.append(sent.copy())
            if retain_trace:
                warm_trace.append(
                    {
                        "time_s": (k - WARM_STEPS) * 0.01,
                        "q": q.tolist(),
                        "qd": qd.tolist(),
                        "measured_q": measured_q.tolist(),
                        "measured_qd": measured_qd.tolist(),
                        "torque_nm": sent.tolist(),
                        "delivered_torque_nm": held.tolist(),
                    }
                )
            env.step_torque(held)
        mq, mqd = interface.measure(env.data.qpos, env.data.qvel, WARM_STEPS)
        if retain_trace:
            warm_trace.append(
                {
                    "time_s": 0.0,
                    "q": env.data.qpos.tolist(),
                    "qd": env.data.qvel.tolist(),
                    "measured_q": mq.tolist(),
                    "measured_qd": mqd.tolist(),
                    "torque_nm": None,
                    "delivered_torque_nm": None,
                }
            )
        history = np.array(sent_history[-4:])
        warmup = {
            "physical_steps": WARM_STEPS * env.substeps,
            "diagnostics": env.diagnostics(),
            "interface": interface.diagnostics(),
            "controller_saturated_steps": saturated,
        }
        env.steps, env._done = 0, False
        env.position_excess = env.velocity_excess = 0.0
        env.contact_substeps = env.saturated_steps = 0
        env.max_torque = np.zeros(7)
        env.max_speed = np.zeros(7)
    return env, task, config, profile, history, held, warmup, warm_trace


def run_episode(seed, scenario, arm, *, retain_trace=False, bank_factory=None):
    if arm not in ARMS:
        raise ValueError(f"unknown variable-delay arm: {arm}")
    if bank_factory is not None and (arm != "bank_align" or not callable(bank_factory)):
        raise ValueError("bank factories require the bank_align arm and a callable")
    config = condition_config(scenario)
    env, task, _, profile, history, held, warmup, warm_trace = _prepare(
        seed, scenario, retain_trace
    )
    initial_q, initial_qd = env.data.qpos.copy(), env.data.qvel.copy()
    initial_time = float(env.data.time)
    ages = applied_ages(profile, env.horizon)
    interface = PandaPacketInterface(config, seed, profile, history, held)
    controller = None
    if arm == "bank_align":
        controller = PandaDelayController(predict=False) if bank_factory is None else bank_factory()
        if not isinstance(controller, PandaDelayController) or controller.predict:
            raise ValueError(
                "bank factories must return a Panda delay controller without prediction"
            )
        controller.sent_history = [u.copy() for u in history]
    elif arm == "known_align":
        controller = KnownAgeDiagnostic(ages, history)
    record, trace = adaptation_episode(
        seed,
        "payload_friction",
        "adaptive" if controller is not None else arm,
        retain_trace=retain_trace,
        interface_config=config,
        controller_override=controller,
        prepared_env=env,
        prepared_interface=interface,
        phase_split_step=SWITCH_STEP,
    )
    actual_ages = tuple(item["applied_age_steps"] for item in interface.delivery_trace)
    if actual_ages != ages:
        raise RuntimeError("packet delivery differs from the independent timestamp schedule")
    record.update(
        scenario=scenario,
        arm=arm,
        plant_scenario="payload_friction",
        posture="wide",
        task_sampling=asdict(task),
        privileged_delay=arm == "known_align",
        warmup=warmup,
        initial_q=initial_q.tolist(),
        initial_qd=initial_qd.tolist(),
        initial_simulation_time_s=initial_time,
        initial_command_history=history.tolist(),
        initial_held_command=held.tolist(),
        scored_physical_steps=record["physical_steps"],
        scored_controller_saturated_steps=record["controller_saturated_steps"],
    )
    if warmup:
        d = warmup["diagnostics"]
        record["physical_steps"] += warmup["physical_steps"]
        record["controller_saturated_steps"] += warmup["controller_saturated_steps"]
        record["contact_substeps"] += d["contact_substeps"]
        for key in ("joint_position_excess_rad", "joint_velocity_excess_rad_s"):
            record[key] = max(record[key], d[key])
        for key in ("max_torque_nm", "max_speed_rad_s"):
            record[key] = np.maximum(record[key], d[key]).tolist()
        record["unsafe_episode"] = bool(record["unsafe_episode"] or d["unsafe_episode"])
        record["success"] = bool(record["success"] and not record["unsafe_episode"])
    if controller is not None:
        selected = controller.selected_history
        correct = np.array(selected) == ages
        final_stable = next(i for i in range(len(ages)) if all(a == ages[-1] for a in ages[i:]))
        incorrect = [i for i in range(final_stable, len(ages)) if selected[i] != ages[-1]]
        acquisition = (
            ((incorrect[-1] + 2) * 0.01 if incorrect else (final_stable + 1) * 0.01)
            if correct[-1]
            else None
        )
        record["delay_estimator"].update(
            final_delay_correct=bool(correct[-1]),
            correct_selection_fraction=float(np.mean(correct)),
            before_switch_correct_fraction=float(np.mean(correct[:SWITCH_STEP])),
            after_switch_correct_fraction=float(np.mean(correct[SWITCH_STEP:])),
            final_regime_start_s=final_stable * 0.01,
            final_regime_acquired_at_s=acquisition,
            final_regime_acquisition_delay_s=None
            if acquisition is None
            else acquisition - final_stable * 0.01,
        )
    if retain_trace:
        trace[0]["warmup_trace"] = warm_trace
        for sample, metadata in zip(trace[:-1], interface.delivery_trace, strict=True):
            sample.update(metadata)
    return record, trace


def summarize(records, *, arms=ARMS):
    summary = adaptation_summary(records, scenarios=CONDITIONS, arms=arms)
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in records}
    seeds = sorted({r["seed"] for r in records})
    for seed in seeds:
        reference = lookup[CONDITIONS[0], seed, "fixed"]
        hashes = {n: set() for n in ("clean", "noise")}
        for condition, (startup, noise, profile) in CONDITION_PARTS.items():
            ref = lookup[condition, seed, "fixed"]
            for arm in arms:
                row = lookup[condition, seed, arm]
                for field in ("start_q", "goal_q", "true_parameters", "task_sampling"):
                    if row[field] != reference[field]:
                        raise ValueError(f"unpaired planned {field}")
                for field in (
                    "initial_q",
                    "initial_qd",
                    "initial_command_history",
                    "initial_held_command",
                    "warmup",
                ):
                    if row[field] != ref[field]:
                        raise ValueError(f"unpaired handoff {field}")
                if row["interface_config"] != asdict(condition_config(condition)):
                    raise ValueError("incorrect initial interface configuration")
                expected_hash = hashlib.sha256(
                    np.array(applied_ages(profile), dtype="<i8").tobytes()
                ).hexdigest()
                if row["applied_age_sha256"] != expected_hash:
                    raise ValueError("unpaired or incorrect applied-command age schedule")
                if (
                    row["privileged_delay"] != (arm == "known_align")
                    or row["privileged_parameters"]
                ):
                    raise ValueError("incorrect variable-delay privilege label")
                if bool(row["warmup"]) != (startup == "warm"):
                    raise ValueError("incorrect startup label")
                hashes[noise].add(row["measurement_noise_sha256"])
        if any(len(values) != 1 for values in hashes.values()):
            raise ValueError("unpaired scored measurement noise")
        for startup in ("cold", "warm"):
            for noise in ("clean", "noise"):
                for changing, constant in (("up", "fixed10"), ("down", "fixed40")):
                    a = lookup[f"{startup}_{noise}_{changing}", seed, "fixed"]
                    b = lookup[f"{startup}_{noise}_{constant}", seed, "fixed"]
                    for field in ("initial_q", "initial_qd", "initial_command_history", "warmup"):
                        if a[field] != b[field]:
                            raise ValueError(
                                "unpaired initial state across matched packet profiles"
                            )
    for condition in CONDITIONS:
        for arm in arms:
            rows = [lookup[condition, seed, arm] for seed in seeds]
            cell = summary["cells"][f"{condition}/{arm}"]
            for phase in ("before", "after"):
                cell[f"mean_{phase}_tracking_rms_rad"] = float(
                    np.mean([np.mean(r["tracking_phase_rms_rad"][phase]) for r in rows])
                )
            if rows[0]["estimator"] is not None:
                cell["mean_bound_hit_fraction"] = float(
                    np.mean([r["estimator"]["bound_hits"] / 300 for r in rows])
                )
            if "delay_estimator" in rows[0]:
                cell.update(
                    final_delay_correct_episodes=sum(
                        r["delay_estimator"]["final_delay_correct"] for r in rows
                    ),
                    mean_correct_selection_fraction=float(
                        np.mean([r["delay_estimator"]["correct_selection_fraction"] for r in rows])
                    ),
                    mean_before_switch_correct_fraction=float(
                        np.mean(
                            [r["delay_estimator"]["before_switch_correct_fraction"] for r in rows]
                        )
                    ),
                    mean_after_switch_correct_fraction=float(
                        np.mean(
                            [r["delay_estimator"]["after_switch_correct_fraction"] for r in rows]
                        )
                    ),
                    final_regime_acquisition_delays_s=[
                        r["delay_estimator"]["final_regime_acquisition_delay_s"] for r in rows
                    ],
                )
    summary["paired_contrasts"] = {}
    for condition in CONDITIONS:
        for treatment, baseline in (("bank_align", "adaptive"), ("known_align", "bank_align")):
            pairs = [
                (lookup[condition, seed, treatment], lookup[condition, seed, baseline])
                for seed in seeds
            ]
            summary["paired_contrasts"][f"{condition}/{treatment}-{baseline}"] = {
                "success_count_difference": sum(a["success"] - b["success"] for a, b in pairs),
                "mean_paired_final_joint_error_difference_rad": float(
                    np.mean([a["joint_error_max_rad"] - b["joint_error_max_rad"] for a, b in pairs])
                ),
                "mean_paired_tracking_rms_difference_rad": float(
                    np.mean(
                        [
                            np.mean(a["tracking_rms_rad"]) - np.mean(b["tracking_rms_rad"])
                            for a, b in pairs
                        ]
                    )
                ),
            }
    summary["warmup_physical_steps"] = sum(
        r["warmup"]["physical_steps"] for r in records if r["warmup"]
    )
    summary["startup_contrasts"] = {}
    for noise in ("clean", "noise"):
        for profile in PROFILES:
            for arm in arms:
                pairs = [
                    (
                        lookup[f"warm_{noise}_{profile}", seed, arm],
                        lookup[f"cold_{noise}_{profile}", seed, arm],
                    )
                    for seed in seeds
                ]
                summary["startup_contrasts"][f"{noise}_{profile}/{arm}"] = {
                    "success_count_difference": sum(a["success"] - b["success"] for a, b in pairs),
                    "mean_paired_final_joint_error_difference_rad": float(
                        np.mean(
                            [a["joint_error_max_rad"] - b["joint_error_max_rad"] for a, b in pairs]
                        )
                    ),
                    "mean_paired_after_tracking_rms_difference_rad": float(
                        np.mean(
                            [
                                np.mean(a["tracking_phase_rms_rad"]["after"])
                                - np.mean(b["tracking_phase_rms_rad"]["after"])
                                for a, b in pairs
                            ]
                        )
                    ),
                }
    return summary
