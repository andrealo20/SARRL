"""A frozen-controller delay grid on matched trajectories at two posture ranges."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from sarrl.controllers.panda_delay import PandaDelayController
from sarrl.envs.panda_interface import PandaInterfaceConfig
from sarrl.envs.panda_reach import PandaReachEnv
from sarrl.evaluation.panda_adaptation import run_episode as adaptation_episode
from sarrl.evaluation.panda_adaptation import summarize as adaptation_summary
from sarrl.evaluation.panda_delay import ARMS, CONTRASTS, delay_protocol

POSTURES = ("narrow", "wide")
SHIFT_HALF_RANGE = (1.0, 0.65, 1.0, 0.65, 1.0, 0.55, 1.0)
MIN_SHIFT_RMS = 0.35
CONDITION_PARTS = {
    f"{posture}_{noise}_d{lag * 10:02d}": (posture, noise, lag)
    for posture in POSTURES
    for noise in ("clean", "noise")
    for lag in range(5)
}
CONDITIONS = tuple(CONDITION_PARTS)


def condition_config(condition):
    if condition not in CONDITION_PARTS:
        raise ValueError(f"unknown posture/delay condition: {condition}")
    _, noise, lag = CONDITION_PARTS[condition]
    sd = 0.001 if noise == "noise" else 0.0
    return PandaInterfaceConfig(sd, sd, lag)


@dataclass(frozen=True)
class PandaTask:
    start: tuple[float, ...]
    goal: tuple[float, ...]
    shift: tuple[float, ...]
    attempts: int


@lru_cache(maxsize=256)
def sample_task(seed: int, posture: str) -> PandaTask:
    if posture not in POSTURES:
        raise ValueError(f"unknown Panda posture range: {posture}")
    env = PandaReachEnv()
    _, info = env.reset(seed=seed)
    start, goal = env.start.copy(), env.goal.copy()
    if posture == "narrow":
        return PandaTask(tuple(start), tuple(goal), (0.0,) * 7, info["sampling_attempts"])
    rng = np.random.default_rng(seed ^ 0x7A5C0)
    half_range = np.array(SHIFT_HALF_RANGE)
    for attempt in range(1000):
        shift = rng.uniform(-half_range, half_range)
        shifted = (start + shift, goal + shift)
        if np.sqrt(np.mean(shift**2)) < MIN_SHIFT_RMS:
            continue
        if not all(np.all((q > env.lower + 0.15) & (q < env.upper - 0.15)) for q in shifted):
            continue
        if env._path_clear(*shifted):
            return PandaTask(tuple(shifted[0]), tuple(shifted[1]), tuple(shift), attempt + 1)
    raise RuntimeError("cannot sample the predeclared translated Panda path")


def posture_delay_protocol(episodes=20, seed_start=9804900):
    protocol = delay_protocol(episodes, seed_start)
    protocol.update(
        kind="exploratory_panda_posture_delay_pilot",
        scenarios=list(CONDITIONS),
        interfaces={name: asdict(condition_config(name)) for name in CONDITIONS},
        posture={
            "ranges": list(POSTURES),
            "narrow": "original sampler, with original episode seed",
            "wide": "translate both narrow endpoints by the same accepted offset",
            "shift_half_range_rad": list(SHIFT_HALF_RANGE),
            "minimum_shift_rms_rad": MIN_SHIFT_RMS,
            "sample_rng": "default_rng(episode_seed ^ 0x7A5C0)",
            "rejection": "RMS shift, 0.15 rad joint clearance, original 21-point contact check",
            "maximum_attempts": 1000,
            "pairing": "same displacement, duration, parameters and noise across postures",
        },
        frozen_controller_reference="results/panda_delay_pilot/manifest.json",
        reproduction={
            **protocol["reproduction"],
            "episodes": 560,
            "reference": "results/panda_delay_pilot/episodes.json",
        },
    )
    return protocol


def run_episode(seed, scenario, arm, *, retain_trace=False):
    if arm not in ARMS:
        raise ValueError(f"unknown Panda posture/delay arm: {arm}")
    config = condition_config(scenario)
    posture, _, _ = CONDITION_PARTS[scenario]
    task = sample_task(seed, posture)
    privileged = arm in ("known_align", "known_predict")
    controller = None
    if arm in ("known_align", "bank_align", "known_predict", "bank_predict"):
        controller = PandaDelayController(
            known_delay_steps=config.command_delay_steps if privileged else None,
            predict=arm.endswith("predict"),
        )
    task_kwargs = {"task_start": task.start, "task_goal": task.goal} if posture == "wide" else {}
    record, trace = adaptation_episode(
        seed,
        "payload_friction",
        "adaptive" if controller is not None else arm,
        retain_trace=retain_trace,
        interface_config=config,
        controller_override=controller,
        **task_kwargs,
    )
    record.update(
        scenario=scenario,
        arm=arm,
        plant_scenario="payload_friction",
        privileged_delay=privileged,
        posture=posture,
        task_sampling=asdict(task),
    )
    if controller is not None:
        selected = controller.selected_history
        truth = config.command_delay_steps
        incorrect = [i for i, lag in enumerate(selected) if lag != truth]
        record["delay_estimator"].update(
            final_delay_correct=selected[-1] == truth,
            correct_selection_fraction=sum(lag == truth for lag in selected) / len(selected),
            final_correct_run_start_s=(
                (incorrect[-1] + 2) * controller.delay_config.control_dt
                if incorrect
                else controller.delay_config.control_dt
            )
            if selected[-1] == truth
            else None,
        )
    return record, trace


def summarize(records):
    summary = adaptation_summary(records, scenarios=CONDITIONS, arms=ARMS)
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in records}
    seeds = sorted({r["seed"] for r in records})
    for seed in seeds:
        physical = lookup[CONDITIONS[0], seed, "fixed"]["true_parameters"]
        noise_hashes = {noise: set() for noise in ("clean", "noise")}
        for posture in POSTURES:
            reference = lookup[f"{posture}_clean_d00", seed, "fixed"]
            for condition, (pose, noise, _) in CONDITION_PARTS.items():
                if pose != posture:
                    continue
                for arm in ARMS:
                    row = lookup[condition, seed, arm]
                    for field in ("start_q", "goal_q", "target_xyz_m", "task_sampling"):
                        if row[field] != reference[field]:
                            raise ValueError(f"unpaired {field} within posture")
                    if row["true_parameters"] != physical:
                        raise ValueError("unpaired physical parameters across postures")
                    if row["interface_config"] != asdict(condition_config(condition)):
                        raise ValueError("incorrect posture/delay interface")
                    if row["posture"] != posture:
                        raise ValueError("incorrect posture label")
                    if row["privileged_delay"] != (arm in ("known_align", "known_predict")):
                        raise ValueError("incorrect privileged-delay label")
                    if row["privileged_parameters"] != (arm == "oracle"):
                        raise ValueError("incorrect privileged-parameter label")
                    noise_hashes[noise].add(row["measurement_noise_sha256"])
        if any(len(values) != 1 for values in noise_hashes.values()):
            raise ValueError("unpaired exogenous noise across posture/delay arms")
        narrow = lookup["narrow_clean_d00", seed, "fixed"]
        wide = lookup["wide_clean_d00", seed, "fixed"]
        shift = np.array(wide["task_sampling"]["shift"])
        if not (
            np.all(np.abs(shift) <= SHIFT_HALF_RANGE)
            and np.sqrt(np.mean(shift**2)) >= MIN_SHIFT_RMS
            and np.allclose(
                np.array(wide["start_q"]) - narrow["start_q"], shift, atol=1e-14, rtol=0
            )
            and np.allclose(np.array(wide["goal_q"]) - narrow["goal_q"], shift, atol=1e-14, rtol=0)
        ):
            raise ValueError("postures do not preserve the declared translated trajectory")
    summary["delay_contrasts"] = {}
    for condition in CONDITIONS:
        for arm in ARMS:
            rows = [lookup[condition, seed, arm] for seed in seeds]
            cell = summary["cells"][f"{condition}/{arm}"]
            if rows[0]["estimator"] is not None:
                cell["mean_bound_hit_fraction"] = float(
                    np.mean(
                        [r["estimator"]["bound_hits"] / r["estimator"]["updates"] for r in rows]
                    )
                )
            if "delay_estimator" in rows[0]:
                cell.update(
                    final_delay_correct_episodes=sum(
                        r["delay_estimator"]["final_delay_correct"] for r in rows
                    ),
                    mean_correct_selection_fraction=float(
                        np.mean([r["delay_estimator"]["correct_selection_fraction"] for r in rows])
                    ),
                    prediction_fallbacks=sum(
                        r["delay_estimator"]["prediction_fallbacks"] for r in rows
                    ),
                    prediction_virtual_substeps=sum(
                        r["delay_estimator"]["prediction_virtual_substeps"] for r in rows
                    ),
                )
        for treatment, baseline in CONTRASTS:
            pairs = [
                (lookup[condition, seed, treatment], lookup[condition, seed, baseline])
                for seed in seeds
            ]
            summary["delay_contrasts"][f"{condition}/{treatment}-{baseline}"] = {
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
    summary["prediction_virtual_substeps"] = sum(
        r.get("delay_estimator", {}).get("prediction_virtual_substeps", 0) for r in records
    )
    summary["posture_contrasts"] = {}
    for noise in ("clean", "noise"):
        for lag in range(5):
            for arm in ARMS:
                pairs = [
                    (
                        lookup[f"wide_{noise}_d{lag * 10:02d}", seed, arm],
                        lookup[f"narrow_{noise}_d{lag * 10:02d}", seed, arm],
                    )
                    for seed in seeds
                ]
                summary["posture_contrasts"][f"{noise}_d{lag * 10:02d}/{arm}"] = {
                    "success_count_difference": sum(a["success"] - b["success"] for a, b in pairs),
                    "unsafe_episode_count_difference": sum(
                        a["unsafe_episode"] - b["unsafe_episode"] for a, b in pairs
                    ),
                    "mean_paired_final_joint_error_difference_rad": float(
                        np.mean(
                            [a["joint_error_max_rad"] - b["joint_error_max_rad"] for a, b in pairs]
                        )
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
    return summary
