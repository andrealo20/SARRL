"""Descriptive engineering pilot for the Panda computed-torque baseline."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from sarrl.controllers.panda_nominal import PandaComputedTorqueController
from sarrl.envs.panda_reach import PandaReachConfig, PandaReachEnv


def run_episode(seed: int, config: PandaReachConfig | None = None, *, retain_trace=False):
    env = PandaReachEnv(config)
    controller = PandaComputedTorqueController()
    _, reset_info = env.reset(seed=seed)
    start, goal = env.start.copy(), env.goal.copy()
    tracking = np.zeros(7)
    saturated = 0
    trace = []
    for _ in range(env.horizon):
        reference = env.reference()
        q, qd = env.data.qpos.copy(), env.data.qvel.copy()
        tracking = np.maximum(tracking, np.abs(q - reference[0]))
        torque = controller.command(q, qd, *reference)
        saturated += int(np.any(np.abs(torque - controller.last_raw_torque) > 1e-12))
        if retain_trace:
            trace.append(
                {
                    "time_s": env.steps * env.config.control_dt,
                    "q": q.tolist(),
                    "qd": qd.tolist(),
                    "reference": reference[0].tolist(),
                    "torque_nm": torque.tolist(),
                    "tip": env.data.site_xpos[env.site].tolist(),
                }
            )
        env.step_torque(torque)
    final = env.diagnostics()
    if retain_trace:
        trace.append(
            {
                "time_s": env.steps * env.config.control_dt,
                "q": env.data.qpos.tolist(),
                "qd": env.data.qvel.tolist(),
                "reference": env.goal.tolist(),
                "torque_nm": None,
                "tip": env.data.site_xpos[env.site].tolist(),
            }
        )
    return {
        "seed": seed,
        "start_q": start.tolist(),
        "goal_q": goal.tolist(),
        "target_xyz_m": env.target.tolist(),
        "final_q": env.data.qpos.tolist(),
        "sampling_attempts": reset_info["sampling_attempts"],
        "tracking_error_max_rad": tracking.tolist(),
        "controller_saturated_steps": saturated,
        **final,
    }, trace


def summarize(episodes: list[dict], config: PandaReachConfig) -> dict:
    if not episodes:
        raise ValueError("the Panda pilot needs at least one episode")
    return {
        "scope": "nominal engineering pilot; no adaptation, learning or safety certificate",
        "config": asdict(config),
        "episodes": len(episodes),
        "successes": sum(r["success"] for r in episodes),
        "unsafe_episodes": sum(r["unsafe_episode"] for r in episodes),
        "contact_episodes": sum(r["contact_substeps"] > 0 for r in episodes),
        "saturated_episodes": sum(r["controller_saturated_steps"] > 0 for r in episodes),
        "physical_steps": sum(r["steps"] for r in episodes)
        * round(config.control_dt / config.timestep),
        "max_final_joint_error_rad": max(r["joint_error_max_rad"] for r in episodes),
        "max_final_tip_error_m": max(r["tip_error_m"] for r in episodes),
        "max_final_speed_rad_s": max(r["final_speed_max_rad_s"] for r in episodes),
        "max_tracking_error_rad": max(max(r["tracking_error_max_rad"]) for r in episodes),
        "max_position_excess_rad": max(r["joint_position_excess_rad"] for r in episodes),
        "max_velocity_excess_rad_s": max(r["joint_velocity_excess_rad_s"] for r in episodes),
    }
