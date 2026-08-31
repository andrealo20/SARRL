#!/usr/bin/env python3
"""Evaluate a trained residual SAC policy through the composed SARRL runtime stack."""

from __future__ import annotations

import argparse

import numpy as np

from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm
from sarrl.envs import DomainRandomization, FaultSpec, PlanarReachEnv
from sarrl.rl import SACAgent
from sarrl.runtime import ControlStackConfig, SARRLControlStack
from sarrl.safety import HOCBFSafetyFilter, SafetyConfig


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=20_000)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--motor-fault", type=float, default=None, help="joint-2 gain after step 80")
    p.add_argument("--safety", action="store_true")
    args = p.parse_args()
    if args.episodes <= 0:
        raise SystemExit("episodes must be positive")

    dr = (
        DomainRandomization(
            mass_fraction=0.15,
            friction_fraction=0.30,
            motor_gain_fraction=0.15,
            payload_range=(0.0, 1.0),
            action_delay_max=2,
        )
        if args.randomize
        else DomainRandomization()
    )
    fault = None
    if args.motor_fault is not None:
        fault = FaultSpec(start_step=80, motor_gain_multiplier=(1.0, args.motor_fault))
    env = PlanarReachEnv(mode="torque", randomization=dr, fault=fault)
    agent = SACAgent.from_checkpoint(args.checkpoint, seed=0, load_optimizers=False)
    if agent.obs_dim != 8 or agent.action_dim != 2:
        raise SystemExit("stack evaluator currently expects a base 8-D observation checkpoint")
    nominal = PlanarArm()
    baseline = ComputedTorqueController(nominal)
    safety = HOCBFSafetyFilter(nominal, SafetyConfig()) if args.safety else None
    stack = SARRLControlStack(
        baseline,
        agent,
        ControlStackConfig(require_safety=args.safety),
        safety_filter=safety,
    )

    successes = 0
    failures = 0
    rewards = []
    interventions = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total = 0.0
        episode_intervention = 0.0
        while True:
            result = stack.command(obs, env.state, env.q_des, deterministic=True)
            if not result.executable:
                failures += 1
                break
            obs, reward, terminated, truncated, info = env.step_torque(
                result.torque, baseline=result.baseline_torque
            )
            total += reward
            episode_intervention += result.safety_correction
            if terminated or truncated:
                successes += int(info["success"])
                rewards.append(total)
                interventions.append(episode_intervention)
                break
    completed = len(rewards)
    print(f"success: {successes}/{args.episodes} = {100.0 * successes / args.episodes:.1f}%")
    print(f"safety infeasible episodes: {failures}")
    if completed:
        print(f"reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
        print(f"safety correction/episode: {np.mean(interventions):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
