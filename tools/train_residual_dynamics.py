#!/usr/bin/env python3
"""Generate randomized plant data and fit an ensemble residual dynamics model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sarrl.dynamics import PlanarArm
from sarrl.envs import DomainRandomization, PlanarReachEnv
from sarrl.models import (
    ResidualDynamicsEnsemble,
    residual_acceleration_target,
    train_residual_ensemble,
)


def collect(samples: int, seed: int):
    env = PlanarReachEnv(
        mode="torque",
        randomization=DomainRandomization(
            mass_fraction=0.25,
            friction_fraction=0.4,
            motor_gain_fraction=0.2,
            payload_range=(0.0, 1.5),
        ),
    )
    nominal = PlanarArm()
    rng = np.random.default_rng(seed)
    states, torques, targets = [], [], []
    env.reset(seed=seed)
    for i in range(samples):
        if i % 64 == 0:
            env.reset(seed=seed + i)
        state = env.state.copy()
        commanded = rng.uniform(-30.0, 30.0, size=2)
        applied = commanded * env.motor_gain
        actual_qdd = env.arm.forward_dynamics(state[:2], state[2:], applied)
        # The learned model receives the commanded torque at runtime. Keep
        # motor-gain error inside the residual target rather than hiding it by
        # replacing the command with the already-degraded applied torque.
        states.append(state.astype(np.float32))
        torques.append(commanded.astype(np.float32))
        targets.append(residual_acceleration_target(nominal, state, commanded, actual_qdd))
        env.state = env.arm.step_rk4(state, applied, env.dt)
    return np.asarray(states), np.asarray(torques), np.asarray(targets)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="results/residual_dynamics/ensemble.pt")
    args = p.parse_args()
    states, actions, targets = collect(args.samples, args.seed)
    model = ResidualDynamicsEnsemble(seed=args.seed)
    stats = train_residual_ensemble(
        model, states, actions, targets, steps=args.steps, seed=args.seed
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    np.savez_compressed(path.with_suffix(".npz"), states=states, actions=actions, targets=targets)
    print(f"residual dynamics loss: {stats.initial_loss:.6f} -> {stats.final_loss:.6f}")
    print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
