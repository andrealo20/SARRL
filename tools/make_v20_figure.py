#!/usr/bin/env python3
"""Draw the v2.0 figure: one motor-fault episode and the time-constant identification.

The left panels replay one decision-block episode on the MuJoCo plant with the
frozen v2.0 configuration, fixed nominal against identified nominal behind the
same HOCBF filter. The seed is the first motor-fault decision seed on which the
retained campaign log records a timeout for the fixed arm and a clean success
for the identified one, so the episode is typical rather than chosen. The right
panel reads the retained episode table and plots the selected actuator time
constant against the true one for every adaptive decision episode.

Usage: python -m tools.make_v20_figure [--output assets/v20_mujoco_results.png]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sarrl.controllers.adaptive_nominal import AdaptiveNominalController
from sarrl.dynamics import PlanarArm
from sarrl.evaluation.adaptive_pilot import MeasuredState, build_stack, make_env
from sarrl.evaluation.mujoco_campaign import (
    V20_ACTUATOR_GRID,
    V20_COMPENSATE_DELAY,
    V20_OUTPUT,
    V20_PLANT,
    V20_PLANT_OPTIONS,
    V20_PRIMARY_SEED_START,
    v20_config,
)
from sarrl.evaluation.planar_v12 import planar_safety_config
from sarrl.evaluation.planar_v13 import v13_scenarios
from sarrl.evaluation.safety_audit import evaluate_safety_episodes
from sarrl.safety import HOCBFSafetyFilter

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / V20_OUTPUT / "episodes.csv"
SCENARIO = "motor_fault"
FAULT_STEP = 20
COLORS = {"fixed_hocbf": "#b5533c", "adaptive_hocbf": "#2a6f97"}
LABELS = {"fixed_hocbf": "fixed nominal + HOCBF", "adaptive_hocbf": "identified nominal + HOCBF"}


def read_table() -> list[dict]:
    with TABLE.open(newline="") as handle:
        return list(csv.DictReader(handle))


def pick_seed(rows: list[dict]) -> int:
    by_seed: dict[int, dict[str, dict]] = {}
    for row in rows:
        seed = int(row["seed"])
        if row["scenario"] == SCENARIO and seed >= V20_PRIMARY_SEED_START:
            by_seed.setdefault(seed, {})[row["arm"]] = row
    for seed in sorted(by_seed):
        fixed = by_seed[seed].get("fixed_hocbf")
        adaptive = by_seed[seed].get("adaptive_hocbf")
        if fixed is None or adaptive is None:
            continue
        if (
            fixed["outcome"] == "timeout"
            and adaptive["outcome"] == "success"
            and fixed["unsafe_episode"] == "False"
            and adaptive["unsafe_episode"] == "False"
        ):
            return seed
    raise RuntimeError("no motor-fault decision seed matches the selection rule")


def replay(arm: str, seed: int) -> dict:
    spec = {s.key: s for s in v13_scenarios()}[SCENARIO]
    env = make_env(V20_PLANT, spec, V20_PLANT_OPTIONS)
    controller, stack = build_stack(arm, v20_config(), V20_COMPENSATE_DELAY)
    adaptive = isinstance(controller, AdaptiveNominalController)
    if adaptive:
        controller.reset()
    measured = MeasuredState(env)
    trace = {"distance": [], "torque": [], "time_constant": []}

    def observe(event):
        if event["info"] is None:
            return
        if adaptive:
            controller.observe(event["state"], measured(), event["command"].torque)
            trace["time_constant"].append(controller.time_constant)
        trace["distance"].append(float(event["info"]["distance"]))
        trace["torque"].append(np.asarray(event["command"].torque, dtype=float).copy())

    outcomes, _ = evaluate_safety_episodes(
        stack,
        HOCBFSafetyFilter(PlanarArm(), planar_safety_config()),
        env,
        episodes=1,
        seed=seed,
        scenario=SCENARIO,
        controller=arm,
        transition_callback=observe,
        state_source=measured,
    )
    return {
        "distance": np.asarray(trace["distance"]),
        "torque": np.asarray(trace["torque"]),
        "dt": float(env.dt),
        "success": bool(outcomes[0].success),
        "true_time_constant": float(env.actuator_time_constant),
        "selected_time_constant": float(controller.time_constant) if adaptive else None,
    }


def identification(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    true = []
    selected = []
    for row in rows:
        if row["arm"] == "adaptive_hocbf" and int(row["seed"]) >= V20_PRIMARY_SEED_START:
            true.append(float(row["true_time_constant"]))
            selected.append(float(row["selected_time_constant"]))
    return np.asarray(true), np.asarray(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets" / "v20_mujoco_results.png")
    args = parser.parse_args()

    rows = read_table()
    seed = pick_seed(rows)
    traces = {arm: replay(arm, seed) for arm in ("fixed_hocbf", "adaptive_hocbf")}
    true_tau, selected_tau = identification(rows)
    grid = np.asarray(V20_ACTUATOR_GRID)
    abs_error_ms = float(np.mean(np.abs(selected_tau - true_tau))) * 1e3

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4), constrained_layout=True)

    ax = axes[0]
    for arm, trace in traces.items():
        t = np.arange(len(trace["distance"])) * trace["dt"]
        ax.plot(t, trace["distance"], color=COLORS[arm], lw=1.6, label=LABELS[arm])
    ax.axvline(FAULT_STEP * traces["fixed_hocbf"]["dt"], color="0.5", ls="--", lw=0.9)
    ax.axhline(0.05, color="0.7", ls=":", lw=0.9)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("distance to target [m]")
    ax.set_title(f"motor fault, seed {seed}", loc="left")
    ax.text(
        FAULT_STEP * traces["fixed_hocbf"]["dt"],
        ax.get_ylim()[1] * 0.97,
        " joint 2 gain drops to 0.55",
        color="0.35",
        va="top",
        fontsize=8,
    )
    ax.legend(frameon=False, loc="center right")

    ax = axes[1]
    for arm, trace in traces.items():
        t = np.arange(len(trace["torque"])) * trace["dt"]
        ax.plot(t, trace["torque"][:, 1], color=COLORS[arm], lw=1.4, label=LABELS[arm])
    ax.axvline(FAULT_STEP * traces["fixed_hocbf"]["dt"], color="0.5", ls="--", lw=0.9)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("joint 2 torque after filter [N m]")
    ax.set_title("executed command", loc="left")

    ax = axes[2]
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.0025, 0.0025, size=selected_tau.shape)
    ax.scatter(
        true_tau * 1e3,
        (selected_tau + jitter) * 1e3,
        s=4,
        alpha=0.25,
        color=COLORS["adaptive_hocbf"],
        edgecolors="none",
    )
    ax.plot([10, 50], [10, 50], color="0.3", lw=0.9, ls="--", label="exact")
    ax.legend(frameon=False, loc="upper left")
    ax.set_yticks(grid * 1e3)
    ax.set_xlabel("true actuator time constant [ms]")
    ax.set_ylabel("selected hypothesis [ms]")
    ax.set_title(f"identified time constant, mean error {abs_error_ms:.1f} ms", loc="left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"seed {seed}: fixed success {traces['fixed_hocbf']['success']}, ", end="")
    print(f"adaptive success {traces['adaptive_hocbf']['success']}")
    print(f"true tau {traces['adaptive_hocbf']['true_time_constant'] * 1e3:.1f} ms, ", end="")
    print(f"selected {traces['adaptive_hocbf']['selected_time_constant'] * 1e3:.0f} ms")
    print(f"mean absolute time-constant error {abs_error_ms:.2f} ms")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
