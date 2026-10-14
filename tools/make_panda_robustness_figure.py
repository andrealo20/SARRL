"""Plot saved Panda noise/delay outcomes and the preselected first-seed traces."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.figure_style import apply_style, save_figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/panda_robustness_pilot"))
    parser.add_argument("--output", type=Path, default=Path("assets/panda_robustness_pilot.png"))
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("figure output exists; choose a new path")
    complete = json.loads((args.input / "complete.json").read_text())
    for name, expected in complete["hashes"].items():
        if hashlib.sha256((args.input / name).read_bytes()).hexdigest() != expected:
            parser.error(f"saved evidence hash mismatch: {name}")
    records = json.loads((args.input / "episodes.json").read_text())
    protocol = json.loads((args.input / "manifest.json").read_text())["protocol"]
    traces = json.loads(gzip.decompress((args.input / "traces.json.gz").read_bytes()))
    seed = protocol["seeds"][0]
    arms, conditions = protocol["arms"], protocol["scenarios"]
    colors = {"fixed": "#ca6d20", "adaptive": "#087e8b", "oracle": "#70757a"}
    saved = {(row["cell"][0], row["cell"][2]): row["trace"] for row in traces}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    ax = axes[0, 0]
    for index, condition in enumerate(conditions):
        for offset, arm in zip((-0.22, 0.0, 0.22), arms, strict=True):
            rows = sorted(
                (r for r in records if r["scenario"] == condition and r["arm"] == arm),
                key=lambda r: r["seed"],
            )
            errors = [r["joint_error_max_rad"] * 1000 for r in rows]
            ax.scatter(
                index + offset + np.linspace(-0.06, 0.06, len(rows)),
                errors,
                color=colors[arm],
                s=20,
                alpha=0.7,
                label=arm if index == 0 else None,
            )
    ax.axhline(10.0, color=".25", linestyle="--", linewidth=1, label="Joint-error threshold")
    ax.set_yscale("log")
    ax.set_xticks(range(4), [c.replace("_", " + ") for c in conditions])
    ax.set(ylabel="Final maximum joint error, mrad", title="All paired tasks, physical state")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    for arm in arms:
        trace = saved["noise_delay", arm]
        error = np.max(
            np.abs(np.array([t["q"] for t in trace]) - np.array([t["reference"] for t in trace])),
            axis=1,
        )
        ax.plot([t["time_s"] for t in trace], error * 1000, color=colors[arm], label=arm)
    ax.set(
        xlabel="Time, s",
        ylabel="Maximum reference error, mrad",
        title=f"Noise + delay, seed {seed}",
    )
    ax.axvline(2.0, color=".5", linestyle=":")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    for condition in conditions:
        trace = saved[condition, "adaptive"]
        ax.plot(
            [t["time_s"] for t in trace],
            [t["estimated_parameters"][0] for t in trace],
            label=condition.replace("_", " + "),
        )
    true_mass = next(r["true_parameters"]["payload_mass"] for r in records if r["seed"] == seed)
    ax.axhline(true_mass, color=".2", linestyle="--", label="True mass, diagnostic only")
    ax.set(xlabel="Time, s", ylabel="Estimated payload, kg", title=f"Unchanged RLS, seed {seed}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    trace = [t for t in saved["noise_delay", "adaptive"] if t["time_s"] <= 0.3]
    times = [t["time_s"] for t in trace]
    ax.step(
        times,
        [t["torque_nm"][1] for t in trace],
        where="post",
        color=colors["adaptive"],
        label="Sent joint-2 command",
    )
    ax.step(
        times,
        [t["delivered_torque_nm"][1] for t in trace],
        where="post",
        color="#97518c",
        label="Delivered after 20 ms",
    )
    ax.set(xlabel="Time, s", ylabel="Torque, N m", title="Zero-initialized delay, first 300 ms")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.suptitle("Panda sensor-noise / command-delay engineering pilot", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.output)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
