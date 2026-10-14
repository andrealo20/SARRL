"""Plot the saved Panda adaptation pilot without new simulator steps."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.figure_style import apply_style, save_figure


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/panda_adaptation_pilot"))
    parser.add_argument("--output", type=Path, default=Path("assets/panda_adaptation_pilot.png"))
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("figure output already exists; choose a new path")
    complete = json.loads((args.input / "complete.json").read_text())
    for name, expected in complete["hashes"].items():
        if hashlib.sha256((args.input / name).read_bytes()).hexdigest() != expected:
            parser.error(f"retained file hash mismatch: {name}")
    manifest = json.loads((args.input / "manifest.json").read_text())
    records = json.loads((args.input / "episodes.json").read_text())
    traces = json.loads(gzip.decompress((args.input / "traces.json.gz").read_bytes()))
    scenarios, arms = manifest["protocol"]["scenarios"], manifest["protocol"]["arms"]
    seed = manifest["protocol"]["seeds"][0]
    colors = {"fixed": "#ca6d20", "adaptive": "#087e8b", "oracle": "#70757a"}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    ax = axes[0, 0]
    for index, scenario in enumerate(scenarios):
        for offset, arm in zip((-0.22, 0.0, 0.22), arms, strict=True):
            rows = sorted(
                (r for r in records if r["scenario"] == scenario and r["arm"] == arm),
                key=lambda r: r["seed"],
            )
            values = np.array([r["joint_error_max_rad"] * 1000 for r in rows])
            x = index + offset + np.linspace(-0.06, 0.06, len(rows))
            ax.scatter(
                x,
                np.maximum(values, 1e-10),
                s=18,
                alpha=0.65,
                color=colors[arm],
                label=arm if index == 0 else None,
            )
    ax.axhline(10.0, color="0.25", linestyle="--", linewidth=1, label="Joint-error threshold")
    ax.set_yscale("log")
    ax.set_xticks(range(len(scenarios)), [s.replace("_", " + ") for s in scenarios])
    ax.set(ylabel="Final maximum joint error, mrad", title="Every paired pilot episode")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, ncol=2)

    combined = {
        row["cell"][2]: row["trace"]
        for row in traces
        if row["cell"][:2] == ["payload_friction", seed]
    }
    ax = axes[0, 1]
    for arm in arms:
        trace = combined[arm]
        times = [row["time_s"] for row in trace]
        errors = np.max(
            np.abs(
                np.array([row["q"] for row in trace])
                - np.array([row["reference"] for row in trace])
            ),
            axis=1,
        )
        ax.plot(times, errors * 1000, color=colors[arm], label=arm)
    ax.axvline(2.0, color="0.5", linestyle=":")
    ax.set(
        xlabel="Time, s",
        ylabel="Maximum reference error, mrad",
        title=f"First combined case, seed {seed}",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)

    first = next(
        r
        for r in records
        if r["scenario"] == "payload_friction" and r["seed"] == seed and r["arm"] == "adaptive"
    )
    trace = combined["adaptive"]
    times = [row["time_s"] for row in trace]
    estimates = np.array([row["estimated_parameters"] for row in trace])
    ax = axes[1, 0]
    ax.plot(times, estimates[:, 0], color=colors["adaptive"], label="Online estimate")
    ax.axhline(
        first["true_parameters"]["payload_mass"],
        color="0.3",
        linestyle="--",
        label="True mass (diagnostic only)",
    )
    ax.set(
        xlabel="Time, s",
        ylabel="Added payload mass, kg",
        title=f"Payload identification, seed {seed}",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    adapted = [r for r in records if r["scenario"] == "payload_friction" and r["arm"] == "adaptive"]
    true = np.array([r["true_parameters"]["viscous_extra"] for r in adapted])
    estimated = np.array([r["estimator"]["estimated_parameters"][1:] for r in adapted])
    for joint in range(7):
        ax.scatter(true[:, joint], estimated[:, joint], s=24, alpha=0.7, label=f"Joint {joint + 1}")
    upper = max(3.0, float(estimated.max()) + 0.1)
    ax.plot([0.0, upper], [0.0, upper], color="0.4", linestyle="--", linewidth=1)
    ax.set(
        xlabel="True extra damping, N m s/rad",
        ylabel="Estimated extra damping, N m s/rad",
        title="Final friction estimates, every combined case",
        xlim=(0, upper),
        ylim=(0, upper),
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Panda payload/friction engineering pilot", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.output)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
