"""Plot the retained posture/delay pilot without executing physical episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.figure_style import apply_style, save_figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/panda_posture_delay_pilot"))
    parser.add_argument("--output", type=Path, default=Path("assets/panda_posture_delay_pilot.png"))
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("figure output already exists; choose a new path")
    complete = json.loads((args.input / "complete.json").read_text())
    for name, digest in complete["hashes"].items():
        if hashlib.sha256((args.input / name).read_bytes()).hexdigest() != digest:
            parser.error(f"retained evidence hash mismatch: {name}")
    rows = json.loads((args.input / "episodes.json").read_text())
    summary = json.loads((args.input / "summary.json").read_text())

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"adaptive": "#4c78a8", "bank_align": "#54a24b", "bank_predict": "#087e8b"}
    measures = (
        ("mean_final_joint_error_rad", 1000.0, "Mean final maximum joint error, mrad"),
        ("mean_tracking_rms_rad", 1000.0, "Mean joint tracking RMS, mrad"),
        ("mean_payload_absolute_error_kg", 1000.0, "Mean payload absolute error, g"),
    )
    for ax, (field, scale, title) in zip(axes.flat, measures, strict=False):
        for posture, style in (("narrow", "--"), ("wide", "-")):
            for arm, color in colors.items():
                values = [
                    summary["cells"][f"{posture}_noise_d{d:02d}/{arm}"][field] * scale
                    for d in range(0, 41, 10)
                ]
                ax.plot(
                    range(0, 41, 10),
                    values,
                    linestyle=style,
                    marker="o",
                    color=color,
                    label=f"{posture} / {arm.replace('_', ' ')}",
                )
        ax.set(
            xlabel="Constant command delay, ms",
            ylabel=title,
            title=title,
            xticks=range(0, 41, 10),
            yscale="log",
        )
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    ax = axes[1, 1]
    for posture, color in (("narrow", "#4c78a8"), ("wide", "#e45756")):
        selected = [
            r for r in rows if r["scenario"] == f"{posture}_clean_d00" and r["arm"] == "fixed"
        ]
        points = np.array([r["start_q"] for r in selected])
        ax.scatter(points[:, 0], points[:, 1], color=color, label=posture, s=30)
    ax.set(
        xlabel="Start joint 1, rad",
        ylabel="Start joint 2, rad",
        title="Every paired task, two coordinates of seven",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.suptitle("Panda frozen-controller posture/delay grid: noisy conditions", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.output)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
