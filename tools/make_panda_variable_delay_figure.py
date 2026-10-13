"""Plot the saved variable-latency pilot without new physical episodes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/panda_variable_delay_pilot"))
    parser.add_argument(
        "--output", type=Path, default=Path("assets/panda_variable_delay_pilot.png")
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("figure output already exists; choose a new path")
    complete = json.loads((args.input / "complete.json").read_text())
    for name, digest in complete["hashes"].items():
        if hashlib.sha256((args.input / name).read_bytes()).hexdigest() != digest:
            parser.error(f"retained evidence hash mismatch: {name}")
    s = json.loads((args.input / "summary.json").read_text())
    traces = json.loads(gzip.decompress((args.input / "traces.json.gz").read_bytes()))
    lookup = {(t["cell"][0], t["cell"][2]): t["trace"] for t in traces}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    profiles = ("fixed10", "fixed40", "up", "down")
    colors = {"adaptive": "#4c78a8", "bank_align": "#54a24b", "known_align": "#b279a2"}
    for ax, field, title in (
        (axes[0, 0], "mean_final_joint_error_rad", "Mean final maximum joint error, mrad"),
        (axes[0, 1], "mean_after_tracking_rms_rad", "Mean joint RMS after 1 s, mrad"),
    ):
        for startup, style in (("cold", "--"), ("warm", "-")):
            for arm, color in colors.items():
                values = [1000 * s["cells"][f"{startup}_noise_{p}/{arm}"][field] for p in profiles]
                ax.plot(
                    range(4),
                    values,
                    linestyle=style,
                    color=color,
                    marker="o",
                    label=f"{startup} / {arm.replace('_', ' ')}",
                )
        ax.set(
            xticks=range(4),
            xticklabels=("10 ms", "40 ms", "10 to 40", "40 to 10"),
            ylabel=title,
            title=title,
            yscale="log",
        )
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, ncol=2)
    for ax, profile in zip(axes[1], ("up", "down"), strict=True):
        trace = lookup[f"warm_noise_{profile}", "bank_align"]
        ax.step(
            [t["time_s"] for t in trace[:-1]],
            [t["applied_age_steps"] * 10 for t in trace[:-1]],
            where="post",
            color=".25",
            linestyle=":",
            label="Applied age, diagnostic only",
        )
        ax.step(
            [t["time_s"] for t in trace[:-1]],
            [t["selected_delay_steps"] * 10 for t in trace[:-1]],
            where="post",
            color=colors["bank_align"],
            label="Frozen bank selection",
        )
        ax.axvline(1.0, color=".6", linewidth=1)
        ax.set(
            xlabel="Scored time, s",
            ylabel="Delay, ms",
            title=f"First-seed warm noisy {profile} transition",
            yticks=range(0, 41, 10),
            ylim=(-2, 42),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Panda populated queue and changing packet latency", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
