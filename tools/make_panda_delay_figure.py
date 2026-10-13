"""Plot every delay-pilot outcome and the preselected first combined-case traces."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/panda_delay_pilot"))
    parser.add_argument("--output", type=Path, default=Path("assets/panda_delay_pilot.png"))
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("figure output already exists; choose a new path")
    complete = json.loads((args.input / "complete.json").read_text())
    for name, expected in complete["hashes"].items():
        if hashlib.sha256((args.input / name).read_bytes()).hexdigest() != expected:
            parser.error(f"retained evidence hash mismatch: {name}")
    records = json.loads((args.input / "episodes.json").read_text())
    protocol = json.loads((args.input / "manifest.json").read_text())["protocol"]
    traces = json.loads(gzip.decompress((args.input / "traces.json.gz").read_bytes()))
    saved = {(r["cell"][0], r["cell"][2]): r["trace"] for r in traces}
    seed = protocol["seeds"][0]
    arms = protocol["arms"]
    colors = dict(
        zip(
            arms,
            ["#ca6d20", "#4c78a8", "#b279a2", "#54a24b", "#e45756", "#087e8b", "#70757a"],
            strict=True,
        )
    )
    labels = {a: a.replace("_", " ") for a in arms}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    for index, condition in enumerate(protocol["scenarios"]):
        for offset, arm in zip(np.linspace(-0.32, 0.32, len(arms)), arms, strict=True):
            rows = sorted(
                (r for r in records if r["scenario"] == condition and r["arm"] == arm),
                key=lambda r: r["seed"],
            )
            ax.scatter(
                index + offset + np.linspace(-0.025, 0.025, len(rows)),
                [r["joint_error_max_rad"] * 1000 for r in rows],
                s=13,
                alpha=0.65,
                color=colors[arm],
                label=labels[arm] if index == 0 else None,
            )
    ax.axhline(10.0, color=".25", linestyle="--", linewidth=1, label="Joint-error threshold")
    ax.set_yscale("log")
    ax.set_xticks(range(4), [s.replace("_", " + ") for s in protocol["scenarios"]])
    ax.set(ylabel="Final maximum joint error, mrad", title="All paired pilot episodes")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    displayed = ("adaptive", "known_align", "bank_align", "known_predict", "bank_predict", "oracle")
    for arm in displayed:
        trace = saved["noise_delay", arm]
        error = np.max(
            np.abs(np.array([t["q"] for t in trace]) - np.array([t["reference"] for t in trace])),
            axis=1,
        )
        ax.plot(
            [t["time_s"] for t in trace],
            error * 1000,
            color=colors[arm],
            linestyle="--" if arm.startswith("known") else "-",
            label=labels[arm],
        )
    ax.axvline(2.0, color=".5", linestyle=":")
    ax.set(
        xlabel="Time, s",
        ylabel="Maximum reference error, mrad",
        title=f"Noise + delay, seed {seed}",
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    for arm in displayed[:-1]:
        trace = saved["noise_delay", arm]
        ax.plot(
            [t["time_s"] for t in trace],
            [t["estimated_parameters"][0] for t in trace],
            color=colors[arm],
            linestyle="--" if arm.startswith("known") else "-",
            label=labels[arm],
        )
    truth = next(r["true_parameters"]["payload_mass"] for r in records if r["seed"] == seed)
    ax.axhline(truth, color=".2", linestyle=":", label="True mass, diagnostic only")
    ax.set(
        xlabel="Time, s",
        ylabel="Selected payload estimate, kg",
        title="Command alignment and identification",
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    for arm in ("bank_align", "bank_predict"):
        trace = saved["noise_delay", arm]
        ax.step(
            [t["time_s"] for t in trace],
            [t["selected_delay_steps"] * 10 for t in trace],
            where="post",
            color=colors[arm],
            label=labels[arm],
        )
    ax.axhline(20.0, color=".3", linestyle=":", label="True delay, diagnostic only")
    ax.set(
        xlabel="Time, s",
        ylabel="Selected delay, ms",
        yticks=range(0, 41, 10),
        ylim=(-2, 42),
        title="Causal delay selection, whole episode",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.suptitle("Panda command alignment / delay identification / prediction", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
