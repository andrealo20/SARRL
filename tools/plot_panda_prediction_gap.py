"""Plot the retained zero-delay position regressions without physical simulation."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from tools.campaign_guards import valid_completion_marker

ROOT = Path(__file__).resolve().parents[1]


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from sarrl.robots.panda import load_panda_model

    directory = ROOT / "results/panda_prediction_gap_position_v26"
    if not valid_completion_marker(
        directory / "complete.json", ("manifest.json", "diagnostics.json", "curves.json.gz")
    ):
        raise ValueError("completed position-gap hashes required")
    data = json.loads((directory / "diagnostics.json").read_text())
    curves = json.loads(gzip.decompress((directory / "curves.json.gz").read_bytes()))
    reports = {tuple(r["cell"]): r for r in data["cases"]}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "panda_prediction_gap",
        }
    )
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, sharey=True)
    bound = load_panda_model().jnt_range[3, 1]
    for axis, item in zip(axes, curves, strict=True):
        samples = item["samples"]
        report = reports[tuple(item["cell"])]
        times = np.array([sample["time_s"] for sample in samples])
        axis.plot(
            times,
            [sample["actual_position"][3] for sample in samples],
            color="#bd4b2f",
            lw=2,
            label="Recorded position",
        )
        axis.plot(
            times,
            [sample["witness_position"][3] for sample in samples],
            color="#286581",
            lw=1.8,
            label="Accepted-model prediction",
        )
        axis.axhline(bound, color="#333333", ls="--", lw=1, label="Upper joint limit")
        axis.axvline(
            report["first_violation"]["time_s"],
            color="#bd4b2f",
            ls=":",
            lw=1,
            label="First position violation",
        )
        axis.set_title(
            "Clean measurements" if "clean" in item["cell"][0] else "Noisy measurements",
            loc="left",
            fontsize=11,
        )
        axis.set_ylabel("Joint 4 position (rad)")
        axis.set_xlim(0, 3)
        axis.set_ylim(-1.45, 0.05)
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    for extension in ("svg", "png"):
        path = ROOT / f"assets/figures/panda_prediction_gap.{extension}"
        figure.savefig(
            path,
            dpi=220,
            bbox_inches="tight",
            metadata={"Date": None} if extension == "svg" else None,
        )
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(figure)


if __name__ == "__main__":
    main()
