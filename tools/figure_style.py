"""Shared typography and layout checks for repository figures."""

from __future__ import annotations

import textwrap
from pathlib import Path

FONT = "STIXGeneral"


def apply_style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": FONT, "font.size": 12, "mathtext.fontset": "stix",
        "axes.titlesize": 16, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def font_path():
    from matplotlib.font_manager import FontProperties, findfont

    return findfont(FontProperties(family=FONT), fallback_to_default=False)


def check_bounds(fig, output):
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()
    if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > width or bounds.y1 > height:
        raise RuntimeError(f"figure text extends beyond canvas: {output}")


def save_svg(fig, output: Path):
    fig.savefig(output)
    # Keep the vector path data while removing serializer trailing spaces.
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")


def save_figure(fig, output: Path):
    """Reserve space for every panel's labels and place legends outside data."""
    rows = max(ax.get_subplotspec().get_gridspec().nrows for ax in fig.axes)
    height = 6.2 if rows == 1 else 10.8
    fig.set_size_inches(13.5, height)
    for ax in fig.axes:
        for loc in ("left", "center", "right"):
            title = ax.get_title(loc=loc)
            if title:
                columns = ax.get_subplotspec().get_gridspec().ncols
                wrapped = "\n".join(textwrap.fill(line, 28 if columns >= 3 else 45)
                                    for line in title.splitlines())
                ax.set_title(wrapped,
                             loc=loc, fontsize=16, pad=16)
        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_fontsize(14)
        for annotation in ax.texts:
            annotation.set_fontsize(max(11, annotation.get_fontsize()))
        legend = ax.get_legend()
        if legend:
            handles, labels = ax.get_legend_handles_labels()
            legend.remove()
            columns = 2
            if all(label.startswith("Joint ") or label == "Settling starts" for label in labels):
                columns = 4
            elif len(labels) <= 3 and all(len(label) <= 10 for label in labels):
                columns = len(labels)
            if ax.get_subplotspec().get_gridspec().ncols >= 3:
                columns = 1
            ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.23),
                      ncol=columns, fontsize=12, frameon=False)
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(22)
    fig.set_layout_engine("constrained")
    fig.get_layout_engine().set(w_pad=0.12, h_pad=0.12, wspace=0.05, hspace=0.03)
    check_bounds(fig, output)
    renderer = fig.canvas.get_renderer()
    from matplotlib.transforms import Bbox

    panels = []
    for ax in fig.axes:
        if ax.get_visible():
            boxes = [ax.get_tightbbox(renderer)]
            if ax.get_legend() is not None:
                boxes.append(ax.get_legend().get_window_extent(renderer))
            panels.append(Bbox.union(boxes))
    for i, left in enumerate(panels):
        for right in panels[i + 1:]:
            overlap_w = min(left.x1, right.x1) - max(left.x0, right.x0)
            overlap_h = min(left.y1, right.y1) - max(left.y0, right.y0)
            if overlap_w > 2 and overlap_h > 2:
                raise RuntimeError(f"panel labels overlap: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    save_svg(fig, output.with_suffix(".svg"))
