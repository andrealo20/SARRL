"""Render the first retained payload/friction pair without new simulator steps."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np

from tools.figure_style import apply_style, check_bounds, font_path

ARMS = ("fixed", "adaptive")
COLORS = {"fixed": "#ca6d20", "adaptive": "#087e8b"}


def load_pair(directory: Path):
    """Verify completion hashes before reading the first, unselected paired case."""
    complete = json.loads((directory / "complete.json").read_text())
    for name, expected in complete["hashes"].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("evidence path escapes its directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"retained file hash mismatch: {name}")
    required = {"manifest.json", "episodes.json", "traces.json.gz"}
    if not required.issubset(complete["hashes"]):
        raise ValueError("completion marker must cover manifest, episodes and traces")
    manifest = json.loads((directory / "manifest.json").read_text())
    seed = manifest["protocol"]["seeds"][0]
    episodes = json.loads((directory / "episodes.json").read_text())
    saved = json.loads(gzip.decompress((directory / "traces.json.gz").read_bytes()))
    records, traces = {}, {}
    for arm in ARMS:
        rows = [r for r in episodes if (r["scenario"], r["seed"], r["arm"])
                == ("payload_friction", seed, arm)]
        cells = [r for r in saved if r["cell"] == ["payload_friction", seed, arm]]
        if len(rows) != 1 or len(cells) != 1:
            raise ValueError("expected one retained record and trace per controller")
        records[arm], traces[arm] = rows[0], cells[0]["trace"]
    for field in ("start_q", "goal_q", "true_parameters"):
        if records["fixed"][field] != records["adaptive"][field]:
            raise ValueError(f"unpaired {field}")
    times = np.array([r["time_s"] for r in traces["fixed"]])
    if not np.array_equal(times, [r["time_s"] for r in traces["adaptive"]]):
        raise ValueError("trace timestamps differ")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("trace timestamps must be finite and increasing")
    return seed, records, traces, times


def make_gif(video: Path, output: Path):
    import imageio_ffmpeg

    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-n", "-i", str(video),
        "-filter_complex",
        "fps=8,scale=800:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a",
        "-loop", "0", str(output),
    ], check=True)


def draw_label(draw, position, value, font, fill, bounds):
    """Fail before writing a frame if a label crosses its reserved text band."""
    box = draw.textbbox(position, value, font=font, anchor="lt")
    left, top, right, bottom = bounds
    if box[0] < left or box[1] < top or box[2] > right or box[3] > bottom:
        raise ValueError(f"label does not fit its band: {value}")
    draw.text(position, value, font=font, fill=fill, anchor="lt")


def render(directory: Path, output: Path, poster: Path, gif: Path | None = None):
    import imageio_ffmpeg
    import matplotlib
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    from sarrl.robots.panda import load_panda_model

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()
    seed, records, traces, times = load_pair(directory)
    errors = {arm: 1000 * np.max(np.abs(
        np.array([r["q"] for r in traces[arm]])
        - np.array([r["reference"] for r in traces[arm]])), axis=1) for arm in ARMS}
    fig, ax = plt.subplots(figsize=(9.6, 2.6), dpi=100)
    for arm in ARMS:
        ax.plot(times, errors[arm], color=COLORS[arm], label=arm.capitalize(), linewidth=2)
    ax.set(xlabel="Physical time (s)", ylabel="Maximum error (mrad)", xlim=(0, times[-1]))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1.18), ncol=2, fontsize=12)
    fig.subplots_adjust(left=0.105, right=0.975, top=0.83, bottom=0.27)
    check_bounds(fig, "replay chart")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    chart = Image.open(buffer).convert("RGB")
    plt.close(fig)
    heading = ImageFont.truetype(font_path(), 30)
    text = ImageFont.truetype(font_path(), 22)
    model = load_panda_model()
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.2, 0.0, 0.5]
    camera.distance, camera.azimuth, camera.elevation = 1.6, 140, -20
    option = mujoco.MjvOption()
    option.geomgroup[3] = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    poster.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output), (960, 880), fps=15, codec="libx264", pix_fmt_out="yuv420p",
        output_params=["-crf", "20", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        with mujoco.Renderer(model, height=360, width=480) as renderer:
            for frame, index in enumerate(np.rint(np.linspace(0, len(times) - 1, 360)).astype(int)):
                canvas = Image.new("RGB", (960, 880), "#fafaf8")
                draw = ImageDraw.Draw(canvas)
                draw_label(draw, (24, 12), "Panda: fixed model and online identification",
                           heading, "#18232f", (0, 0, 960, 48))
                subtitle = f"Seed {seed}. Three physical seconds replayed over 24 s."
                draw_label(draw, (24, 54), subtitle,
                           text, "#374151", (0, 48, 960, 82))
                for column, arm in enumerate(ARMS):
                    state = traces[arm][index]
                    data.qpos[:], data.qvel[:] = state["q"], state["qd"]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    canvas.paste(Image.fromarray(renderer.render()), (480 * column, 120))
                    draw_label(draw, (24 + 480 * column, 90), arm.capitalize(),
                               text, COLORS[arm], (480 * column, 82, 480 * (column + 1), 120))
                    draw_label(draw, (24 + 480 * column, 494),
                               f"Error at {times[index]:.2f} s: {errors[arm][index]:.3f} mrad",
                               text, COLORS[arm], (480 * column, 480, 480 * (column + 1), 530))
                canvas.paste(chart, (0, 540))
                x = round(ax.bbox.x0 + ax.bbox.width * times[index] / times[-1])
                cursor_top = round(540 + chart.height - ax.bbox.y1)
                cursor_bottom = round(540 + chart.height - ax.bbox.y0)
                draw.line((x, cursor_top, x, cursor_bottom), fill="#374151", width=2)
                footer = "Saved-data replay. Noise-free, no delay. Single episode."
                draw_label(draw, (24, 827), footer,
                           text, "#374151", (0, 810, 960, 880))
                writer.send(np.asarray(canvas))
                if frame == 359:
                    canvas.save(poster)
                if frame % 60 == 0:
                    print(f"Rendered {frame}/360 frames", flush=True)
    finally:
        writer.close()
    if gif is not None:
        make_gif(output, gif)
    print(json.dumps({"seed": seed, "source": str(directory), "video": str(output),
                      "poster": str(poster), "physical_simulation_steps": 0,
                      "final_errors_mrad": {a: float(errors[a][-1]) for a in ARMS}}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/panda_adaptation_pilot"))
    parser.add_argument("--output", type=Path, default=Path("assets/replays/panda_control.mp4"))
    parser.add_argument("--poster", type=Path, default=Path("assets/replays/panda_control.png"))
    parser.add_argument("--gif", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.poster.exists() or (args.gif and args.gif.exists()):
        parser.error("output already exists; choose new video and poster paths")
    if args.output.suffix.lower() != ".mp4" or args.poster.suffix.lower() != ".png":
        parser.error("use an .mp4 video and a .png poster")
    if args.gif and args.gif.suffix.lower() != ".gif":
        parser.error("use a .gif animation path")
    render(args.input, args.output, args.poster, args.gif)


if __name__ == "__main__":
    main()
