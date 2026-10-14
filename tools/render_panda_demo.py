"""Render the first retained payload/friction pair without new simulator steps."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

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


def render(directory: Path, output: Path, poster: Path):
    import imageio_ffmpeg
    import matplotlib
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    from sarrl.robots.panda import load_panda_model

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed, records, traces, times = load_pair(directory)
    errors = {arm: 1000 * np.max(np.abs(
        np.array([r["q"] for r in traces[arm]])
        - np.array([r["reference"] for r in traces[arm]])), axis=1) for arm in ARMS}
    fig, ax = plt.subplots(figsize=(9.6, 2.2), dpi=100)
    for arm in ARMS:
        ax.plot(times, errors[arm], color=COLORS[arm], label=arm.capitalize(), linewidth=2)
    ax.set(xlabel="Physical time (s)", ylabel="Maximum joint error (mrad)", xlim=(0, times[-1]))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.subplots_adjust(left=0.105, right=0.975, top=0.95, bottom=0.25)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    chart = Image.open(buffer).convert("RGB")
    plt.close(fig)
    font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    heading = ImageFont.truetype(str(font_path), 24)
    text = ImageFont.truetype(str(font_path), 16)
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
        str(output), (960, 720), fps=15, codec="libx264", pix_fmt_out="yuv420p",
        output_params=["-crf", "20", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        with mujoco.Renderer(model, height=360, width=480) as renderer:
            for frame, index in enumerate(np.rint(np.linspace(0, len(times) - 1, 360)).astype(int)):
                canvas = Image.new("RGB", (960, 720), "#f4f6f8")
                draw = ImageDraw.Draw(canvas)
                draw.text((24, 10), "Panda: fixed model vs online identification", font=heading,
                          fill="#18232f")
                draw.text((24, 43), f"Retained seed {seed} | 3 s trajectory replayed over 24 s",
                          font=text, fill="#374151")
                for column, arm in enumerate(ARMS):
                    state = traces[arm][index]
                    data.qpos[:], data.qvel[:] = state["q"], state["qd"]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    canvas.paste(Image.fromarray(renderer.render()), (480 * column, 95))
                    draw.text((24 + 480 * column, 72), arm.capitalize(), font=text,
                              fill=COLORS[arm])
                    draw.text((24 + 480 * column, 462),
                              f"Error at {times[index]:.2f} s: {errors[arm][index]:.3f} mrad",
                              font=text, fill=COLORS[arm])
                canvas.paste(chart, (0, 490))
                x = round(100.8 + 835.2 * times[index] / times[-1])
                draw.line((x, 501, x, 655), fill="#374151", width=2)
                draw.text((24, 696), "Saved-data replay | Noise-free, no delay | Single episode",
                          font=text, fill="#374151")
                writer.send(np.asarray(canvas))
                if frame == 359:
                    canvas.save(poster)
                if frame % 60 == 0:
                    print(f"Rendered {frame}/360 frames", flush=True)
    finally:
        writer.close()
    print(json.dumps({"seed": seed, "source": str(directory), "video": str(output),
                      "poster": str(poster), "physical_simulation_steps": 0,
                      "final_errors_mrad": {a: float(errors[a][-1]) for a in ARMS}}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/panda_adaptation_pilot"))
    parser.add_argument("--output", type=Path, default=Path("assets/panda_demo.mp4"))
    parser.add_argument("--poster", type=Path, default=Path("assets/panda_demo.png"))
    args = parser.parse_args(argv)
    if args.output.exists() or args.poster.exists():
        parser.error("output already exists; choose new video and poster paths")
    if args.output.suffix.lower() != ".mp4" or args.poster.suffix.lower() != ".png":
        parser.error("use an .mp4 video and a .png poster")
    render(args.input, args.output, args.poster)


if __name__ == "__main__":
    main()
