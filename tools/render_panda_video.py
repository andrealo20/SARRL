"""Render retained Panda states as native 3840-by-2160 H.264 videos."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from tools.figure_style import apply_style, check_bounds, font_path
from tools.render_panda_demo import ARMS, COLORS, draw_label, load_pair, scene_markers, tip_overlay

SIZE = (3840, 2160)


def load_nominal(directory: Path):
    complete = json.loads((directory / "complete.json").read_text())
    for name, expected in complete["hashes"].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("evidence path escapes its directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"retained evidence hash mismatch: {name}")
    required = {"manifest.json", "episodes.json", "first_episode_trace.json"}
    if not required.issubset(complete["hashes"]):
        raise ValueError("completion marker must cover the record and trace")
    return (json.loads((directory / "episodes.json").read_text())[0],
            json.loads((directory / "first_episode_trace.json").read_text()))


def retained_model(directory: Path, width: int, height: int):
    from sarrl.robots.panda import load_panda_model, model_provenance

    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["model"]["files"] != model_provenance()["files"]:
        raise ValueError("packaged model differs from the retained trajectory model")
    model = load_panda_model()
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    return model


def video_writer(output: Path, frames: int, seconds: int):
    import imageio_ffmpeg

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output), SIZE, fps=frames / seconds, codec="libx264", pix_fmt_out="yuv420p",
        output_params=["-profile:v", "high", "-level:v", "5.1", "-crf", "18",
                       "-movflags", "+faststart"],
    )
    writer.send(None)
    return writer


def comparison_chart(times, errors, offsets):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()
    fig = plt.figure(figsize=(12, 17.4), dpi=100)
    ax = fig.add_axes((0.23, 0.68, 0.70, 0.24))
    zoom = fig.add_axes((0.23, 0.18, 0.70, 0.24))
    for arm in ARMS:
        ax.plot(times, errors[arm], color=COLORS[arm], label=arm.capitalize(), linewidth=6)
    ax.set(xlabel="Physical time (s)", ylabel="Maximum joint\nerror (deg)",
           title="Joint-reference error", xlim=(0, times[-1]))
    final_points = np.concatenate([points[times >= 2][:, [0, 2]]
                                   for points in offsets.values()] + [np.zeros((1, 2))])
    centre = (final_points.min(axis=0) + final_points.max(axis=0)) / 2
    span = max(60.0, float(np.ptp(final_points, axis=0).max()) + 12)
    zoom.set(xlim=(centre[0] - span / 2, centre[0] + span / 2),
             ylim=(centre[1] - span / 2, centre[1] + span / 2),
             xlabel="X offset from target (mm)", ylabel="Z offset from\ntarget (mm)",
             title="Tip close-up, final approach", aspect="equal")
    zoom.scatter([0], [0], marker="x", color="#18232f", s=800, linewidths=6,
                 label="Target", zorder=4)
    trails, dots = {}, {}
    for arm in ARMS:
        trails[arm], = zoom.plot([], [], color=COLORS[arm], linewidth=6)
        dots[arm], = zoom.plot([], [], "o", color=COLORS[arm], markersize=18)
    for panel in (ax, zoom):
        panel.set_title(panel.get_title(), fontsize=54, pad=24)
        panel.xaxis.label.set_size(50)
        panel.yaxis.label.set_size(50)
        panel.tick_params(labelsize=42, length=10, width=2, pad=10)
        for spine in panel.spines.values():
            spine.set_linewidth(2)
        panel.grid(alpha=0.2, linewidth=2)
        panel.legend(loc="upper center", bbox_to_anchor=(0.5, -0.40),
                     ncol=2, fontsize=40)
    cursor = ax.axvline(0, color="#374151", linewidth=3)
    check_bounds(fig, "Panda comparison chart")
    return fig, cursor, trails, dots


def render_comparison(directory: Path, output: Path, poster: Path | None = None, frames=360):
    import matplotlib
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    from tools.render_panda_demo import tip_paths

    seed, records, traces, times = load_pair(directory)
    model = retained_model(directory, 1272, 1510)
    paths = tip_paths(model, traces)
    target = np.asarray(records["fixed"]["target_xyz_m"])
    tip_errors = {arm: 1000 * np.linalg.norm(paths[arm] - paths["reference"], axis=1)
                  for arm in ARMS}
    errors = {arm: np.rad2deg(np.max(np.abs(
        np.array([row["q"] for row in traces[arm]])
        - np.array([row["reference"] for row in traces[arm]])), axis=1)) for arm in ARMS}
    offsets = {arm: 1000 * (paths[arm] - target) for arm in ARMS}
    fig, cursor, trails, dots = comparison_chart(times, errors, offsets)
    heading = ImageFont.truetype(font_path(), 100)
    text = ImageFont.truetype(font_path(), 58)
    arm_font = ImageFont.truetype(font_path(), 72)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    wide_lookat = np.array([0.2, 0, 0.5])
    camera.azimuth, camera.elevation = 140, -20
    options = mujoco.MjvOption()
    options.geomgroup[3] = 1
    writer = video_writer(output, frames, 24)
    try:
        with mujoco.Renderer(model, height=1510, width=1272) as renderer:
            indices = np.rint(np.linspace(0, len(times) - 1, frames)).astype(int)
            for frame, index in enumerate(indices):
                fraction = np.clip((times[index] - 2) / 0.35, 0, 1)
                fraction = fraction ** 2 * (3 - 2 * fraction)
                camera.lookat[:] = (1 - fraction) * wide_lookat + fraction * target
                camera.distance = 1.6 - fraction
                canvas = Image.new("RGB", SIZE, "#fafaf8")
                draw = ImageDraw.Draw(canvas)
                draw_label(draw, (40, 30), "Panda: fixed model and online identification",
                           heading, "#18232f", (0, 0, 3840, 150))
                draw_label(draw, (40, 160),
                           f"Seed {seed}. Three physical seconds replayed over 24 s.",
                           text, "#374151", (0, 150, 3840, 240))
                for column, arm in enumerate(ARMS):
                    left = 1296 * column
                    state = traces[arm][index]
                    data.qpos[:], data.qvel[:] = state["q"], state["qd"]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=options)
                    scene_markers(renderer.scene, paths[arm][:index + 1], target,
                                  matplotlib.colors.to_rgba(COLORS[arm]))
                    view = Image.fromarray(renderer.render())
                    tip_overlay(view, renderer.scene, paths[arm][index], target,
                                COLORS[arm], scale=3)
                    canvas.paste(view, (left, 350))
                    draw_label(draw, (left + 40, 264), arm.capitalize(), arm_font,
                               COLORS[arm], (left, 240, left + 1272, 350))
                    draw_label(draw, (left + 40, 1904),
                               f"Tip tracking error: {tip_errors[arm][index]:.3f} mm",
                               text, COLORS[arm], (left, 1860, left + 1272, 1980))
                    draw_label(draw, (left + 40, 1990),
                               f"Max joint error: {errors[arm][index]:.4f} deg",
                               text, COLORS[arm], (left, 1980, left + 1272, 2070))
                    points = offsets[arm][:index + 1][times[:index + 1] >= 2]
                    trails[arm].set_data(points[:, 0], points[:, 2])
                    dots[arm].set_data([offsets[arm][index, 0]], [offsets[arm][index, 2]])
                cursor.set_xdata([times[index], times[index]])
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png")
                buffer.seek(0)
                canvas.paste(Image.open(buffer).convert("RGB"), (2640, 280))
                draw_label(draw, (40, 2080),
                           f"t = {times[index]:.2f} s. Gold cross: target. Colored trails: tip. "
                           "Saved states. No noise or delay.",
                           text, "#374151", (0, 2070, 3840, 2160))
                writer.send(np.asarray(canvas))
                if poster is not None and frame == frames - 1:
                    poster.parent.mkdir(parents=True, exist_ok=True)
                    canvas.save(poster)
                if frame % 60 == 0:
                    print(f"Rendered {frame}/{frames} comparison frames", flush=True)
    finally:
        writer.close()
        import matplotlib.pyplot as plt

        plt.close(fig)
    print(json.dumps({"seed": seed, "source": str(directory), "video": str(output),
                      "video_size": SIZE, "frames": frames,
                      "physical_simulation_steps": 0,
                      "final_tip_errors_mm": {arm: float(tip_errors[arm][-1]) for arm in ARMS}},
                     indent=2))


def render_motion(directory: Path, output: Path, poster: Path | None = None, frames=180):
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    record, trace = load_nominal(directory)
    model = retained_model(directory, 3840, 1740)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.2, 0, 0.45]
    camera.distance, camera.azimuth, camera.elevation = 1.7, 135, -22
    options = mujoco.MjvOption()
    options.geomgroup[3] = 1
    heading = ImageFont.truetype(font_path(), 120)
    text = ImageFont.truetype(font_path(), 88)
    footer = ImageFont.truetype(font_path(), 72)
    points = np.array([row["tip"] for row in trace])
    writer = video_writer(output, frames, 12)
    try:
        with mujoco.Renderer(model, height=1740, width=3840) as renderer:
            indices = np.rint(np.linspace(0, len(trace) - 1, frames)).astype(int)
            for frame, index in enumerate(indices):
                row = trace[index]
                data.qpos[:], data.qvel[:] = row["q"], row["qd"]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=options)
                scene = renderer.scene
                for left, right in zip(points[:index + 1:5][:-1], points[:index + 1:5][1:],
                                       strict=True):
                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                       np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                       np.array([0.1, 0.8, 0.7, 1]))
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004, left, right)
                    scene.ngeom += 1
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                   np.full(3, 0.012), np.array(record["target_xyz_m"]),
                                   np.eye(3).ravel(), np.array([1, 0.6, 0.2, 1]))
                scene.ngeom += 1
                canvas = Image.new("RGB", SIZE, "#fafaf8")
                draw = ImageDraw.Draw(canvas)
                draw_label(draw, (48, 28), "Panda: a seven-joint trajectory", heading,
                           "#18232f", (0, 0, 3840, 180))
                draw_label(draw, (48, 202),
                           f"Seed {record['seed']}. Three physical seconds replayed over 12 s.",
                           text, "#374151", (0, 180, 3840, 340))
                canvas.paste(Image.fromarray(renderer.render()), (0, 340))
                draw_label(draw, (48, 2092),
                           f"t = {row['time_s']:.2f} s. Teal: recorded tip path. Orange: target.",
                           footer, "#374151", (0, 2080, 3840, 2160))
                writer.send(np.asarray(canvas))
                if poster is not None and frame == frames - 1:
                    poster.parent.mkdir(parents=True, exist_ok=True)
                    canvas.save(poster)
                if frame % 60 == 0:
                    print(f"Rendered {frame}/{frames} motion frames", flush=True)
    finally:
        writer.close()
    print(json.dumps({"seed": record["seed"], "source": str(directory), "video": str(output),
                      "video_size": SIZE, "frames": frames,
                      "physical_simulation_steps": 0}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("comparison", "motion"), default="comparison")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--poster", type=Path)
    args = parser.parse_args(argv)
    name = "panda_control" if args.mode == "comparison" else "panda_motion"
    source = "panda_adaptation_pilot" if args.mode == "comparison" else "panda_nominal_pilot"
    output = args.output or Path(f"assets/replays/{name}.mp4")
    directory = args.input or Path(f"results/{source}")
    if output.exists() or (args.poster is not None and args.poster.exists()):
        parser.error("output already exists; choose new paths")
    if output.suffix.lower() != ".mp4":
        parser.error("use an .mp4 video path")
    if args.poster is not None and args.poster.suffix.lower() != ".png":
        parser.error("use a .png poster path")
    renderer = render_comparison if args.mode == "comparison" else render_motion
    renderer(directory, output, args.poster)


if __name__ == "__main__":
    main()
