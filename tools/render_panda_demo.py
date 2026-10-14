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
    if not np.array_equal([r["reference"] for r in traces["fixed"]],
                          [r["reference"] for r in traces["adaptive"]]):
        raise ValueError("paired reference trajectories differ")
    return seed, records, traces, times


def tip_paths(model, traces):
    """Project saved joint states and their shared reference at physical scale."""
    import mujoco

    data = mujoco.MjData(model)
    site = model.site("attachment_site").id
    paths = {}
    for arm in (*ARMS, "reference"):
        rows = traces["fixed" if arm == "reference" else arm]
        points = []
        for row in rows:
            data.qpos[:] = row["reference" if arm == "reference" else "q"]
            mujoco.mj_forward(model, data)
            points.append(data.site_xpos[site].copy())
        paths[arm] = np.asarray(points)
    return paths


def scene_markers(scene, path, target, color):
    """Draw the recorded tip trail and a target cross without changing the plant."""
    import mujoco

    indices = list(range(0, len(path), 5))
    if indices[-1] != len(path) - 1:
        indices.append(len(path) - 1)
    segments = [(path[left], path[right], 0.0025, color)
                for left, right in zip(indices[:-1], indices[1:], strict=True)]
    for axis in np.eye(3):
        segments.append((target - 0.012 * axis, target + 0.012 * axis,
                         0.0015, [0.96, 0.78, 0.37, 1]))
    for left, right, radius, rgba in segments:
        if scene.ngeom >= scene.maxgeom:
            raise ValueError("replay markers exceed scene capacity")
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                           np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, left, right)
        scene.ngeom += 1
    if scene.ngeom >= scene.maxgeom:
        raise ValueError("replay markers exceed scene capacity")
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                       np.full(3, 0.005), path[-1], np.eye(3).ravel(), np.asarray(color))
    scene.ngeom += 1


def project_marker(scene, point, width, height):
    """Project a world point using the renderer's actual perspective camera."""
    cameras = scene.camera
    position = np.mean([camera.pos for camera in cameras], axis=0)
    forward = np.mean([camera.forward for camera in cameras], axis=0)
    up = np.mean([camera.up for camera in cameras], axis=0)
    forward /= np.linalg.norm(forward)
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    relative = point - position
    depth = np.dot(relative, forward)
    if depth <= 0:
        return None
    camera = cameras[0]
    scale = height * camera.frustum_near / (camera.frustum_top - camera.frustum_bottom)
    return (width / 2 + scale * np.dot(relative, right) / depth,
            height / 2 - scale * np.dot(relative, up) / depth)


def tip_overlay(image, scene, tip, target, color):
    """Keep projected tip and target markers visible over occluding wrist meshes."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    tip_pixel = project_marker(scene, tip, *image.size)
    target_pixel = project_marker(scene, target, *image.size)
    if tip_pixel is not None and target_pixel is not None:
        draw.line((*tip_pixel, *target_pixel), fill="#f6c85f", width=2)
    if target_pixel is not None:
        x, y = target_pixel
        draw.line((x - 9, y, x + 9, y), fill="#f6c85f", width=3)
        draw.line((x, y - 9, x, y + 9), fill="#f6c85f", width=3)
    if tip_pixel is not None:
        x, y = tip_pixel
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="white", width=1)


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

    from sarrl.robots.panda import load_panda_model, model_provenance

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()
    seed, records, traces, times = load_pair(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["model"]["files"] != model_provenance()["files"]:
        raise ValueError("packaged model differs from the retained trajectory model")
    model = load_panda_model()
    paths = tip_paths(model, traces)
    target = np.asarray(records["fixed"]["target_xyz_m"])
    tip_errors = {arm: 1000 * np.linalg.norm(paths[arm] - paths["reference"], axis=1)
                  for arm in ARMS}
    errors = {arm: np.rad2deg(np.max(np.abs(
        np.array([r["q"] for r in traces[arm]])
        - np.array([r["reference"] for r in traces[arm]])), axis=1)) for arm in ARMS}
    fig = plt.figure(figsize=(9.6, 3.6), dpi=100)
    ax = fig.add_axes((0.085, 0.29, 0.37, 0.55))
    zoom = fig.add_axes((0.62, 0.29, 0.35, 0.55))
    for arm in ARMS:
        ax.plot(times, errors[arm], color=COLORS[arm], label=arm.capitalize(), linewidth=2)
    ax.set(xlabel="Physical time (s)", ylabel="Maximum joint error (deg)",
           title="Joint-reference error", xlim=(0, times[-1]))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2, fontsize=12)
    offsets = {arm: 1000 * (paths[arm] - target) for arm in ARMS}
    final_points = np.concatenate([points[times >= 2][:, [0, 2]]
                                   for points in offsets.values()] + [np.zeros((1, 2))])
    centre = (final_points.min(axis=0) + final_points.max(axis=0)) / 2
    span = max(60.0, float(np.ptp(final_points, axis=0).max()) + 12)
    zoom.set(xlim=(centre[0] - span / 2, centre[0] + span / 2),
             ylim=(centre[1] - span / 2, centre[1] + span / 2),
             xlabel="X offset from target (mm)", ylabel="Z offset from target (mm)",
             title="Tip close-up, final approach", aspect="equal")
    zoom.scatter([0], [0], marker="x", color="#18232f", s=70, label="Target", zorder=4)
    zoom.grid(alpha=0.2)
    trails, dots = {}, {}
    for arm in ARMS:
        trails[arm], = zoom.plot([], [], color=COLORS[arm], linewidth=2)
        dots[arm], = zoom.plot([], [], "o", color=COLORS[arm], markersize=6)
    zoom.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), fontsize=12)
    cursor = ax.axvline(0, color="#374151", linewidth=1)
    check_bounds(fig, "replay chart")
    heading = ImageFont.truetype(font_path(), 30)
    text = ImageFont.truetype(font_path(), 22)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    wide_lookat = np.array([0.2, 0.0, 0.5])
    camera.lookat[:] = wide_lookat
    camera.distance, camera.azimuth, camera.elevation = 1.6, 140, -20
    option = mujoco.MjvOption()
    option.geomgroup[3] = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    poster.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output), (960, 1008), fps=15, codec="libx264", pix_fmt_out="yuv420p",
        output_params=["-crf", "20", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        with mujoco.Renderer(model, height=360, width=480) as renderer:
            for frame, index in enumerate(np.rint(np.linspace(0, len(times) - 1, 360)).astype(int)):
                zoom_fraction = np.clip((times[index] - 2.0) / 0.35, 0, 1)
                zoom_fraction = zoom_fraction ** 2 * (3 - 2 * zoom_fraction)
                camera.lookat[:] = (1 - zoom_fraction) * wide_lookat + zoom_fraction * target
                camera.distance = 1.6 - zoom_fraction
                canvas = Image.new("RGB", (960, 1008), "#fafaf8")
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
                    color = matplotlib.colors.to_rgba(COLORS[arm])
                    scene_markers(renderer.scene, paths[arm][:index + 1], target, color)
                    robot_view = Image.fromarray(renderer.render())
                    tip_overlay(robot_view, renderer.scene, paths[arm][index],
                                target, COLORS[arm])
                    canvas.paste(robot_view, (480 * column, 120))
                    draw_label(draw, (24 + 480 * column, 90), arm.capitalize(),
                               text, COLORS[arm], (480 * column, 82, 480 * (column + 1), 120))
                    draw_label(draw, (24 + 480 * column, 494),
                               f"Tip error: {tip_errors[arm][index]:.3f} mm",
                               text, COLORS[arm], (480 * column, 480, 480 * (column + 1), 530))
                    selected = (times[:index + 1] >= 2)
                    points = offsets[arm][:index + 1][selected]
                    trails[arm].set_data(points[:, 0], points[:, 2])
                    dots[arm].set_data([offsets[arm][index, 0]], [offsets[arm][index, 2]])
                draw_label(draw, (24, 532),
                           f"t = {times[index]:.2f} s. Gold cross: target. "
                           "Colored trails: actual tip.",
                           text, "#374151", (0, 530, 960, 572))
                cursor.set_xdata([times[index], times[index]])
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png")
                buffer.seek(0)
                canvas.paste(Image.open(buffer).convert("RGB"), (0, 580))
                footer = "Saved-data replay. Noise-free, no delay. Single episode."
                draw_label(draw, (24, 970), footer,
                           text, "#374151", (0, 950, 960, 1008))
                writer.send(np.asarray(canvas))
                if frame == 359:
                    canvas.save(poster)
                if frame % 60 == 0:
                    print(f"Rendered {frame}/360 frames", flush=True)
    finally:
        writer.close()
        plt.close(fig)
    if gif is not None:
        make_gif(output, gif)
    print(json.dumps({"seed": seed, "source": str(directory), "video": str(output),
                      "poster": str(poster), "physical_simulation_steps": 0,
                      "final_errors_mrad": {a: float(np.deg2rad(errors[a][-1]) * 1000)
                                            for a in ARMS},
                      "final_tip_errors_mm": {a: float(tip_errors[a][-1]) for a in ARMS}},
                     indent=2))


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
