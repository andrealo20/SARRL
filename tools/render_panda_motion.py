"""Animate the first retained nominal trajectory and its recorded tip path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.figure_style import font_path
from tools.render_panda_demo import REPLAY_SCALE, draw_label, make_gif


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/panda_nominal_pilot"))
    parser.add_argument("--output", type=Path, default=Path("assets/replays/panda_motion.mp4"))
    parser.add_argument("--poster", type=Path, default=Path("assets/replays/panda_motion.png"))
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--gif-width", type=int, default=1200)
    args = parser.parse_args(argv)
    if any(path.exists() for path in (args.output, args.poster, args.gif) if path is not None):
        parser.error("output already exists; choose new paths")
    if args.output.suffix != ".mp4" or args.poster.suffix != ".png":
        parser.error("use .mp4 and .png output paths")
    if args.gif and args.gif.suffix != ".gif":
        parser.error("use a .gif animation path")
    if not 1 <= args.gif_width <= 1920:
        parser.error("GIF width must be between 1 and 1920 pixels")
    complete = json.loads((args.input / "complete.json").read_text())
    for name, expected in complete["hashes"].items():
        path = args.input / name
        if not path.resolve().is_relative_to(args.input.resolve()):
            parser.error("evidence path escapes its directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            parser.error(f"retained evidence hash mismatch: {name}")
    required = {"manifest.json", "episodes.json", "first_episode_trace.json"}
    if not required.issubset(complete["hashes"]):
        parser.error("completion marker must cover the record and trace")
    record = json.loads((args.input / "episodes.json").read_text())[0]
    trace = json.loads((args.input / "first_episode_trace.json").read_text())
    manifest = json.loads((args.input / "manifest.json").read_text())

    import imageio_ffmpeg
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    from sarrl.robots.panda import load_panda_model, model_provenance

    if manifest["model"]["files"] != model_provenance()["files"]:
        parser.error("packaged model differs from the retained trajectory model")
    model = load_panda_model()
    model.vis.global_.offwidth = 960 * REPLAY_SCALE
    model.vis.global_.offheight = 640 * REPLAY_SCALE
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.2, 0, 0.45]
    camera.distance, camera.azimuth, camera.elevation = 1.7, 135, -22
    options = mujoco.MjvOption()
    options.geomgroup[3] = 1
    heading = ImageFont.truetype(font_path(), 30 * REPLAY_SCALE)
    text = ImageFont.truetype(font_path(), 22 * REPLAY_SCALE)
    points = np.array([row["tip"] for row in trace])
    for path in (args.output, args.poster):
        path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(args.output), (960 * REPLAY_SCALE, 800 * REPLAY_SCALE),
        fps=15, codec="libx264", pix_fmt_out="yuv420p",
        output_params=["-crf", "20", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        with mujoco.Renderer(model, height=640 * REPLAY_SCALE,
                             width=960 * REPLAY_SCALE) as renderer:
            for frame, index in enumerate(np.rint(np.linspace(0, len(trace) - 1, 180)).astype(int)):
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
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                   np.full(3, 0.012), np.array(record["target_xyz_m"]),
                                   np.eye(3).ravel(), np.array([1, 0.6, 0.2, 1]))
                scene.ngeom += 1
                canvas = Image.new("RGB", (960 * REPLAY_SCALE, 800 * REPLAY_SCALE), "#fafaf8")
                draw = ImageDraw.Draw(canvas)
                draw_label(draw, (24, 12), "Panda: a seven-joint trajectory", heading,
                           "#18232f", (0, 0, 960, 50), scale=REPLAY_SCALE)
                draw_label(draw, (24, 54),
                           f"Seed {record['seed']}. Three physical seconds replayed over 12 s.",
                           text, "#374151", (0, 50, 960, 90), scale=REPLAY_SCALE)
                canvas.paste(Image.fromarray(renderer.render()), (0, 90 * REPLAY_SCALE))
                draw_label(draw, (24, 744),
                           f"t = {row['time_s']:.2f} s. Teal: recorded tip path. Orange: target.",
                           text, "#374151", (0, 730, 960, 800), scale=REPLAY_SCALE)
                writer.send(np.asarray(canvas))
                if frame == 179:
                    canvas.save(args.poster)
    finally:
        writer.close()
    if args.gif:
        make_gif(args.output, args.gif, width=args.gif_width)
    print(json.dumps({"seed": record["seed"], "frames": 180,
                      "physical_simulation_steps": 0, "video": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
