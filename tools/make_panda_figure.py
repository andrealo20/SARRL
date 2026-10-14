"""Render the retained first Panda episode without advancing the simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sarrl.robots.panda import load_panda_model, model_provenance, mujoco
from tools.figure_style import apply_style, save_figure


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/panda_nominal_pilot"))
    parser.add_argument("--output", type=Path, default=Path("assets/panda_nominal_pilot.png"))
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("figure output already exists; choose a new path")
    complete = json.loads((args.input / "complete.json").read_text())
    for relative, expected in complete["hashes"].items():
        if hashlib.sha256((args.input / relative).read_bytes()).hexdigest() != expected:
            parser.error(f"retained file hash mismatch: {relative}")
    manifest = json.loads((args.input / "manifest.json").read_text())
    if manifest["model"]["files"] != model_provenance()["files"]:
        parser.error("the packaged Panda model differs from the retained episode's model")
    trace = json.loads((args.input / "first_episode_trace.json").read_text())
    episodes = json.loads((args.input / "episodes.json").read_text())

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()

    model = load_panda_model(manifest["config"]["timestep"])
    data = mujoco.MjData(model)
    middle = min(trace, key=lambda row: abs(row["time_s"] - 1.0))
    data.qpos[:] = middle["q"]
    data.qvel[:] = middle["qd"]
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.2, 0.0, 0.35]
    camera.distance, camera.azimuth, camera.elevation = 1.65, 135, -22
    options = mujoco.MjvOption()
    options.geomgroup[3] = 1
    model.vis.global_.offwidth, model.vis.global_.offheight = 800, 700
    with mujoco.Renderer(model, height=700, width=800) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=options)
        # Show the recorded attachment-site path and its Cartesian endpoint.
        points = np.array([row["tip"] for row in trace])
        for left, right in zip(points[::10][:-1], points[::10][1:], strict=True):
            scene = renderer.scene
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).ravel(),
                np.array([0.1, 0.8, 0.7, 1.0]),
            )
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004, left, right)
            scene.ngeom += 1
        snapshot = renderer.render().copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.1]})
    axes[0].imshow(snapshot)
    axes[0].axis("off")
    axes[0].set_title("Panda, seven torque-controlled joints\nRecorded episode at t = 1 s")
    times = np.array([row["time_s"] for row in trace])
    errors = 1000 * (
        np.array([row["q"] for row in trace]) - np.array([row["reference"] for row in trace])
    )
    for joint in range(7):
        axes[1].plot(times, errors[:, joint], label=f"Joint {joint + 1}", linewidth=1.3)
    axes[1].axvline(2.0, color="0.5", linestyle=":", label="Settling starts")
    axes[1].set(xlabel="Time, s", ylabel="Reference tracking error, mrad")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8, ncol=2)
    first = episodes[0]
    axes[1].set_title(
        f"Seed {first['seed']}; contact substeps: {first['contact_substeps']}\n"
        f"Final tip error: {1000 * first['tip_error_m']:.6f} mm"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.output)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
