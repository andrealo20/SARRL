"""Draw the project header and the two evaluated control paths."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.figure_style import apply_style, check_bounds, save_svg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/figures"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    names = ("project_header", "control_paths")
    if any((args.output / f"{name}.{ext}").exists() for name in names for ext in ("png", "svg")):
        parser.error("overview output already exists; choose a new directory")

    import matplotlib
    import mujoco

    from sarrl.robots.panda import HOME, load_panda_model

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    apply_style()
    model = load_panda_model()
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 480
    data = mujoco.MjData(model)
    data.qpos[:] = HOME
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.2, 0, 0.45]
    camera.distance, camera.azimuth, camera.elevation = 1.7, 135, -22
    options = mujoco.MjvOption()
    options.geomgroup[3] = 1
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=options)
        robot = renderer.render().copy()
    fig = plt.figure(figsize=(13.5, 3.3))
    ax = fig.add_axes((0.035, 0.08, 0.62, 0.84))
    ax.axis("off")
    ax.text(0, 0.88, "SARRL", fontsize=48, color="#18232f", va="top")
    ax.text(0, 0.49, "Safe Adaptive Residual Reinforcement Learning", fontsize=18,
            color="#18232f", va="top")
    ax.text(0, 0.28, "Robotic control under model mismatch", fontsize=17,
            color="#46535e", va="top")
    ax.text(0, 0.08, "Planar learning. Seven-joint control and identification.", fontsize=15,
            color="#46535e", va="top")
    ax = fig.add_axes((0.69, 0.12, 0.28, 0.72))
    ax.imshow(robot)
    ax.axis("off")
    ax.set_title("Packaged Panda model", fontsize=15, pad=10)
    check_bounds(fig, "project header")
    fig.savefig(args.output / "project_header.png", dpi=160)
    save_svg(fig, args.output / "project_header.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13.5, 5.4))
    ax.set(xlim=(0, 12), ylim=(-0.4, 5))
    ax.axis("off")
    fig.subplots_adjust(left=0.035, right=0.965, top=0.96, bottom=0.07)
    ax.text(0, 4.7, "Planar: residual learning and safety filtering", fontsize=20)
    ax.text(0, 2.3, "Panda: nominal control and online identification", fontsize=20)

    def box(x, y, label, width=2.3):
        ax.add_patch(Rectangle((x, y), width, 0.8, facecolor="#f6f7f5",
                               edgecolor="#607078", linewidth=1.2))
        ax.text(x + width / 2, y + 0.4, label, fontsize=16, ha="center", va="center")

    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "->", "color": "#087e8b", "lw": 1.6})

    for x, label in zip((0, 3.1, 6.2, 9.3),
                        ("Model-based\ncontrol", "Add bounded\nSAC correction",
                         "HOCBF\nfilter", "2-DoF\nplant"),
                        strict=True):
        box(x, 3.4, label)
    for x in (2.3, 5.4, 8.5):
        arrow((x, 3.8), (x + 0.8, 3.8))
    ax.text(0, 2.9, "The policy corrects the nominal command before projection.",
            fontsize=14, color="#46535e")
    box(0, 1.1, "Joint\nreference")
    box(4.3, 1.1, "Computed-torque\ncontrol")
    box(9.3, 1.1, "Seven-joint\nPanda plant")
    arrow((2.3, 1.5), (4.3, 1.5))
    arrow((6.6, 1.5), (9.3, 1.5))
    box(4.1, -0.1, "Payload / friction estimate", width=3.0)
    ax.plot([10.45, 10.45, 7.1], [1.1, 0.3, 0.3], color="#087e8b", linewidth=1.5)
    arrow((7.6, 0.3), (7.1, 0.3))
    ax.text(7.6, 0.52, "Measured transitions", fontsize=13, color="#46535e")
    arrow((5.6, 0.7), (5.6, 1.1))
    check_bounds(fig, "control paths")
    fig.savefig(args.output / "control_paths.png", dpi=160)
    save_svg(fig, args.output / "control_paths.svg")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
