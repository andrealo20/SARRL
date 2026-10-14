# Panda comparison demo

The video replays the first retained payload/friction pair, seed `9804300`,
from the [adaptation pilot](panda-adaptation.md). It uses fixed and adaptive
controllers on the same initial pose, target, payload and seven added
friction coefficients. The case is selected by its position in the original
protocol, without ranking outcomes. The adaptive controller's recorded inputs
are causal positions, velocities and sent commands; plant parameters remain
diagnostics.

Both MP4s are rendered at 3840 by 2160 pixels, with native scenes, captions
and curves. The existing portrait GIF previews remain 1200 pixels wide.
See [visual reproduction](visuals.md) for the dimensions and preview settings.

The trace covers three seconds in 301 samples. The 24-second video slows
playback to one eighth of physical speed and repeats the nearest saved state.
Rendering calls forward kinematics only, without advancing the physical plant
or fitting an estimator. The original records, traces and completion marker
remain unchanged. Every completion hash is verified before rendering.

Both arms follow the same commanded trajectory. Their actual joint states
differ, but the shared camera view makes a few centimetres of endpoint offset
hard to see. The revised replay adds a gold target cross and colored tip trails
to each robot view. During the final settling second, both cameras zoom
toward the same target with identical settings. In the landscape MP4, plots to the
right report joint error in degrees and tip position in the X-Z plane relative
to the target, in mm. The portrait GIF keeps these plots below the robot views.
Its black cross marks the target. The axes have equal physical scale; no
joint displacement is amplified.
The tip dot, target cross and connecting line are projected with the rendered
camera and drawn over the scene so the wrist mesh cannot hide them. The
attachment site lies inside the collision geometry; these overlays mark
its coordinates rather than adding a physical gripper or payload.
The close-up retains the last physical second's paths. Instantaneous markers
are shown earlier when they fall inside that view. The on-screen tip error
is the full three-dimensional distance to the current reference tip, computed
by forward kinematics of the saved joint reference.

The graph shows maximum absolute error across the seven joints relative to
the reference at each physical timestamp. Final errors for this single case
are 136.334532 mrad for fixed and 0.029262 mrad for adaptive. They are not the
twenty-pair averages reported in the pilot. The corresponding final tip errors
are 37.918834 mm and 0.008527 mm, and the largest joint errors are 7.811393
and 0.001677 degrees. The target is stationary during the final second.
There is no sensor noise or command
delay in this case, and no residual learning or HOCBF filtering on the Panda.
The video displays the model's collision meshes, without a gripper.

## Reference environment

Use Python 3.12 and a checkout of release `v2.3.2` or later. The reference file
pins the direct
numerical and rendering dependencies used for the demo. Transitive dependencies
are resolved by pip; the file is not a complete environment lock. The checked
Linux runtime uses Python 3.12.3 and a Torch `2.12.0+cu130` wheel, compatible
with the `torch==2.12.0` requirement. The replay does not use Torch or a GPU for
learning. Graphics-driver differences can change pixels without changing
the saved trajectories or numerical results.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements/demo-python312.txt -e '.[demo]'
python examples/panda_reach.py
```

For other supported Python versions, install `.[demo]` with its version ranges.
Exact retained-data audits require each manifest's recorded runtime, as described
in the [release guide](v2.3.md).

## Regenerate the video

The renderer and retained data require a repository checkout of `v2.3.2` or later.
Choose new paths because existing outputs are never overwritten:

```bash
python -m tools.render_panda_video --mode comparison \
  --output demo-output/panda-control.mp4 \
  --poster demo-output/panda-control.png
```

On headless Linux with EGL support, set `MUJOCO_GL=egl` before starting Python.
The renderer requires a working OpenGL backend. FFmpeg is supplied by the
`imageio-ffmpeg` dependency. The command prints its evidence directory,
episode seed, final errors and zero physical simulation steps.

The README also includes a nominal motion replay with its recorded tip path.
See [visual reproduction](visuals.md) for both animations and all plot commands.
