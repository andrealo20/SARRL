# Panda comparison demo

The video replays the first retained payload/friction pair, seed `9804300`,
from the [adaptation pilot](panda-adaptation.md). It uses fixed and adaptive
controllers on the same initial pose, target, payload and seven added
friction coefficients. The case is selected by its position in the original
protocol, without ranking outcomes. The adaptive controller's recorded inputs
are causal positions, velocities and sent commands; plant parameters remain
diagnostics.

The trace covers three seconds in 301 samples. The 24-second video slows
playback to one eighth of physical speed and repeats the nearest saved state.
Rendering calls forward kinematics only, without advancing the physical plant
or fitting an estimator. The original records, traces and completion marker
remain unchanged. Every completion hash is verified before rendering.

The graph shows maximum absolute error across the seven joints relative to
the reference at each physical timestamp. Final errors for this single case
are 136.334532 mrad for fixed and 0.029262 mrad for adaptive. They are not the
twenty-pair averages reported in the pilot. There is no sensor noise or command
delay in this case, and no residual learning or HOCBF filtering on the Panda.
The video displays the model's collision meshes, without a gripper.

## Reference environment

Use Python 3.12 and a checkout of v2.3.1. The reference file pins the direct
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

The tools and retained data require the repository checkout, available from
the release's full source archive. Choose new paths because existing outputs
are never overwritten:

```bash
python -m tools.render_panda_demo \
  --output demo-output/panda.mp4 \
  --poster demo-output/panda.png
```

On headless Linux with EGL support, set `MUJOCO_GL=egl` before starting Python.
The renderer requires a working OpenGL backend. FFmpeg is supplied by the
`imageio-ffmpeg` dependency. The command prints its evidence directory,
episode seed, final errors and zero physical simulation steps.
