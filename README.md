# SARRL

[![CI](https://github.com/andrealo20/SARRL/actions/workflows/ci.yml/badge.svg)](https://github.com/andrealo20/SARRL/actions/workflows/ci.yml)
[![CodeQL](https://img.shields.io/github/actions/workflow/status/andrealo20/SARRL/dynamic%2Fgithub-code-scanning%2Fcodeql?branch=main&label=CodeQL)](https://github.com/andrealo20/SARRL/actions/workflows/dynamic/github-code-scanning/codeql)
[![release](https://img.shields.io/github/v/release/andrealo20/SARRL?sort=semver&color=6f42c1&label=release)](https://github.com/andrealo20/SARRL/releases/latest)
[![Python](https://img.shields.io/badge/python-3.10%2B-FFD43B.svg?logo=python&logoColor=white&labelColor=3776AB)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-f39c12.svg)](LICENSE)

![SARRL banner](assets/banner.png)

**SARRL** (Safe Adaptive Residual Reinforcement Learning) is a research stack for robotic control under model mismatch. It combines model-based control with bounded residual reinforcement learning, learned dynamics context, uncertainty-aware policy authority and HOCBF safety filtering.

Version 2.3 adds a **seven-joint Panda in MuJoCo** with torque control,
online payload/friction identification and command-delay alignment. The
reference platform for residual learning and HOCBF remains the reproducible
**2-DoF planar arm**, available as an analytical model and a MuJoCo plant.
The Panda release covers joint-trajectory engineering benchmarks across two
sampled posture ranges. Prediction and changing-latency variants remain
experimental. See the [v2.3 guide](docs/v2.3.md) for scope, a short example
and frozen-data audits. Hardware and sim-to-real results are not claimed.

## Panda demo

[![Fixed and adaptive Panda replay](assets/replays/panda_control.gif)](assets/replays/panda_control.mp4)

[Watch the 24-second comparison](assets/replays/panda_control.mp4). Both controllers
follow the same joint reference with the same added payload and friction.
The video replays the first retained paired case, seed `9804300`; three
seconds of physical time are slowed to twenty-four seconds. The graph shows
the maximum joint-reference error. This single, noise-free episode illustrates
the [adaptation pilot](docs/panda-adaptation.md), which retained twenty pairs
per condition. See [demo reproduction](docs/demo.md) for hashes, dependencies
and rendering commands.

### Robot motion and recorded path

[![Nominal Panda motion and tip path](assets/replays/panda_motion.gif)](assets/replays/panda_motion.mp4)

[Watch the 12-second motion replay](assets/replays/panda_motion.mp4). The teal
line follows the recorded attachment-site path; the orange point marks the
target. This is the first nominal pilot episode, seed `9804000`, with three
physical seconds slowed to twelve seconds. It demonstrates joint-trajectory
tracking with a matched model, without adaptation, residual learning or a
safety certificate. Both animations use saved states rather than new physical
simulation steps. Full-resolution MP4 files are linked from the GIFs.

## Selected evidence

| Question | Retained comparison | Scope |
|---|---|---|
| Does residual learning help the fixed planar baseline? | Residual SAC: 56.4% ± 7.0 pp success; fixed nominal: 11.0% | v1.2 held-out benchmark; mean ± training-seed SD over five policies |
| Does online identification help on the MuJoCo planar plant? | 11.7% to 90.9% success in distribution | v2.0, 1,900 paired decision episodes |
| Does an uncertainty margin reduce physical violations? | 11.1% to 0.9% unsafe episodes in distribution | v2.2, 1,900 paired decision episodes |
| Does Panda adaptation handle added payload and friction? | Adaptive 20/20 successful; fixed 0/20 | Descriptive seven-joint pilot, noise-free and without delay |

These are separate comparisons, with different protocols and seed blocks.
The [complete results](docs/results.md) include OOD/fault tests, failed
acceptance decisions, safety tradeoffs and the seven Panda pilots.
Protocols and raw evidence are indexed in [experiments](docs/experiments.md);
executed checks are recorded in [verification](docs/verification.md).

## How it works

[![Planar and Panda control paths](assets/figures/control_paths.png)](assets/figures/control_paths.svg)

The two paths have different evaluated scope. The planar stack combines
model-based control, residual SAC and HOCBF projection. Panda identification
updates the computed-torque model from causal measured transitions and sent
commands. The diagram shows these components, not a seven-joint safety claim.


SARRL starts from a physics-based controller and lets SAC learn only a bounded correction:

```math
\tau_{candidate}=\tau_{nominal}+\tau_{residual}
```

This narrows the learning problem to compensating for model mismatch. The repository also includes domain randomization, fault injection, a causal GRU context encoder, residual-dynamics ensembles, uncertainty gating and hard HOCBF projection. Components are modular and independently testable.

## Experiments in pictures

Each figure below comes from retained evidence. Click a plot for its scalable
SVG version. Full protocols describe the conditions, sampling and limits.

### Nominal model and seven-joint tracking

[![Panda model, recorded tip path and joint tracking errors](assets/figures/panda_nominal.png)](assets/figures/panda_nominal.svg)

The first nominal episode shows the packaged collision geometry and tip path.
The adjacent panel separates the tracking error of all seven joints.
[Nominal protocol](docs/panda.md).

### Payload and friction identification

[![Paired controller results and online parameter estimates](assets/figures/panda_adaptation.png)](assets/figures/panda_adaptation.svg)

The pilot separates payload and added friction, then combines them. Every
paired endpoint error is shown, together with a first-seed trace and parameter
estimates. Accurate tracking does not imply accurate identification of every
joint's friction. [Adaptation protocol](docs/panda-adaptation.md).

### Sensor noise and command delay

[![Noise and delay effects on Panda tracking and estimates](assets/figures/panda_robustness.png)](assets/figures/panda_robustness.svg)

Noise and transport delay stress the same fixed and adaptive controllers.
The plots distinguish tracking from parameter estimation under the tested
conditions. [Noise and delay protocol](docs/panda-robustness.md).

### Command alignment and queued-command prediction

[![Panda delay identification and prediction comparison](assets/figures/panda_delay.png)](assets/figures/panda_delay.svg)

The comparison separates alignment, selection of a delay hypothesis and
prediction through the command queue. Prediction is experimental; alignment
alone remains the reference. [Delay protocol](docs/panda-delay.md).

### Wider postures and constant delays

[![Tracking and identification across two postures and five delays](assets/figures/panda_posture_delay.png)](assets/figures/panda_posture_delay.svg)

Two sampled posture ranges and delays from 0 to 40 ms test the frozen
controllers. The noisy-condition plots show final error, tracking RMS and
payload error; the final panel shows two coordinates of the paired start poses.
[Posture and delay protocol](docs/panda-posture-delay.md).

### Changing packet latency

[![Warm startup and changing-latency comparison](assets/figures/panda_variable_delay.png)](assets/figures/panda_variable_delay.svg)

Reaching the target and identifying the active delay are separate outcomes.
The cumulative bank tracks the tasks while retaining the initial delay in the
changing cases. Applied command age is a labelled diagnostic, not an input
available to the bank. [Variable-latency protocol](docs/panda-variable-delay.md).

### Finite score memory and its tracking cost

[![Cumulative, exponential and window score-memory comparison](assets/figures/panda_score_memory.png)](assets/figures/panda_score_memory.svg)

Shorter memory recovers the final age in some changing cases, but both tested
variants worsen noisy tracking. The first-seed transition panels show why a
correct final selection alone is insufficient to assess the controller.
[Score-memory protocol](docs/panda-score-memory.md).

### Planar adaptive control under a motor fault

[![Planar motor-fault replay and actuator time-constant identification](assets/figures/planar_mujoco.png)](assets/figures/planar_mujoco.svg)

An illustrative motor-fault replay compares fixed and identified control
behind the same HOCBF filter. Its seed is selected by the documented
timeout-versus-success rule; it is not an unselected population sample.
The third panel includes all 5,700 adaptive decision episodes from v2.0.
[Campaign results](docs/results.md) and [visual reproduction](docs/visuals.md).

## Quick start

```bash
git clone https://github.com/andrealo20/SARRL.git
cd SARRL
python -m pip install -e '.[dev]'
pytest -q
```

Run one nominal Panda trajectory after installing the simulator extra:

```bash
python -m pip install -e '.[mujoco]'
python examples/panda_reach.py
```

The example prints a JSON result and writes no files. It is an installation
check using an existing engineering-test seed. See the [release guide](docs/v2.3.md)
for the controller scope, complete evidence and source/runtime requirements
for audits.

For the Python 3.12 reference dependency versions and saved-data video
renderer, follow the [demo reproduction guide](docs/demo.md).

The planar stack requires NumPy, SciPy and PyTorch. The MuJoCo plant needs the optional `mujoco` extra (`pip install -e '.[mujoco]'`); its tests skip without it. Training and evaluation commands are documented in [`docs/experiments.md`](docs/experiments.md).

## Repository guide

- `sarrl/`: dynamics, controllers, RL, adaptation and safety
- `tools/`: training, evaluation and sweep commands
- `tests/`: automated verification
- `artifacts/` and `results/`: retained experimental evidence
- `docs/`: [design](docs/design.md), [mathematics](docs/mathematics.md), [experiments](docs/experiments.md), [verification](docs/verification.md) and [changelog](docs/changelog.md)

## Limitations

Residual learning and HOCBF are evaluated on the planar stack. The Panda
supports joint-trajectory control and online payload/friction identification;
constant on-grid delay alignment is the reference. Changing latency remains
unresolved, and finite score memory worsened noisy tracking in the retained
comparison. Panda residual learning, seven-joint HOCBF, hardware and sim-to-real
remain future work.

HOCBF guarantees are relative to the controller model; physical violations
remain possible with model mismatch and delayed actuation. Ensemble
disagreement is not calibrated probability. See [complete results](docs/results.md)
for campaign-specific limitations and [the roadmap](docs/roadmap.md) for next work.

## License and citation

Released under the [MIT License](LICENSE) by [Andrea Loroni](https://github.com/andrealo20).

```bibtex
@software{loroni_sarrl_2026,
  author  = {Andrea Loroni},
  title   = {SARRL: Safe Adaptive Residual Reinforcement Learning for Robotic Manipulation},
  year    = {2026},
  version = {2.3.1},
  url     = {https://github.com/andrealo20/SARRL}
}
```
