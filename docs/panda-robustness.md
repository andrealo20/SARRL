# Panda sensor-noise and command-delay pilot

This descriptive engineering pilot measures the existing fixed, adaptive
and true-parameter diagnostic controllers under noisy measurements and
delayed actuation. It keeps the task, gains and eight-parameter estimator
from the [payload/friction pilot](panda-adaptation.md). No controller is
tuned on these new episodes and there is no statistical acceptance rule.

## Paired design

Every episode has unknown payload mass uniformly drawn in 0.5 to 2.5 kg
and seven extra dampings uniformly drawn in 0.5 to 3 N m s/rad, using the
previous pilot's independent parameter generator. For each seed, all four
conditions and all three arms share the initial state, target and physical
parameters.

| Condition | Position noise SD | Velocity noise SD | Command delay |
|---|---:|---:|---:|
| `ideal` | Zero | Zero | Zero |
| `noise` | 0.001 rad | 0.001 rad/s | Zero |
| `delay` | Zero | Zero | 20 ms |
| `noise_delay` | 0.001 rad | 0.001 rad/s | 20 ms |

These values define engineering perturbations, not measurements of a
particular robot's sensors or drive electronics. The delay is a pure FIFO
of two 10 ms command intervals, not a first-order actuator time constant.

Sensors draw independent Gaussian noise on each joint's position and
velocity at every control boundary, including the initial and final
states. The generator is `default_rng(seed ^ 0x50A07)` and draws fourteen
standard normals in position-then-velocity order, even in zero-noise
conditions. A cached sample is shared by feedback, the transition estimator
and the next control step. Repeated reads do not consume another draw.
Sensor generators are independent of task and parameter generators.
The two noisy conditions use identical exogenous noise sequences across
conditions and arms, although their physical states can differ.

The command chain clips the sent torque to the usual limits, queues it,
then applies the command from two intervals earlier for all five physical
substeps. The FIFO starts with zeros, so the first 20 ms has zero commanded
torque, including gravity compensation. Commands still queued at the
three-second horizon are not flushed. This startup behaviour is part of
the tested condition and is visible in the saved traces.

## Controller inputs and scoring

Every controller receives the same measurement interface. The adaptive
controller remains the previous bounded RLS with a midpoint regressor
and finite-difference acceleration. It receives the current sent command
and cached measured pre/post positions and velocities. It receives no
delivered torque, delay value, sensor-noise sample, simulator acceleration
or true parameters. It has no noise filter or delay hypothesis bank.

The oracle has the true payload and friction parameters, but uses the
same noisy feedback and delayed actuation. It has no advance knowledge of
future noise and no delay compensation. It is a diagnostic reference for
physical-model error, not an optimal controller or a robustness bound.

Tracking, contacts, limits and final success are scored on physical state,
using the original task's thresholds and substep observations. Measurement
errors do not alter the ground-truth success test. Records distinguish
sent and delivered torque, count command mismatches, record observed noise
RMS and hash every exogenous noise sequence. Estimator diagnostics retain
parameter errors, rank/conditioning and bound hits. The covariance remains
an inverse-information matrix without a calibrated uncertainty claim.

## Execution and evidence

Reserve `9804500..9804599` before opening data; this pilot uses the first
twenty seeds, `9804500..9804519`. Tests use `9804400..9804499`. The design
has 240 new episodes and 360,000 physical steps. Before any new seed, the
runner reproduces all 240 retained adaptation episodes on every field,
adding another 360,000 steps. Numeric comparisons keep relative tolerance
`1e-9` and absolute tolerance `1e-10`, including nested estimator records.
Counters and flags must agree exactly. The retained reference's MuJoCo
3.13.0 and NumPy 2.5.2 versions are required for this reproduction.

Use Python 3.12 for the retained numerical runtime. The runner requires
committed clean sources, checks the seed block across
all reachable Git history, verifies the reference's completion hashes,
and records code, protocol and runtime fingerprints. Reproduction cells
and new cells are saved atomically with individual hashes under an
OS-level lock. A mismatch stops execution before any new episode.

```bash
python -m pip install -e '.[mujoco,dev]' 'mujoco==3.13.0' 'numpy==2.5.2'
python -m tools.run_panda_robustness --episodes 20 --output results/panda_robustness_pilot
```

An interrupted run resumes with the same arguments plus `--resume`, only
with matching protocol, commit, source hashes and runtime. Existing cells
are checked before any pending case; malformed evidence stops the run.
A completed pilot cannot be overwritten. Final retained files are the
manifest, reproduction report, all 240 episode records, twelve compressed
first-seed traces, summary and completion hashes. Duplicate per-cell files
remain local.

Report all conditions and all arms, including failures. Compare adaptive
against fixed, each perturbed interface against ideal, and parameter
error separately from task success. Twenty tasks at one noise/delay
setting do not establish a tolerance envelope, population generalisation
or hardware safety. No residual learning or seven-joint safety filter is
introduced in this pilot.
