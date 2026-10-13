# Panda command-delay identification and prediction pilot

This engineering pilot separates alignment of the identification target,
online delay selection and prediction through the command queue. It builds
on the [noise/delay pilot](panda-robustness.md), which retained 20/20
adaptive successes at every condition but showed increased tracking and
parameter error under a 20 ms delay. This is descriptive, with no statistical
acceptance rule or controller tuning on the new episode block.

## Paired controllers and conditions

The task, physical parameters, noise sequences, torque limits, gains,
three-second horizon and physical-state success criteria stay unchanged.
The four interface conditions are ideal, noise only, 20 ms delay only,
and noise plus delay. Noise SDs remain 0.001 rad and 0.001 rad/s. The delay
FIFO starts with two zeros. The bank contains candidate delays of 0, 10,
20, 30 and 40 ms, although this pilot's true delays are only 0 and 20 ms.

| Arm | Physical parameters | Delay for identification | Predict queued commands |
|---|---|---|---|
| `fixed` | Original nominal | None | No |
| `adaptive` | Original online RLS | Assumes zero | No |
| `known_align` | Online RLS | True delay, privileged diagnostic | No |
| `bank_align` | One RLS per candidate delay | Selected online | No |
| `known_predict` | Online RLS | True delay, privileged diagnostic | Yes |
| `bank_predict` | One RLS per candidate delay | Selected online | Yes |
| `oracle` | True payload/friction, privileged diagnostic | None | No |

Only `known_align` and `known_predict` receive the true delay. Only
`oracle` receives the true payload and friction. Operational bank arms
receive sent clipped commands and the same cached measured pre/post states
as the original adaptive controller. They receive no delivered torque,
plant object, simulator acceleration or true delay. Every episode starts
fresh estimators and empty sent-command histories.

## Identification and prediction

After transition `k`, each delay candidate `L` regresses on `u[k-L]`, or
zero when that command predates the episode. The shared eight-column
regressor uses midpoint measured state and finite-difference acceleration,
as before. Each candidate has the same original RLS prior, bounds and
innovation scale. Its score accumulates the squared norm of the seven-joint
innovation computed before that transition's update. The lowest score
selects the model for the next command; ties choose the lowest delay.
There is no forgetting, process noise, smoothing, score gate or hysteresis.

For a selected delay `L`, prediction propagates the measured state through
the `L` commands already sent that would execute before the new command.
Missing startup commands are zero. Forward acceleration is computed from
the selected payload/friction model's mass matrix and bias, using private
nominal and unit-payload basis models. RK4 uses five 2 ms substeps per
10 ms queued interval. The control reference is evaluated at the same
future time. Tracking is scored against the original current-time reference.

Prediction falls back to the current measured state and current-time
reference if its state becomes non-finite, its dynamics solve fails, or
its position/velocity displacement exceeds 0.5 rad or 3 rad/s from the
initial measured state. Each fallback is counted. These are numerical
guards, not safety certificates. Prediction integration is recorded as
virtual substeps and never added to the physical simulation-step count.

Report final delay correctness, fraction of correctly selected updates,
selection switches and the start of the final uninterrupted correct
selection run. Parameter diagnostics refer to the final selected RLS;
each candidate's estimates and bound-hit counts are also retained.
All covariance semantics remain unchanged and uncalibrated.

Six paired contrasts separate the components: known alignment versus
original adaptive; bank alignment versus original adaptive and known
alignment; known prediction versus known alignment; bank prediction versus
bank alignment and known prediction. Report task success, final and RMS
tracking error, parameter error, clipping and prediction fallbacks for all
arms, including regressions.

## Execution and retained evidence

Reserve `9804700..9804799` before data; this pilot opens `9804700..9804719`.
Engineering tests use `9804600..9804699`. There are 560 new episodes:
twenty paired seeds, four conditions and seven arms, totalling 840,000
physical steps. Before any new seed, reproduce all 240 retained noise/delay
episodes on every field, adding 360,000 steps. Total physical execution is
1,200,000 steps; prediction steps are counted separately.

The runtime must match the reference's NumPy 2.5.2 and MuJoCo 3.13.0.
Reproduction preserves `rtol=1e-9`, `atol=1e-10`, with exact counters and
flags, including nested records. Committed clean sources, history-wide
seed checks, reference completion hashes and an OS-level output lock are
required. Each completed reproduction or new cell is saved atomically
with a hash and validated before pending episodes on resume. A completed
run cannot be overwritten. Resume requires identical source commit,
hashes, protocol and runtime.

```bash
python -m pip install -e '.[mujoco,dev]' 'mujoco==3.13.0' 'numpy==2.5.2'
python -m tools.run_panda_delay --episodes 20 --output results/panda_delay_pilot
```

Use Python 3.12 for the recorded numerical runtime. Retain the manifest,
reproduction report, all 560 episode records, twenty-eight compressed
first-seed traces, summary and completion hashes. Duplicate per-cell
files remain local. Test other true delays, broader postures, actuator
time constants, Cartesian reaching and safety filtering in separate
increments. A successful result here does not establish those capabilities.
