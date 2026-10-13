# Panda posture and command-delay grid

This descriptive engineering pilot tests the previous controller without
retuning it. The [delay-identification pilot](panda-delay.md) retained
accurate selection at 0 and 20 ms around one posture. The new pilot covers
all five candidate delays at the original posture range and at translated
postures, with the same two sensor-noise levels.

## Frozen controller and paired task

The controller, estimator, private model, plant and sensor/FIFO interface
must match the exact source hashes in the previous retained manifest
(source freeze `b7440fe`, results published in `d891c49`). The runner
rejects changes to any of these six source files. Gains, RLS priors and
bounds, delay scores, prediction integration and fallback thresholds stay
unchanged. The evaluation loop gains only optional explicit task endpoints;
its default path must reproduce all 560 previous episode records before
any new pilot seed is opened.

For each seed, first sample the original collision-screened narrow task.
For the wide task, translate both endpoints by the same offset, drawn
independently with `default_rng(seed ^ 0x7A5C0)`. Offset half ranges are
`[1.0, 0.65, 1.0, 0.65, 1.0, 0.55, 1.0]` rad. Reject offsets with RMS
below 0.35 rad, endpoints outside the original 0.15 rad joint clearance,
or contacts at any of the original 21 path samples. Try at most 1,000
offsets. This is a geometric sampling screen, not continuous-path safety
certification. Record the offset and attempt count.

The two posture ranges have identical joint displacement, quintic velocity
and acceleration, two-second motion and one-second settling period.
Translation changes absolute configuration, gravity and Cartesian motion.
Payload/friction draws and exogenous sensor noise are identical across
all conditions for a given seed. The payload and friction distributions
stay unchanged; no task is rejected based on controller performance.

## Factorial comparison

- Two posture ranges: original narrow and translated wide.
- Five constant FIFO delays: 0, 10, 20, 30 and 40 ms, always starting with
  zero commands. No changing latency or off-grid delay is tested.
- Two noise levels: zero, or independent Gaussian SDs of 0.001 rad and
  0.001 rad/s, with the previous cached-sample semantics.
- Seven unchanged arms: `fixed`, `adaptive`, `known_align`, `bank_align`,
  `known_predict`, `bank_predict`, `oracle`. Only the known-delay arms
  receive true delay; only `oracle` receives true physical parameters.
- Twenty fresh paired seeds, `9804900..9804919`, in the reserved block
  `9804900..9804999`. Tests use `9804800..9804899`.

There are 2,800 new episodes, 4,200,000 physical steps. Reproducing all 560
previous episodes adds 840,000 physical steps, for 5,040,000 total.
Prediction integration remains a separately counted virtual operation.
Four independent spawned worker processes each use one numerical thread.
Saved records are ordered by the predeclared cells regardless of completion
order; all existing cells are hash-checked before pending work on resume.

Report every cell's task success, physical violations and contacts,
saturation, final maximum joint error, per-joint tracking RMS averaged over
joints, parameter error, bound hits, final delay correctness, selection
fraction and prediction fallbacks. Preserve the same six component
contrasts as the previous pilot and paired wide-minus-narrow differences
for each controller/noise/delay combination. Report regressions and any failures.
There is no statistical acceptance rule or population inference.

## Execution and retained evidence

Commit clean sources and this protocol before opening new pilot seeds.
Validate previous completion hashes, frozen source hashes, numerical
runtime and seed reuse across every reachable Git tree. Reproduction
retains `rtol=1e-9`, `atol=1e-10` and exact discrete fields. A reproduction
mismatch prevents new cells. An OS-level lock protects the output;
atomic hashed cells permit resumption with identical source, protocol,
runtime and worker count. Completed outputs cannot be replaced.

```bash
python -m pip install -e '.[mujoco,dev]' 'mujoco==3.13.0' 'numpy==2.5.2'
python -m tools.run_panda_posture_delay --episodes 20 --workers 4 \
  --output results/panda_posture_delay_pilot
```

Retain manifest, reproduction report, 2,800 episode records, 140 compressed
first-seed traces, summary and completion hashes. Keep duplicate per-cell
files local. Generate figures from saved data only. Broader loads, moving
queue initialization, changing latency, actuator time constants, Cartesian
tasks and seven-joint safety filtering remain separate increments.
