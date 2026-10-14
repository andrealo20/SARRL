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

## Retained result

The pilot ran once from clean commit `a5041c7`, with Python 3.12.3,
NumPy 2.5.2, MuJoCo 3.13.0, four spawned processes and one numerical
thread per process. The seed scan checked 167 reachable commits, 160
distinct trees and 515 evidence blobs. All 560 previous episodes reproduced
on every field before the new seeds opened. The six frozen source hashes
match the previous controller, plant and interface exactly.

All 2,800 planned cells completed. Physical execution totals 5,040,000
steps, including reproduction. The new episodes used 2,392,000 virtual
prediction substeps; reproduction repeats another 239,400, for 2,631,400
virtual substeps in total. All twenty translated tasks were accepted in
one to three attempts, with offset RMS from 0.352537 to 0.586052 rad.

Each of the four delay-aware arms and the privileged physical oracle
achieved 400/400 success. Fixed-model success was 0/400; original adaptive
success was 374/400. The original adaptive controller achieved 20/20 in
every narrow condition and in wide conditions through 20 ms. At wide
postures it achieved 17/20 at 30 ms and 10/20 at 40 ms, both with and
without noise. All 26 failures exceeded the final joint-error threshold;
the other terminal criteria passed.

No contact or joint-position/speed violation was observed in any new
episode, and no prediction fallback occurred. The original adaptive
controller saturated in 46 episodes. No other arm saturated:

| Posture | Delay, ms | Saturated adaptive episodes, clean | Saturated adaptive episodes, noisy |
|---|---:|---:|---:|
| Narrow | 40 | 4/20 | 4/20 |
| Wide | 20 | 2/20 | 2/20 |
| Wide | 30 | 7/20 | 7/20 |
| Wide | 40 | 10/20 | 10/20 |

For the wide noisy tasks, the mean final maximum joint error and mean
tracking RMS are below, in mrad. RMS is calculated per joint over the
episode, then averaged over joints and seeds.

| Delay, ms | Adaptive final | Bank align final | Bank predict final | Adaptive RMS | Bank align RMS | Bank predict RMS |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.364478 | 0.364478 | 0.364478 | 0.983684 | 0.983684 | 0.983684 |
| 10 | 2.075585 | 0.342138 | 0.348129 | 2.597414 | 1.592613 | 1.502551 |
| 20 | 4.980286 | 0.343773 | 0.356199 | 5.539575 | 2.379150 | 2.263529 |
| 30 | 7.458338 | 0.343809 | 0.357915 | 8.024402 | 3.257160 | 3.193863 |
| 40 | 10.054279 | 0.359496 | 0.366779 | 10.633881 | 4.284080 | 4.320690 |

At wide noisy 40 ms, bank alignment reduces payload absolute error from
155.988 to 2.381 g, mean damping absolute error from 0.759689 to
0.095367 N m s/rad, and updates hitting parameter bounds from 49.27% to
2.12%. Bank prediction gives 2.518 g payload error. Known alignment and
bank alignment have identical tracking and parameter-error metrics in
every condition. Without delay, all four new arms match the original
adaptive metrics within each posture/noise condition.

Prediction is not uniformly better than alignment. Its RMS decreases in
14 of the 16 delayed conditions, but at wide 40 ms it rises by 0.65%
without noise and 0.85% with noise. Its final error rises in 13 of the
16 delayed conditions, including every wide delayed condition and every
clean delayed condition. The privileged known-delay predictor reaches
3.610816 mrad RMS in the wide noisy 40 ms cell, against 4.320690 for the
online bank predictor. These results support bank alignment as the next
baseline; adding prediction needs a separate tracking tradeoff assessment.

Each bank selects the correct final delay in all 400 episodes. Correct
selection starts at the first update for 0 and 10 ms, then at 0.02, 0.03
and 0.04 s for 20, 30 and 40 ms, respectively. Corresponding correctly
selected update fractions are 1, 1, 299/300, 298/300 and 297/300. The
zero-initialized FIFO exposes latency early. Selection from a moving
state or a nonempty queue and adaptation to changing delay remain untested.

The [retained files](../results/panda_posture_delay_pilot) contain every
condition, arm and paired contrast. Completion hashes and all source
hashes were checked against retained bytes and the frozen Git blobs.
A separate saved-data audit checked all 2,800 task and parameter records
and recomputed terminal kinematics, success flags and summary. It checked every
sensor sample, FIFO-delivered command, current and future reference,
sent command, saturation counter and tracking RMS in all 140 traces.
It also performed 280 sampled private-model prediction replays without
new physical episodes. The completed-run resume command refused replacement;
completion hashes remained intact. The seed block is now marked used.

![Panda posture and delay-grid pilot](../assets/figures/panda_posture_delay.png)

The figure plots all twenty paired tasks' noisy-condition means for the
three operational adaptive variants, plus two coordinates of every start
posture. All other arms and clean conditions remain in the retained
summary. Regenerate from saved data at a new output path, with the
`figures` extra installed:

```bash
python -m tools.make_panda_posture_delay_figure \
  --output results/panda_posture_delay_preview.png
```
