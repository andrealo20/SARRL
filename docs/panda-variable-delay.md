# Panda populated-queue and variable-latency pilot

This descriptive pilot tests the frozen `bank_align` controller after the
[posture/delay grid](panda-posture-delay.md). That grid retained success
and accurate final delay selection but exposed latency through a zero-start
queue. The new inputs remove that startup cue and change latency during
motion. No controller or estimator is tuned on the new data.

## Task, startup and controller inputs

Use the same translated wide tasks, payload/friction distributions, gains,
estimator priors and bounds, three-second scored horizon, limits and
physical-state success criteria. The six controller, plant and original
sensor-interface files and model provenance must match the previous pilot.
The evaluation loop adds prepared-state/interface inputs and optional
phase RMS; its default path must reproduce every previous record.

Cold episodes start at the planned initial posture with zero velocity and
four known zero prior commands. Warm episodes first execute twenty 10 ms
intervals of a common fixed nominal controller holding the planned initial
posture. That controller receives noisy state, knows neither payload nor
friction, and uses the initial profile's constant FIFO delay. Each arm
performs the same physical warmup independently, without rejecting outcomes.

At handoff, retain the physical state and last delivered command, and
provide the last four sent commands to both alignment controllers.
Neither receives warmup state transitions or fitted parameters. All
estimators and scores start fresh. The bank receives no true latency,
packet-arrival metadata, delivered torque or plant object. Main-reference
endpoints stay unchanged, so the warm handoff can start with position and
velocity error. Warmup changes both physical state and queue; its contrast
with cold startup does not isolate queue population alone.

Warmup sensing uses `seed ^ 0xA117` as its interface seed. At handoff the
main interface starts the original episode-seed noise stream, giving all
conditions identical 301 scored sensor draws. Common warmup and handoff
state are paired across arms within a condition. Profiles with different
initial latency can have different warm states. The task, true parameters
and main exogenous noise stay paired across the whole factorial.

## Packet transport and privileged diagnostic

A command sent at interval `k` arrives at `k + d(k)`. At each control
boundary, use the arrived command with the newest sent timestamp. Keep
the last accepted command if nothing new arrives; discard packets older
than a command already accepted. Simultaneous arrivals also keep only
the newest timestamp. There is no queue reset at a latency change.

Four profiles provide controls for both change directions:

| Profile | Sent-command latency before 1 s | Sent-command latency from 1 s |
|---|---:|---:|
| `fixed10` | 10 ms | 10 ms |
| `fixed40` | 40 ms | 40 ms |
| `up` | 10 ms | 40 ms |
| `down` | 40 ms | 10 ms |

Sent-packet latency differs from the age of the command actually applied.
For `up`, applied ages at intervals 100 through 104 are 10, 20, 30,
40 and 40 ms. For `down`, age is 40 ms at interval 100 and 10 ms from
101, while three older packets are discarded. Record both quantities.
The frozen bank still considers fixed ages 0, 10, 20, 30 and 40 ms, with
cumulative pre-update scores and no forgetting or resets.

Compare four arms: `fixed`, original `adaptive`, frozen `bank_align`, and
`known_align`. The last is a privileged diagnostic using the same original
single RLS, with the true applied-command age for each transition. It
aligns the target to known sent history, receives no true physical
parameters, and performs no prediction. It is excluded from operational
claims. Every arm observes the same transport schedule.

## Design and reporting

Cross two startups, two sensor-noise levels (zero or the previous SDs
0.001 rad and 0.001 rad/s), four profiles and four arms on twenty paired
fresh seeds `9805100..9805119`. Reserve the whole `9805100..9805199` block
before data; engineering tests use `9805000..9805099`.

The 1,280 scored episodes use 1,920,000 physical steps; 640 warmups add
64,000. Reproducing all 2,800 previous records adds 4,200,000, for
6,184,000 physical steps. No new arm predicts state. Reproduction's
2,392,000 virtual prediction substeps are counted separately.

Report every cell's success, final maximum joint error, per-joint tracking
RMS, parameter errors, bound hits, clipping and physical violations.
Tracking RMS covers only the scored three seconds, also split at 1 s.
Safety and total physical-step counts include warmup. Retain warmup
diagnostics separately. Report paired bank-minus-adaptive and privileged
known-minus-bank contrasts and compare warm and cold cells explicitly.

For alignment arms, compare each post-transition selected age with the
age applied in that transition. Report final correctness, whole/pre/post
switch correctness fractions and the acquisition time of the final
uninterrupted correct run. Final-regime acquisition delay is measured
from the first interval with the stable final applied age to the
post-transition observation at acquisition; the minimum is one control
period. If selection is wrong at the end, report no acquisition. Preserve
failures and regressions. There is no statistical acceptance rule,
population inference or tuning after the new seeds open.

## Execution and evidence

Commit clean sources before data. Check reference completion hashes,
frozen source/model provenance, NumPy 2.5.2, MuJoCo 3.13.0 and seed reuse
over all reachable history. Reproduce all 2,800 previous episodes on
every field before opening new seeds, retaining the original tolerances.
Four spawned workers each use one numerical thread. Atomic hashed cells
are validated before pending work; resume requires identical source,
runtime, protocol and worker count. A completed output cannot be replaced.

```bash
python -m tools.run_panda_variable_delay --episodes 20 --workers 4 \
  --output results/panda_variable_delay_pilot
```

Retain manifest, reproduction report, all episode records, 64 compressed
first-seed scored traces with their warmup traces, summary and completion
hashes. Duplicate cells stay local. Figures use saved data only. Other
switch times, stochastic/off-grid latency, residual learning, Cartesian
control and seven-joint safety filtering remain separate increments.

Audit the saved files with the recorded NumPy/MuJoCo runtime and frozen
Python sources, without new physical episodes:

```bash
python -m tools.audit_panda_variable_delay
```
