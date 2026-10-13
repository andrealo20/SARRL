# Panda delay-score memory pilot

The [previous pilot](panda-variable-delay.md) reached every target with
the cumulative-score bank but kept the initial delay after all 160 latency
changes. This descriptive comparison changes only the score memory and
uses fresh paired tasks. There is no acceptance rule, statistical
population claim, parameter sweep or tuning after new seeds open.

## Frozen variants

Keep the original `PandaDelayController` unchanged as `bank_align`.
Add two alignment-only controllers with the same five age hypotheses,
eight-parameter candidate RLS estimators, priors, bounds, parameter
forgetting, gains and torque clipping. Each innovation is calculated
before updating its candidate estimator. The squared innovation is the
sum over all seven torque components, as in the cumulative baseline.

For exponential scores (`bank_exp`), use
`score[k] = (49/51) * score[k-1] + innovation_squared[k]`, starting at zero.
For windowed scores (`bank_window`), sum the latest fifty innovations,
including the current transition; use every available transition until
fifty exist. Select the smallest score, with the original lowest-age tie
break. Fifty intervals equal a 0.5 s window. The exponential effective
sample size `(1 + decay) / (1 - decay)` is also fifty. These are two fixed
engineering choices, selected before tests and new data, without a search
over memory lengths.

Candidate RLS estimators retain their own complete causal update history.
There are no score or estimator resets, switch detectors, hysteresis or
changes to parameter forgetting. The controller receives measured state,
sent clipped commands and the control period. It receives no true age,
packet arrivals, delivered torque, switch time, profile or plant object.
Prediction is disabled. The new class uses the original update arithmetic;
tests compare all candidate diagnostics on identical exogenous transitions.

## Paired design

Reuse the previous wide tasks, payload/friction sampling, three-second
scored horizon, packet semantics, change at 1 s, sensor noise and common
0.2 s physical warmup. Keep the physical warmup/main handoff continuous,
with fresh estimators and the four known prior sent commands. Main noise
is paired across all conditions; warmup changes both state and queue.
The variable-delay evaluator adds an opt-in bank factory and arm-list
summary argument. Its default path must reproduce every previous record.
Baseline controller, plant, original interface, packet transport and
model provenance must retain their previous hashes.

Cross cold/warm startup, clean/noisy sensing, fixed 10 ms/fixed 40 ms/
10 to 40 ms/40 to 10 ms profiles and six arms: `fixed`, original `adaptive`,
frozen `bank_align`, privileged `known_align`, `bank_exp`, `bank_window`.
Use twenty fresh paired seeds `9805300..9805319`, reserving the whole
`9805300..9805399` block. Engineering tests use `9805200..9805299`.
Keep the fixed and original adaptive controls even if their previous
performance suggests no benefit. The known-age diagnostic receives actual
applied ages and is excluded from operational claims.

The 1,920 new scored episodes require 2,880,000 physical steps; 960 warmups
add 96,000. Reproduction of all 1,280 previous records adds 1,984,000,
for 4,960,000 physical steps. No arm uses virtual prediction integration.
Four spawned workers each use one numerical thread.

## Metrics and interpretation

Report every cell's success, final maximum joint error, mean per-joint
tracking RMS over all three seconds and before/after 1 s, parameter
errors, bound hits, clipping and physical violations. Safety and total
physical counts include warmup. Report paired memory-minus-cumulative
differences in success, final age correctness, final joint error and
tracking RMS after 1 s, together with warm-minus-cold contrasts.

For all banks, report final age correctness, whole/pre/post-switch correct
selection fractions, final uninterrupted correct-run acquisition delay,
and total/post-switch selection switches. Use the previous definition of
applied command age, including the transitional holds/discards. Acquisition
delay starts at the first interval with stable final applied age and ends
at the post-transition observation establishing the final correct run.
If final selection is wrong, retain absent acquisition. A correct final
selection alone does not imply stable identification; noisy switching and
constant-profile regressions remain part of the result. Better age
selection does not guarantee better tracking because it also changes the
candidate parameters used for control.

## Execution and retained evidence

Commit clean code and this design before new data. Verify previous
completion hashes, source/model provenance, NumPy 2.5.2 and MuJoCo 3.13.0,
and scan all reachable evidence for seed reuse. Reproduce all 1,280
previous episodes on every field before opening new seeds, using the
existing tolerances. Validate every hashed saved cell before pending
work. Resume requires identical commit, sources, runtime, worker count
and protocol; completed output cannot be replaced.

```bash
python -m tools.run_panda_score_memory --episodes 20 --workers 4 \
  --output results/panda_score_memory_pilot
```

Retain manifest, reproduction report, all records, summary, completion
hashes and 96 compressed first-seed traces, including 48 warmups. New
variant traces also contain pre-update scores and each transition's
squared innovations. The public audit reconstructs score weights
independently, replays all eighty first-seed adaptive estimators and checks
saved commands, packet ages, noise and metrics without physical episodes.
Figures use saved data only. Duplicate hashed cells stay local.

```bash
python -m tools.audit_panda_score_memory
python -m tools.make_panda_score_memory_figure
```

This pilot retains one switch time and two on-grid latencies. Multiple
switches, off-grid or stochastic delays, memory calibration, prediction,
residual learning and seven-joint safety filtering are separate work.
