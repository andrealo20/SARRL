# Panda candidate parameter memory

The [saved-trace diagnosis](panda-latency-diagnosis.md) found that suffix
scores still failed in clean decreasing-delay cases, while true parameters
and separate phase fits distinguished the final age. This pilot changes
candidate parameter memory while keeping exponential score memory fixed.

## Frozen comparison

Twenty paired seeds, `9805400` through `9805419`, cover the same sixteen
conditions as the previous pilot: cold/warm startup, clean/noisy sensors,
constant 10/40 ms latency and 10-to-40/40-to-10 ms changes. Six arms yield
1,920 new episodes:

| Arm | Score memory | Candidate parameter memory |
|---|---|---|
| Fixed | None | No identification |
| Original adaptive | None | Cumulative, without command alignment |
| Cumulative bank | Cumulative | Cumulative |
| Known-age diagnostic | Privileged applied age | Cumulative |
| Score-only bank | Exponential, effective size 50 | Cumulative |
| Joint-memory bank | Exponential, effective size 50 | Exponential, effective size 50 |

The primary comparison is joint memory versus score-only memory. Cumulative
bank results remain a separate reference. The known-age arm receives applied
age only; the three banks receive measured transitions and sent commands.
Prediction is disabled. The new arm receives no switch time, delivered torque
or true physical parameter. Controller gains, priors, bounds, transport,
measurement noise and task sampling are unchanged.

The forgetting factor is fixed at `49/51`, matching the score rule. Before
the seven scalar RLS observations in each control interval, the inverse
information matrix is divided by this factor and accumulated data information
is multiplied by it. Forgetting occurs once per interval. All seven current
rows have equal age. Initial prior precision also decays. Joseph-form row
updates and end-of-interval parameter clipping retain their original arithmetic.

The inverse-information matrix is not calibrated uncertainty. Under missing
excitation it can grow; no uniform upper bound or convergence guarantee is
claimed. Invalid input is rejected, and a nonfinite numerical update raises
an error without changing estimator state. Weighted batch equivalence is
tested where clipping is inactive. Projection can change the parameter path.
The underlying exponential recurrence is described in the
[MIT identification notes](https://ocw.mit.edu/courses/2-160-identification-estimation-and-learning-spring-2006/0223ccdb1348e84e3a5f04fc0af46edf_lecture_3.pdf).

## Execution and evidence

Commit implementations, this protocol, tests and the seed registry before
execution. First reproduce all 1,920 retained score-memory episodes. Every
record field must match within the existing reproduction tolerances before
opening a new seed. Sources, numerical runtime and the packaged robot model
are checked against the retained reference.
Saved-state estimator replay uses the same numerical tolerances
(`rtol=1e-9`, `atol=1e-10`); hashes and categorical outcomes remain exact.

The new run retains every episode, first-seed traces for all 96 cells,
reproduction and decision reports, source hashes and completion hashes.
Interrupted cells resume only with the same source commit and runtime.
Completed output is never replaced. Diagnostic data and this descriptive
pilot remain separate from later confirmation.
The output directory is fixed to prevent reopening the same seed block
under a different destination.

```bash
python -m tools.run_panda_estimator_memory \
  --output results/panda_estimator_memory_pilot --workers 4
python -m tools.audit_panda_estimator_memory \
  --input results/panda_estimator_memory_pilot
```

## Exploratory decision rule

All conditions below must pass to advance the candidate to separate
confirmation. They do not change the default controller:

1. Final-age correctness improves over score-only memory by at least
   20 percentage points across changing conditions and separately in each
   cold/warm clean decreasing-delay cell.
2. Each constant-profile cell has no fewer correct final ages than either
   the cumulative or score-only reference.
3. Every cell has no fewer task successes than either reference, and the
   candidate has zero physically unsafe episodes.
4. In every cell, each joint's mean RMS after 1 s and mean final maximum joint
   error increase by at most 0.1 mrad relative to each reference. These are
   paired mean differences, with all seven joints and the final-error metric
   checked independently.

Report all cells, including selection switches, parameter errors, clipping,
acquisition failures and physical violations. Conditional recovery times
exclude absent recoveries and must not be presented as population recovery.
Any failed criterion retains the cumulative reference. Fresh confirmation
requires its own frozen protocol and unopened seeds.
