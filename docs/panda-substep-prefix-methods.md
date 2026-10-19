# Held substeps for the shared-parameter prefix

The [remainder split](panda-remainder-split-results.md) attributes a median
0.656 of the last velocity residual of the
[shared-parameter screen](panda-mean-value-results.md) to the held-flow Taylor
remainder. Under the rule fixed before that split, one final phase on this
remainder is justified. This document fixes that phase before any substep is
computed on the Panda fixture.

## Change

Each held command period of 10 ms is divided into four substeps of 2.5 ms.
The command is still computed once per period from the state at its start,
queued once and delivered with the same lag, and it is held through all four
substeps. For every substep the Picard tube, the mean-value acceleration at
its start, the jerk bound on its tube and the Taylor endpoint are computed
with the substep length. The velocity Taylor term of one period therefore
becomes `dt^2 J_jerk / 8` instead of `dt^2 J_jerk / 2`, with each `J_jerk`
taken on a shorter tube.

Everything else is unchanged: the bounded-friction context, gains, physical
box, sixteen mass cells, five constant lags, six held periods, torque limits,
domains, precision, Picard budget and minimum inflation. The four substep
lengths are binary64 values that sum exactly to the period; other substep
counts must satisfy the same condition. With one substep the code path
equals the retained screen.

## Order of execution

1. For every mass cell, rerun one substep and require the result to equal the
   retained report field for field. Any difference stops the run before a
   substep result exists.
2. Run four substeps per period for the sixteen cells and five lags.
3. Audit every substep record with the existing auditor: start identity, one
   send per period shared by its substeps, causal FIFO, torque limits, tube
   domains, Picard propagation over the substep, initial acceleration, jerk
   bound and exact-rational Taylor endpoints with the substep length.

No substep was computed on the Panda fixture before this document was
committed. Development used the manufactured held-acceleration model only.
A one-substep timing probe on the first mass cell reproduced the retained
result in 3.3 seconds; it opens no new data.

## Decision fixed before computation

This is the final phase of the shared-parameter prefix line.

- If all eighty branches complete the six held periods, the line continues
  to the open terminal, sensor, runtime and simulator conditions. Those
  conditions still block integration, task comparison and release.
- If any branch fails, the line stops. The current chain of numerical changes
  is prepared as a release of methods and negative results. A partial gain
  in completed periods does not count.

The rule does not change after the result. No physical step, seed or task is
used.

## Command

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m tools.assess_panda_substep_prefix \
  --output results/panda_substep_prefix/assessment.json.gz
```

`--audit` recomputes the whole report under the same source fingerprints and
requires equality. Runtime: the pinned Panda environment, Python 3.12.3,
NumPy 2.5.2, SciPy 1.18.1, MuJoCo 3.13.0 and python-flint 0.9.0.
