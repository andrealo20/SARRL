# Panda constant-delay reference validation

This comparison validates the unchanged cumulative bank on fresh paired
tasks before further changing-latency work. It covers both posture ranges,
clean/noisy measurements and constant delays of 0, 10, 20, 30 and 40 ms.
Fifty seeds, `9805500` through `9805549`, give twenty conditions and four
arms, for 4,000 episodes. The seed is the pairing unit across conditions.

Fixed control, original adaptation, cumulative bank alignment and a
known-age diagnostic use identical physical parameter draws and matched
tasks/noise. Prediction is disabled. The diagnostic receives constant
applied age; no controller receives true parameters. Original controllers,
plant, transport, gains, RLS priors and bounds remain frozen.

## Execution

Commit code, this protocol and the reserved registry before execution.
Reproduce all 1,920 parameter-memory records before opening the new block.
Numerical versions are Python 3.12, NumPy 2.5.2 and MuJoCo 3.13.0, with
one numerical thread per worker. Only the canonical output is allowed.
An interruption resumes with the same commit and runtime; completed
outputs are never replaced.

```bash
python -m tools.run_panda_reference_validation --workers 4 --reproduce-only
python -m tools.run_panda_reference_validation --workers 4 --resume
python -m tools.audit_panda_reference_validation
```

The run retains every record, eighty first-seed traces, reproduction and
decision reports, source hashes and completion hashes. It uses 6,000,000
fresh physical substeps plus 2,976,000 for reproduction, without prediction
virtual steps. All conditions are retained regardless of outcome.

## Reference acceptance rule

All criteria must pass:

1. Cumulative-bank task success and final-age correctness are each at least
   98% across its 1,000 episodes and at least 95% in every fifty-episode cell.
2. The bank has zero measured physically unsafe episodes.
3. In each cell, the paired mean full-episode RMS increase for every joint
   and mean final maximum joint error increase are at most 0.1 mrad against
   the known-age diagnostic.

These are engineering thresholds for the declared grid. They do not certify
general safety or statistically establish a population noninferiority claim.
Point estimates and every failed criterion are reported separately.
The later latency comparison starts only after this gate has been evaluated.
A failed reference gate requires investigation before that comparison.

Paired tracking/final-error contrasts against original adaptation and the
known-age diagnostic report 95% percentile bootstrap intervals using
5,000 resamples. Resampling selects complete seed blocks across twenty
conditions, preserving dependence. The analysis RNG seed is 2305.
The fifty task/parameter draws define the uncertainty unit; the 1,000 bank
episodes are not treated as independent observations. Zero failures do not
produce a claimed zero failure-probability bound.
The [SciPy bootstrap reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
describes paired resampling and percentile intervals.

## Saved-input profiling

After the completed validation, profiling replays the twenty bank traces
sequentially in one process. One complete pass is discarded as warmup;
five measured passes retain all 30,000 control-cycle durations. Each cycle
includes state preparation, reference evaluation, command computation and
the causal estimator update. Plant stepping, construction, file I/O and
diagnostic serialization are excluded.

The report retains per-cycle durations, p50/p95/p99/max, fractions exceeding
the 10 ms control period, runtime, CPU identity and input/source hashes.
The p99-versus-period check describes this host and workload. Desktop
scheduling and the omitted plant/transport costs prevent a hard real-time
or hardware claim. Profiling opens no seed and performs no physical step.
Timing uses Python's [performance counter](https://docs.python.org/3/library/time.html#time.perf_counter_ns).
The canonical directory records the attempt before measurement; completed
timings are never replaced or selected from repeated attempts.

```bash
python -m tools.profile_panda_reference
python -m tools.profile_panda_reference --audit
```
