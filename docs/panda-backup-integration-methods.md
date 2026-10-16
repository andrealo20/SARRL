# Panda backup integration on retained tasks

This phase checks the causal supervisor and its independent auditor on
already opened tasks. It does not open a seed block, train a policy, tune a
gain or establish candidate advancement. The [backup design](panda-backup-design.md)
retains its 10 ms period, five constant delay hypotheses, gains 25/10,
200-interval horizon and original joint reserves and QP.

## Command commitment and fallback

The new controller preserves the five cumulative RLS estimators and nominal
alignment. Before the first task proposal, verify the current backup command
and its continuation. An unsuccessful initial check latches an unrecoverable
initialization. Its clipped, frozen initial policy remains an emergency
command, with no witnessed-fallback claim. If no valid model exists, send a
zero-torque emergency command. Neither mode can count as controller success.

For each task proposal, require both the unchanged joint QP and the backup
witness. On acceptance retain that witness's immutable anchor, inverse mass,
drift and feedback law. On rejection send the previously retained feedback
at the current scheduled send and latch backup for the rest of the episode.
Do not replace its context with the newly rejected model. The FIFO is never
cleared. Retain raw and clipped fallback torques and every witness case,
including failed initial and proposal checks.

One command must precede exactly one observed transition. All five estimators
receive the actual sent torque, never the delayed delivered torque. Once
backup latches, command generation does not recompute nominal/model dynamics;
estimator updates continue from observed transitions.

## Fixed selection and reproduction

Use the original 480-record joint-filter archive, completion SHA-256
`e7f62d3732b20c9ef49dbdaa6777ebd3ee50672ae616d5ab18ebce411c0318c8`.
Take all twelve conditions at first seed `9805800`, plus all twenty tasks
with an unsafe `bank_filtered` record. There are 32 distinct scenario/seed
pairs, all within the already used `9805800..9805819` interval. Retain every
outcome and full substep trace.

Before executing any canonical candidate cell, reproduce both original arms
on all 32 tasks from unchanged production source
`6891dd27edf8b28514160d1ade350e3213826058` in a separate interpreter/checkout.
Compare all 64 original records using `rtol=1e-9`, `atol=1e-10` for finite
floats and exact discrete values/types. Preserve any mismatch and stop before
candidate execution. A reference replay has 300 intervals and 1,500 physical
integration calls, so reproduction retains 96,000 calls. The candidate adds
48,000 calls on the same 32 tasks. The canonical total is 144,000.

One preliminary first-seed `slow_clean_d00` run was retained locally before
this freeze, adding 1,500 diagnostic physical calls outside the canonical
budget. Its audit reuses saved states. No gain or horizon was changed from
the mathematical prototype. Completed-cell counts exclude repeated work from
an interrupted cell; preserve interrupted output rather than replacing it.

Commit implementation, tests and this protocol before the canonical run.
Record all Python package/tool source hashes and the source commit, fixed
selection, original completion identity and runtime. Use Python 3.12,
NumPy 2.5.2, SciPy 1.18.1 and MuJoCo 3.13.0, one worker and one numerical
thread with CUDA disabled. Original model provenance must match.
Completed output is never replaced. Resume requires the same source and
manifest; completed public records/traces audit without local cell shards.

## Independent audit and timing

The audit prohibits physical stepping. Reconstruct all 32 causal controllers,
their five estimators, noise, references, FIFO delivery, retained backup
context and aggregate diagnostics. Compare regenerated floating quantities at
the original tolerance. Check policy hashes against their retained arrays and
require unchanged hash identity across fallback steps. Hash strings themselves
are not compared between floating replays on different CPUs.

For every recorded witness, independently rebuild the sampled map from basis
trajectories. Solve its Lyapunov equation through a Kronecker linear system,
not the verifier's solver. Build the full seven-joint support metric and
recompute terminal radii. Execute each frozen-model continuation sequentially,
checking exact interval extrema, raw torque bounds, terminal energy and
earliest-event acceptance. Check feasible QP projections independently too.
Reconstruct measured physical limit/contact flags, maxima, final task outcome
and tracking RMS from saved states, without generating a replacement plant
trajectory.

Measure each complete control computation as command generation plus the
post-transition estimator update. Record all 300 samples per episode, median,
p99, maximum and every sample exceeding 10 ms. Exclude plant integration,
interface bookkeeping and serialization; record controller construction
separately. The earlier verifier-only timing does not establish this cycle's
cost. Simulation remains fixed-step and does not pretend to model a real
deadline miss as transport delay. Timing observations are not a hardware
real-time guarantee.

## Interpretation and commands

Report successes, aborts, emergency initializations, physical violations,
witness reasons and timing by condition, including failures. This selection
deliberately includes all earlier unsafe filtered cases; its aggregate is not
a population estimate or an independent comparison. The output always records
`candidate_advances: false`. The original bank and v2.6 release remain the
reference regardless of this integration result. A future comparison needs
its own frozen protocol, unused seeds and the original advancement criteria.

From the frozen integration checkout, with the separate original source:

```bash
python -m tools.run_panda_backup_integration --source-checkout ../SARRL-filter-source
python -m tools.run_panda_backup_integration --source-checkout ../SARRL-filter-source --resume
python -m tools.run_panda_backup_integration --source-checkout ../SARRL-filter-source --audit
```

The canonical output is `results/panda_backup_integration_v26`. Preserve the
earlier failure diagnosis, recovered traces, supplied banner and local models.
