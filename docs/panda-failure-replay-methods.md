# Recovery of retained Panda failure traces

This is a diagnostic reproduction of twenty already used episodes, not a new
candidate comparison or confirmation set. Select every `bank_filtered`
episode with a recorded physical violation in the original 480-row joint-filter
study. There are ten clean and ten noisy fast-move cases at 40 ms delay.
Their original completion marker is fixed by SHA-256 in the recovery command.
No seed range is opened or changed, and no controller setting is tuned.

## Frozen execution

Use production commit `6891dd27edf8b28514160d1ade350e3213826058` in a separate
checkout. Verify every recorded Python source hash and the model provenance
before importing it in an isolated child interpreter. That process runs the
original `run_episode(..., retain_trace=True)`. The recovery implementation,
tests and this protocol are committed before execution. Use one worker and
one numerical thread, with NumPy 2.5.2, SciPy 1.18.1 and MuJoCo 3.13.0.

Each full episode retains 300 control intervals, 1,500 physical integration
calls and 301 control-boundary frames. Each interval retains all five physical
substeps. Compare every regenerated episode field with the original, using
`rtol=1e-9`, `atol=1e-10` for finite floats and exact values/types for discrete
fields. Retain a mismatching cell and stop before the next case. Do not change
tolerances, selection, controller, braking or physics to obtain agreement.

The twenty completed replays contain 30,000 retained physical integration
calls. These are additional simulation steps on used seeds. They are separate
from the original campaign's budget and add no fresh-task evidence. An
interrupted cell can repeat on resume; retained-step counts describe completed
cells rather than all computation spent on interrupted attempts.

## Audit and event ordering

Only after all twenty records reproduce, reconstruct the first measured
physical violation from every substep. Locate it relative to the command
interval where braking latches and the interval where that braking command
first arrives through the retained FIFO. Record which previously sent command
is delivered during the first violating interval and which joints exceed
velocity limits. Samples at an interval's right boundary belong to the command
that generated that interval; classify by control-step identity, not rounded
timestamp comparisons.

A separate audit prohibits physical stepping and replays the original
controller, estimator, noise and reference against all twenty saved traces.
It checks recorded FIFO equality and substep continuity exactly, floating
replays at the existing tolerance, final estimator/filter/bank diagnostics,
and independently checks feasible QP projections. Reconstructed physical
violation maxima and flags must agree with every original episode.

Command provenance and event order do not establish the effect of a different
fallback or identify a unique cause. The original failed acceptance decision
and reference controller remain unchanged. Any new design needs its own
protocol and unused confirmation seeds.

## Commands

From the frozen recovery checkout:

```bash
git worktree add --detach ../SARRL-filter-source 6891dd27edf8b28514160d1ade350e3213826058
python -m tools.recover_panda_failures --source-checkout ../SARRL-filter-source
# Resume only an interrupted output with identical source and runtime:
python -m tools.recover_panda_failures --source-checkout ../SARRL-filter-source --resume
python -m tools.recover_panda_failures --source-checkout ../SARRL-filter-source --audit
```

Output is `results/panda_failure_replay_v26`. Original evidence is read-only;
completed recovery output is never replaced. Public records, compressed traces,
diagnostics, audit totals and completion hashes reproduce without local
resumable cell shards. Later audits require the recovery source commit recorded
in the manifest and the original frozen production checkout.
