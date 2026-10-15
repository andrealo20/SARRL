# Saved-data control failure diagnosis

This analysis is retrospective. It uses all 2,100 planar and 480 Panda
episodes from v2.6, plus every retained first-seed trace (21 planar and 24
Panda). Their completion-marker hashes are fixed in the runner. No training,
new task seed, policy selection, physical step or candidate evaluation is
performed. Source, runtime, input identities and output hashes are retained.

## Planar analysis

For each scenario, pair all five policies with their adaptive baseline on
the same task. Report success gains/losses, the outcomes of lost successes,
and descriptive changes in reward, intervention fraction and steps. These
are model/task pairs with a repeated shared baseline, not independent trials;
no new confidence interval or acceptance decision is computed.

For all 21 traces, reconstruct reward from physical next states, the saved
target joint configuration and unit link lengths: distance, squared velocity,
sent-torque cost and success bonus. Verify the reconstructed total and final
distance against the episode. Count near-bound residual attempts (at least
95% of the registered 8 N m bound), projection interventions on executed
transitions, and target-radius visits with excessive speed. Classify timeout
endpoints by distance and speed. Report last-half mean residuals as descriptive
features; they do not establish a causal mechanism across the task population.

Training adds -500 on an abort transition. Held-out evaluation records only
executed physical-step rewards and scores aborts separately. Report an
explicit penalty-adjusted diagnostic return to expose that convention;
do not compare unadjusted training and evaluation returns as one objective.
The frozen v2.6 success/safety decision remains unchanged.

## Panda analysis

Summarize every filtered episode's recorded rejection reason, abort step and
physical violation type. On all first-seed traces, reconstruct position,
velocity and contact violations from every physical substep and compare
maxima and unsafe flags with the original records. Locate first physical
violations relative to the command step where braking latches. Commands
already in flight may continue after that event; ordering alone does not
identify which command or estimation error caused a violation.

At each retained filtered abort, reconstruct estimated affine acceleration
from the recorded measured state and pre-update parameters. Rebuild constraint
construction for maximum delay bounds 0, 1, 2, 3 and 4 on that same saved
state and command history. Verify that bound four reproduces the rejection
reason. Also report the bound corresponding to the recorded true delay.
True delay is privileged diagnostic information, never a controller input.
Reaching QP construction is not proof of QP feasibility, safe braking,
successful tracking or a successful alternative trajectory. No QP is solved
and no counterfactual plant trajectory is generated.

The recorded Panda runtime (NumPy 2.5.2, SciPy 1.18.1, MuJoCo 3.13.0)
reconstructs model quantities. The runner prohibits `mj_step`. Trace arithmetic
uses no policy inference. The analysis is committed before canonical output
is produced; existing diagnostic output is never overwritten.

## Run and audit

From the frozen diagnostic checkout, with one numerical thread:

```bash
python -m tools.diagnose_control_failures
python -m tools.diagnose_control_failures --audit
```

Output is `results/control_failure_diagnosis_v26`. Later audits must use the
analysis commit recorded in its manifest, because source hashes are checked.
Public reproduction requires the original retained data and simulator model,
but no local policy tensors. Findings may motivate a new protocol on unused
seeds. They cannot confirm a design developed from these same held-out data.
