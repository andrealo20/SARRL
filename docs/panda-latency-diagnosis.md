# Panda latency diagnosis

This analysis replays the retained score-memory inputs before choosing a
new controller comparison. It opens no new episode seeds and performs no
physical simulation steps. The cumulative bank remains the reference.

## Questions and scope

The retained pilot reached every task with all three bank variants. Final
delay selection differed, and finite score memory increased tracking RMS
in every noisy changing condition. The next diagnosis separates three
possible mechanisms:

- Scores retain innovations from the initial delay profile.
- Candidate parameter fits absorb part of a command-alignment error.
- The moving and settling trajectory phases provide different information
  about payload and friction.

All 48 retained bank traces are included: three arms across sixteen
conditions, using the protocol's first seed, `9805300`. These traces cover
one task and one parameter draw. The 1,920 episode records provide the
existing pilot totals; a trace-level observation is not a population estimate.
Clean/noisy and cold/warm conditions remain separate.

## Frozen analysis

The analysis implementation is committed before execution. Input completion
hashes, frozen controller sources, model provenance and the recorded
NumPy/MuJoCo versions must match. Every bank is replayed causally from measured
states and sent commands. Selected parameters, final candidate estimates,
scores, bound hits and the entire selection history must reproduce the
retained records. No delivered torque or true parameter enters that replay.

Retrospective diagnostics use the recorded applied ages and true parameters
only after replay validation:

1. Reset cumulative score accumulation at transition 100, while retaining
   each candidate's original pre-update innovations. This isolates score
   persistence on the saved history. The reset knows the switch time and
   is a privileged diagnostic.
2. Compare each candidate's innovations with residuals evaluated using the
   same true parameter vector. Fit a separate bounded least-squares model
   within each phase, plus a known-applied-age diagnostic. These fits use
   future samples within their phase and are not online controllers.
3. Report design-column norms, rank and singular values from measured and
   recorded physical states. Also report column-normalized conditioning.
   Physical-state regressors are privileged diagnostics. Sensor noise can
   change measured rank or conditioning without adding physical excitation.

The phases are transitions `[0, 100)`, `[100, 200)` and `[200, 300)`.
They cover initial motion, motion after the scheduled latency change and
settling. Constant-profile conditions use the same splits as controls.
Actual applied ages are retained for each phase because packet delivery
does not change instantaneously at the scheduled switch.

Suffix selection and phase fits keep the original commands and measured
states. A different online selection would change later commands and
trajectories, so these diagnostics cannot establish tracking improvements.
Rank and conditioning do not establish calibrated parameter uncertainty.

## Reproduce

Use the recorded numerical runtime and a clean repository checkout:

```bash
python -m tools.diagnose_panda_latency \
  --input results/panda_score_memory_pilot \
  --output diagnostic-output/panda-latency
```

The output records source hashes, the analysis commit, runtime, input hashes,
per-trace diagnostics and completion hashes. Existing outputs are refused.

## Retained findings

Execution from frozen analysis commit `98134bd` reproduced every bank's
selected parameters, candidate estimates, scores, bound hits and selection
history in all 48 traces. It performed zero physical simulation steps and
opened no new seeds. The [manifest](../results/panda_latency_diagnosis/manifest.json),
[diagnostics](../results/panda_latency_diagnosis/diagnostics.json) and
[completion hashes](../results/panda_latency_diagnosis/complete.json) are retained.

The following counts cover the eight changing-profile traces per arm,
all from the same first seed. Suffix scores are cumulative sums starting
at transition 100 on each arm's original innovation history.

| Original arm | Correct final age | Correct final age with suffix scores |
|---|---:|---:|
| Cumulative | 1/8 | 6/8 |
| Exponential | 5/8 | 6/8 |
| Window | 5/8 | 6/8 |

Suffix scoring still selects the wrong final age in both clean 40-to-10 ms
conditions for every arm. In all six of these traces, the correct candidate
also fails to minimize mean pre-update innovation in both post-switch phases.
Using the shared true parameter vector or fitting parameters separately
within each phase instead ranks the final applied age correctly in every
changing trace, in both motion and settling (48 phase rankings per diagnostic).
Inherited candidate fits therefore warrant a separate comparison. These
observations do not establish that parameter forgetting improves control.

The physical-state design remains full rank in those post-switch phases.
Its column-normalized condition ranges from 1.15 to 3.23, while raw condition
ranges from 574 to 23,743. The combined norm of its seven friction columns
drops during settling to a median 1.33% of the preceding moving phase
(range 0.47% to 2.95%). Normalized conditioning alone would miss this loss
of absolute friction information. The present traces cannot establish
the result across other task or parameter draws.

## Next comparison

Retain the cumulative reference. The next targeted comparison should hold
the score-memory rule fixed and change candidate estimator memory, with
constant-profile controls. This isolates the inherited information update
from another score-length choice. A privileged switch reset remains a
diagnostic, not an online controller feature.

The next protocol must freeze the controller change, constant-profile
controls, paired seeds, tracking and identification metrics, and promotion
criteria before opening new episodes. Improving final delay selection alone
does not justify replacing the reference. Confirmation uses a separate
seed block from diagnostic development.
