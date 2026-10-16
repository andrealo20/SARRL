# Saved-state Panda prediction-gap methods

This post-hoc diagnosis follows the failed backup integration. Its eighteen
unsafe episodes and two zero-delay regressions have already been inspected.
It is not a preregistered confirmation study or a calibration dataset for a
new safety margin. All 26 aborted episodes are reviewed; all six ordinary
successful episodes remain in the input but have no latched fallback tail.
Detailed velocity curves are retained for the clean/noisy zero-delay
regressions at seed `9805800`.

Pin integration completion SHA-256
`351674a90849d64e7359fa101f72aee86ea1536ff2713b1b79d558b1d2f9831a`
and verify every input artifact hash. Select the last accepted policy from
its recorded origin, control step and SHA-256. Require its context to remain
unchanged throughout the latched backup. Never change the controller, gains,
horizon, task seeds or original acceptance decision.

## Three saved-state calculations

Start from the measured state at acceptance. Reconstruct the accepted
candidate and then its frozen feedback under the recorded constant delay.
The recorded age is privileged diagnostic information, not a controller
input. Predict at the retained 2 ms sample times using the frozen affine
acceleration and exact constant-acceleration intervals. The command FIFO
includes its preceding sent commands and startup zero torques.

Separately integrate that same frozen model with the actual saved delivered
torques. Compare both predictions with the saved physical velocity. At every
physical substep verify the telescoping identity

```text
v_physical - v_witness = initial measurement offset
                      + accumulated observed-model discrepancy
                      + accumulated model response to command differences.
```

The second term sums actual velocity increments minus the frozen-model
increments under recorded torques. The third sums `B (u_recorded - u_witness)`
times the substep. Require absolute closure within `1e-9 rad/s`. Their values
are algebraic components, not independent causal effects. Feedback reacts to
the physical trajectory, so its command difference can oppose model error.
Retain first-violation vectors, full-tail extrema, local one-period error and
the maximum closure residual. Check the recorded first violation lies after
policy acceptance. Predictions beyond the original finite witness horizon
continue the same unsaturated linear feedback; they are model calculations,
not replacement physical trajectories.

At every unsafe episode's first violating physical substep, also evaluate
three instantaneous accelerations under the saved applied torque: frozen
coefficients, refreshed physical state with retained estimated parameters,
and refreshed state with true payload/friction parameters. The last uses
privileged parameters solely in this offline calculation. Verify the retained
parameters reproduce the accepted coefficients at the acceptance measurement.
Report the state-replacement difference, parameter-replacement difference
and remaining observed difference as vectors. Their sum equals the observed
substep-average acceleration minus the frozen prediction. The remaining term
can include integration, damping discretization and within-substep dynamics;
it must not be labelled pure numerical error. State replacement also includes
the effect of initial measurement noise. No replacement is executed in closed
loop, and no unique failure mechanism follows from these point substitutions.

## Retention and verification

Freeze diagnostic implementation, tests and these methods before retaining
the canonical output. This freeze follows exposure to the original outcomes.
Retain source/input hashes, numerical runtime, all 26 reports and both focus
curves. Require NumPy 2.5.2, SciPy 1.18.1, MuJoCo 3.13.0 and unchanged Panda
model provenance. Prohibit physical stepping during calculation and audit.
The core trajectory identities are tested independently on synthetic linear
systems, including sensor offsets, FIFO delay and injected discrepancy.
Completed output is never replaced. A separate audit recomputes both reports
and curves from the source commit recorded in the manifest.

```bash
python -m tools.diagnose_panda_prediction_gap
python -m tools.diagnose_panda_prediction_gap --audit
```

Output is `results/panda_prediction_gap_v26`. No fresh seed, training run,
physical integration step or candidate advancement is part of this phase.
