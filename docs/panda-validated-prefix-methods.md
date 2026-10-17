# Validated numerical prefix screen

This protocol consolidates continuous parameter solves, raw feedback bounds,
finite-prefix arithmetic and a structural auditor in one offline increment.
It does not introduce a task comparison or authorize production integration.
The earlier controller regressions and rejected candidates remain unchanged.

## Numerical model and solver

Use the [normalized compiled-constant model](panda-center-arithmetic-design.md)
and its [regional coefficient enclosures](panda-regional-arithmetic-design.md).
Compiled binary64 constants are exact inputs; frames and axes are normalized
in ball arithmetic. This model excludes contacts and joint-limit forces.
Physical mass/force remainders are explicit assumptions, separate from the
controller's nominal coefficient functions. Packaged damping is already in
the nominal bias; uncertain additional viscous damping is included once.

Partition the full payload interval into 16 adjacent exact rational cells.
Within every cell use the same continuous mass ball in `M0 + m Mp` and
`u - h0 - m hp - diag(b)v`. Solve this interval linear system with the
preconditioned [Arb solver](https://python-flint.readthedocs.io/en/latest/arb_mat.html#flint.arb_mat.solve).
The approximate solver option is never used. Any unverified or nonfinite
cell rejects the entire query. Union the outward binary64 answer boxes over
all cells. Correlations can be lost inside a cell; subdivision retains shared
mass localization, not an exact affine dependency representation.

No sampled mass extrema, ordinary generalized eigenvectors, numerical padding
or sample-fitted error reserve supplies the verification decision. Arb endpoint
export retains the exact rational rounding checks from the center evaluator.
The backend is python-flint 0.9.0 with 128-bit precision. Its global precision
must not be changed by unrelated concurrent code. All operations here are
offline, with no 10 ms execution claim.

Raw backup torque uses the immutable anchor, estimate and gains (25, 10) from
the [structured law](panda-structured-backup-design.md). Bounds cover fresh
measured-state boxes and add command implementation error once. The first
candidate and recorded history are known sent values. They receive no second
command error. Every subsequent send must fit the raw actuator limits;
clipping cannot turn a failed enclosure into an accepted one.

## Held intervals and independent structural checks

Recenter the regional coefficient evaluator on each trial state box. Bound
every partial interval by the [Picard images](panda-structured-prefix-design.md)
using Arb additions, products, time powers and error inflation. Interpret the
binary64 control period as an exact input; compute its square and division by
two in ball arithmetic. Export endpoints outward. Search inflation may be
trimmed only after the full reachable image satisfies domain and constraint
bounds. End boxes must also fit the admitted tube.

Keep all four oldest-to-newest queue slots. Enumerate every constant lag
0 through 4 with the same feedback law. Startup slots missing from history
are zero; no pending command is erased. Delay switching remains outside scope.

The auditor does not call the Picard search. It checks complete delay coverage,
first sends, FIFO delivery, actuator/state limits, completion flags and retained
endpoints. It checks the propagation inequalities using separate exact rational
arithmetic, including all-partial-time images. It recomputes force and torque
enclosures with the same trusted rigid-body provider. Thus queue and propagation
checks are independent of search, while model equations and the numerical
backend remain shared dependencies. This is not an independent physical
witness auditor or a production handoff.

## Fixed static cases and decision

Before executing this screen, freeze its code and this document in a commit.
No task seed is reserved or opened. Static HOME fixtures already appeared in
the preceding numerical diagnostics. An exploratory check with a 12-iteration
budget could not establish the first interval for any lag. The canonical
screen uses the existing 20-iteration default, without a subsequent budget
search. Its outputs are numerical diagnostics, not confirmation data.

1. At HOME and velocity center zero, evaluate the same three joint-position
   radii 0.0001, 0.001 and 0.01 rad, with velocity bounds +/-0.01 rad/s.
   Use mass [0.5,2.5] kg and each added viscous parameter [0.5,3] throughout.
   Hold the known torque computed by the existing regressor's HOME gravity
   estimate at 1.5 kg. Its difference from the normalized model is not hidden
   or used to change that model.
2. Request two 10 ms intervals for each constant lag 0 through 4. Start at
   HOME with zero velocity, the same candidate, and four known prior sends
   equal to it. The history is a declared algebraic fixture, not a recovered
   physical episode. Retain all successful intervals and every failure reason.
3. Use position domains HOME +/-0.5 rad, velocity domains +/-1 rad/s and motor
   limits [87,87,87,87,12,12,12] Nm. Sensor, command and partial-propagation
   defects are zero solely for the ideal continuous-model screen. Force/mass
   remainders are zero under the same assumption. These values do not bind
   noisy measurements or MuJoCo integration. Audit all retained intervals.
4. Preserve source fingerprints (all package Python files, assessment tool and
   this document, CRLF normalized to LF), model hashes, versions, exact input
   boxes, output boxes and failure reasons. A changed source rejects replay.
   Rerunning the frozen calculation must reproduce every retained field.

A finite-prefix pass is only a numerical prerequisite. Integration additionally
requires a terminal set in augmented position, velocity and pending-command
state for each constant lag. Under the same nonlinear law and uncertainty,
every state in that set must stay within the physical and raw torque limits
throughout a control interval and return to the set at its end. Every admitted
prefix must enter the corresponding set. The old frozen-model ellipsoid does
not prove this condition for the new state-dependent law.

No terminal set or physical binding is established in this increment. The
existing untruncated Gaussian benchmark supplies no finite deterministic sensor
bound. Contacts/limit-force scope, bounded partial-runtime discrepancies,
coefficient implementation error, nonlinear terminal invariance and causal
production handoff remain open. Labels or Boolean flags cannot validate them.
The decision therefore forbids integration, a task comparison, candidate
promotion and release, even if a finite numerical prefix passes. A future
comparison needs its own frozen protocol after these prerequisites are met.

Run the screen and exact replay with:

```bash
python -m tools.assess_panda_validated_prefix --output results/panda_validated_prefix/assessment.json
python -m tools.assess_panda_validated_prefix --audit --output results/panda_validated_prefix/assessment.json
```

The CLI refuses to overwrite a retained report. It executes zero physical
steps and uses no task seed. All learned models and earlier evidence remain
untouched. [Results and the resulting stop decision](panda-validated-prefix-results.md)
are recorded after the frozen execution.
