# Local coefficients and persistent payload branches

The [preceding screen](panda-validated-prefix-results.md) verified no Panda
interval. This design changes the numerical enclosure representation while
preserving the controller, full uncertainty, constraints and decision gates.
It remains an offline diagnosis on already used static fixtures.

## Direct state-box coefficients

The existing normalized-model kinematic evaluator accepts state balls through
an explicit internal argument. Its point-input path is unchanged. The new
provider constructs these balls from outward endpoint unions and evaluates
forward transforms, first/second kinematic derivatives, mass, mass derivatives,
gravity and the signed Christoffel sums over the entire state box.

In particular, the bias is evaluated as

```text
h_i(q,v) = g_i(q) + damping_i v_i + sum_jk Gamma_ijk(q) v_j v_k
Gamma_ijk = (M_ij,k + M_ik,j - M_jk,i)/2.
```

The negative term and actual local mass derivatives are retained until the
ball calculation. Global absolute gravity/Hessian majorants no longer replace
these coefficient functions. Repeated occurrences still lose dependencies
in ball arithmetic; tighter results are not assumed for every domain.
No derivative or error bound is fitted to samples. No MuJoCo dynamics,
Jacobian, finite-difference or step call supplies the coefficient enclosure.

The model semantics match the [center evaluator](panda-center-arithmetic-design.md):
exact compiled binary64 constants, normalized frames/axes and exact sphere
inertia formula. The new path checks packaged model/source identity, preserves
the precision context and exports all endpoints outward. This does not enclose
MuJoCo compilation, contact/limit forces, physical identification uncertainty
or its discrete runtime.

## Keep the same parameter branch through the prefix

The earlier engine subdivided mass only inside each acceleration query, then
merged all answers before propagation. The next step therefore combined states
reachable under one payload with the dynamics of another payload. This was a
valid outer enclosure but could grow too quickly to establish containment.

Partition the original continuous mass range into the same 16 exact rational
cells. Round subbox endpoints outward, retaining overlap where binary64 cannot
represent a boundary. Keep the full seven viscous-parameter intervals in every
cell. Run the complete finite prefix and all five constant lags within each
cell, with one continuous mass cell per inverse solve. Aggregate only the final
containment flags; keep the state and FIFO records separate throughout.

Every cell uses the same immutable anchor, estimate and feedback function.
Physical states and resulting feedback boxes may differ. The controller never
receives the true mass or delay. Covering all cells is a universal numerical
check, not an oracle selector. Friction-state dependencies and within-cell
mass/state dependencies can still be lost; no exact Taylor-model claim is made.

The parameter auditor reconstructs the rational partition, checks full mass
and friction coverage, rejects missing cells/lags and independently replays
each branch's causal queue and rational propagation inequalities. It reuses
the coefficient/torque provider. Thus it does not independently derive the
rigid-body equations or certify physical/runtime assumptions.

## Frozen diagnostic comparison

Freeze implementation and this protocol before executing the complete screen.
The baseline report is retained unchanged with normalized-LF SHA-256
`cd67880ee7ad6e578a8fee33c158f73d18db98b3b35b7750f5ef6713df3fad46`.
Read its exact inputs rather than reconstructing rounded numbers from prose.

Repeat all three static position domains (radii 0.0001, 0.001, 0.01 rad;
velocity +/-0.01 rad/s). Compare the direct local coefficient solves with the
retained baseline availability and intervals. Static queries retain 16 inner
continuous mass cells. These are diagnostic comparisons, not new task data.

For the prefix, retain two 10 ms intervals, every constant lag 0 through 4,
20 Picard iterations, the same inflation settings, initial HOME/zero velocity,
known gravity candidate and four recorded equal prior sends. Keep mass
[0.5,2.5] kg, each added viscous parameter [0.5,3], position domains HOME +/-0.5
rad, velocity +/-1 rad/s, torque limits and all zero ideal-model error/remainder
assumptions unchanged. The only parameter-partition change is to carry those
16 cells through the entire prefix instead of merging their states every step.

Development probes used the already observed HOME fixture: the direct provider
verified one interval at zero delay before its second interval was inconclusive;
three retained payload subranges (.5-.625, 1.5-1.625 and 2.375-2.5 kg) then
verified two intervals at zero delay. These are exploratory method checks, not
independent confirmation. No cell count, budget or threshold is selected after
the complete frozen comparison. Report every failed branch.

Store all 16 parameter boxes, all five lag records per box, successful interval
boxes, failure reasons, exact input definitions, runtime/model identities and
source fingerprints in a compressed JSON report. Hash every package Python
file, this tool and this protocol with CRLF normalized to LF. The CLI refuses
to overwrite a report; an audit must reproduce every field under the same
frozen source and runtime. Earlier results remain auditable at their own
frozen source, not under a modified evaluator.

```bash
python -m tools.assess_panda_parameter_prefix --output results/panda_parameter_prefix/assessment.json.gz
python -m tools.assess_panda_parameter_prefix --audit --output results/panda_parameter_prefix/assessment.json.gz
```

A complete numerical prefix requires two admitted intervals in all 80 cell/lag
branches and a successful structural audit. This only establishes finite-time
containment for the declared normalized continuous model under its assumptions.
No physical step or task seed is executed. Even a complete pass cannot authorize
integration, a task comparison, candidate promotion or release. Nonlinear delayed
terminal invariance, force/contact/limit scope, compatible Gaussian sensor/horizon
treatment, partial-runtime defects and a causal production handoff remain open.
The preceding unsafe episodes and zero-delay regressions are not superseded.

The [retained results](panda-parameter-prefix-results.md) record the complete
screen and its limited numerical scope.
