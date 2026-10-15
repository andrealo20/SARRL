# Retained control failure diagnosis

The [saved-data analysis](control-failure-methods.md) examines all 2,580
v2.6 episode records and all 45 retained first-seed traces without new
physical steps, task seeds or policy inference. The diagnostic implementation
was frozen at `2e88aa0c6284418859cf8ec37c7e25c4cfc5b81e` before output.
The [results directory](../results/control_failure_diagnosis_v26) retains
input/source hashes, reconstructed quantities and the complete per-trace reports.

## Planar success losses mostly become timeouts

Each row below pairs five residual models with the same adaptive reference
on 100 tasks. Counts are model/task pairs, with a shared baseline repeated
across models. They are not independent experimental replications.

| Scenario | Adaptive success lost | Lost to timeout | Lost to abort | Residual-only success gained |
|---|---:|---:|---:|---:|
| ID | 141 | 137 | 4 | 1 |
| Compound OOD | 51 | 47 | 4 | 9 |
| Motor fault | 106 | 100 | 6 | 0 |
| Total | 298 | 284 | 14 | 10 |

Timeouts account for 284/298 lost successes (95.3%). Aborts alone cannot
explain the registered success regression. Mean intervention fraction falls
by 5.01 percentage points on ID, rises by 2.11 points under motor fault and
by 9.89 points on OOD. More frequent projection is therefore not a uniform
explanation across scenarios.

All five failed residual first-seed traces end outside the 0.05 m target
radius with speed below the 0.35 rad/s success threshold. Their blocker is
distance, rather than excessive terminal speed. Nonzero late residuals are
consistent with a settling offset, but five traces do not establish that
mechanism across all timeout episodes.

| Scenario, model | Final distance | Final speed norm | Last-half mean residual, joints 1/2 | Projected/executed steps |
|---|---:|---:|---:|---:|
| ID, 9806001 | 0.217 m | 0.002 rad/s | +2.48 / +4.79 N m | 74/250 |
| ID, 9806003 | 0.076 m | 0.010 rad/s | -3.78 / +0.10 N m | 73/250 |
| Fault, 9806000 | 0.764 m | 0.055 rad/s | +7.41 / -6.67 N m | 250/250 |
| Fault, 9806002 | 0.688 m | 0.054 rad/s | -0.19 / +7.98 N m | 249/250 |
| Fault, 9806003 | 0.077 m | 0.011 rad/s | -3.79 / +0.09 N m | 61/250 |

In the two heavily projected fault traces, at least one raw residual joint
is within 5% of its bound on 107/250 and 194/250 attempts. This locates a
policy/projection interaction worth testing. It does not show that removing
the filter, reducing the bound or changing the reward would improve outcomes.
Residual RMS in the raw report is the RMS vector norm; the 8 N m action bound
applies separately to each joint.

Every first-seed reward and final distance reconstructs from the recorded
physical states and commands. Training adds a -500 abort-transition penalty;
evaluation sums executed-step rewards and records aborts separately. The
report makes that difference explicit. It is not a change to the original
success/safety decision, and raw training/evaluation returns should not be
treated as interchangeable. The policy also lacks estimator state and the
current nominal command in its observation. That is a design constraint,
not evidence that additional inputs would resolve the failure.

## Panda aborts precede QP construction

All 107 filtered fast-move aborts are `pending_command_violation`, between
control steps 5 and 13 (0.05 to 0.13 s). The filter rejects a predicted
pending-command interval before constructing the QP. No optimizer failure
accounts for these aborts. Ordinary moves never project or abort in the
retained 120 filtered episodes.

The 20 remaining filtered unsafe episodes are all velocity violations at
40 ms delay. No filtered episode has a position violation or contact.
This narrows the observed failure type, without establishing its cause.

For the six fast first-seed filtered traces, the same saved abort state
gives these maximum-delay constraint-construction results. Clean and noisy
traces have the same pattern:

| Recorded delay | Abort step | First maximum-delay bound rejected | Construction at recorded-delay bound |
|---|---:|---:|---|
| 0 ms | 6 | 3 periods | Reaches QP construction |
| 20 ms | 6 | 4 periods | Reaches QP construction |
| 40 ms | 11 | 3 periods | Pending-command rejection |

In four traces, the robust 0..40 ms assumption introduces a pending-prefix
rejection absent with the privileged recorded-delay bound. This demonstrates
conservatism at those saved states. No QP is solved, and narrower construction
does not establish feasible control or a safe alternative trajectory. The
recorded delay must not become a controller input.

All six fast filtered first-seed trajectories have zero measured physical
violations, including the two at 40 ms. Their paired unfiltered trajectories first violate velocity
limits at 0.084 s, 0.100 s and 0.116 s for delays 0, 20 and 40 ms. These are
illustrative paired records. The 20 filtered unsafe episodes belong to other
seeds for which complete physical substep traces were not retained. Their
violation timing relative to braking cannot be recovered from the available
episode summaries. It would be incorrect to attribute all of them to braking
delay or model error using this trace subset.

## Next design work

1. For the planar stack, specify one mechanism that preserves nominal settling
   while allowing residual authority during mismatch transients. Terminal
   attenuation or estimator-aware input are hypotheses, not accepted remedies.
   Choose one before a new protocol and independent seed block; do not tune
   several variants on these held-out outcomes.
2. Before redesigning Panda fallback, recover full traces for the already
   retained unsafe cases by a separately documented replay of the unchanged
   source. Require record reproduction before interpreting new trace fields.
   This would be additional simulation on used seeds, not fresh confirmation.
3. A future delay-aware filter needs a defensible causal delay set and explicit
   treatment of commands in flight. An accurate final delay estimate on the
   ordinary reference tasks does not establish that the set may be narrowed
   safely during the first 0.05 to 0.13 s of a fast move.

Neither candidate advances and no new training campaign starts from this
diagnosis. Version 3.0 integration remains conditional on a new controlled
comparison; the original v2.6 tag and reference controllers remain unchanged.
