# Parameter-preserving prefix results

The [frozen numerical screen](panda-parameter-prefix-methods.md) verifies both
requested intervals in all 80 continuous-payload/constant-lag branches. Its
structural audit checks 160 intervals. The preceding method verified no first
interval on the same fixture. This is a finite 20 ms result for the declared
normalized continuous model, with zero ideal-model error assumptions.
It does not authorize production integration or release.

The corrected implementation and protocol are frozen in
`0601f7242fd076a423f085b2aeeaa3133b8e2881`. The earlier frozen run stopped on
a feedback-box replay discrepancy; its treatment is described in the methods.
The complete [compressed JSON assessment](../results/panda_parameter_prefix/assessment.json.gz)
retains all 16 parameter boxes, five lag cases per box, interval enclosures,
model/runtime identities, source fingerprints and every input. The original
negative [baseline report](../results/panda_validated_prefix/assessment.json)
remains unchanged.

## Complete finite-prefix result

| Constant lag | Payload cells | Admitted intervals | Largest position-tube width | Largest velocity-tube width | Maximum Picard iterations |
|---|---:|---:|---:|---:|---:|
| 0 periods | 16 | 32/32 | 0.003192 rad | 0.367031 rad/s | 4 |
| 1 period | 16 | 32/32 | 0.002758 rad | 0.261704 rad/s | 4 |
| 2 periods | 16 | 32/32 | 0.002758 rad | 0.261704 rad/s | 4 |
| 3 periods | 16 | 32/32 | 0.002758 rad | 0.261704 rad/s | 4 |
| 4 periods | 16 | 32/32 | 0.002758 rad | 0.261704 rad/s | 4 |

Widths are rounded upward and describe enclosure spans, not measured tracking
errors. Every cell retains the full seven added viscous intervals [0.5,3].
Together the cells cover the full mass range [0.5,2.5] kg. The estimate, anchor,
gains, constraints, control period and 20-iteration budget are unchanged.
No mass, friction, gain or sensor bound is selected to exclude a failed branch.

The initial state is HOME/zero velocity and all four known prior sends equal
the fixed candidate. Positive-lag results therefore coincide during these two
intervals: their delivered torques remain equal. This fixture does not exercise
delivery of newly computed feedback after a four-period delay. Arbitrary
histories, longer prefixes, other initial poses and noisy/discrete plants need
their own checked witnesses. No delay-switching guarantee is inferred.

Direct local coefficients retain signed derivative terms, and the persistent
parameter branches avoid combining a state reached under one payload with the
full payload range at the next step. Interval dependencies within a cell and
between state coordinates can still be lost. A complete finite pass on this
fixture does not establish a terminal invariant set.

## Static coefficient/solve comparison

| Position radius | Baseline largest acceleration width | Local largest acceleration width |
|---|---:|---:|
| 0.0001 rad | 9.850 rad/s² | 9.797 rad/s² |
| 0.001 rad | 11.047 rad/s² | 10.602 rad/s² |
| 0.01 rad | Unverified | Unverified |

These are full continuous-parameter solve enclosures with the same +/-0.01
rad/s velocity domain and held input. The largest static box still rejects
with `validated_mass_solve_unavailable`; this neither proves singularity nor
contradicts the successful much smaller state tubes used by the prefix.

## Verification and remaining gates

Twenty new tests cover continuous state-box coefficient inclusion, unchanged
point results, source/context handling, exact parameter coverage, all-cell
aggregation, record modifications and feedback measurement-box replay. All
170 distinct focused tests pass across the new and existing numerical modules.
The structural audit shares the coefficient/torque evaluator and checks queues
and partial-time propagation with separate exact rational calculations. It is
not a separate derivation of the rigid-body dynamics.

The report is reproduced field for field under Python 3.12.3, NumPy 2.5.2,
SciPy 1.18.1, MuJoCo 3.13.0 and python-flint 0.9.0. Source fingerprints and
the baseline identity are checked; retained reports are not overwritten.

The numerical prerequisite is met only for this two-interval ideal-model
fixture. Nonlinear delayed terminal invariance, force/contact/limit exclusions
or remainders, a compatible sensor/horizon guarantee, partial-runtime errors
and an audited causal production handoff remain open. The existing untruncated
Gaussian benchmark has no finite deterministic sensor bound. No physical step,
task seed, candidate promotion, training, task comparison or release is added.
The earlier eighteen unsafe backup episodes and two zero-delay regressions
remain unresolved. Version 2.6.0 remains the reference.
