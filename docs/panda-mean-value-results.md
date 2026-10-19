# Shared-parameter propagation results

Shared-parameter propagation verifies 281 intervals against 229 for the
preceding bounded-friction screen on the same fixture. Fifty-two branches
gain one interval, twenty-eight are unchanged and none loses an interval.
No branch completes all six requested intervals. This improves a numerical
enclosure result and does not admit the candidate to integration or release.

The corrected implementation and [protocol](panda-mean-value-methods.md) were
frozen in `f21fa2cd59d533b1b335c168fb366644e8bc7c67`. The initial complete
attempt stopped on an audit-record format error, as recorded in the methods.
The [compressed assessment](../results/panda_mean_value_prefix/assessment.json.gz)
retains every affine center, shared generator, residual, tube, derivative
bound, causal record, failure and model/runtime/source identity. Its SHA-256 is
`18746bc56a77f70bde364ec068f866b6c9316f57a09ea0df24c52ec3478c19d6`
(678,026 bytes). The preceding reports remain unchanged.

## Complete same-fixture comparison

Both methods use the bounded-friction context, fixed payload estimate 1.5 kg,
compensation 0.5, gains 25/10, sixteen continuous payload cells covering
[0.5,2.5] kg, seven full viscous intervals [0.5,3], the same known commands,
HOME/zero velocity and zero ideal-model errors. Limits, 10 ms control periods
and the 20-iteration Picard budget are unchanged. Each lag requests 96
intervals across sixteen branches.

| Constant lag | Prior admitted intervals | Shared-coordinate admitted intervals | Branches gaining one interval | New-feedback delivery, prior | New-feedback delivery, shared |
|---|---:|---:|---:|---:|---:|
| 0 periods | 38/96 | 47/96 | 9/16 | 16/16 | 16/16 |
| 1 period | 46/96 | 55/96 | 9/16 | 14/16 | 16/16 |
| 2 periods | 47/96 | 59/96 | 12/16 | 0/16 | 11/16 |
| 3 periods | 49/96 | 60/96 | 11/16 | 0/16 | 0/16 |
| 4 periods | 49/96 | 60/96 | 11/16 | 0/16 | 0/16 |
| Total | 229/480 | 281/480 | 52/80 | 30/80 | 43/80 |

Successful prefixes contain two or three intervals at zero lag and three or
four at positive lags. All eighty branches still stop with
`whole_interval_bounds_not_established`; no branch is omitted. The maximum
Picard count among accepted intervals is four. The remaining 199 requested
intervals are unverified, rather than measured violations. Newly computed
feedback is now delivered in every lag-one branch and eleven lag-two
branches, but none at lags three or four. A universal six-interval witness
and full delayed-feedback delivery therefore remain unavailable.

The method retains first-order physical-parameter correlations across states
and sent commands. It still uses independent residual enclosures and
whole-box derivative/partial-time bounds. At each branch's last successful
endpoint, the largest velocity residual radius exceeds the largest shared
generator radius by factors from about 8.27 to 50.04 (median 13.48). These
descriptive ratios come from the retained representation, not measured model
error or a calibrated safety reserve. They identify residual growth as a
remaining limitation without attributing it to a single bound or proving
that the physical trajectory is safe beyond the admitted prefix.

## Verification and decision

All 281 retained intervals pass the structural audit. It checks exact
parameter and shared-coordinate identities, all queue transitions, physical
tube domains, independent endpoint boxes and exact-rational partial-time and
Taylor-remainder inequalities. The independent Picard endpoint box and the
affine Taylor endpoint both enclose the actual endpoint; their box hulls need
not nest. Neither enclosure is clipped to force acceptance. Coefficient,
derivative and mean-value evaluations remain shared trusted primitives.

Thirty-one new tests pass, including exact manufactured trajectories for
five parameter fibers, all five lags and six intervals, derivative
cross-checks, modified-record rejection and the crossing-zero magnitude
regression. All 115 distinct focused checks pass across this increment and
the preceding numerical modules. The complete report is reproduced field
for field under Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1, MuJoCo 3.13.0 and
python-flint 0.9.0. Existing reports are not overwritten.

The numerical improvement is insufficient for admission. The prior
[local stability result](panda-terminal-admission-results.md) for the bounded
candidate remains a pointwise result. Nonlinear delayed terminal invariance,
force/contact/limit binding, a valid sensor/horizon contract, partial-runtime
error bounds and an audited causal production handoff remain open. The
untruncated Gaussian benchmark and numerical simulator are unchanged.

No physical step, task seed, training run, controller activation, candidate
promotion, task comparison or release is added. Version 2.6.0 remains the
reference. Before another numerical design, isolate the contributions of
mean-value derivative variation, independent residual propagation and held-flow
Taylor remainders; these selected diagnostics cannot certify a new safety
reserve or substitute for the missing physical and terminal arguments.
