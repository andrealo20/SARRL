# Terminal admission results

The retained structured backup is locally unstable at an admissible rest
equilibrium for every constant lag zero through four. Limiting its viscous
compensation removes this point instability, but neither arm establishes the
requested six-interval prefix over the full physical parameter box. Neither
candidate advances to integration, a task comparison or release.
The subsequent [shared-parameter propagation study](panda-mean-value-results.md)
keeps the bounded context and physical uncertainty, changing only its offline
enclosure representation.

The [protocol and implementation](panda-terminal-admission-methods.md) were
frozen in `ba62a182321687bbefd0b5194d9d93f41075edff` before the complete run.
The [compressed assessment](../results/panda_terminal_admission/assessment.json.gz)
retains all matrix/eigenvalue enclosures, both contexts, all eighty branches
per arm, successful intervals, failure reasons, source fingerprints and model
identities. Its SHA-256 is
`a909e5e551bb3badb59a690c6a999ad41524fe1ef0f729a6218a08c46f655bec`
(238,817 bytes). All preceding reports and models remain unchanged.

## Certified sampled equilibrium result

At HOME/zero velocity, physical and estimated payload mass both equal 1.5 kg.
All seven physical added viscous coefficients equal 0.5, within the original
[0.5,3] intervals. The retained compensation is 1.75. The separate bounded
candidate compensates 0.5, the declared physical lower bound, with the same
mass estimate, anchor and gains. Both have strict unsaturated equilibrium
torque margins. Every multiplier is isolated using validated Rump arithmetic.

| Constant lag | Retained spectral-radius enclosure | Bounded-compensation spectral-radius enclosure |
|---|---|---|
| 0 periods | [1.009020099963, 1.009020099964] | [0.956932145660, 0.956932145661] |
| 1 period | [1.008742441765, 1.008742441766] | [0.960097028819, 0.960097028820] |
| 2 periods | [1.008454481131, 1.008454481132] | [0.961932528260, 0.961932528261] |
| 3 periods | [1.008179501192, 1.008179501193] | [0.963265897818, 0.963265897819] |
| 4 periods | [1.007925137015, 1.007925137016] | [0.964721490263, 0.964721490264] |

Displayed endpoints are rounded outward. These are exact held-flow sampled
Jacobian bounds for the declared normalized continuous model. The retained
law has two certified unstable multipliers at lags zero through three and
four at lag four. Excess viscous compensation can remove damping in small
inertia modes. The separate cap removes this mechanism and meets the Schur
condition at this point for every lag.

The instability refutes local asymptotic stability around this equilibrium.
It does not establish physical limit violations, rule out every larger
invariant set, or apply to the deployed reference controller. The bounded
candidate's pointwise result does not certify nonlinear terminal invariance,
the continuous mass/friction box, or a causal estimator-acquisition history.

## Complete six-interval screen

Each arm requests 16 continuous payload cells times five constant lags times
six intervals, or 480 intervals. Every cell retains all seven full friction
ranges. The uncertainty, 20-iteration Picard budget, limits and ideal-model
errors remain identical to the earlier two-interval fixture.

| Constant lag | Retained admitted intervals | Bounded admitted intervals | Retained branches delivering new feedback | Bounded branches delivering new feedback |
|---|---:|---:|---:|---:|
| 0 periods | 37/96 | 38/96 | 16/16 | 16/16 |
| 1 period | 46/96 | 46/96 | 14/16 | 14/16 |
| 2 periods | 47/96 | 47/96 | 0/16 | 0/16 |
| 3 periods | 49/96 | 49/96 | 0/16 | 0/16 |
| 4 periods | 49/96 | 49/96 | 0/16 | 0/16 |
| Total | 228/480 | 229/480 | 30/80 | 30/80 |

No branch admits all six intervals. Every branch stops after two to four
successful intervals with `whole_interval_bounds_not_established`.
This is an inconclusive enclosure calculation, not a measured constraint
violation or proof that the physical trajectory escapes. No failed branch
is omitted and no bounds, gains or budgets are changed after observing it.
The structural audits check all 457 successful interval records, exact
parameter coverage, measurement boxes, queues and partial-time propagation.
They share the coefficient/torque evaluator and are not an independent
physical derivation of the dynamics.

The [earlier twenty-millisecond result](panda-parameter-prefix-results.md)
remains valid at its frozen source. The longer request does not complete
delivery of newly computed feedback under lags two through four. Limiting
friction compensation alone therefore does not resolve the universal prefix
prerequisite.

## Verification and decision

All 158 focused tests pass, including 34 new stability, queue, compensation,
snapshot-equivalence and report-guard checks. The complete report is
reproduced field for field with Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1,
MuJoCo 3.13.0 and python-flint 0.9.0, under the frozen source and protocol.
The replay does not overwrite the retained report.

The original candidate fails the local stability screen. The separate cap
passes only the pointwise screen and fails full-prefix admission. Neither
provides a nonlinear delayed terminal set. Force/contact/limit binding,
partial-runtime discrepancies, a compatible sensor/horizon contract and the
causal production handoff remain unverified. In particular, untruncated
Gaussian noise admits no finite deterministic bound, and the simulator's
2 ms `implicitfast` integration is not the exact continuous flow analyzed here.

This assessment executes zero physical steps and opens zero task seeds.
It changes no training, estimator bank, simulator, production controller or
release. The eighteen unsafe backup integration episodes, including two
zero-delay regressions, remain unresolved. Version 2.6.0 remains the reference.
