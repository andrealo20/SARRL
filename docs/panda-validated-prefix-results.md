# Validated prefix assessment

The [frozen static screen](panda-validated-prefix-methods.md) does not establish
a Panda prefix for any constant lag. The combined numerical implementation
and structural checks are retained, but the candidate stops before production
integration, task comparison or release. No physical trajectory was generated.

Code and protocol were frozen in
`57ebc0f8aa31a92f32f8b251c7b7902c002315bb`. The complete
[assessment](../results/panda_validated_prefix/assessment.json) retains source
fingerprints, model identity, exact binary64 inputs, output intervals and all
five rejection reasons. It uses the full mass [0.5,2.5] kg and seven added
viscous-parameter intervals [0.5,3], 16 continuous payload cells and 128-bit
ball arithmetic. Gains and error assumptions follow the protocol unchanged.

| Position radius at HOME | Held-input linear solve | Largest acceleration-box width |
|---|---|---:|
| 0.0001 rad | Available | 9.850 rad/s² |
| 0.001 rad | Available | 11.047 rad/s² |
| 0.01 rad | Unverified | Not available |

Widths are rounded upward. The largest box rejects with
`validated_mass_solve_unavailable`. This is an inconclusive interval solve,
not evidence that the actual Panda mass matrix is singular. All matrix and
force arithmetic covers continuous parameter cells rather than sampled
mass extrema.

The two-interval fixture starts at HOME with zero velocity, one known gravity
candidate and four identical known prior sends. Each lag 0 through 4 rejects
before retaining its first interval with
`whole_interval_bounds_not_established`. The search cannot establish a tube
inside the declared position/velocity limits under its conservative bounds.
It establishes no actual physical limit crossing. All sensor, command and
propagation errors were already zero for this ideal-model screen; reducing
them cannot resolve this result. Bounds are not tuned on the failed fixture.

No Panda interval is accepted. The structural audit therefore checks zero
successful Panda intervals and reports `prefix_contained = false`; this is
not a positive robot-witness audit. Separate tests check fifteen successful
manufactured intervals across all five lags and reject altered records.
Three manufactured parameter fixtures are checked against independent dense
trajectories, and six static higher-precision Panda point solves lie inside
the smaller regional acceleration enclosure. These checks test implementation
behavior; they do not change the failed Panda admission decision.

All 150 focused numerical, prefix, point-model and release tests pass,
including 30 new cases. The canonical report is reproduced field for field
by the frozen audit command. The calculation uses Python 3.12.3, NumPy 2.5.2,
SciPy 1.18.1, MuJoCo 3.13.0 and python-flint 0.9.0. Every source fingerprint
must match before replay; no retained report is overwritten.

The earlier eighteen unsafe backup episodes and both new zero-delay regressions
remain unresolved physical evidence. The next design review must assess a
representation that retains more state/parameter/input dependencies before
another integration attempt. A verified nonlinear delayed terminal set,
physical force exclusions or remainders, compatible sensor/horizon treatment,
partial-runtime errors and an audited causal production handoff remain
prerequisites. No task seed, training, gain adjustment, parameter narrowing,
candidate promotion or new release follows this diagnostic. Version 2.6.0
remains the reference.

The separately frozen [parameter-preserving follow-up](panda-parameter-prefix-results.md)
uses direct local coefficients and retains the same continuous payload cells
throughout the prefix. Its finite numerical result does not alter the report
or the physical/terminal limitations recorded here.
