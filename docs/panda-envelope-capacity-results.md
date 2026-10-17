# Panda backup error-capacity screen

All 26 retained backup policies fail the conditional scalar-tube criterion
at every constant lag from zero through four. The pointwise necessary error
floor already requires a radius 456 to 3,807 times the available radius.
This sufficient formulation does not proceed to finite-prefix integration or
a new task comparison. It supplies no impossibility result for other control
laws, uncertainty descriptions or invariant sets.

The [method](panda-envelope-capacity-methods.md) was frozen at
`0cc0e7e299e2c3798772d11177d78a52ce962934` before evaluating the static points.
The [retained evidence](../results/panda_envelope_capacity_v26) contains all
26 policy identities, 52 static parameter points, row-specific extremizing
commands, 416 direct forward checks and all 130 terminal calculations.
No new robot trajectory, integration step, task seed or training run is used.

## The necessary floor exceeds terminal capacity

| Constant lag | Feasible policies | Smallest `R_min / R_cap` | Largest `R_min / R_cap` |
|---|---:|---:|---:|
| 0 ms | 0/26 | 456.0 | 1,650.6 |
| 10 ms | 0/26 | 588.8 | 2,131.0 |
| 20 ms | 0/26 | 778.7 | 2,818.3 |
| 30 ms | 0/26 | 1,035.3 | 3,747.2 |
| 40 ms | 0/26 | 1,051.8 | 3,806.8 |

Every admissible radius is positive. Every rejection is
`disturbance_exceeds_terminal_capacity`, rather than an empty nominal
admissible set. The radius uses the augmented-state Lyapunov metric and has
no single physical unit; these ratios are dimensionless.

At the clean zero-delay regression's retained policy, the necessary
componentwise acceleration-error floor is approximately
`[22.77, 8.69, 28.51, 26.40, 76.23, 66.01, 15.33] rad/s²`.
For its zero-lag calculation, `R_min = 251.319` and `R_cap = 0.180151`.
The noisy episode's retained policy gives `250.561` and `0.181749` under
the same optimistic zero-sensor-error screen. This does not reinterpret the
episode's original noisy measurements as clean data.

## Why a larger domain calculation is not the next step

The static points use each retained anchor, zero velocity, two admissible
payloads and the full actuator torque box required by the present envelope
contract. A bound over the entire declared state/parameter domain must also
cover these points. Increasing that error bound cannot lower `R_min` or
increase `R_cap`. Shrinking the model domain cannot increase `R_cap`.
The screen already gives the formulation the largest available physical
domain, exact sensors and zero propagation defects.

Consequently a full-domain enclosure satisfying that contract cannot rescue
this particular sufficient test for these retained policies. This conclusion
does not require selecting a maximum observed trajectory error as a bound.
The numerical calculations are checked at fixed floating-point tolerances;
they are not interval-certified evaluations of MuJoCo.

The row extrema can occur under different torques and parameter points.
The componentwise box and scalar norm discard those dependencies. The
result therefore identifies a limitation of this conservative description,
not a measured disturbance acting simultaneously in every joint. A failed
sufficient inequality does not prove that every terminal set is empty or
that the physical plant lacks a safe backup.

## Verification and next design question

Nine new synthetic tests compare the pointwise formula with all 128 torque
vertices, check attaining commands and invalid inputs, and verify capacity
monotonicity as error increases or the domain shrinks. Together with the
existing tube and release checks, 38 focused tests pass. All static points
have no active contacts/constraints, preserve model time, and pass direct
forward-acceleration checks at `rtol=1e-9, atol=1e-9`. Source/input/output
hashes and a separate frozen-source audit make the screen reproducible.

The next candidate needs an uncertainty description that retains the
dependence on state, parameters and admissible backup inputs, with a
compatible terminal argument. Any narrower parameter/input set needs its
own causal justification and invariance checks. The existing RLS covariance
is not a calibrated parameter enclosure. No torque box or error bound is
reduced here to obtain a passing result.

The original references, failed integration, Gaussian-noise interface and
conditional prototype remain available. No candidate advances, no release
changes and no fresh seed block is opened.
