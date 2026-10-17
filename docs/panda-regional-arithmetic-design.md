# Outward Panda regional coefficients

Status: whole-box nominal and unit-payload coefficient enclosures for the
normalized compiled-constant model defined by the
[center arithmetic audit](panda-center-arithmetic-design.md). This extends
arithmetic control to global derivative majorants, regional derivative
bounds and coefficient variation. It does not evaluate an inverse matrix,
acceleration, torque admissibility, delayed prefix or terminal set.
The production controller and previous experimental evidence are unchanged.

## Keep the same model semantics

The [implementation](../sarrl/safety/panda_regional_arithmetic.py) uses the
same exact binary64 constants, normalized frame quaternions/hinge axes and
spherical unit payload inertia `2/5 r²` as the point evaluator. The compiled
constant signature must match the center's signature. Source integrity is
checked again after evaluation. The optional backend remains
`python-flint==0.9.0`; calls obey the same offline precision-context contract.

The result contains the actual center, position/velocity domain, enclosing
radii, mass/bias intervals and nonnegative derivative majorants. Arrays are
immutable. The MuJoCo compiler and physical model are not validated by these
arithmetic enclosures. Additional uncertain viscous coefficients, physical
force remainders, contacts and joint-limit responses are outside the result.

## Recompute global majorants outward

Compute link-offset and COM lengths and gravity magnitude as ball-valued
Euclidean norms. For a normalized inertial frame, the largest stored
principal inertia is the exact spectral-norm bound; selecting that binary64
component requires no rounded arithmetic. Apply the existing
[serial product-rule derivation](panda-analytic-region-design.md) using ball
addition, multiplication and division throughout the sums. This encloses
the first/second mass-derivative majorants and gravity-derivative majorants
for every angle in the unconstrained normalized model.

Export their upper endpoints outward to binary64. These are known upper
bounds, rather than nominal float estimates of a bound. The guarantees of
the [Arb scalar operations](https://python-flint.readthedocs.io/en/stable/arb.html)
carry numerical errors through the calculation. The derivative inequalities
and their implemented index assignments remain part of the trusted
mathematical argument; regression sampling does not prove them.

## Cover the domain and the rounded center

Form each midpoint from the exact rational average of its two binary64
endpoints, convert it to binary64 and verify that it belongs to the domain.
This avoids half-value underflow on subnormal singleton intervals. The
enclosing radius is an upward-rounded exact-rational calculation:

```text
r_i >= max(abs(lower_i-center_i), abs(upper_i-center_i)).
```

Using half the rounded interval width alone can miss the farther endpoint
when the midpoint rounds to one of two adjacent binary64 values. Both that
case and the subnormal singleton case have explicit regression tests.

Let `A_ijk` be the largest absolute endpoint of the enclosed center mass
derivative, and `H_ijkl` the outward global Hessian bound. Compute

```text
L_ijk = min(global_first_upper_ijk,
            upward(A_ijk + sum_l H_ijkl r_l)).
```

Both operands of the minimum are proven upper bounds. The selection never
compares uncertain ball midpoints. Apply the same process separately to
nominal and unit-payload mass. From these bounds construct Christoffel and
velocity-bias derivatives using the maximum absolute domain velocities,
global Hessian/gravity bounds and nominal damping. Every sum and product is
enclosed before exporting its upper endpoint.

## Expand point coefficients over the full box

For center interval `[M_low,M_high]`, expand each mass component by
`sum_k L_ijk r_qk`, outward on both sides. Expand the center bias interval
using its full position/velocity derivative majorants and enclosing state
radii. The mean-value inequality covers every point of the convex box,
including the departure from the actual stored center. Point coefficient
evaluation error is already included in the center interval.

`ArrayEnclosure.absolute_error(reference)` then bounds the difference from
any supplied numerical center array uniformly over this box. This includes
state variation and center-reference discrepancy. It does not bound
MuJoCo's coefficient evaluation at every state or establish a physical
model-error allowance. No empirical discrepancy is used to set a reserve.

## Reproduce the static audit

```console
pip install -e '.[mujoco,numerics]'
python -m tools.audit_panda_regional_arithmetic
```

The audit uses HOME, zero center velocity, velocity radii `0.01 rad/s` and
three previously declared position radii. Numerical references are nominal
and unit-payload MuJoCo coefficients at the stored center. The largest
componentwise uniform reference-error bounds are summarized upward below.
Payload columns concern the coefficient per unit payload mass.

| position radius (rad) | nominal mass | nominal bias | unit-payload mass | unit-payload bias |
|---:|---:|---:|---:|---:|
| 0.0001 | 0.000175324 | 0.0292965 | 0.0000541221 | 0.00474274 |
| 0.001 | 0.00191299 | 0.193471 | 0.000620534 | 0.0444351 |
| 0.01 | 0.0351053 | 1.83579 | 0.0141367 | 0.441358 |

Mass entries have inertia units and bias entries have torque units. These
are full coefficient-variation bounds, not rounding-error magnitudes alone.
Recorded with Python 3.12.3, NumPy 2.5.2, MuJoCo 3.13.0 and python-flint 0.9.0.
The JSON retains actual endpoints, centers, enclosing radii, source hashes,
compiled-constant identity and versions. There is no physics integration,
new task seed, training, uncertainty narrowing or candidate advancement.

Coefficient enclosure at radius `0.01 rad` does not supersede the earlier
inconclusive inverse comparison at that radius. This audit performs no
inversion and makes no claim that such a box supports a safe prefix.

## Verification and next gate

Tests check exact two-link product-rule sums and closed-form derivative
amplitudes, exact-rational radii, subnormal midpoints, input/majorant
rejection, zero-radius reduction, immutable arrays and reference-error
inequalities. Eight deterministic corner patterns have their full point
mass/bias intervals and mass derivatives checked against the regional
enclosure at higher precision. Model-constant/source mismatches and injected
evaluation failures reject without leaving the precision context changed.
Dynamics and integration calls are prohibited during coefficient evaluation.

These checks exercise the implementation; the analytic inequalities and
ball arithmetic establish coverage under the declared model. Next, validate
the inverse/linear-system calculation, mass-parameter composition, torque
boxes and held-interval prefix images. Then resolve physical-force,
sensor/horizon and partial-runtime propagation contracts. A nonlinear
delayed terminal argument, independent witness auditor and causal handoff
still precede any separately frozen used-task protocol or release.
