# Panda center coefficient arithmetic

Status: an optional, offline enclosure of coefficients at one declared
position/velocity center. It addresses center evaluation error for the
[regional derivative design](panda-regional-derivatives-design.md). The point
evaluator does not enclose regional Hessians, inverse comparisons or prefix
propagation. This increment supplies no complete numerical or physical
backup certificate and changes no controller.
The subsequent [regional arithmetic implementation](panda-regional-arithmetic-design.md)
now encloses derivative majorants and coefficient variation over full boxes.
Inverse and prefix arithmetic remain separate requirements.

## Define the enclosed model

The [evaluator](../sarrl/safety/panda_center_arithmetic.py) loads a fresh
hash-checked packaged Panda and applies the supported fixed-base chain
checks. Every binary64 compiled mass, COM position, principal inertia,
armature, damping and gravity component is interpreted as an exact number.
Body/inertial quaternions and hinge axes are normalized inside the ball
calculation. This defines unit rotations and axes in exact real arithmetic.
The unit payload uses the attachment COM and spherical inertia `2/5 r²`,
where `r` is the exact binary64 value of `PAYLOAD_RADIUS`.

These semantics define a normalized compiled-constant rigid-body model.
They do not validate the model compiler, recover exact decimal XML inputs,
or bound physical identification error. The result records the model XML
hash, a canonical hash of all consumed compiled constants, the position and
velocity snapshot, precision and backend version. The diagnostic also hashes
its implementation files. Both coefficient values and their provenance can
therefore be inspected without an episode or plant object.

The previous ordinary evaluator uses the stored frame components directly
and rounded constants in its payload formula. Reference discrepancies can
include normalization and formula differences as well as arithmetic error.
They must not be described as a universal MuJoCo roundoff constant.

## Track arithmetic error through the center calculation

Install the optional numerical backend:

```console
pip install -e '.[mujoco,numerics]'
```

The dependency is pinned to `python-flint==0.9.0`. Its Arb real balls enclose
the exact results of operations on the input balls, including square roots
and trigonometric evaluation, subject to that library's guarantees.
See the [FLINT arithmetic specification](https://flintlib.org/doc/arb.html)
and [Python interface](https://python-flint.readthedocs.io/en/stable/arb.html).

Import inputs through exact binary64 integer ratios. At the default 128-bit
precision, normalize frames, evaluate hinge sine/cosine and propagate
homogeneous transforms and their first/second joint derivatives. Accumulate
COM and body-frame angular Jacobians, nominal/unit-payload mass matrices,
mass derivatives, gravity and Christoffel velocity bias. No MuJoCo dynamics,
Jacobian, finite-difference or integration call supplies these enclosures.
The same product-rule equations as the regional design define the calculation.

For each resulting ball, obtain lower/upper bounds directed outward at the
working precision. Convert those exact dyadic endpoints into rational
numbers, then binary64. Compare the converted value with the rational and
move by one representable number when needed; verify the final inequality.
Overflow or a nonfinite ball rejects the result. This export preserves
enclosure instead of silently rounding both ends to the nearest float.

`ArrayEnclosure.absolute_error(reference)` bounds the distance between any
value in the enclosure and the supplied numerical reference. It computes
endpoint distances as exact rationals and rounds their maximum upward.
This is a pointwise error bound relative to the defined normalized model,
not a tolerance fitted from discrepancies at other states.

The backend is an offline diagnostic. Its precision context is restored on
success and failure. A lock serializes its own calls; the process must not
change FLINT's global precision concurrently through unrelated code. No
10 ms control-loop timing or supported hardware interface is claimed.

## Reproduce the center audit

```console
python -m tools.audit_panda_center_arithmetic
```

The audit uses the same three static centers as the regional regression
checks: HOME plus offsets `0`, `0.2` and `-0.3` times `sin(arange(7))`, and
velocity `linspace(-0.2,0.15,7)`. It compares both nominal and unit-payload
enclosures against ordinary kinematics and direct MuJoCo coefficient values.
Its JSON retains all center values, source hashes and runtime versions.
It performs no physical integration and opens no task seeds.

The largest componentwise error bounds across those three centers are
summarized upward below. Each row concerns its own coefficient units; payload
columns are per unit payload mass.

| coefficient | ordinary kinematics reference | MuJoCo reference |
|---|---:|---:|
| nominal mass | 1.12e-15 | 3.11e-15 |
| unit-payload mass | 5.00e-16 | 7.78e-16 |
| nominal mass derivative | 8.89e-16 | not evaluated |
| unit-payload mass derivative | 4.45e-16 | not evaluated |
| nominal bias | 1.43e-14 | 2.85e-14 |
| unit-payload bias | 4.45e-15 | 7.11e-15 |

Recorded with Python 3.12.3, NumPy 2.5.2, MuJoCo 3.13.0 and
python-flint 0.9.0. These are bounds for the reported centers and references.
They do not calibrate a global model, sensor or integration reserve.

## Verification and remaining numerical work

Tests check endpoint direction with exact rationals, positive/negative
subnormals, rounding at the binary64 integer boundary, overflow, invalid
arrays and references, immutable inputs, missing/wrong backend versions and
nonfinite balls. At the three Panda centers, 128-bit and 256-bit enclosures
overlap; exact-rational checks verify the exported reference-error bounds.
Dynamics and integration entry points are prohibited during enclosure
evaluation. Context restoration is checked after an injected failure.
The higher-precision overlap is a regression check; the enclosure guarantee
comes from the arithmetic and declared equations.

The regional companion described above encloses global Hessian/gravity
bounds, combines center derivative intervals and remainders outward, and
bounds differences from numerical center references. The generalized
eigensolve, inverse comparison, linear solve, torque boxes
and Picard images also need validated treatment. Inserting the reported
point errors into the existing floating-point engine would not complete
that work. Physical-force, sensor/runtime, terminal and independent-audit
requirements remain separate. No candidate advances and no release starts.
