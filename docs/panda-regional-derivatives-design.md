# Regional Panda mass derivatives

Status: tighter analytic derivative bounds for the ideal unconstrained Panda
model. This extends the [global geometry bounds](panda-analytic-region-design.md).
Physical-force, sensor/runtime and validated-arithmetic contracts remain
open, as does the nonlinear delayed terminal argument. No production
supervisor uses these calculations.
The later [center arithmetic audit](panda-center-arithmetic-design.md)
encloses point coefficients and reference discrepancies for an explicitly
normalized model. It does not yet validate this entire regional calculation.
The subsequent [outward regional evaluator](panda-regional-arithmetic-design.md)
recomputes the majorants under that model and covers whole-box coefficients;
it supplies no inverse or physical prefix witness.

## Differentiate the center kinematics

The [kinematic coefficient implementation](../sarrl/safety/panda_kinematic_coefficients.py)
loads a fresh hash-checked packaged model and the same supported-chain checks
as the global provider. It reads compiled geometry, COM masses/inertias,
gravity, constant armature and nominal viscous damping. It does not evaluate
MuJoCo forward/inverse dynamics, Jacobians or finite differences to obtain
its coefficients.
The supported topology contains the fixed base, seven moving link bodies
and the attachment only. The checks reject an altered base chain or added
bodies whose inertias would otherwise be omitted from the sum.

For each body, form a homogeneous local transform from its fixed offset and
rotation and its hinge rotation at `q-qpos0`. Differentiate the hinge
rotation algebraically through second order, including repeated derivatives
in one joint. At a parent/child product `T = P A`, propagate

```text
T_k  = P_k A + P A_k
T_kl = P_kl A + P_k A_l + P_l A_k + P A_kl.
```

COM position derivatives give the linear Jacobian `Jv` and its derivatives.
World joint axes are transformed into each body's frame to obtain the
angular Jacobian `W` and its derivatives. Ancestor masks come from the body
tree, independently of numerical derivative values. The local-coordinate
and hinge conventions follow the
[MuJoCo model reference](https://mujoco.readthedocs.io/en/3.13.0/XMLreference.html).

Accumulate each moving body's kinetic-energy contribution and its derivative:

```text
M_b = m_b Jv_b^T Jv_b + W_b^T I_b W_b
partial_k M_b = m_b [(partial_k Jv_b)^T Jv_b + Jv_b^T partial_k Jv_b]
                + (partial_k W_b)^T I_b W_b + W_b^T I_b partial_k W_b.
```

Include constant joint armature only in nominal `M`. The unit-payload column
uses the attachment COM and the loader's spherical unit inertia. Gravity
and its first derivative follow from the COM Jacobians and Hessians.
The Christoffel formula reconstructs the quadratic velocity force; nominal
damping is added once. These independently computed mass, gravity and bias
values support cross-checks against MuJoCo. The regional provider keeps the
existing MuJoCo center coefficients and controller law unchanged.

## Enclose a derivative over the entire position box

Let `G_ijk = partial_k M_ij(q_center)` and let the global tensor `D2_ijkl`
bound `|partial_l partial_k M_ij|`. For position radii `r_l`, the mean-value
inequality gives

```text
|partial_k M_ij(q)| <= |G_ijk| + sum_l D2_ijkl r_l.
```

Take the componentwise minimum of this bound and the existing global first
derivative bound. Both hold on the whole convex box in exact arithmetic.
The product-rule calculation at the center preserves cancellations before
taking absolute values; the Hessian remainder covers departure from that
center. It is not a derivative estimate fitted to sampled positions.

Apply the calculation separately to nominal and unit-payload mass matrices.
The [regional provider](../sarrl/safety/panda_regional_bounds.py) substitutes
these bounds into the existing state-region interface. Its Coriolis
velocity-derivative bounds also improve through the Christoffel expression.
Global mass Hessian and gravity-derivative bounds are retained, so the bias
position-derivative bound is unchanged. The remainder and state boxes still
discard correlations between coordinates.

The mathematical result requires exact center derivatives or a justified
allowance for their evaluation error. The implementation uses ordinary
floating-point arithmetic and supplies no validated allowance or directed
rounding. Regression tolerances do not establish that allowance. Physical
mass/force remainders and an assumption statement remain required caller
inputs; no force exclusion or runtime binding is inferred from a derivative.

## Reproduce the static comparison

```console
python -m tools.assess_panda_regional_bounds
```

This reruns the global diagnostic and evaluates regional bounds on exactly
the same five boxes. The position center is HOME, the velocity center is
zero, and all velocity radii are `0.01 rad/s`. Payload remains `[0.5,2.5] kg`
and added damping remains `[0.5,3] N m s/rad` per joint. The held torque is
nominal-plus-1.5-kg gravity compensation at the center. Zero physical
remainders describe this ideal unconstrained model calculation only.
There is no sensor, queue, prefix or terminal contract and no integration.
The report retains the model XML hash and both outcomes for every box.

Recorded with MuJoCo 3.13.0, NumPy 2.5.2 and SciPy 1.18.1:

| position radius (rad) | global inverse comparison norm | regional inverse comparison norm | regional max acceleration width (rad/s²) |
|---:|---:|---:|---:|
| 0.00000001 | 0.000014726 | 0.000000568 | 8.2098 |
| 0.000001 | 0.0014726 | 0.000056775 | 8.2119 |
| 0.0001 | 0.14726 | 0.0058121 | 8.4210 |
| 0.001 | 1.4726 | 0.070362 | 10.4225 |
| 0.01 | 14.726 | 1.9277 | unavailable |

At radius `0.001 rad`, the sufficient inverse condition now passes and the
ideal-model acceleration enclosure is available. At `0.01 rad` it still
returns `mass_variation_unverified`. Neither outcome establishes a physical
trajectory, a collision-free box, torque admissibility, delayed-prefix
containment or terminal invariance. These diagnostics change no gains,
uncertainty ranges, seed allocation or candidate advancement decision.

## Executed checks and next gate

Closed-form tests cover sine derivative bounds over entire intervals,
repeated-angle transform derivatives, monotonicity, invalid inputs and
immutable arrays. At three Panda configurations, independent MuJoCo dynamics
checks mass, gravity and velocity bias, while independent finite differences
check mass/gravity derivatives. The analytic evaluator runs with dynamics
and integration entry points prohibited. Finite differences are regression
checks, not the source of the regional guarantee.

Eight deterministic corner patterns check regional derivatives, coefficient
variation and direct MuJoCo-coefficient acceleration solves for six
payload/damping fixtures. A broader box remains inconclusive; an outside
query is rejected. No robot episode, task seed or training is introduced.

Before a Panda prefix campaign, establish a compatible force, sensor and
partial-interval runtime contract and validated coefficient/derivative
arithmetic. Check whether the resulting tubes fit the available model
regions. Broad-box refinement and a nonlinear delayed terminal argument
remain separate tasks. An independent witness auditor and causal handoff
must precede a separately frozen used-task protocol.
