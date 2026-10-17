# Geometry-derived Panda coefficient bounds

Status: analytic derivative bounds for the packaged unconstrained rigid-body
model, with a conditional [state-region provider](../sarrl/safety/panda_analytic_region.py).
The sensor, physical-force and discrete-runtime contracts remain open. The
structured backup has no nonlinear terminal argument or production handoff.

The bounds use link geometry and inertias, rather than maxima of sampled
derivatives. The packaged model is hash-checked before loading. Its seven
serial revolute joints have zero joint offsets, and a spherical unit payload
is attached to the final fixed body. The implementation checks these chain
assumptions and rejects springs, dry friction, fluid forces, gravity
compensation, tendons and model plugins. No supplied plant or randomized
parameter is read. The center coefficients come from fresh nominal and
unit-payload models; no physical integration is performed.

## Bound derivatives of the kinetic-energy coefficients

For body `b` and ancestor joint `i`, define the tail-length bound

```text
L_bi = ||COM_offset_b|| + sum_{r=i+1..b} ||link_offset_r||.
```

The payload includes the attachment offset. A COM position is a sum of
rotated constant offset vectors. Each joint angle occurs once in each
rotation product. Any positive-order derivative of a unit-axis rotation has
operator norm at most one, including repeated derivatives in the same angle.
For an ordered collection `S` of angle indices, therefore,

```text
||partial_S (Jv_bi)|| <= L_b,max({i} union S).
```

A derivative vanishes when an index is outside the body's ancestors. Empty
`S` gives the ordinary linear Jacobian bound. Write the angular Jacobian in
the body's frame as `W_bi`. Its norm and every nonzero derivative norm are at
most one. It depends only on descendant joints after `i`; a derivative in
an index at or before `i` vanishes. Fixed frame rotations preserve norms.

The body's mass-matrix contribution is

```text
M_bij = m_b Jv_bi^T Jv_bj + W_bi^T I_b W_bj.
```

For each first or second derivative, apply the product rule to the two
Jacobians. Sum every assignment of the ordered derivative occurrences to
the left and right factor, including multiplicities for repeated indices.
Each linear term is bounded by the product of the two tail lengths times
`m_b`. Each angular term is bounded by the largest principal COM inertia
when both assigned derivative collections can be nonzero. Summing bodies
gives componentwise nonnegative tensors `D1[i,j,k]` and `D2[i,j,k,l]`.
Constant joint armature has zero angle derivatives.

The nominal tensors use the seven moving link masses and COM inertias.
The payload tensors use unit mass and spherical inertia
`0.4 * 0.04^2 kg m^2`, with the same attachment used by the loader. These
are global angle bounds; joint boxes do not tighten them. They can discard
large cancellations between terms.

## Bound gravity, Coriolis and nominal damping

The same position-derivative bound gives

```text
|partial_k g_i| <= sum_b m_b ||gravity|| L_b,max(i,k).
```

Use the Christoffel representation of the quadratic velocity force:

```text
Gamma_ijk = (partial_k M_ij + partial_j M_ik - partial_i M_jk)/2
c_i = sum_jk Gamma_ijk v_j v_k.
```

The triangle inequality applied to `D1` bounds `|Gamma|`; the same combination
of `D2` bounds `|partial_l Gamma|`. For `|v_j| <= V_j`, this yields

```text
|dh_i/dq_l| <= G1_il + sum_jk Gamma1_ijkl V_j V_k
|dh_i/dv_l| <= sum_k (Gamma0_ilk + Gamma0_ikl) V_k + D_il.
```

Here `D` is nominal diagonal viscous damping. The payload column has no
damping. The seven uncertain additional viscous coefficients remain in the
existing structured parameter calculation, so they are not counted again
in the payload bias. The kinetic-energy equations are described in the
[MIT multibody notes](https://underactuated.csail.mit.edu/multibody.html).
MuJoCo's [continuous dynamics and passive-force conventions](https://mujoco.readthedocs.io/en/stable/computation/)
determine the signs in the existing nominal coefficient calculation.

## Build a conditional region

`analytic_panda_region` uses the position/velocity box midpoint for its
coefficient center and the maximum absolute domain velocities for `V`.
It returns the existing `PandaStateRegion`, which applies the mean-value
inequality and inverse comparison to query boxes.

Mass and force remainders and an assumption statement are required inputs.
The provider does not establish absence of collision or joint-limit forces
throughout a box. Those forces must be excluded or enclosed separately.
It also supplies no sensor or partial-interval integration-defect bounds.
Zero remainders are appropriate to the explicitly ideal unconstrained
coefficient diagnostic below, not a physical certification claim.

Both coefficient evaluation and the analytic formulas use ordinary floating
point. The existing padding is a numerical precaution; there is no directed
rounding proof, independently bounded coefficient evaluation error, immutable
runtime binding or validated interval arithmetic. These limits remain even
when a region query returns an acceleration box.

## Static conservatism check

Reproduce the diagnostic with:

```console
python -m tools.assess_panda_analytic_region
```

The center is `HOME`, zero velocity. All joint velocity radii are `0.01 rad/s`,
payload is `[0.5,2.5] kg`, and every added damping coefficient is
`[0.5,3] N m s/rad`. The fixed held torque equals the nominal-plus-1.5-kg
gravity compensation at the center. Physical model remainders are zero for
this ideal-model diagnostic. No sensor, queue, prefix or terminal contract
is asserted. All position coordinates use the same radius. The recorded
runtime uses MuJoCo 3.13.0, NumPy 2.5.2 and SciPy 1.18.1.

| position radius (rad) | inverse comparison infinity norm | largest acceleration-box width (rad/s²) |
|---:|---:|---:|
| 0.00000001 | 0.000014726 | 8.6243 |
| 0.000001 | 0.0014726 | 8.6288 |
| 0.0001 | 0.14726 | 9.1305 |
| 0.001 | 1.4726 | unavailable |
| 0.01 | 14.726 | unavailable |

The last two queries fail the sufficient inverse-comparison condition and
return `mass_variation_unverified`. This is inconclusive about the actual
mass matrix and physical safety. The bounds provide an analytic starting
point but lose too much dependence for these larger boxes. Before a robot
prefix campaign, tighten regional variation while retaining joint and
payload dependencies, then establish the missing force, sensor and runtime
contracts. A terminal proof and independent causal auditor still precede
production integration. No gains, parameter range, old evidence or seed
allocation is changed by this diagnostic.

## Verification

The tests compare the formulas with closed-form derivative amplitudes for
a two-link planar arm and an anisotropic rotor. They cover a zero-length
rotor, damping, invalid inputs, immutable copies and rejected model changes.
Packaged Panda checks compare independent finite differences at three states
with the analytic first-derivative bounds. Additional deterministic corner
patterns check coefficient variation and direct acceleration solves over
payload/damping endpoints and midpoint mass. These numerical checks can find
implementation mistakes; they do not establish a whole-domain bound. The
derivation above supplies that mathematical bound for the declared model.
No physical integration, new task seeds, training or release is introduced.
