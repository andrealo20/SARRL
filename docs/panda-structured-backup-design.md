# State-dependent Panda backup formulation

Status: pointwise algebra and continuous-parameter enclosure prototype. This
defines a different feedback law from the rejected frozen-coefficient backup.
No existing witness or terminal metric certifies it. No production controller
uses it and no task comparison starts here.
The subsequent [finite-prefix engine](panda-structured-prefix-design.md)
implements conditional interval and FIFO propagation. It still requires a
Panda-specific region provider and a new terminal argument before integration.

The [previous screen](panda-envelope-capacity-results.md) rejected one
global torque-box/error-box/scalar-radius sufficient criterion. This design
retains the shared physical parameters and evaluates the backup's own torque
at the current state, rather than independently maximizing every actuator.

## Fixed parameter context, current-state dynamics

For the packaged payload/viscous model, write

```text
M(q,m) = M0(q) + m Mp(q)
h(q,v,m,b) = h0(q,v) + m hp(q,v) + diag(v) b.
```

The nominal term includes the packaged passive forces. The existing
regressor supplies the payload column and the seven viscous coefficients.
This acceleration model assumes no active contact or joint-limit force.
Constraint responses would add a generalized-force term and are outside
the present point enclosure.
Freeze the anchor `q_a` and estimate `(m_hat,b_hat)` at policy acquisition;
evaluate the coefficients at every new measured state. The proposed law is

```text
a_d(q,v) = -25 (q-q_a) - 10 v
kappa(q,v) = M(q,m_hat) a_d(q,v) + h(q,v,m_hat,b_hat).
```

This law refreshes state dependence while keeping the parameter context
fixed. It changes the earlier law, which froze mass and bias coefficients.
It cannot inherit an accepted trajectory or invariant set from that law.
The point API requires caller-supplied coefficients for its stated q/v;
production provenance binding remains unimplemented.

## Keep the common mass and friction parameters

At one exact state, when delivered torque equals `kappa(q,v)`, acceleration
obeys the identity

```text
a = a_d + M(q,m)^-1 [ (m_hat-m) (Mp a_d + hp) + diag(v) (b_hat-b) ].
```

The same mass occurs in both the inverse and the payload error. The same
friction vector affects every joint through that inverse. The full existing
parameter box remains `m in [0.5,2.5] kg`, `b_j in [0.5,3.0] Nm s/rad`.
No RLS covariance is interpreted as a smaller uncertainty set. Parameter
truth and delivered command age are not inputs to the proposed law.

For a pointwise enclosure, use generalized eigenvectors `E` satisfying
`E^T Mc E = I`, `E^T Mp E = diag(lambda)`, where `Mc=M0+m_c Mp` and
`m_c` is the midpoint of the mass interval. Then

```text
M(m)^-1 = E diag(1 / (1 + (m-m_c) lambda)) E^T.
```

Require positive denominators across the entire mass interval. With friction
at its midpoint, each modal residual is an affine numerator divided by a
positive affine denominator, so its extrema occur at the interval endpoints.
Bound the remaining friction deviations using interval inverse entries and
their row supports. Partition mass into 16 fixed cells and take the union of
their enclosures. This improves dependency bounds without treating a finite
mass grid as a continuous-parameter proof. Tests compare a grid independently;
the grid is not the derivation. Floating-point eigensolves and padding are
not machine-checked interval arithmetic.

The implementation is [panda_structured_backup.py](../sarrl/safety/panda_structured_backup.py).
Its output is a pointwise parameter interval, not an enclosure over uncertain
state, sensor noise or a full control interval. Its raw torque is reported
without clipping. The interval API requires an explicit delivered-minus-law
torque difference. Zero represents ideal same-state delivery; a supplied
point difference accounts for clipping or a known alternative command.
This input does not establish the contents or evolution of a command queue.

## Delayed delivery adds a separate term

When an older command is executing, the identity becomes

```text
a = a_d + M(q,m)^-1 [ (m_hat-m) (Mp a_d + hp)
                     + diag(v) (b_hat-b) + u_delivered-kappa(q,v) ].
```

The queue term cannot be omitted or replaced by a true-age selector online.
For measured-state law `kappa_m` and true-state law `kappa`, decompose it as
`u_delivered-kappa = (u_delivered-kappa_m) + (kappa_m-kappa)`.
The second term contains measurement effects; it must not be counted again
if the complete delivered-minus-true-state difference is already enclosed.
At one exact point, the interval calculation includes the declared input
difference in the modal numerator. It does not bound that difference over
uncertain state, sensors or a changing queue. A point result therefore
cannot certify a lagged backup.

## Gates before robot integration

1. Test the algebra and continuous-parameter bounds on coupled synthetic
   systems, including singleton uncertainty and varying mass. At retained
   engineering points, verify the packaged model's affine-parameter identity
   through direct static forward dynamics. Retain raw torque violations.
2. Derive a state/parameter/input enclosure over each held interval, with the
   sent-command FIFO and numerical propagation defects. Establish a compatible
   sensor/horizon contract and exclude or enclose constraint forces; the
   present Gaussian interface remains unchanged.
3. Define a terminal condition for this nonlinear delayed feedback and an
   independent verifier. Reusing the frozen double-integrator maps would
   certify different dynamics. Check startup recoverability and causal
   retention before accepting any task command.
4. Freeze the implementation, protocol, timing budget and reference
   reproduction before used-task integration. A later fresh comparison needs
   its own unused seed block and unchanged physical/task advancement gates.

## Verification of this increment

The structured, existing tube and release test files pass 55 cases. There
are 26 new cases: 24 synthetic checks and two optional Panda static fixtures.
Coupled synthetic matrices are tested against all 128 friction corners at
33 mass values, including estimates outside the true parameter box and
nonzero delivered-command differences. These grids validate the analytical
enclosure; the modal endpoint calculation covers the continuous mass cells.
Other checks cover singleton boxes, partition refinement, zero velocity,
clipping, alternate delivered commands, invalid models and input copies.

The Panda fixtures use `HOME` and a declared small position offset, with
three payloads and two friction vectors each. Twelve direct forward checks
match the affine model and structured identity at `rtol=1e-9, atol=1e-9`.
All have zero contacts/constraints, unchanged q/v and time, and bounded raw
delivered torque. Integration entry points are prohibited. No task seed,
trajectory replay, physical integration or held-out comparison is used.
Retained engineering-point assessment and all subsequent gates remain open.

No candidate advances through pointwise interval containment alone. Existing
references, prior failures, banner, README and model artifacts stay intact.
The computed-torque relationship is described in the
[MIT manipulator-control notes](https://manipulation.mit.edu/force.html);
the parameter-enclosure calculation above is specific to this prototype.
