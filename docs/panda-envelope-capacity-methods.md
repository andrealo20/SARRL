# Necessary error floor for the Panda backup envelope

This engineering screen follows the [conditional tube prototype](panda-backup-tube-design.md).
It does not fit an enclosure from failed trajectories. It asks whether that
particular sufficient terminal criterion can survive a necessary error floor
at the retained anchor alone. Freeze implementation and this method before
evaluating the static Panda points. No new task seed or physical integration
is part of the calculation.

## Fixed selection and static points

Use all 26 final retained policies from the aborted episodes in
`results/panda_backup_integration_v26`. Verify its completion marker and all
input hashes against completion SHA-256
`351674a90849d64e7359fa101f72aee86ea1536ff2713b1b79d558b1d2f9831a`.
Preserve accepted-policy hashes and causal retention checks. Selection follows
the previous diagnostic phase and has no population or confirmation meaning.

For each anchor `q_a`, evaluate zero velocity at both payload endpoints
0.5 and 2.5 kg, with extra viscous friction 0.5 Nm s/rad on all seven joints.
These are members of the existing parameter box, not random draws or a claim
that endpoints maximize the full uncertainty. Damping contributes no viscous
force at zero velocity. Check all anchors are within the existing reserved
joint limits and have no active contact or joint constraint. Otherwise stop
the screen rather than apply an unconstrained affine torque calculation.

Use the retained numerical runtime and verified packaged model. Compute
mass and passive/bias forces through static forward dynamics, with no advance
in model time. Physical stepping and alternative integration entry points
are prohibited throughout the calculation and its audit.

## An exact torque-box maximum at each point

For one static model, let acceleration be `B_p u + d_p`, with the retained
prediction `B_f u + d_f`. The full componentwise actuator box is
`|u_j| <= U_j`, exactly as assumed by the current envelope contract. Then

```text
L_i = |d_p,i - d_f,i| + sum_j |B_p,ij - B_f,ij| U_j.
```

This is the exact maximum absolute discrepancy for joint `i` at that point.
Retain an attaining torque for each row and verify its acceleration through
direct `mj_forward`, independently of the matrix inversion route. Each row
may require a different torque. Componentwise bounds over the full box must
cover all these extrema even when they are not simultaneous.

Take the componentwise maximum across the two static parameter models.
Any uniform acceleration-error envelope covering the declared parameter box,
anchor, zero velocity and full torque box must cover this vector. This is
a necessary floor, not an upper enclosure over state or parameter domains.
The claim concerns the continuous acceleration part of the specified
contract; it does not identify a discrete integration defect with acceleration.

## An optimistic terminal calculation

Evaluate the existing scalar tube with that floor, zero sensor errors, zero
propagation defects and the largest domain allowed by the existing reserved
physical position/velocity limits. Preserve gains, period, delays and torque
limits. Do not revise the policy, reserves or metric after seeing results.

Require conditional terminal feasibility for every constant lag zero through
four. A rejection means this particular full-box, componentwise-error,
scalar-radius sufficient criterion cannot certify that retained policy under
the stated assumptions. A smaller domain, larger valid error bound, sensor
error or propagation defect cannot restore its capacity. It does not prove
that the physical policy is unsafe everywhere, that all invariant sets are
empty or that a state/input-dependent formulation would fail.

A passing case remains inconclusive: it supplies no full-domain enclosure,
finite-prefix witness, sensor guarantee or controller validation. Publish all
26 outcomes and all per-lag radii. Stop this formulation before integration
if all retained policies fail this necessary screen. Any alternative requires
a separately declared design; do not shrink the torque/error box to fit these
selected results.

## Numerical and provenance checks

Test the box formula against all 128 torque vertices on synthetic fixtures,
check attaining commands, invalid inputs and monotonic capacity. At Panda
points compare direct forward accelerations with the affine reconstruction
using `rtol=1e-9, atol=1e-9`. Retain runtime, source commit and hashes, input
identity, static model coefficients, attaining commands and direct results.
These are floating-point calculations, not a machine-checked enclosure proof.

The audit repeats every static evaluation and terminal calculation from the
frozen source, verifies output hashes and compares finite values at the same
tolerance with exact discrete identities. It requires no local episode
shards. It must prohibit physical stepping and leave all old evidence intact.
Forward dynamics and time integration are separate stages in the
[MuJoCo computation reference](https://mujoco.readthedocs.io/en/stable/computation/).
