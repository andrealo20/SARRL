# Queue-aware Panda backup design

Status: verifier and causal supervisor implemented; the separately frozen
[used-task integration check](panda-backup-integration-results.md) failed.
The [bounded-error design](panda-backup-tube-design.md) adds a separate
conditional terminal calculation. It is not integrated into this controller.
The cumulative bank and the published v2.6 filter remain unchanged.

The [recovered failure traces](panda-failure-replay-results.md) place every
first violation in a command interval sent before abort. Checking a command's
arrival interval alone does not establish that a later fallback can keep the
plant inside its limits. This prototype tests one extension: require a
bounded, queue-aware backup trajectory ending in an invariant terminal set
before accepting a proposed torque.

The implementation is [panda_backup.py](../sarrl/safety/panda_backup.py).
It supplies the new [supervisor](../sarrl/safety/panda_supervisor.py) and
[controller](../sarrl/controllers/panda_backup.py). The original controller and
QP remain available unchanged. Prediction does not step MuJoCo.

## One observable-state backup law

At a proposed acceptance, freeze the measured position as an anchor `q_a`,
the estimated inverse mass matrix `B` and drift `d`. Let `M = inverse(B)` and
`u_eq = -M d`. The backup requests

```text
u_b(q, v) = u_eq - M (25 (q - q_a) + 10 v).
```

The gains describe a critically damped, delay-free double integrator with
natural frequency 5 rad/s. They are design constants, without fitting on the
twenty recovered failures. The same frozen law applies to every delay
hypothesis. Its arguments contain measured position and velocity, with no
true delay, delivered torque, physical parameter or task phase.

In the verifier, send the candidate at the current cycle and this feedback
from the next cycle onward. Roll out each possible constant FIFO delay
`0..4`, using its actual known sent-command prefix. All scenarios use the
same policy parameters; their future torques differ only because their
predicted observable states differ. A scenario-specific policy selected using
the true age would not be a valid witness.

Each interval uses the frozen acceleration `B u_delivered + d`. Check exact
quadratic position extrema, linear velocity endpoints and both sent and
pending torque bounds. The 0.03 rad position reserve and 0.10 rad/s velocity
reserve are inherited from the existing filter. The control period is 10 ms.
All hypotheses must reach their terminal set within 200 intervals, including
the candidate interval. A timeout, saturation requirement or interval failure
rejects the witness. No alternative gain, horizon or candidate search is used.

The implementation precomputes powers of the constant backup maps and batches
the predicted states and interval checks. The first candidate interval is
handled separately before applying the backup maps. Earliest-event ordering
rejects any torque or interval failure before terminal entry. Predictions
beyond that entry do not invalidate an already verified infinite tail.
Sequential FIFO/feedback tests check the batched calculation independently.

## Terminal condition includes the queue

For each joint, form the augmented state

```text
z = [q - q_a, v, a_(t-4), a_(t-3), a_(t-2), a_(t-1)],
a_sent = B u_sent + d.
```

Before startup, missing torques are zeros, so their transformed accelerations
are `d`, not zero. The state keeps all four queue entries even for a shorter
delay. The exact sampled backup map for a fixed lag is `z_next = A_l z`.
Check every map's spectral radius below one. With
`Q = diag(25, 1, 0.0001, 0.0001, 0.0001, 0.0001)`, solve

```text
A_l^T P_l A_l - P_l = -Q.
E_l = sum_j z_j^T P_l z_j.
```

Independently require a finite symmetric positive definite `P_l` and a small
Lyapunov-identity residual. The terminal condition is `E_l <= r_l^2`.
Its radius is inscribed using support bounds for every joint position,
velocity, raw feedback torque and queued torque. Coupled mass-matrix rows
enter torque support bounds across all seven joints. An anchor outside the
reserved position interval or an equilibrium requiring an actuator's full
limit leaves no accepted terminal set.

The support calculation bounds entire control intervals. If `s_x`, `s_v`
and `s_a` are the ellipsoid supports of position error, velocity and the
delivered acceleration, its position bound is
`r_l (s_x + dt s_v + 0.5 dt^2 s_a)` and its velocity bound is
`r_l (s_v + dt s_a)`. Torque bounds include every pending slot. Thus a tail
inside the terminal ellipsoid remains admissible under the unsaturated
feedback in that same frozen model. A low terminal speed alone would leave
potentially accelerating commands in flight and does not replace this check.

The separate `P_l` matrices establish tails for constant delays. They do not
establish invariance when the delay switches between hypotheses. Measurement
noise, changing estimates and nonlinear plant mismatch also fall outside this
terminal argument. Fixed engineering reserves are not calibrated uncertainty
bounds. Physical safety must still be measured independently.

## Controller integration requirements

The verifier must precede any command commitment. A later supervisor must
retain the immutable policy from the last accepted witness. If a subsequent
proposal fails, it must execute that retained policy from the next scheduled
send, with the FIFO intact. Replacing its mass, drift or anchor with a new
estimate would invalidate the previous witness. RLS may keep observing sent
commands without changing the retained backup context.

At startup there is no previous witness. Verify an initial backup before
accepting a task command. If the initial backup is not witnessed, reject task
commitment and record an unrecoverable initialization and a bounded emergency
command; it cannot count as a successful or certified fallback. Likewise,
clipping after model mismatch keeps actuator limits but does not extend the
unsaturated terminal argument.

The supervisor implements startup verification, latched rejection and
retained-context identity. Tests cover rejection on the following cycle,
simultaneous updates of the five estimators, sent-command ownership and
emergency saturation. It retains every witness failure and distinguishes
requested from clipped torques. The integration protocol specifies complete
FIFO checks and independent sequential audits on saved traces.

## Gates before a fresh comparison

1. Complete the causal supervisor and an independent witness auditor. Check
   its finite prefixes and terminal inequalities independently of the
   acceptance flags. Preserve the original filter as a separate arm.
2. Use already opened engineering cases only for integration checks. Keep
   these outcomes diagnostic, retain every rejection and measure computation
   time. If the design changes, record that change before reserving new seeds.
3. Freeze implementation, runtime, reproduction tolerance, task grid,
   physical-step budget, full-trace selection and an unused seed block in a
   separate comparison protocol. Reproduce the retained reference outcomes
   before opening that block. No new block is reserved or opened here.
4. Preserve the existing advancement requirements: informative fast baseline,
   at least a ten-point paired unsafe reduction, zero measured candidate
   violations, no cell success regression, zero ordinary-task aborts and no
   ordinary per-joint mean RMS regression over 5 mrad. Rejected commands and
   backup execution cannot erase task failures. Only a passing screen can
   justify a separate confirmation study.

The verifier and supervisor are one candidate design. This phase does not
add a delay-set estimator, tune the planar residual or change the reference.

## Related formulations

[Chen et al., Backup Control Barrier Functions](https://arxiv.org/abs/2104.11332)
describe defining admissible states through a fixed backup policy and its
forward trajectories. [Singletary et al., sampled-data systems with input delays](https://arxiv.org/abs/2005.06418)
treat held inputs, delays and state uncertainty. These motivate checking a
backup continuation and whole intervals. Their guarantees do not transfer
to this estimated-model prototype without establishing their assumptions.
