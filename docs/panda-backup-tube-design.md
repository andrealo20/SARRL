# Bounded-error Panda backup design

Status: conditional mathematical prototype. No Panda model-error envelope has
been established, no supervisor uses this calculation and no new episode is
authorized by its output. The [failed integration](panda-backup-integration-results.md)
and [prediction discrepancy](panda-prediction-gap-results.md) remain unchanged.

The prototype is [panda_backup_tube.py](../sarrl/safety/panda_backup_tube.py).
It retains the existing anchor, estimated coefficients, feedback gains, period
and constant-delay hypotheses. It asks whether a supplied error bound leaves
room for an invariant terminal set. It does not refresh the accepted model.

## A model envelope is a separate prerequisite

For each accepted frozen `B, d, q_a`, an envelope must specify a physical
position box, absolute velocity limits and componentwise bounds. The domain
must cover all allowed plant parameters and every delivered torque within
the actuator limits. It must remain valid throughout a control interval,
including changes in state inside that interval. A fitted point estimate or
the largest error in the selected traces does not provide this enclosure.

The assumptions are explicit inputs:

| Input | Meaning |
|---|---|
| `acceleration_error = w_bar` | Bound on actual acceleration minus `B u_delivered + d` throughout the domain |
| `position_error = eps_q` | Bound on each position measurement error at every send |
| `velocity_error = eps_v` | Bound on each velocity measurement error at every send |
| `position_defect = eta_q` | Bound on additional position propagation error over every partial control interval |
| `velocity_defect = eta_v` | Bound on additional velocity propagation error over every partial control interval |

The propagation contract, for every `0 <= h <= dt`, is

```text
a_0 = B u_delivered + d,              |w(s)| <= w_bar
q(t+h) = q(t) + h v(t) + h² a_0/2 + integral_0^h (h-s) w(s) ds + delta_q(h)
v(t+h) = v(t) + h a_0 + integral_0^h w(s) ds + delta_v(h)
|delta_q(h)| <= eta_q,                |delta_v(h)| <= eta_v.
```

Defects include numerical propagation or interface discrepancies not already
covered by the acceleration term. Setting them to zero needs justification;
a continuous double-integrator calculation alone does not justify MuJoCo's
discrete state updates. Contact responses must either be excluded from the
declared domain or enclosed too. Joint constraints do not establish collision
avoidance.

The envelope's text label records an assumption. It does not validate its
source, bind it to a production policy or certify the plant. A future
integration must hash the bound derivation, domain, plant/runtime versions,
parameter range, sensor contract and frozen policy together. This prototype
deliberately has no command-acceptance API.

## Error propagation keeps the queue

Use the existing augmented coordinates `z = [q-q_a, v, a_sent_(t-4:t-1)]`
and the per-lag matrices `A_l, P_l`. The total norm across seven joints is
`||z||_P = sqrt(sum_j z_j^T P_l z_j)`. The exact physical sent-command queue
belongs to the state, including previous measurement-induced torque errors.
Missing startup entries remain transformed zero torques, including drift.

Measurement errors cause a new transformed sent-command error `n` with
`|n| <= n_bar = kp eps_q + kd eps_v`. For lag zero, this error affects the
current position/velocity and newest queue entry. For positive lag, it enters
only the newest queue entry. The selector `g_l` is therefore

```text
g_0 = [dt²/2, dt, 0, ..., 1]    (omit queue coordinates when maximum lag is zero)
g_l = [0, 0, 0, ..., 1]         (l > 0).
```

Compute the induced contraction and a sufficient disturbance radius:

```text
alpha_l = sqrt(lambda_max(A_l^T P_l A_l, P_l)) < 1
b_q = dt² w_bar/2 + eta_q,       b_v = dt w_bar + eta_v
beta_l = sqrt(P_l[0,0]) ||b_q||_2 + sqrt(P_l[1,1]) ||b_v||_2
         + sqrt(g_l^T P_l g_l) ||n_bar||_2
r_(k+1) = alpha_l r_k + beta_l.
```

Triangle inequalities make this sufficient bound conservative. It permits
time-varying disturbances and measurement errors within the envelope. Tiny
numerical guards inflate contraction/disturbance estimates and shrink the
admissible radius. They are floating-point precautions, not validated interval
arithmetic or a formal proof of numerical rounding bounds.

The first candidate send uses a separate map. Both paths execute the same
fixed candidate and recorded sent-command prefix. Initial queue error is
zero, while initial position/velocity errors remain. Thus

```text
c_q = eps_q + dt eps_v + dt² w_bar/2 + eta_q
c_v = eps_v + dt w_bar + eta_v
r_1² <= P_l[0,0] ||c_q||² + P_l[1,1] ||c_v||²
        + 2 |P_l[0,1]| dot(c_q, c_v).
```

`candidate_radius` implements this bound. `backup_radii` applies only from
subsequent feedback intervals. Reusing the backup contraction for the initial
arbitrary task command would apply the wrong dynamics.

## The terminal set must absorb persistent error

Let `R_cap` be the largest radius retained by the prototype's sufficient
support inequalities. They intersect the model domain with the existing
reserved physical position/velocity limits, bound every partial interval,
and retain coupled mass-matrix torque supports and all pending torque slots.
For zero lag, partial-interval acceleration includes both `w_bar` and
`n_bar`; for positive lag, current measurement error affects the queue only.
All lags include propagation defects. Raw feedback torque needs the extra
reserve `abs(M) n_bar`; saturation cannot silently replace the checked law.

An origin-centered terminal ball can satisfy the sufficient invariance test
only if

```text
R_min = beta_l / (1 - alpha_l)
0 < R_cap,        R_min <= R_cap,        alpha_l R_cap + beta_l <= R_cap.
```

Then any state in that ball stays in it under the assumed envelope and
constant lag, while its whole intervals remain within the checked supports.
A nominal predicted center enters only when `||z_hat||_P + r <= R_cap`.
Separate metrics cover separate constant lags; switching delays need a new
argument. A feasible mathematical case supplies no evidence that the input
envelope holds on the physical plant.

The finite prefix is not implemented here. It must check domain and physical
constraints for every tube before terminal entry, including the candidate
interval, sensor uncertainty, pending torques and intersample extrema. A
terminal feasibility result alone cannot accept a proposal or repair an
unrecoverable startup state.

## The present noisy benchmark has no deterministic sensor envelope

[PandaControlInterface](../sarrl/envs/panda_interface.py) samples untruncated
Gaussian errors. No finite `eps_q, eps_v` bound holds for all future samples
when the configured standard deviations are nonzero. Consequently an
infinite-horizon deterministic terminal claim cannot cover that interface
merely by choosing a multiple of its standard deviation.

A finite-horizon probability statement would need its own risk budget,
sample count and tail calculation, and would remain finite horizon. A
bounded sensor model would define a different evaluation condition. Neither
change is made here. The existing noisy records retain their original
meaning. Clean measurements allow zero sensor bounds but still need a
justified dynamics and propagation envelope.

## Gates before integration or fresh seeds

1. Derive an enclosure over a declared domain and parameter set, including
   mass-inverse conditioning, state dependence, passive forces and runtime
   propagation defects. Validate the derivation independently. If the
   sufficient terminal capacity is empty, retain that result and stop this
   candidate instead of reducing the bound to fit the known failures.
2. Resolve the sensor guarantee and horizon before declaring a protocol.
   Keep physical parameter truth and true delay out of online decisions.
3. Implement and independently audit the finite prefix, robust terminal
   entry and causal supervisor handoff. Bind the envelope to the immutable
   accepted policy. Retain the original controller as a separate arm.
4. Check used engineering cases and full command-plus-update timing under a
   separately frozen protocol. Preserve every failure and every overrun.
5. Freeze code, derivation, runtime, reference reproduction, decision rule,
   physical-step budget and an unused seed block before a new comparison.
   Preserve the existing zero-violation and task-regression gates. No seed
   block is reserved, opened or trained in this phase.

The tests exercise synthetic bounded systems, coupled torques, zero/positive
delays, candidate ownership and unusable envelopes. They establish the
calculation's behavior under its assumptions, without new robot simulation.
The tube, existing backup and release test files passed 56 cases:

```bash
pytest tests/test_panda_backup_tube.py tests/test_panda_backup.py tests/test_release.py
```

The use of nominal centers plus bounded error sets is consistent with the
linear additive-disturbance setting described by
[Rakovic et al., Simple Homothetic Tube Model Predictive Control](https://users.ox.ac.uk/~engs0169/pdf/rakovic_mtns10.pdf).
The inequalities above are derived for this frozen delayed backup. The paper
does not establish the missing Panda plant or sensor bounds.
