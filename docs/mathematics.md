# Mathematics

## Rigid-body plant

The reference system is

```text
M(q) qdd + C(q, qd) qd + g(q) + tau_f(qd) = tau.
```

An endpoint payload is included directly in `M(q)` and `g(q)`. The implementation exposes mass matrix, Coriolis matrix, gravity, friction, kinematics and integration independently.

For the selected Christoffel-consistent Coriolis representation,

```text
Mdot(q,qd) - 2 C(q,qd)
```

is skew-symmetric. The test suite checks this numerically using central differences.

## Computed torque

The baseline acceleration command is

```text
qdd_cmd = qdd_des + Kd (qd_des - qd) + Kp wrap(q_des - q)
```

and the nominal torque is

```text
tau_nom = M(q) qdd_cmd + C(q,qd) qd + g(q) + tau_f(qd).
```

Torque limits are applied after inverse dynamics.

## Residual action

The RL policy outputs `a` in `[-1,1]^2`:

```text
tau_candidate = tau_nom + rho .* a
```

where `rho` is the per-joint residual authority.

Direct-torque mode instead uses

```text
tau_candidate = tau_limit .* a.
```

## Nonlinear MPC

The MPC decision variable is a horizon of torque commands

```text
U = [u_0, ..., u_(H-1)].
```

The analytical RK4 model rolls the state forward. The objective penalises wrapped joint-position error, joint velocity and torque, with an increased terminal state weight.

Hard optimisation constraints include:

```text
q_min <= q_k <= q_max
-v_max <= qd_k <= v_max
-tau_max <= u_k <= tau_max.
```

SARRL currently uses SLSQP as a reference nonlinear optimiser.

## Soft Actor-Critic

The critic target is

```text
y = r + gamma (1-d) [ min(Q1_target, Q2_target) - alpha log pi(a'|s') ].
```

The actor uses a reparameterised diagonal Gaussian followed by `tanh`.

If

```text
u ~ Normal(mu, sigma)
a = tanh(u),
```

then

```text
log pi(a|s) = log Normal(u; mu, sigma)
              - sum log(1 - tanh(u)^2).
```

The implementation evaluates the Jacobian term in a numerically stable form using `softplus`.

Automatic entropy tuning optimises `alpha` against target entropy `-|A|`.

## Dynamics context

For transition feature

```text
zeta_t = [o_t, a_t, o_(t+1)-o_t],
```

a GRU maps a causal history

```text
z_t = encoder(zeta_(t-H+1:t))
```

to a bounded latent. An auxiliary head predicts the physical context during supervised encoder training. The latent can be appended to the policy observation.

## Learned residual dynamics

The nominal model predicts

```text
qdd_nom = f_nom(x, tau_commanded).
```

The training target is

```text
Delta qdd = qdd_observed - qdd_nom.
```

An ensemble predicts

```text
Delta qdd_k = f_k(x, tau_commanded).
```

The mean gives a learned correction and the elementwise ensemble standard deviation is used as an epistemic-uncertainty signal.

The runtime uncertainty gate computes a scalar authority factor of the form

```text
scale = max(scale_min, 1 / (1 + gain * ||sigma||)).
```

and applies it only to the learned residual command.

## High-order CBF constraints

### Joint position

For lower joint limit `q_i >= q_min`, define

```text
h = q_i - q_min.
```

With relative degree two, SARRL imposes

```text
h_ddot + (gamma1 + gamma2) h_dot + gamma1 gamma2 h >= 0.
```

Since

```text
qdd = B(q) tau + drift(q,qd),    B = M(q)^(-1),
```

the barrier becomes affine in torque.

The upper position barrier is derived analogously.

### Cartesian obstacle

For end-effector position `p(q)`, obstacle centre `c` and required radius `R`,

```text
h(q) = ||p(q)-c||^2 - R^2.
```

Using

```text
p_dot  = J qd
p_ddot = J qdd + Jdot qd,
```

the same relative-degree-two HOCBF condition yields an affine torque inequality.

### Exact 2-D projection

The runtime solves

```text
min_tau  0.5 ||tau - tau_candidate||^2
subject to A tau >= b.
```

For a strictly convex quadratic in two variables, an optimum of a non-empty polytope is represented by either the unconstrained candidate or an active set of at most two linearly independent inequalities. SARRL enumerates those cases directly and rejects infeasible problems instead of adding hidden slack.
