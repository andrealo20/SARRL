# Mathematics

## Plant

SARRL starts from the rigid-body model

```text
M(q) qdd + C(q, qd) qd + g(q) + tau_f(qd) = tau.
```

The implementation exposes `M`, `C`, `g`, forward/inverse dynamics, kinematics and RK4 integration independently so each identity can be tested.

For the chosen Christoffel-consistent Coriolis matrix, `Mdot - 2 C` is skew-symmetric. Tests verify this numerically with central differences rather than assuming it from the derivation.

## Computed-torque baseline

```text
qdd_cmd = qdd_des + Kd (qd_des - qd) + Kp wrap(q_des - q)

tau_nom = M(q) qdd_cmd + C(q,qd) qd + g(q) + tau_f(qd)
```

This is a deliberately strong nominal controller: residual RL is only interesting if it improves a competent physics baseline under model mismatch.

## Residual action

In residual mode the RL policy does not command the whole actuator torque:

```text
tau_candidate = tau_nom + rho * a_RL,     a_RL in [-1, 1]^2.
```

The candidate is then clipped to the actuator limits. Later milestones will replace this final clipping stage with an explicit safety projection.

## SAC objective

The critic target is

```text
y = r + gamma (1-d) [ min(Q1_target, Q2_target) - alpha log pi(a'|s') ].
```

The actor is a reparameterized diagonal Gaussian followed by `tanh`. Its log probability includes the exact change-of-variables correction. Automatic entropy tuning learns `alpha` against target entropy `-|A|`.
