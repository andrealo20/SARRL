# Mathematics

This file holds the full mathematical formulation. `README.md` keeps only the headline residual-control
equation; every derivation below is what backs it.

## Rigid-body plant

The analytical planar arm follows the standard rigid-body manipulator equation

```math
M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q) + f(\dot{q}) = \tau
```

where $q \in \mathbb{R}^2$ is the joint configuration, $\dot q$ and $\ddot q$ are joint velocity and
acceleration, $M(q)$ is the inertia matrix, $C(q,\dot q)\dot q$ contains Coriolis and centrifugal terms,
$g(q)$ is the gravity vector, $f(\dot q)$ models viscous and smoothed Coulomb friction, and $\tau$ is the
applied joint torque. An endpoint payload is included directly in $M(q)$ and $g(q)$. The implementation
exposes the mass matrix, Coriolis matrix, gravity, friction, forward/inverse kinematics, RK4 integration,
the analytical Jacobian and $\dot J(q,\dot q)\dot q$ independently.

For the selected Christoffel-consistent Coriolis representation,

```math
\dot M(q,\dot q) - 2C(q,\dot q)
```

is skew-symmetric. The test suite checks this numerically using central differences.

## Computed torque

The nominal controller uses feedback linearization:

```math
\tau_{\mathrm{nom}}
=
M(q)
\left[
\ddot q_d
+
K_d(\dot q_d-\dot q)
+
K_p(q_d-q)
\right]
+
C(q,\dot q)\dot q
+
g(q)
```

Torque limits are applied after inverse dynamics. Under an accurate model this approximately reduces the
tracking error dynamics to

```math
\ddot e + K_d\dot e + K_p e = 0, \qquad e = q_d - q
```

The same nominal controller is retained as the non-learned comparison baseline throughout the release.

## Residual control

Instead of asking reinforcement learning to replace the entire controller ($\tau = \pi(s)$), SARRL begins
from the competent physics-based controller above and lets the learned policy produce only a bounded
correction:

```math
\tau_{\mathrm{candidate}} = \tau_{\mathrm{nom}} + \tau_{\mathrm{res}}
```

With normalized SAC action $a_{\mathrm{RL}}\in[-1,1]^n$,

```math
\tau_{\mathrm{res}} = \tau_{\mathrm{res,max}} \odot a_{\mathrm{RL}}
```

Direct-torque mode instead uses $\tau_{\mathrm{candidate}} = \tau_{\mathrm{limit}} \odot a_{\mathrm{RL}}$ and
is retained as an ablation. The residual decomposition gives the policy a much narrower task: compensate
for model mismatch rather than rediscover the complete robot controller from scratch.

## Nonlinear MPC

The MPC decision variable is a horizon of torque commands $U = [u_0, \ldots, u_{H-1}]$. The analytical RK4
model rolls the state forward; the objective penalizes wrapped joint-position error, joint velocity and
torque, with an increased terminal-state weight. Hard optimization constraints include

```math
q_{\min} \le q_k \le q_{\max}, \qquad -v_{\max} \le \dot q_k \le v_{\max}, \qquad -\tau_{\max} \le u_k \le \tau_{\max}
```

SARRL currently uses SciPy/SLSQP as a reference nonlinear optimizer, not a hard real-time solver.

## Soft Actor-Critic

SARRL implements Soft Actor-Critic directly in PyTorch rather than using SB3 or RLlib. The stochastic actor
is a tanh-squashed Gaussian policy:

```math
u_\theta(s,\epsilon) = \mu_\theta(s) + \sigma_\theta(s)\odot\epsilon, \qquad \epsilon\sim\mathcal N(0,I), \qquad a = \tanh(u_\theta)
```

If $u \sim \mathcal N(\mu,\sigma)$ and $a=\tanh(u)$, the corrected log-probability is

```math
\log\pi(a|s) = \log\mathcal N(u;\mu,\sigma) - \sum \log(1-\tanh(u)^2)
```

evaluated in a numerically stable form using `softplus`. The policy objective is

```math
J_\pi(\theta) = \mathbb E_{s\sim\mathcal D,\, a\sim\pi_\theta}\left[\alpha\log\pi_\theta(a|s) - \min_{i\in\{1,2\}} Q_{\phi_i}(s,a)\right]
```

and the critic target is

```math
y = r + \gamma(1-d)\left[\min_{i\in\{1,2\}} Q_{\bar\phi_i}(s',a') - \alpha\log\pi_\theta(a'|s')\right]
```

Automatic entropy tuning optimizes $\alpha$ against target entropy $-|A|$. The implementation includes twin
critics, target critics, Polyak averaging, reparameterized Gaussian sampling, the exact tanh
change-of-variables correction, automatic entropy-temperature tuning, reproducible replay sampling,
architecture-safe checkpoints, deterministic inference that does not advance the stochastic PyTorch RNG, and
exact off-policy training-session reconstruction (see [`docs/design.md`](design.md)).

## Domain randomization and faults

The analytical environment can independently vary link mass and inertia, joint friction, endpoint payload,
motor gain, sensor noise and actuator-command delay. The retained v1.1.0 training campaign uses

```math
\Delta m = \pm 15\%, \qquad \Delta f = \pm 30\%, \qquad \Delta k_{\mathrm{motor}} = \pm 15\%
```

```math
m_{\mathrm{payload}} \in [0,1]\ \mathrm{kg}, \qquad d \in \{0,1,2\}\ \text{steps}
```

Abrupt in-episode motor-gain and payload faults are also supported for controlled fault-recovery
experiments.

## Causal dynamics context

The adaptation module uses only transition history available at runtime. Each history element has the form

```math
h_t = (o_t,\ a_t,\ o_{t+1}-o_t)
```

A GRU encoder maps a finite causal history $H_t = (h_{t-L},\ldots,h_{t-1})$ to a latent context

```math
z_t = f_{\mathrm{GRU}}(H_t)
```

Ground-truth dynamics parameters can be used as auxiliary supervision during training or diagnostics, but
are **not required as runtime policy inputs** — this prevents privileged physical information from leaking
directly into the deployed controller.

## Learned residual dynamics

The nominal rigid-body model predicts acceleration $\ddot q_{\mathrm{nom}} = f_{\mathrm{nom}}(x,\tau_{\mathrm{cmd}})$.
The actual plant acceleration is represented as

```math
\ddot q_{\mathrm{real}} = \ddot q_{\mathrm{nom}} + \Delta\ddot q
```

Each learned ensemble member approximates $\hat r_k(x,\tau_{\mathrm{cmd}}) \approx \Delta\ddot q$. The
commanded torque is intentionally used as model input rather than an already-degraded applied torque, so
actuator-gain mismatch remains visible in the residual target.

## Epistemic uncertainty

For an ensemble of $K$ residual models, the mean prediction is

```math
\bar r(x,\tau) = \frac{1}{K}\sum_{k=1}^{K}\hat r_k(x,\tau)
```

with disagreement measured as

```math
\sigma_r^2(x,\tau) = \frac{1}{K}\sum_{k=1}^{K}\left\|\hat r_k(x,\tau)-\bar r(x,\tau)\right\|_2^2
```

The runtime uncertainty gate computes a scalar authority factor $\mathrm{scale} = \max(\mathrm{scale}_{\min},\ 1/(1+\mathrm{gain}\cdot\|\sigma_r\|))$
and applies it only to the learned residual command, giving an uncertainty-dependent gate $0 \le \lambda(\sigma_r) \le 1$ and

```math
\tau_{\mathrm{candidate}} = \tau_{\mathrm{nom}} + \lambda(\sigma_r)\,\tau_{\mathrm{res}}
```

The uncertainty gate is a **robustness heuristic**. It is **not** a safety certificate.

## High-order CBF constraints

The safety layer modifies the candidate command only when required to satisfy hard constraints. The
projection has the generic form

```math
\tau^\star = \underset{\tau}{\operatorname{argmin}}\ \frac{1}{2}\left\|\tau-\tau_{\mathrm{candidate}}\right\|_2^2
\quad \text{subject to} \quad A(x)\tau \le b(x)
```

For a relative-degree-two safety function $h(q)$, SARRL uses a high-order Control Barrier Function condition
of the form

```math
\ddot h + \alpha_1\dot h + \alpha_0 h \ge 0
```

### Joint position

For lower joint limit $q_i \ge q_{\min}$, define $h = q_i - q_{\min}$. With relative degree two, SARRL
imposes

```text
h_ddot + (gamma1 + gamma2) h_dot + gamma1 gamma2 h >= 0.
```

Since $\ddot q = B(q)\tau + \mathrm{drift}(q,\dot q)$ with $B = M(q)^{-1}$, the barrier becomes affine in
torque. The upper position barrier is derived analogously.

### Cartesian obstacle

For end-effector position $p(q)$, obstacle centre $c$ and required radius $R$,

```text
h(q) = ||p(q)-c||^2 - R^2.
```

Using $\dot p = J\dot q$ and $\ddot p = J\ddot q + \dot J\dot q$, the same relative-degree-two HOCBF
condition yields an affine torque inequality.

### Exact 2-D projection

The runtime solves

```text
min_tau  0.5 ||tau - tau_candidate||^2
subject to A tau >= b.
```

For a strictly convex quadratic in two variables, an optimum over a non-empty polytope is represented by
either the unconstrained candidate or an active set of at most two linearly independent inequalities. SARRL
enumerates those cases directly and rejects infeasible problems instead of adding hidden slack. The planar
implementation constructs constraints for joint-position limits, one-step joint-velocity limits, circular
Cartesian obstacles and actuator torque limits.

## Quantified safety metrics

For each observed state, v1.4 measures physical envelope excess independently
from the nominal HOCBF certificate. For joint $i$,

```math
e_{q,i}=\max(q_{\min,i}-q_i,\ q_i-q_{\max,i},\ 0),\qquad
e_{v,i}=\max(|\dot q_i|-v_{\max,i},\ 0).
```

The reported physical maxima retain radians and radians per second. A
dimensionless severity score divides each one-sided position excess by its
corresponding boundary magnitude and each velocity excess by $v_{\max,i}$,
then takes the largest component. The unsafe-state fraction is the number of
observations with positive severity divided by all observations. The severity
integral is its rectangular time integral using the environment step size.

Command-level diagnostics use the signed nominal constraint margin
$m(\tau)=\min(A\tau-b)$. A negative candidate margin marks a command that
violates at least one nominal HOCBF or torque constraint. For executable
filtered commands, the same margin is recomputed after projection. These
command certificates do not include actuator delay or plant mismatch, so they
must not be substituted for the independently measured physical-state metrics.
