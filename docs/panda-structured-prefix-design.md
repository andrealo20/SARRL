# Conditional held-input prefixes

Status: a finite-prefix engine under supplied state-region and error
contracts. No Panda state-derivative provider has been established. This
increment does not verify a terminal set or authorize a production command.
The [point model](panda-structured-backup-design.md) and every earlier
negative result remain available.

The implementation separates [regional dynamics](../sarrl/safety/panda_state_region.py)
from [causal prefix propagation](../sarrl/safety/panda_structured_prefix.py).
It evaluates all configured constant delays, zero through four by default,
with one immutable anchor, parameter estimate and feedback law. No true-age
selector or plant parameter enters that law.

## State-region assumptions

A region supplies a convex position/velocity box, coefficient values at a
declared center, and componentwise absolute derivative bounds for `M0(q)`,
`Mp(q)`, `h0(q,v)` and `hp(q,v)`. The bounds must hold throughout the region.
Matrix derivatives have shape `(7,7,7)` and bias derivatives `(7,14)`.
The mean-value inequality then bounds coefficient changes between the center
and any state box contained in that region.

Uniform mass and bias remainders cover physical model discrepancies not
already included in those coefficient functions. Contact and joint-limit
forces must be excluded from the domain or bounded in the force remainder.
The region's assumption label records these inputs; it does not validate
their provenance. Sampling MuJoCo or differentiating at isolated points does
not establish a derivative bound over the whole region.

Controller arithmetic has a separate torque-error allowance. Physical model
remainders do not alter the declared nominal feedback function. A production
provider would need to bind coefficient/derivative accuracy, parameter range,
force scope and runtime to the retained policy before these assumptions could
support a robot witness. No such provider or binding is implemented here.

## Enclose acceleration over a state and held-command box

Use the structured point calculation at the region center with the midpoint
of the held torque box. It encloses `a0(m,b)` over the full parameter box,
preserving the common mass in the inverse and payload terms. This algebraic
reference uses zero estimated parameters and zero desired acceleration; its
explicit input difference is `torque_midpoint-h0`. It does not change the
backup law used to bound future sent commands.

Let `E` bound mass variation over the query, `H` bound force variation, and
`N0` bound the absolute inverse mass at the center for every allowed payload.
The generalized mass modes provide `N0`. Require

```text
R = N0 E >= 0,       ||R||_infinity < 1
N = (I-R)^-1 N0.
```

The Neumann comparison bounds the absolute inverse throughout the region.
For held-command radius `r_u`, the change from the center acceleration obeys

```text
|a-a0| <= N (r_u + H + E |a0|).
```

`E` includes nominal/payload derivative terms and the mass remainder. `H`
includes bias derivatives, the full viscous-parameter bound times velocity
displacement, and the force remainder. This is a sufficient outer enclosure;
it can lose dependencies and fail on a region that a tighter method could
verify. Failure of the inverse comparison reports an unverified region,
without claiming a physical singularity or an unsafe trajectory.

## Whole intervals require a self-contained state tube

For initial state boxes `Q0,V0`, a trial tube `Q,V` and held acceleration
enclosure `A(Q,V,U)`, bound every partial interval `0 <= h <= dt` by

```text
Q_image = Q0 + [0,dt] V0 + [0,dt²/2] A + [-eta_q,eta_q]
V_image = V0 + [0,dt] A + [-eta_v,eta_v].
```

The propagation-defect bounds must hold for every partial interval, including
discrete-runtime discrepancies. Zero defects require justification. A
numerical Picard search expands a trial until both images lie strictly
inside it. The tube must remain within the supplied model region and the
declared position/velocity constraints. Under the smooth-model and error
contracts this gives a whole-interval enclosure, rather than an endpoint
check alone. End boxes use `dt` in place of `[0,dt]` and retain the defects.

Search inflation defaults to 10% plus `1e-9` in each coordinate, with a
20-iteration budget. These are enclosure-search settings, not changes to
controller gains or physical reserves. Surplus trial inflation may be trimmed
at the declared domain/constraint boundary. An escaping image is rejected
before that operation; the reachable image is never clipped to claim
containment. Exhausting the search or failing an outer bound is inconclusive,
not proof that the physical trajectory violates a limit.

The implementation uses floating-point arithmetic, small padding and residual
checks. It is not a validated interval-arithmetic solver or a machine-checked
physical certificate. Whole-time reach sets and validated numerical solvers
are described in the [ReachabilityAnalysis documentation](https://juliareach.github.io/ReachabilityAnalysis.jl/stable/tutorials/linear_methods/introduction/).
That library's numerical guarantees are not transferred to this prototype.

## A causal FIFO and fresh measurements

The first send is one known fixed candidate, with its actual bounded torque.
Subsequent sends bound the same structured feedback over the predicted state
box inflated by the next measurement-error allowance. Coefficient bounds
must cover that measured-state box too. Command implementation error is
included once at the send. A raw torque box outside actuator limits rejects
the prefix; silently clipping it would change the checked law.

Each lag hypothesis keeps all configured queue slots. Previously sent
commands remain fixed, missing startup entries are zero torques, and each
new send shifts the queue once. Lag zero executes the current send; positive
lags execute the corresponding previous slot. Each held-command box is then
propagated through the full physical interval. The first candidate does not
use an assumed backup-feedback error map. Scenario trajectories differ, but
their future sends use the same state-feedback law and retained parameters.
Switching delays are outside this constant-lag calculation.

Initial physical boxes include the measurement errors once. Future feedback
uses fresh bounded measurement errors, while physical propagation does not
add sensor noise again. `PandaPrefixErrors` separately requires position and
velocity measurement bounds, partial-interval position/velocity defects and
command error. It accepts no infinite sensor bound. The existing untruncated
Gaussian benchmark has no finite deterministic bound and is unchanged; any
finite-horizon risk contract would require its own protocol.

## Executed checks and remaining gates

The prefix, point-model and release test files pass 55 focused cases. There
are 26 new prefix/region cases. Three manufactured nonlinear, coupled
seven-joint fixtures run eight control intervals for each of five constant
lags. Independent `solve_ivp` trajectories, with bounded fresh sensor and
command errors, are checked at 21 times per interval and at the endpoints.
Their derivative bounds come from the declared analytic functions, not fits
to samples. These are synthetic tests, not new Panda episodes or task seeds.

Other cases check intersample violations with safe endpoints, unerasable
pending commands, startup drift under zero torque, nonzero propagation
defects, physical coefficient/force remainders, singular mass, insufficient
regions, torque limits, iteration budgets, invalid assumptions and immutable
input copies. The full-interval images and endpoint boxes remain available
in every successful lag record for a later independent auditor.

`prefix_contained` describes only the requested finite prefix under its
contracts. `terminal_verified` is always false. No supervisor uses this
result, and a finite prefix cannot authorize an indefinitely latched backup.
Before robot integration, establish the Panda regional coefficient and
derivative enclosures, a compatible sensor/runtime/force contract, and a
terminal argument for the nonlinear delayed law. Freeze and independently
audit the resulting witness and causal handoff before a used-task protocol.
No new comparison, parameter narrowing, seed block or release is introduced.
