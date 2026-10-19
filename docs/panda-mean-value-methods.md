# Shared-parameter held-flow propagation

The [preceding admission screen](panda-terminal-admission-results.md) finds
local instability in the retained structured law. Its separate bounded-friction
candidate passes only a pointwise stability test; all eighty six-interval
branches remain unverified. This follow-up changes the offline enclosure
representation for that bounded candidate. It changes no controller, estimator,
gain, physical uncertainty, simulator or task protocol.

## Persistent coordinates and arithmetic

Represent every state and sent torque as `c + G epsilon + r`, where the same
eight coordinates `epsilon in [-1,1]^8` describe physical payload mass and
seven viscous coefficients throughout a complete branch. Initial measurement
errors and nonlinear remainders enter independent bounded residuals `r`.
All four known prior sends remain in the FIFO. New sends retain their
generators when queued and when delivered. There is no parameter redraw,
oracle plant selection, residual covariance claim or delay-dependent law.

Store centers and generators as binary64 values and residual endpoints as
outward enclosures. All operations and conversions use python-flint 0.9.0
at 128 bits. Before rounding a center or generator, transfer its complete
rounding uncertainty into the residual. Recenter the residual and symmetrize
it so that the representation center belongs to the hull. Each scalar remains
an enclosure of the same physical-parameter fiber, not merely the global box.

For a smooth map f on a checked convex hull D, use a validated derivative
enclosure J(D) and any point matrix J0 chosen from its midpoints:

```text
f(c + G epsilon + r) is contained in
f(c) + J0 G epsilon + J(D) r + (J(D)-J0) G epsilon.
```

The last two terms enter the residual. The final term may lose dependencies,
but the first-order generators persist. Midpoints choose a representation;
they never supply an acceptance bound. Hulls remain necessary for whole-box
derivative evaluation and physical checks. This is a first-order enclosure,
not a claim that all nonlinear dependencies are retained.

## Analytic differential bounds

Use the same checked normalized compiled-constant model as the prior screen.
Direct local kinematics supply M, Mq, g, Gq and signed Christoffel terms.
For `h=C(q,v)v+g(q)+D0 v`, use exact signed velocity derivatives and bound
the Coriolis position derivative with the previously derived outward global
mass-Hessian majorant. This majorant can remain conservative at nonzero
velocity. No finite differences or measured residuals set the bounds.

The ideal held-input acceleration satisfies
`M(q,m) a = u-h(q,v,m)-diag(b)v`. Implicit differentiation gives

```text
a_q = -M^-1 (Mq a + h_q)
a_v = -M^-1 (h_v + diag(b))
a_m = -M^-1 (Mp a + hp)
a_b = -M^-1 diag(v)
a_u = M^-1.
```

All terms cover the full current convex state/parameter/held-torque box.
Validated preconditioned matrix solves enclose every derivative column; a
failed solve rejects the query. Differentiate the unchanged raw structured
feedback `Mhat[-kp(q-anchor)-kd v]+hhat+diag(bc)v` in the same way before
applying the mean-value map. The compensated friction `bc=0.5` and mass
estimate 1.5 remain fixed across cells and lags, as in the preceding report.

Unstructured mass/bias remainders do not provide the derivatives this argument
needs. Nonzero such remainders therefore reject construction. The checked
compiled snapshot, source provenance and bounded cache have the same offline
identity requirements as the earlier evaluator; there is no real-time claim.

## Held-flow endpoint and whole-interval proof

The existing 20-iteration Picard search still checks whole-interval position,
velocity and delivered-input boxes. Its tube must stay inside every declared
physical and coefficient domain and contain the complete partial-time image.
No image or escaping physical bound is clipped. Raw sends must satisfy the
same actuator limits.

Within one held interval, torque and physical parameters are constant. Thus
`a_dot = a_q v + a_v a`. Enclose this derivative over the validated tube by
the componentwise magnitude J. Export J outward and use those exact exported
upper endpoints in the Taylor remainder calculation:

```text
v_next = v_start + dt a_start + Rv,  |Rv| <= dt^2 J / 2
q_next = q_start + dt v_start + dt^2 a_start / 2 + Rq,
                                              |Rq| <= dt^3 J / 6.
```

The initial acceleration is a shared-coordinate mean-value enclosure. These
are integral Taylor bounds for the declared continuous ODE with held input,
not a MuJoCo integration formula. Nonzero partial-runtime defects reject this
derivative argument rather than being silently treated as smooth disturbances.
Finite bounded measurement and command errors can enter residuals; this does
not establish a bound for the benchmark's untruncated Gaussian noise.

Store both the independent Picard endpoint box and the affine Taylor endpoint.
The actual endpoint belongs to their intersection. Their box hulls need not
nest: each can include different extraneous states. The next interval starts
from the complete affine enclosure, and must again pass the whole-interval
physical checks. No affine residual is clipped to force nesting or acceptance.
The finite-prefix result supplies no terminal entry or physical model binding.

## Audit and development exposure

The structural auditor never calls the Picard search. It checks parameter
identity, every start/end transition, raw sends, shared-coordinate FIFO slots,
physical tube domains and exact-rational partial-time propagation. It verifies
Taylor endpoint center/generator rounding and residual inequalities with
Fractions, using the exported derivative bound. Coefficients, differential
bounds and mean-value maps remain shared trusted primitives; this is not an
independent rigid-body derivation.

Tests use exact manufactured held acceleration, five parameter fibers, all
five lags, six intervals and dense partial times. They check nonzero shared
generators in queued commands and reject changed endpoints, generators,
parameters, queues, derivative bounds and terminal flags. Independent ordinary
kinematics and central differences cross-check the Panda formulas; they do
not supply any acceptance bound.

A development check on the first payload cell requested all six intervals.
An initial extra requirement that the affine hull nest in the Picard tube
rejected the first endpoint. That requirement was removed before freezing:
both independently derived endpoint enclosures are retained, with the
whole-flow physical checks unchanged. The resulting development check admits
two intervals at zero lag and three at positive lags before becoming
inconclusive, matching the prior first-cell counts. These are exposed method
diagnostics, not confirmation data. No gains, boxes, horizon, cell count or
iteration budget are changed in response.

The first complete attempt, frozen in
`20eeae546ea15da9b14fbec26fb6373af9976866`, stopped before writing a report:
the auditor required a nonnegative lower endpoint for a magnitude record,
whereas Arb's ball enclosure of an absolute value could retain a negative
lower endpoint. Recalculation confirmed identical upper endpoints. Store the
magnitude as `[0, exported_upper]`; the upper bound used by Taylor propagation
is unchanged. Add a crossing-zero regression test, refreeze the implementation
and protocol, and repeat the complete run. No numerical tolerance or physical
acceptance criterion is relaxed.

## Frozen complete comparison

Freeze implementation and this protocol before the full run. Read the exact
prior inputs, bounded context and model identities from the compressed
terminal-admission report with SHA-256
`a909e5e551bb3badb59a690c6a999ad41524fe1ef0f729a6218a08c46f655bec`.
Request six 10 ms intervals in all sixteen continuous payload cells and all
five constant lags, keeping every full viscous interval, prior send, initial
state, physical limit and Picard setting unchanged. All 480 intervals and
delivery of newly computed feedback in every branch are necessary to pass
this finite numerical screen. Switching lags remain outside scope.

Retain all eighty branches, every successful interval and every failure
reason, including an outcome that is unchanged or worse than the baseline.
Source/asset changes during the run reject the report. Save source hashes,
runtime identities, all affine coordinates, Taylor bounds and causal records.
The full replay must reproduce every field and may not overwrite the report.

```bash
python -m tools.assess_panda_mean_value_prefix --output results/panda_mean_value_prefix/assessment.json.gz
python -m tools.assess_panda_mean_value_prefix --audit --output results/panda_mean_value_prefix/assessment.json.gz
```

The [retained outcome](panda-mean-value-results.md) is a numerical diagnostic
on an already exposed fixture. Terminal invariance, force/contact/limit
binding, sensor/horizon and partial-runtime contracts, causal production
handoff and independent task confirmation remain separate gates. This screen
executes zero physical steps and opens zero task seeds. It cannot promote
either candidate or authorize integration, a task comparison or release.
