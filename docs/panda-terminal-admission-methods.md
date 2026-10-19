# Sampled stability and terminal admission

The [20 ms numerical prefix](panda-parameter-prefix-results.md) does not test
delivery of new feedback under every lag or establish a stabilizing terminal
region. This protocol checks those prerequisites before any production handoff.
It compares the retained law with one separately declared backup candidate.
All earlier reports, models and physical regressions remain unchanged.

## Equilibrium and exact sampled linearization

Use HOME, zero velocity, physical mass 1.5 kg and all added viscous parameters
0.5. These values lie in the original mass [0.5,2.5] kg and seven friction
[0.5,3] intervals. Retain the previous estimate (mass 1.5 and friction 1.75),
anchor, gains 25/10 and 10 ms control period. Matching physical/estimated mass
makes HOME an exact rest equilibrium of the normalized law; friction mismatch
has zero force there. Check strict raw torque margin at the equilibrium.

Let M be the enclosed mass, G the gravity Jacobian, D0 the packaged damping,
b the physical added friction, and bc the compensated friction. At this rest
point the held-input linearization is

```text
A = [[0, I], [-M^-1 G, -M^-1 diag(D0+b)]]
B = [[0], [M^-1]]
K = [G-kp M, diag(D0+bc)-kd M].
```

Mass-derivative terms multiplying equilibrium acceleration and Coriolis terms
linear in velocity vanish at rest. Obtain G from the same differentiated
normalized kinematics, using the COM Jacobian derivatives already computed.
The ordinary point path remains unchanged. Enclose inverse operations and
the 21 by 21 augmented matrix exponential in Arb:

```text
exp(dt [[A, B], [0, 0]]) = [[F, H], [0, I]].
```

Zero-delay feedback gives J=F+HK. This is the exact derivative of the sampled
nonlinear continuous-model flow at its equilibrium, not a continuous-feedback
approximation, Euler step or MuJoCo integration step. For constant lag L>0,
augment with the last L sent torques, apply H to the oldest slot, shift the
queue and append K times the current state. Unused older slots in the original
four-slot FIFO add only zero eigenvalues; they are omitted solely from the
spectral calculation and never erased by the prefix engine.

Use python-flint 0.9.0, 128-bit arithmetic, directed binary64 export and
[Rump eigenvalue isolation](https://python-flint.readthedocs.io/en/latest/acb_mat.html#flint.acb_mat.eig).
Never use the approximate eigensolver. Isolate every multiplier for all five
lags; any isolation failure is inconclusive. A modulus lower bound above one
refutes local asymptotic stability at this admissible point. A modulus upper
bound below one verifies the pointwise Schur condition. Report sorting uses
midpoints only for order, never for a stability decision.

This is a necessary screen for a locally stabilizing terminal design around
the declared rest equilibrium. Instability here does not disprove every larger
invariant set, prove physical limit violations, or diagnose the deployed
reference controller. A pointwise pass does not certify the full parameter
range or provide an explicit nonlinear invariant set.

## Separate compensation candidate

The retained law compensates the estimated added viscous friction. If that
estimate exceeds the physical damping, it can remove rather than preserve
dissipation. At matched mass the continuous linearized damping is
`kd M + diag(b-bc)`. The seven-joint model has small inertia modes, so the
negative term cannot be discarded in a terminal argument.

Define a new backup context with
`bc_j = min(retained_estimate_j, declared_physical_lower_j)` for the seven
viscous parameters. Keep the estimated mass, anchor and gains unchanged.
The retained estimator and five cumulative RLS banks are not modified.
The compensation limit follows the declared full physical interval, not an
estimated confidence interval or a selected true plant value. Every payload
branch uses the same new context. All physical uncertainty remains unchanged.

This makes `b_j-bc_j >= 0` throughout the declared viscous box. At matched mass
it removes this particular continuous local negative-damping mechanism. It
does not establish sampled/delayed nonlinear stability under mass error, a
terminal set, physical model bounds or a valid estimator-acquisition history.
Treat it as a separate, unpromoted candidate. The known first candidate and
four prior sends remain identical because they were defined at zero velocity.

Development checks used these already observed inputs: ordinary arithmetic
found the retained point's zero-delay multiplier modulus near 1.00902 and
the bounded-compensation point below one at all five lags; a 128-bit check
confirmed the retained zero-delay instability. A first-payload-cell six-step
development check of the new candidate admitted two intervals at lag zero
and three at positive lags before becoming inconclusive. These exposures
are method diagnostics, not independent confirmation. No gain, physical box,
cell count, iteration budget or failure threshold is tuned afterward.

## Six-interval universal prefix screen

Freeze code and this protocol before the complete execution. Read exact inputs
from the preceding compressed report with SHA-256
`ac77c689474f861fa08cfeefe7be7b22a6262b101f99b8371ead41d02e8619d0`.
Compare retained and bounded compensation, each with sixteen continuous payload
branches, all seven full viscous intervals and every constant lag 0 through 4.
Request six 10 ms intervals, keeping all earlier domains, limits, 20 Picard
iterations and zero ideal-model error assumptions unchanged. The horizon is
fixed because the first newly computed feedback sent at step 1 is delivered
at step 5 under lag 4. A complete pass therefore needs 480 admitted intervals
per arm and delivered new feedback in every branch, not just warm-history
torques. Switching delays remain outside scope.

Use a checked compiled snapshot for offline coefficient evaluation, with a
128-entry cache of identical state boxes. Each query checks the used compiled
constants, including cache hits; cache keys include precision and all four
position/velocity endpoint arrays. Capture and verify packaged source/asset
provenance before and after the run. A changed model rejects the result.
Memoization changes neither coefficient arithmetic nor enclosure decisions;
tests compare cached/snapshot results bit for bit with the fresh provider.
The snapshot does not follow later disk updates. No runtime performance
guarantee follows from this offline optimization.

Keep every cell/lag and every successful interval, including partial prefixes
and all failure reasons. Audit exact parameter coverage, unchanged feedback
contexts, measurement-box coverage, causal queues and all-partial-time
propagation. Do not clip an escaping image, omit failed branches, reduce the
uncertainty or promote a candidate on a subset. Preserve source fingerprints,
all eigenvalue/matrix enclosures, runtime/model identities and the full report.
The audit must reproduce every field under the frozen source and runtime.

```bash
python -m tools.assess_panda_terminal_admission --output results/panda_terminal_admission/assessment.json.gz
python -m tools.assess_panda_terminal_admission --audit --output results/panda_terminal_admission/assessment.json.gz
```

## Physical and terminal admission gates

Both local stability at the declared point and all six intervals are necessary
checks in this design. They are not a sufficient certificate. No arm advances
to integration, task comparison, promotion or release without an explicit
nonlinear terminal set, justified contact/limit forces or exclusions, bounded
partial-runtime discrepancies and an audited causal production handoff.

The [measurement source](../sarrl/envs/panda_interface.py) adds untruncated
Gaussian noise. At nonzero standard deviation it has no finite deterministic
sensor bound. A finite-horizon risk contract would need a separate failure
probability, horizon and proof; silently truncating the existing benchmark or
calling a few standard deviations a deterministic bound is not permitted.
The [loader](../sarrl/robots/panda.py) uses a 2 ms timestep and the
[model](../sarrl/assets/panda/panda.xml) selects `implicitfast`.
MuJoCo's [numerical integration](https://mujoco.readthedocs.io/en/stable/computation/index.html#numerical-integration)
is not the exact continuous flow used here. No zero discrete-runtime defect
is established. The declared normalized compiled-constant model also differs
from a verified MuJoCo arithmetic/physical binding. These open gates remain
false regardless of a pointwise spectral or finite-prefix pass.

This assessment executes no physical steps or fresh task seeds and introduces
no training, simulator change or production activation. Report all decisions
in the [retained results](panda-terminal-admission-results.md).
