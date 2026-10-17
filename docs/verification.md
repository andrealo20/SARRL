# Verification record

This file distinguishes checks that were actually executed from features that are only implemented or
planned.

## Panda center arithmetic enclosures

The optional [ball-arithmetic audit](panda-center-arithmetic-design.md)
encloses nominal/unit-payload mass, mass derivatives, gravity and bias at
three existing static centers. Exact-rational endpoint checks verify outward
binary64 export and reference-error bounds. Tests cover subnormals, overflow,
invalid input, optional-backend failures, immutable snapshots, nonfinite
balls and precision restoration. Independent reference values are read
before dynamics/integration calls are prohibited during enclosure evaluation.
All 107 focused center, regional, global, prefix, point-model and release
tests pass, including 23 new cases. The MuJoCo CI job installs the pinned
optional numerical backend; base installations retain no new requirement.

The declared model normalizes compiled frame/axis constants and interprets
other binary64 constants exactly. The largest reported nominal mass/bias
errors against MuJoCo are bounded by 3.11e-15 and 2.85e-14, respectively,
at those centers only. Regional Hessians, inverse solves and prefix arithmetic
remain unvalidated; no physical controller or task seed is introduced.

## Regional Panda derivative refinement

The [center/Hessian provider](panda-regional-derivatives-design.md) computes
mass derivatives through second-order kinematics and bounds their regional
variation using global analytic mass Hessians. Independent MuJoCo mass,
gravity and velocity-bias values and finite-difference derivative checks
agree at three static configurations. The analytic evaluator uses no
dynamics or integration entry point. Eight deterministic corner patterns
check derivative and coefficient bounds and independent acceleration solves.
Closed-form tests cover an entire sine-derivative interval and repeated-angle
transform derivatives, input rejection and immutable arrays.
All 84 focused regional, global, prefix, point-model and release tests pass,
including 12 new cases. The supported-model check also rejects a detached
base chain and prevents unrepresented bodies from entering the derivation.

On the same five ideal-model boxes as the global diagnostic, the inverse
comparison norm at a 0.001 rad radius falls from 1.4726 to 0.070362. The
0.01 rad box remains inconclusive at 1.9277. No physical integration, seed,
sensor/runtime or terminal claim is introduced. Validated center-derivative
arithmetic and physical-force contracts remain prerequisites.

## Analytic Panda coefficient regions

The [geometry-derived provider](panda-analytic-region-design.md) is checked
against closed-form two-link and rotor derivative amplitudes, independent
Panda finite differences at three static states, coefficient variation and
direct acceleration solves. Tests also reject unsupported model changes and
invalid inputs. All 72 focused analytic-region, prefix, point-model and
release tests pass, including 17 new cases. No physical integration or task
seed is used. Derivative
bounds follow the analytic product-rule derivation, not sampled extrema.

The reproducible static diagnostic finds inverse-comparison norms of 0.14726
and 1.4726 at position radii of 0.0001 and 0.001 rad around HOME, respectively.
The latter is inconclusive. These global derivative bounds need regional
tightening. The provider establishes no contact/limit-force exclusion,
sensor/runtime contract, validated floating-point enclosure or terminal set.

## Conditional structured held-input prefixes

The [prefix engine](panda-structured-prefix-design.md) passed 55 focused
prefix, structured-point and release tests. Twenty-six new cases include
independent dense trajectories of manufactured nonlinear coupled models for
three parameter fixtures, all five constant lags and eight intervals. Analytic
regional derivative contracts, fresh bounded noise, command uncertainty and
the exact sent-command FIFO are checked. Failure cases cover intersample
violations, pending commands, startup drift, remainders, bad regions, torque
limits, singular mass and search budgets. No Panda trajectory is integrated.

This verifies a conditional floating-point enclosure engine. The later
analytic regional provider is described above. Validated numerical enclosures,
sensor/runtime/force contracts, terminal proof and supervisor handoff remain
unimplemented. A
contained finite prefix has `terminal_verified = false` and cannot advance
the candidate to robot integration.

## Structured Panda backup point model

The [structured formulation](panda-structured-backup-design.md) passed 55
focused structured-model, tube and release checks. Of 26 new cases, 24 use
coupled synthetic models and two use static Panda fixtures. Independent
parameter grids exercise all 128 friction corners, varying mass and input
differences; singleton, invalid-family, refinement and clipping cases check
the calculation's scope. Twelve direct Panda forward checks reproduce the
affine-parameter identity and fall within the derived point intervals without
advancing time or invoking an integration entry point. No robot episode or
new task seed is used.

This is a new pointwise law and parameter calculation. The existing
supervisor is unchanged. Delayed interval propagation, measurement
uncertainty, invariant terminal conditions and retained-policy handoff are
not verified by these tests.

## Static Panda envelope-capacity screen

The [necessary-error screen](panda-envelope-capacity-results.md) checks 26
retained policies at 52 static parameter points, including 416 direct forward
checks and 130 conditional terminal calculations. All fail capacity without
any new physical integration. Nine new synthetic tests verify exact
torque-box extrema against all 128 vertices, attained values, input rejection
and monotonicity. The capacity, existing tube and release files pass 38 tests.
The frozen source, runtime and input/output hashes are retained for audit.
The result stops this sufficient formulation; it does not establish that no
other invariant set or controller can work.

## Conditional bounded-error backup

The [tube prototype](panda-backup-tube-design.md) passed 56 focused tube,
existing backup and release tests. Twenty-six new cases check zero/positive
constant delays, the fixed first candidate, persistent error propagation,
nonzero numerical defects, changing acceleration inside an interval, coupled
torque boundaries with aligned sensor errors, immutable envelope copies and
rejection of invalid or unusable bounds. Synthetic boundary states are
propagated with an explicit FIFO and dense intersample checks. Ruff and
compilation pass. No robot episode, model fitting or physical integration is
performed by these tests.

This verifies a conditional mathematical calculation. A Panda dynamics and
propagation envelope is not established, the untruncated Gaussian interface
has no deterministic sensor bound, and finite-prefix tube verification and
supervisor integration remain pending. No controller or advancement decision
changes.

## Saved-state Panda prediction discrepancy

The [prediction-gap review](panda-prediction-gap-results.md) covers all 26
aborted integration episodes. Its initial velocity report and subsequent
position extension are retained separately from their own frozen source
commits. Fourteen focused tests check delay queues, measurement offsets,
injected transition discrepancy, provenance tampering, algebraic model
substitutions and late positional violations without velocity violations.
Both datasets reproduce through separate audits with physical stepping
prohibited. Maximum position/velocity identity residuals are `3.27e-13 rad`
and `3.13e-13 rad/s`. The chart was inspected for readable labels and full
trajectory coverage. No controller, seed or physical result changed.

## Causal Panda backup integration

The new supervisor retains accepted backup context across model changes,
latches rejection without clearing the FIFO, distinguishes witnessed backup
from emergency initialization and records raw/clipped torques. All five RLS
models continue observing the actual sent command. The independent auditor
uses sequential trajectories, a Kronecker Lyapunov solve and full-joint
terminal supports, with no call to the verifier's acceptance implementation.

Sixty-six focused supervisor, witness, auditor, joint-filter, selection,
timing and release tests passed. One already used first-seed ordinary case
completed successfully without a physical violation or abort. Its saved-state
audit passed 300 causal cycles, 301 witnesses, 1,505 delay cases, 50,288
independent backup intervals and 300 feasible QP checks without physical
stepping. The [frozen integration protocol](panda-backup-integration-methods.md)
defines the separate 32-task run. It is now complete: all 64 original records
matched before 32 candidate episodes. Its audit checked 9,600 causal cycles,
2,078 witnesses, 10,390 delay cases, 429,209 independent backup intervals and
2,046 feasible QP projections without physical stepping. The
[results](panda-backup-integration-results.md) retain eighteen unsafe episodes,
including two zero-delay regressions, and twelve control cycles over 10 ms.
The candidate does not advance.

## Queue-aware Panda backup prototype

The [backup verifier](panda-backup-design.md) has deterministic tests for
exact FIFO dynamics at delays zero through four, stable terminal maps,
independent Lyapunov-identity rejection, coupled mass/bias compensation,
unrecoverable pending commands and a terminal condition beyond one safe
interval. Boundary directions of each terminal ellipsoid are propagated
for 300 intervals, with independent dense checks of positions, velocities
and raw feedback torque. Policy copies preserve the accepted model context.

The backup and existing joint-filter test files passed 44 cases. Batched
matrix-power predictions agree with a sequential FIFO/feedback reconstruction
on moving, coupled-bias, pending-violation and outside-domain fixtures. The new
verifier tests require no physical robot simulation or task seeds. They do
not evaluate closed-loop tracking, noise robustness or candidate advancement.
The subsequent causal integration is described above; this earlier test
record concerns the standalone mathematical verifier.

## Panda failure-trace recovery

Recovery source `f29dd97` reproduced all twenty unsafe filtered v2.6 records
using production source `6891dd2`, with finite floats checked at `rtol=1e-9`,
`atol=1e-10` and exact discrete values/types. This adds 30,000 retained physical
integration calls on used seeds. No new seed, controller or training run was
introduced. A separate public-output audit reproduced all event diagnostics,
twenty causal controller replays, 6,000 control cycles, 30,000 saved substeps
and 206 independently checked feasible QP projections without physical stepping.

Twenty focused recovery, diagnosis and release tests passed. They include
boundary ownership at braking arrival, strict record reproduction, public
auditing without local cell shards and a resume guard that permits only the
recovery's own untracked output. The completed evidence remains tied to the
original frozen recovery source. See the [timing findings](panda-failure-replay-results.md).

## Saved-data control failure diagnosis

Source `2e88aa0` analyzed all 2,100 planar and 480 Panda v2.6 records and all
45 retained traces. A separate audit reproduced the output, including every
planar trace reward/endpoint, Panda substep violation maxima and six fast
filtered abort-prefix reconstructions. Input completion identities and data
hashes are fixed, and diagnostic source hashes are retained. Physical stepping
is prohibited; no new episode, training run or policy inference occurred.
Eight diagnostic tests and three release/command tests passed. They check
input tampering, output preservation, paired loss accounting, reward and
projection reconstruction, abort-return conventions, physical event ordering
and the limited meaning of a narrower saved-state delay bound. The
[findings](control-failure-results.md) distinguish that initial missing-trace
limitation from the subsequent unchanged-source recovery above.

## v2.6 joint filtering and adaptive residuals

The [Panda filter study](panda-joint-filter-results.md), frozen at `6891dd2`,
reproduced all 4,000 reference-validation records before retaining 480 fresh
episodes. The portable audit passed 480 records, 24 causal controller replays,
36,000 recorded physical substeps and 1,846 independently checked feasible
projections. The production run used 6,720,000 physical substeps including
reproduction. The candidate failed four frozen criteria and remains experimental.

The [planar study](adaptive-residual-results.md), frozen at `b41e167`, retained
five complete 200,000-call/195,000-update policies and all 2,100 held-out
episodes. The full local audit passed 2,100 records, 21 first-seed traces,
2,416 actor-action checks and 21 causal controller/estimator replays. It
verified the five policy identities and training artifact hashes. The public
audit separately passed all records, pairing, trace bounds, statistics and
decision, explicitly reporting `policy_inference_verified: false`.

Both audits recomputed the 300 historical comparisons: 300 portable matches
and 299 matches under the original strict tolerance. The pre-training
amendment and all recalculated records retain the strict failure. Discrete
fields, source hashes, data hashes and metadata remain exact. The planar
runtime was Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, MuJoCo 3.14.0 and
PyTorch 2.12.0+cpu, with two workers and one numerical thread per worker.
The residual failed seven advancement checks; neither v2.6 candidate replaces
the established reference or advances to Panda integration.

The eleven Panda campaigns retain 15,380 new records. Completion hashes and
491 frozen-source blob references, including the 30,000-cycle profile, passed
verification against Git history. The planar study's 2,100 held-out records
and 217 frozen-source blob references passed separately. Ignored models and
older archives are preserved. The supplied SVG retains SHA-256
`7479624165f1eda00cca9702ba5eb968dc9654afe7e21803dd878aab5f1b61e4`.
All 127 checked local documentation links resolve, and the README retains
its three selected SVG figures and existing replay assets.

The wheel and source distribution built successfully. All 94 packaged files
matched the checkout; an isolated installation reported version 2.6.0 and
loaded all twelve Panda model files. The new controller and training-wrapper
constructors passed their installation probes. The nominal example completed
300 steps with success, no clipping and no unsafe event on an existing test
seed. Numerical dependencies were reused; this was not a clean dependency
resolution test. The source archive preserves the supplied SVG byte for byte
and excludes local working instructions.

The complete local suite passed 625 tests with two skips: unavailable local
diagnostic episode records and the CPU runtime's absent CUDA device. The
three release/command tests passed separately after making the portable
auditor's help available before source-checkout validation. Lint, compilation
and whitespace checks passed. CI runs Python 3.10, 3.11 and 3.12, MuJoCo and
the pinned archival audits, including planar statistics in a separate NumPy
2.4.6 environment. Numerical replay and decision rules are unchanged by the
command-help adjustment.

## v2.5 reference validation and shared scoring

Source `09858a7` reproduced all 1,920 v2.4 records and retained 4,000 fresh
constant-delay episodes. The reference passed every frozen criterion, with
1,000/1,000 bank successes and correct final ages. Its audit verified all
records, eighty first-seed traces and sixty causal estimator replays. Saved-input
profiling retained and audited 30,000 cycles: p50 0.976287 ms, p95 1.378157 ms,
p99 1.827530 ms and maximum 3.928535 ms, with no duration above the 10 ms
period. Host, runtime and excluded costs are documented in the
[reference results](panda-reference-results.md).

Source `1a47469` then reproduced all 4,000 validation records before retaining
1,920 fresh shared-scoring episodes. The audit verified all records, 96 traces
and eighty causal estimator replays, including pre-update innovations and
the fixed scoring copy. The candidate failed its frozen age, constant-age,
success and tracking criteria. The [complete results](panda-shared-scoring-results.md)
retain every condition and failed comparison. Both phases together used
17,952,000 physical substeps including reproduction, with zero prediction
virtual substeps. Profiling added no physical steps or seeds.

Thirty-five focused tests passed. The complete local suite passed 599 cases
with one skip on Python 3.12.3, NumPy 2.5.2 and MuJoCo 3.13.0. New checks cover
unchanged bootstrap arithmetic, acquisition timing and a read-only independent
parameter copy, causal prefix invariance, one-joint tracking regressions,
unknown arms, reproduction refusal and canonical output/worker guards.
Lint, compilation and whitespace checks passed. All five CI jobs passed for
the frozen comparison sources, including the recorded-runtime archival audit
of reference validation and profiling; the MuJoCo job passed 594 tests with
six skips. Final retained-data CI additionally audits shared scoring from
its recorded source worktree.

The ten retained Panda campaigns contain 14,900 new episode records.
Completion hashes and 418 frozen-source blob references, including profiling,
were verified against Git history. Original Python sources from `09858a7`
remain unchanged; shared scoring adds separate modules. The supplied SVG
retains SHA-256 `7479624165f1eda00cca9702ba5eb968dc9654afe7e21803dd878aab5f1b61e4`.

The wheel and source distribution built successfully. All 87 packaged files
matched the checkout; the installed package reported version 2.5.0 and loaded
the twelve-file Panda model. Its nominal example completed 300 control steps
with success, no clipping and no unsafe event on an existing test seed.
Numerical dependencies were reused, so this was not a clean dependency-resolution
test. The source archive includes the supplied SVG byte for byte and excludes
local working instructions.

## v2.4 parameter-memory comparison

Source `cee94b2` reproduced all 1,920 prior score-memory records, then
completed 1,920 fresh paired episodes. Both stages used 2,976,000 physical
substeps including warmup. The audit verified all 1,920 new records,
96 traces and 80 causal estimator replays, completion hashes, frozen Git
sources and the recomputed decision. No prediction virtual
substeps were performed. The [protocol and results](panda-estimator-memory.md)
retain all conditions and the failed age/per-joint tracking screen.

The local full suite passed 581 cases with one skip for unavailable local
diagnostic records. After applying recorded floating-point replay tolerances,
the diagnostic and parameter-memory tests passed all 21 cases. These check
weighted batch equivalence without clipping, exact legacy arithmetic when
forgetting is disabled, numerical rollback, bounds, covariance positivity,
causal prefix invariance, reproduction refusal and per-joint gate failures.
A replay accepts rounding-scale differences and rejects an altered parameter.
Lint, compilation and whitespace checks passed.

All five CI jobs passed for the frozen campaign sources: Python 3.10,
3.11 and 3.12, MuJoCo tests, and the pinned saved-input replay. CI also
audits the retained parameter-memory data from its recorded source worktree,
keeping that replay separate from future package changes.

The final wheel and source distribution built successfully. All 84 packaged
files matched the checkout, the installed controller reported version 2.4.0,
and the twelve-file Panda model loaded. The nominal installation example
completed 300 control steps with success, no clipping and no unsafe event,
using an existing engineering-test seed. Numerical dependencies were reused;
this was not a clean dependency-resolution test. The source archive includes
the supplied banner byte for byte and excludes local working instructions.

## Panda latency diagnosis

The saved-input analysis reproduced all 48 bank traces from seed 9805300,
including final candidate estimates, scores, bound hits and complete
selection histories. Both output hashes and all three frozen analysis-source
hashes passed. No new physical episode or seed was opened.

The diagnostic, score-memory and delay tests passed 54 cases. The tests
cover score-history removal, parameter/alignment confounding, missing
excitation, output preservation and a retained replay with physical stepping
disabled. Lint passed. A dedicated CI job uses the retained NumPy/MuJoCo
versions and executes the full diagnostic command.

## v2.3.2 demonstration checks

### Release package

The wheel and source distribution built successfully. A fresh installation
outside the checkout, reusing the numerical dependencies, reported version
2.3.2, verified twelve model files and completed the nominal Panda example
with 300 control steps, no clipping and no unsafe event. The example uses
the existing engineering-test seed, without adding benchmark evidence.

The landscape renderer has a neutral command name; the prior command remains
compatible. Its help command and lint passed. Media files and controller
arithmetic are unchanged. All seven pilot completion markers and 173 frozen
source blobs passed, covering 7,060 retained episode records.

### Landscape video playback

Both MP4s were regenerated at 3840 by 2160 pixels with H.264 High level 5.1,
YUV 4:2:0 and CRF 18. The comparison arranges two native robot viewports beside
the error plots. Its 360 frames cover 24 seconds; nominal motion uses 180
frames over 12 seconds. Both remain 15 fps. The encoded files are 3,905,792
and 1,041,916 bytes, each below 10 MB.

All 540 frames decoded successfully. Sampled initial, middle, settling and
final frames were inspected; caption bands and chart canvas bounds passed.
Both videos completed browser playback with decoded dimensions of 3840 by
2160 and no media error. Local links and all seven completion markers passed.
The 32-test targeted suite and lint passed. Final paired tip errors remain
37.918834 mm and 0.008527 mm. Rendering performed no physical simulation steps.

The existing banner, PNG/SVG figures, posters and GIF previews are unchanged.
The portrait preview renderers remain available alongside the new MP4 renderer.

### Native higher-resolution replays and vector figures

Both replays were regenerated from the same hash-verified saved states.
Robot viewports, chart DPI, caption fonts and projected markers render at
twice their previous dimensions. The paired MP4 is 1920 by 2016 pixels;
the nominal MP4 is 1920 by 1600. All 540 encoded frames decoded successfully,
at 15 fps over 24 and 12 seconds. Posters have matching native dimensions.

The GIFs are 1200 by 1260 and 1200 by 1000 pixels. All 288 frames decoded,
with looping enabled and durations of 24.01 and 12.01 seconds. The files
are 8,372,396 and 3,987,580 bytes, both below 10 MB. Caption bands and chart
bounds passed during rendering. Sampled video and GIF frames were inspected.

The three README result figures now embed the existing SVG files directly.
Their XML parsed successfully and contains no raster image elements. Local
documentation links and all seven retained completion markers passed. The
32-test targeted suite and lint passed. Final paired tip errors remain
37.918834 mm for fixed and 0.008527 mm for adaptive. No controller, model
asset, evidence file or original banner changed.

### Focused README and Panda endpoint comparison

The README now selects three result figures and uses native Mermaid for
the two control diagrams. The other figures remain indexed in the result document.
The original banner is unchanged.

The revised paired replay retains seed `9804300` and all saved joint states.
Target crosses, projected tip markers, physical-scale paths and identical
camera zooms expose the endpoint difference. Forward kinematics reproduces
the retained final tip errors: fixed 37.918834 mm, adaptive 0.008527 mm.
The close-up uses equal X/Z scales and the last physical second's paths.
No physical simulation step or parameter fit was performed.

All 360 MP4 frames decoded at 960 by 1008 pixels and 15 fps, giving 24 seconds.
All 192 GIF frames decoded at 800 by 840 pixels, giving 24.01 seconds with
looping enabled. Initial, middle, settling and final frames were inspected;
text bands and chart canvas bounds passed. All local links and the seven
original completion markers passed. The targeted release, demo, model and
adaptation suite passed 32 tests, including endpoint-error reproduction.
Lint passed. Controller sources and retained evidence are unchanged.

### Visual guide refinement

The README visual guide contains ten static figures and two animated GIFs,
with two full-resolution MP4 replays. All eight result plots regenerated with
shared STIX typography and external legends, passing canvas-bound and
neighbouring-panel checks. Header and architecture drawings also passed their
canvas checks. Plot SVGs accompany PNGs for enlargement.

All 540 video frames and 288 GIF frames decoded successfully. Control replay
is 960 by 880 pixels over 24 seconds; nominal motion is 960 by 800 over twelve
seconds. GIFs are 800 pixels wide and use centisecond timing, giving 24.01 and
12.01 seconds. First, middle and final frames were inspected. Text bands are
checked on every rendered frame; the time cursor is confined to the plot.
All visual-guide links and seven original completion markers passed. The
targeted release, demo-integrity, model and adaptation suite passed 31 tests;
lint and compilation passed. No controller or raw evidence file changed.

The planar result figure uses the existing v2.0 outcome-conditioned seed
selection rule. Its two illustrative replays preserve that scope and do not
constitute a new campaign. Panda animations use forward kinematics of saved
states without advancing the physical plant.

The paired video was rendered from the hash-verified adaptation pilot's first
payload/friction case, seed `9804300`, without new physical simulation steps.
Its final error values match the retained episode records. All 360 encoded
frames decoded at 960 by 720 pixels and 15 fps, giving a 24-second video;
the initial, middle and final images were inspected. Documentation links passed.
Rendering used Python 3.12.3, MuJoCo 3.13.0 with EGL, Matplotlib 3.11.2,
Pillow 12.3.0 and imageio-ffmpeg 0.6.0.

The targeted release, demo-integrity, Panda model and adaptation suite passed
31 tests. The complete lint check passed. The demo tests reject modified
evidence and check that the replay uses the first retained paired case.
The reference file's direct dependency versions match the installed runtime.
Only the package version changed under `sarrl/`; controller arithmetic and
original evidence files remain unchanged.

The wheel and source distribution built successfully. The installed wheel,
outside the checkout with numerical dependencies reused, reported package and
distribution version 2.3.1, verified all twelve model files and ran the nominal
example successfully for 300 control steps without clipping or an unsafe event.
All seven pilot completion markers and 173 frozen source blobs passed again,
covering 7,060 retained new episode records.

## v2.3 release checks

The release tree passed 558 tests with one skip for unavailable local
diagnostic records (559 collected), on Python 3.12.3, NumPy 2.5.2 and
MuJoCo 3.13.0. Lint and compilation passed, including the short Panda
example. The changes to score-memory evaluation correct a protocol prose
label; controller arithmetic and retained numerical evidence are unchanged.

A wheel and source distribution were built locally without fetching build
dependencies. The wheel was installed into a clean target directory outside
the checkout, reusing the runtime's numerical dependencies. Imports resolved
to that installed package, with both package and distribution metadata at
2.3.0. On MuJoCo 3.13.0 and isolated 3.15.0, all twelve Panda model/mesh/license
hashes passed, nominal/adaptive/delay/memory APIs loaded, and the planar
inertia matrix remained positive definite. The example copied from the
source distribution completed 300 control steps successfully on both
engines, with no unsafe event or clipping. It uses engineering seed
`9804100`; this is packaging evidence, not a new pilot result.

All seven Panda completion markers passed, covering 7,060 retained new
episode records. Every one of their 173 recorded source blobs matched its
frozen commit and remained reachable from release history. From detached
commit `c4736d4`, the public score-memory audit passed again on all 1,920
records and 96 traces with eighty causal estimator replays and independent
score-weight reconstruction, without new physical episodes. The
[release guide](v2.3.md) documents auditing with each recorded source/runtime.
Original manifests, including the corrected prose label's historical
wording, remain unchanged.

## Test coverage checklist

The automated regression suite is executed across Python 3.10, 3.11 and 3.12.
The exact test count evolves with the codebase; release-specific historical counts are retained below. The CI pipeline runs `ruff check .`, `python -m compileall -q sarrl tests
tools` and `pytest`; a second job installs the optional `mujoco` extra and runs `pytest` again, so the
MuJoCo tests, which skip without the engine, also run in CI. Important regression and numerical tests cover:

- inertia-matrix symmetry and positive definiteness;
- finite-difference verification of rigid-body identities;
- forward/inverse dynamics round trips;
- analytical Jacobian vs finite differences;
- RK4 conservative-energy behavior;
- computed-torque closed-loop convergence;
- nonlinear MPC feasibility and objective improvement;
- deterministic randomization;
- sensor noise;
- command delay;
- injected faults;
- replay-buffer reproducibility;
- SAC Bellman targets;
- target-network updates;
- tanh log-probability correction;
- entropy-temperature optimization;
- deterministic policy evaluation without stochastic RNG consumption;
- architecture-safe SAC checkpoint loading;
- exact replay/environment/RNG training continuation;
- CUDA RNG-state checkpoint restoration;
- causal context-encoder behavior;
- residual-dynamics learning and checkpointing;
- motor-gain mismatch visibility;
- ensemble uncertainty;
- exact 2-D safety projection;
- HOCBF infeasibility semantics;
- runtime-stack composition;
- Wilson intervals;
- paired bootstrap comparisons;
- validation/held-out seed separation.

## Panda nominal integration checks

Eleven simulator-gated tests in `tests/test_panda.py` verify model-file hashes,
seven unit-gear torque controls, symmetric positive-definite mass matrices,
forward/inverse dynamics agreement, stationary gravity compensation, quintic
reference endpoint derivatives, exact repeated rollouts, independent nominal
and plant models, torque limits, invalid-input handling, physical contact
observation and refusal to overwrite outputs or use unregistered pilot seeds.
They passed under Python 3.12.3 and MuJoCo 3.13.0 in WSL.
The full suite passed with 450 tests and one skip for unavailable local
diagnostic records (451 collected); lint and compilation checks passed.

A local wheel was built without fetching dependencies. The Panda was loaded
from the extracted wheel, with all twelve model, collision-mesh and license
hashes verified, without using the checkout's assets. See [`panda.md`](panda.md)
for the pilot's limited scope.

The retained pilot contains 20 successful nominal episodes on
`9804000..9804019`, 30,000 physical steps and no contacts, joint-limit/speed
violations or saturation. Every completion-marker hash was verified. Its
eight source hashes match implementation commit `adb2344`, despite the run
preceding that commit; `source_link.json` records the comparison. The figure
was generated from the first episode's saved trace without new physical
steps.

## Panda adaptation integration checks

Fifteen tests in `tests/test_panda_adaptive.py` cover the eight-parameter
regressor against independent engine dynamics, payload inertia and damping,
excited and unobserved RLS parameters, estimate bounds, nominal equivalence,
paired task and parameter sampling, deterministic repeated rollouts,
reproduction tolerances, adaptive tracking under a modelled load, and checked
cell resume. A corrupt saved cell is rejected before any pending case runs.
The adaptive controller receives neither the true parameters nor the plant.

Before the runtime compatibility follow-up below, the full suite passed
with 464 tests and one skip for unavailable local diagnostic records
(465 collected), under Python 3.12.3 and MuJoCo 3.13.0. After tightening
the resume guards, the then-fourteen adaptation tests passed again.
Lint and compilation checks passed.

CI's MuJoCo 3.15.0 produced a `1.62e-8 N m` difference in the first nominal
reference's maximum torque, outside the frozen reproduction tolerance.
An isolated local 3.15.0 installation reproduced that difference while
keeping NumPy at the reference version. The original XML-loading path and
the zero-parameter extension produced identical episode records and traces
within the same engine. The regression suite now checks that equivalence
on every supported runtime, and compares against retained numerical data
only when MuJoCo and NumPy match their recorded versions. The runner's
strict reproduction tolerances and retained evidence are unchanged.
All fifteen adaptation tests passed under MuJoCo 3.13.0. All Panda tests
under isolated 3.15.0 passed: 25 passed, one skipped for the reference
runtime mismatch. This adds one test, bringing collection to 466 tests.

The pilot ran once from clean implementation commit `3086aeb`. Its history
scan covered 160 reachable commits, 153 distinct trees and 494 evidence
blobs without finding the reserved seed block. Before opening new seeds,
all twenty retained nominal episodes reproduced on every field. The new
240 episodes used 360,000 physical steps, with no observed contact,
position/speed violation or saturation; reproduction adds 30,000 steps.
All completion hashes, 240 unique cell identities and twelve 301-sample
traces were checked. Every recorded source hash matches the exact bytes in
the implementation commit. The figure uses only those saved records.
The completed pilot's `--resume` command was also executed and refused to
replace its outputs; completion hashes were checked again afterwards.

With payload alone or payload plus friction, fixed-model success was 0/20
and adaptive success was 20/20. Both achieved 20/20 on nominal and
friction-only cases; the oracle achieved 20/20 in every condition.
The combined final joint error averaged 78.348 mrad fixed and 0.029043 mrad
adaptive. The design, parameter-error diagnostics and limits of this
descriptive result are in [`panda-adaptation.md`](panda-adaptation.md).

## Panda sensor-noise and command-delay checks

Sixteen collected cases in `tests/test_panda_robustness.py` check invalid
interface settings, cached/copy-safe/repeated sensor samples, the exact
two-interval zero-initialized command queue and clipping, unchanged ideal
episode and trace fields, measured-only adaptive transitions using the
current sent command, shared task/parameter/noise draws, ground-truth
success scoring, deterministic noisy delayed episodes, cross-condition
pairing rejection, nested reproduction comparisons, unchanged previous
summary, runner output/seed guards and refusal after a reproduction mismatch.

The full suite passed with 481 tests and one skip for unavailable local
diagnostic records (482 collected), under Python 3.12.3, NumPy 2.5.2 and
MuJoCo 3.13.0. All Panda tests under isolated MuJoCo 3.15.0 passed: 41
passed and one skipped for the original numerical-reference runtime.
Lint and compilation checks passed. The committed design in
[`panda-robustness.md`](panda-robustness.md) specifies the pilot.

The pilot ran once from clean commit `2fc7a48`, after scanning 163 reachable
commits, 156 distinct trees and 501 evidence blobs and reproducing all 240
retained adaptation episodes on every field. Its 240 new episodes used
360,000 physical steps, plus 360,000 for reproduction. In each condition
(ideal, noise, delay, noise plus delay), fixed success was 0/20 and both
adaptive and oracle success were 20/20. No contacts, position/speed
violations or saturation were observed. Adaptive mean final joint error
rose from 0.033114 mrad ideal to 3.002693 mrad with both perturbations;
parameter error and bound hits increased under delay.

All completion hashes and recorded source hashes match the retained files
and implementation commit. A separate saved-data audit verified all 240
success flags and final errors, cross-condition pairing, twelve 301-sample
traces, exact RNG-derived sensor samples, every delayed command, torque
limits and recomputed tracking RMS. The completed-run resume command was
executed and refused replacement. The result is descriptive at one chosen
noise/delay setting, not a tolerance envelope or safety certificate.

## Panda delay identification and prediction checks

Twenty-seven collected cases in `tests/test_panda_delay.py` check delay
configuration and privileged inputs, exact zero-delay equivalence with the
original adaptive rollout, causal target alignment and zero startup,
forward acceleration against independently configured engine dynamics,
pending-command prediction order, private-model equilibrium, counted
prediction fallback, invalid-transition rejection, identification of all
five grid delays, paired tasks/parameters/noise, privilege labels,
reference lookahead with current-time tracking, exact repeated prediction
rollouts and runner/reproduction guards.

The full suite passed with 508 tests and one skip for unavailable local
diagnostic records (509 collected), under Python 3.12.3, NumPy 2.5.2 and
MuJoCo 3.13.0. All Panda tests also passed on isolated MuJoCo 3.15.0:
68 passed and one skipped for the recorded numerical-reference runtime.
Lint and compilation checks passed. The committed design in
[`panda-delay.md`](panda-delay.md) was frozen before the seven-arm pilot.

The pilot ran once from clean commit `b7440fe`, after scanning 165 reachable
commits, 158 distinct trees and 508 evidence blobs and reproducing all 240
previous noise/delay episodes on every field. It retained 560 unique new
cells, twenty paired seeds per condition across seven arms, using 840,000
physical steps plus 360,000 for reproduction. Prediction used a separately
counted 239,400 virtual substeps. No contact, position/speed violation,
saturation or prediction fallback was observed. Fixed success was 0/20
in each condition and every other arm achieved 20/20.

With noise plus delay, mean final joint error fell from 3.360627 mrad for
original adaptation to 0.341801 mrad for bank alignment and 0.337742 mrad
for bank prediction. Their mean tracking RMS was 4.139280, 2.298004 and
2.076842 mrad, respectively. Both banks selected the correct final delay
in every episode. The retained result includes the small noise-free final
error regression from prediction and the favorable zero-startup queue.

A separate saved-data audit checked every completion and source hash,
all 560 cell identities, recomputed success flags, final errors and summary,
and all twenty-eight 301-sample traces. It verified exact RNG-derived noise,
every FIFO-delivered torque, torque clipping, current-time RMS, selected
delay histories and future references. Every sent command was recomputed
from the recorded control state, selected model and reference. Sampled
replays on all eight predictive traces checked queued-state integration
without new physical episodes. The completed-run resume command refused
replacement; completion hashes were checked again afterwards. The figure
was generated from retained data only.

## Panda posture and delay-grid checks

Twelve collected cases in `tests/test_panda_posture_delay.py` verify
translated-path displacement and clearance, the full delay/noise grid,
exact narrow-task equivalence with the previous rollout, wide noisy
identification at 10, 30 and 40 ms, factorial pairing and translation
checks, rejection of changed frozen controller bytes, identical serial
and spawned-process records/traces, checked resume, reproduction blocking
and seed/worker/output guards. All twelve passed on MuJoCo 3.13.0 and
isolated 3.15.0. The full suite passed with 520 tests and one skip for
unavailable local diagnostic records (521 collected), under Python 3.12.3,
NumPy 2.5.2 and MuJoCo 3.13.0. Lint and compilation checks passed.

The pilot ran once from clean commit `a5041c7`. All six frozen controller,
plant and interface source hashes matched the previous manifest. The seed
scan checked 167 reachable commits, 160 distinct trees and 515 evidence
blobs. All 560 previous records reproduced on every field before the new
block. The 2,800 new episodes used 4,200,000 physical steps, plus 840,000
for reproduction. New prediction substeps total 2,392,000; reproduction
adds another 239,400 virtual substeps.

Each delay-aware arm and the physical oracle achieved 400/400 success;
fixed achieved 0/400 and original adaptive 374/400. The adaptive failures
were at wide postures with 30 and 40 ms delays, both clean and noisy.
No contacts, position/speed violations or prediction fallbacks occurred.
The original adaptive arm saturated in 46 episodes; no other arm did.
Both banks selected the correct final delay in all 400 of their episodes.
Prediction's mean RMS worsened relative to bank alignment at wide 40 ms,
by 0.65% clean and 0.85% noisy; all cell outcomes and final-error regressions
are retained in [`panda-posture-delay.md`](panda-posture-delay.md).

A separate saved-data audit checked every completion and source hash,
all 2,800 task/parameter draws, terminal forward kinematics, success flags,
cell identities and the recomputed summary. All 140 first-seed traces
were verified for exact exogenous noise, FIFO delivery, current/future
references, sent commands, saturation counters and tracking RMS. There
were also 280 sampled private-model prediction replays, without new
physical episodes. The completed-run resume command refused replacement;
completion hashes were checked again afterwards. The saved-data figure
was visually inspected.

## Panda populated-queue and variable-latency checks

Sixteen collected cases in `tests/test_panda_variable_delay.py` verify
constant-transport equivalence with the original FIFO and cold rollout,
packet hold and stale rejection, applied age versus sent-packet delay,
warmup/main-state continuity, shared handoff and scored noise, fresh
estimators, total physical-step accounting and warmup safety inclusion.
They also check that a future switch cannot change pre-switch states or
commands, privileged diagnostic alignment, paired-summary rejection,
serial/parallel identity and execution guards. The full local suite passed
with 536 tests and one skip for unavailable local diagnostic records
(537 collected), on Python 3.12.3, NumPy 2.5.2 and MuJoCo 3.13.0.
All sixteen new cases passed on isolated MuJoCo 3.15.0. Lint and compilation
checks passed; the saved-data audit CLI help was checked.

The pilot ran once from clean source commit `8900bc6`. Its history scan
covered 169 reachable commits, 162 distinct trees and 522 evidence blobs.
All 2,800 previous episodes matched on every compared field before the
new seeds opened. The 1,280 new records contain 273/320 original adaptive
successes, 320/320 for frozen bank alignment and the known-age diagnostic,
and 0/320 for fixed. No contacts or position/speed violations occurred,
including warmup; only original adaptive clipped, in 44 episodes and
152 scored control intervals.

The bank selected the correct final delay in all 160 constant-profile
episodes, but none of the 160 changing-profile episodes. All changing
cases kept the initial age throughout transitions 100 to 299. Tracking
success does not establish identification of the changed regime. See
[`panda-variable-delay.md`](panda-variable-delay.md) for all success cells,
tracking comparisons, the frozen protocol and retained evidence.

The frozen public audit passed on 1,280 records and 64 first-seed traces,
including 32 physical warmups and 48 causal estimator replays. It checked
saved-data hashes, source blobs, model provenance, torque calculations,
noise streams, packet delivery, clipping and tracking/age metrics without
new physical episodes. Total physical steps were 6,184,000 including
reproduction; 2,392,000 reproduction virtual substeps were counted
separately. Completed-output resume refused replacement. Completion
hashes passed again, and the saved-data figure was visually inspected.

## Panda delay-score memory checks

Twenty-two new cases in `tests/test_panda_score_memory.py` cover independent
batch score weights, causal prefixes, dropping an old score advantage,
validation without partial updates, and equality of every candidate RLS
diagnostic with the frozen bank on identical exogenous transitions.
They verify the full six-arm factorial's handoff/noise pairing, score
traces, physical-step accounting, future-profile causality, factory/default
equivalence, serial/parallel identity and execution guards. The new cases
passed on the recorded NumPy 2.5.2 and MuJoCo 3.13.0 runtime and on
isolated MuJoCo 3.15.0. Lint, compilation and the public audit's CLI help
passed.

The full local suite passed with 558 tests and one skip for unavailable
local diagnostic records (559 collected), on Python 3.12.3, NumPy 2.5.2
and MuJoCo 3.13.0. The pilot then ran once from clean commit `c4736d4`.
The seed scan covered 171 reachable commits, 164 distinct trees and 529
evidence blobs. All 1,280 previous records matched on every compared field
before the 1,920 new episodes. No source or memory length changed after
the new seeds opened.

All four alignment arms succeeded in 320/320, original adaptive in 276/320,
fixed in 0/320. Correct final age on changing profiles was 1/160 for
cumulative scores, 74/160 for exponential and 72/160 for the window;
on constant profiles it was 160/160, 159/160 and 152/160. Both memory
variants increased mean final error and RMS after 1 s in every noisy
changing condition. The [full results](panda-score-memory.md) retain these
regressions and the clean/noisy split without promoting a new baseline.

No contacts or position/speed violations occurred, including warmup.
Original adaptive clipped in 48 episodes, 192 scored intervals. Each
alignment arm clipped in the same twelve scenario/seed cases, 74 intervals
per arm; fixed and warmups had zero clipping. Physical steps totaled
4,960,000 including reproduction, with no virtual prediction substeps.
The frozen public audit passed for all 1,920 records and 96 traces,
including 48 continuous warmups, eighty causal estimator replays and
independent batch score-weight reconstruction. Completion hashes passed
again, completed-output resume refused replacement, and the figure was
generated from saved data and visually inspected.

## v2.2.0 uncertainty-margin evidence

The campaign trained nothing. The protocol section of `docs/experiments.md`
and its implementation were committed before any decision seed was opened
(freeze commit `817db04`) and sealed by `b3925ed` in `docs/v22_seal.json`.
Unlike v2.1, the protocol went through no external review round: it was
sealed on the author's decision, with three open points recorded in the
protocol and left as implemented. Before the seal, the whole reproduction
block was run once locally against the retained v2.1 log (300 of 300 rows,
all 39 retained fields).

Before the first episode the runner verified the sealed source state and a
clean tree, scanned the CSV, JSON and JSONL blobs of 142 reachable commits
(140 distinct trees, 478 blobs) for seed-named values in `56200..58099`
(none), checked the registry, and recorded the reproduction reference hash,
the runtime (Python 3.11.15, torch 2.14.1, numpy 2.4.6, MuJoCo 3.14.0 on
Linux), the installed distributions and the four-worker fingerprint in
`manifest.json`. The reproduction block ran first and matched the retained
v2.1 rows on 300 of 300 episodes and every field.

The campaign was executed twice from the sealed state. The first
execution's manifest listed, among the installed distributions of the
execution environment, a package unrelated to the project; it was
uninstalled and the runner started again from the same commit on an empty
output directory. Both executions produced the same `episodes.jsonl`
(SHA-256 `6c66dcc1ed480500a7ecf3b5d2074ff3c33cce234b26460807797ad4a5bd5dee`) and the same decision; the retained files are the
second execution's.

Retained under `results/uncertainty_margin_v22/`: `manifest.json`,
`journal.jsonl` (17,400 session-tagged cell records in completion order),
`episodes.jsonl` (the same records in the planned order), `episodes.csv`,
`decision.json` with the three contrasts in every scenario, the
conditional severities, the margin diagnostics, per-cell estimator
diagnostics and summaries, the reproduction check and the protocol, and
`complete.json` hashing the five files. The decision file records
`confirmed`, with every estimator cell valid and the reproduction check at
300/300. At the v2.2.0 release the suite collects 440 tests.

## v2.1.0 certificate factorial evidence

The campaign trained nothing. The protocol section of `docs/experiments.md`
and its implementation were committed before any decision seed was opened
(freeze commit `d7c20d6`) and sealed by `992329d` in `docs/v21_seal.json`.
The design went through four rounds of external adversarial review: the
first produced ten changes (the controller factor scoped to the whole v2.0
stack, a two-phase runner that checks the reproduction block before any
decision seed, comparison of every retained field, spy tests on the
certificate-only estimator, validity per cell, statements with the lower
bound above the margin, safety endpoints in four families, a seed registry
with a scan of every release tag, atomic outputs), the second six (a scan
of every reachable commit, a journal that survives a truncated final
record, simulator-free guard tests with a phase-gate test, the retained
directory unignored, the prediction contrast reported, the estimator test
run with noise on), the third four (the journal tail repaired on disk, the
ignore rules frozen, wording, scan manifest fields), and the fourth
returned approval with one non-blocking note, an `fsync` after the tail
repair, applied in the freeze commit.

Before the first episode the runner verified the sealed source state and a
clean tree, scanned the CSV, JSON and JSONL blobs of 138 reachable commits
(133 distinct trees, 467 blobs) for seed-named values in `54200..56099`
(none), checked the registry, and recorded the reproduction reference hash,
the runtime (Python 3.12.3, torch 2.12.0, numpy 2.5.2, MuJoCo 3.13.0 under
WSL2), the installed distributions and the six-worker fingerprint in
`manifest.json`. The reproduction block ran first and matched the retained
v2.0 rows on 600 of 600 episodes and every field.

Retained under `results/certificate_factorial_v21/`: `manifest.json`,
`journal.jsonl` (23,700 session-tagged cell records in completion order),
`episodes.jsonl` (the same records in the planned order), `episodes.csv`,
`decision.json` with the success effects and interaction, the four safety
families, the prediction contrast, per-cell estimator diagnostics and
summaries, the reproduction check and the protocol, and `complete.json`
hashing the five files. The decision file records `partial`, with the
failed statement named, every estimator cell valid and the reproduction
check at 600/600. At the v2.1.0 release the suite collects 411 tests.

## v2.0.0 MuJoCo campaign evidence

The campaign trained nothing. The protocol section of `docs/experiments.md`
and its implementation were committed before any decision seed was opened
(freeze commit `e334eb1`) and sealed by `c6fd964` in `docs/v20_seal.json`.
The design went through three rounds of external adversarial review, the
third returning approval; the first round produced five changes (strict
reproduction of every retained A0 field, one sensor sample per plant state,
a per-step guard decision with an estimator-validity condition in the rule,
the prediction-error endpoint named against the analytical-parameter oracle,
the descriptive block restricted to the unfiltered arms) and the second a
true relative tolerance in the reproduction check.

Before the first episode the runner verified the sealed source state and a
clean tree, scanned 440 committed CSV/JSON artifacts for seed-named values in
`52200..54099` (none), and recorded the reproduction reference hash, the
runtime (Python 3.12.3, torch 2.12.0, numpy 2.5.2, MuJoCo 3.13.0 under WSL2),
the installed distributions and the six-worker fingerprint in
`manifest.json`.

Retained under `results/adaptive_mujoco_v20/`: `manifest.json`,
`journal.jsonl` (12,600 session-tagged cell records in completion order),
`episodes.jsonl` (the same records in the planned order), `episodes.csv`,
`decision.json` with contrasts, estimator diagnostics, per-cell summaries for
the decision and descriptive blocks, the transfer check and the protocol,
and `complete.json` hashing the five files. The decision file records `go`,
no veto, non-inferiority in all three scenarios, the estimator valid and the
reproduction check at 300/300 on all seven retained fields.

The MuJoCo plant itself is verified by `tests/test_mujoco_planar.py`: without
dry friction its accelerations match the analytical model within `1e-5` on
100 random states; the same seed yields the same plant draws, target, initial
state, delay and gains on both plants; fault, armature and actuator options
reach the engine; state dictionaries round-trip. Those tests skip where
MuJoCo is not installed, as in the default CI environment. At the v2.0.0
release the suite collects 380 tests.

## v1.9.0 adaptive nominal control evidence

The campaign trained nothing. The protocol section of `docs/experiments.md`
and the code that implements it were committed before any decision seed was
opened (freeze commit `7b18c68`) and sealed by a later commit that records
that hash in `docs/v19_seal.json` (`2570afb`). The design went through five
rounds of external adversarial review before the freeze; the reviewer's
findings and their outcomes are summarised in `docs/experiments.md` and led,
among other changes, to the non-inferiority condition, the move of the
decision seeds to a range no retained artifact had used, the committed-blob
seed scan, the at-least-once journal with an exclusive lock and the mandatory
reproduction check.

Before the first episode the runner verified that the frozen paths of `HEAD`
equalled the sealed commit with no local change, that the seal named an
existing ancestor commit, that 433 committed CSV/JSON artifacts held no
seed-named value in `50200..52099`, and recorded the reproduction reference
hash, the runtime (Python 3.12.3, torch 2.12.0, numpy 2.5.2 under WSL2), the
installed distributions and the six-worker fingerprint in `manifest.json`.

Retained under `results/adaptive_nominal_v19/`: `manifest.json`,
`journal.jsonl` (12,600 cell records in completion order, each tagged with
its session), `episodes.jsonl` (the same records in the planned order),
`episodes.csv`, `decision.json` with contrasts, per-cell summaries for both
blocks, the reproduction check and the protocol, and `complete.json` hashing
the five files. The decision file records `go`, no veto, non-inferiority in
all three scenarios and the reproduction check at 300/300.

The reproduction block is the integrity link to earlier evidence: the
unfiltered fixed arm on seeds `50000..50099` matched every retained v1.3
`A0_computed_torque` row on success and final distance, so the environment,
the nominal controller and the evaluator are byte-for-byte the ones that
produced the v1.3 and v1.4 tables.

Limits carried into the claim: full-state noise-free feedback, identical for
both arms; no compensation of the identified delay in the PD gains; targets
whose shorter angular path crosses a joint limit stay blocked by the filter;
the learned policies were not re-evaluated on the adaptive nominal. The
first launch failed at import and opened no seed; the official run used the
module invocation. At the v1.9.0 release the suite collects 362 tests.

## v1.7.0 and v1.8.0 training campaigns and diagnosis evidence

Both campaigns trained new policies; the diagnosis that follows them did not.
At the v1.8.0 freeze the suite collects 315 tests across the matrix and passes
locally with CUDA available; tests that need a CUDA device or the local
diagnostic artifacts skip on machines without them.

### v1.7 safety-aware training

Ten training runs, five paired seeds `20..24` per condition, 200,000 decisions
each, completed on a single machine with the in-loop arm trained through
`SafetyProjectedEnv`. The evaluation writer initially failed after the full
computation because of a CSV schema mismatch; the writer was corrected and the
whole 7,000-episode evaluation was re-run from the selected checkpoints. The
training code is the source commit recorded in each `run_manifest.json`.

The aggregate decision file records `no_go` with the primary ID success
difference `-0.1728`, 95% interval `[-0.236, -0.098]`, one-sided upper bound
`-0.110`, per-seed differences `-0.150, -0.038, -0.198, -0.220, -0.258`, an
unsafe-episode guardrail difference of `+0.0008` `[-0.0108, +0.0124]` and an
infeasibility guardrail difference of `-0.0004` `[-0.0080, +0.0072]`. The
evaluation manifest binds the six output files by SHA-256 and references the
checkpoint inventory at
`e7ffbac7404abac30f756de8305c966634b972f7a5a6dc0afeaa1d94534ba3a6`.

Retained: `results/safety_aware_training/checkpoint_inventory.json`,
`evaluation/*.json`, `evaluation/*.csv`, per-seed `run_manifest.json`,
`selection.json`, `validation.csv` and `episodes.csv`, and the in-loop
`training_safety.csv` curves. The ten checkpoints, the per-seed evaluation
shards and the training-replay critic diagnostic remain local; the shards
duplicate the top-level CSVs row for row.

### v1.8 terminal-penalty ablation

The frozen protocol is embedded in `results/penalty_ablation/campaign.json`
together with the runtime (Python 3.12.3, torch 2.12.0+cu130, numpy 2.5.2,
five workers, one thread per process) and the source commit
`68df4213e2ddae643bb6bdd841719e2a78b422fd`. The plan hash recorded in the
campaign file is
`0e5456bccac220e94aabc69a1e59db73bb937a6490fb8a8ef409c12775e72dd1` and the
checkpoint inventory hash is
`78f359c13e3146d5b383c522d00809149403af92ed06839895ccf24f47a1754f`.

Before the campaign, 240 tests passed with Ruff and compilation clean, and a
24-episode engineering smoke run used separate output paths and seeds. The
campaign was interrupted once by a GPU driver fault and resumed from retained
checkpoints as described in `docs/experiments.md`; the final inventory lists
the hashes of the checkpoints each run resumed from. `complete.json` certifies
the aggregation and `workflow_complete.json` additionally binds the critic
diagnostic; the presence of a CSV alone is not evidence of completion.

The aggregate records `inconclusive` with ID success difference `+0.0452`
`[-0.0088, +0.1040]`, per-seed `+0.018, +0.140, +0.078, -0.004, -0.006`,
ID unsafe-episode difference `-0.010` `[-0.0268, +0.0072]`, ID infeasibility
`+0.0008` `[-0.0028, +0.0048]`, fault infeasibility `+0.010` with the veto rule
using a strict inequality at `+0.01`, and a late-training abort difference of
`+0.00555`.

The failure audit under `results/penalty_ablation/failure_audit/` fixed its
60-episode protocol before execution, verified 208 hashes listed in the v1.7
and v1.8 inventories and completion chain, classified all 7,000 official
episodes and retained the 60 trajectories. Its `audit.py` refuses to overwrite
a completed audit. The audit manifest records `device = cpu` as a constant
while the SAC loader selected CUDA when available; a separate provenance check
(`results/residual_device_provenance_20260906/`, completion hash
`4c505487376c0d35dbc34502f08429eaaf19e5d2491d7376c6cef0d506fd9b00`) reproduced
all 60 historical first actions exactly on CUDA and only 8 on CPU, so the audit
is treated as CUDA inference and the manifest field as an unverified
declaration that was not rewritten.

### Exploratory diagnosis

The null-residual campaign (`results/residual_diagnosis_20260905/`, completion
hash `76961ce34cc81aff4df6d8db227409f12140003cef2eac730345a3f1dede8c18`)
executed 66 episodes, 15,582 physical steps against a preregistered ceiling of
16,500, with no invalid or repeated cell. Its manifest binds the protocol, the
device amendment, 43 source and configuration files and the ten checkpoints;
all 60 policy episodes reproduce the audit rows within `rtol = atol = 1e-10`.
One deviation is recorded: during synthetic testing of the runner a pytest
selector repeated the fixtures, so 305 technical physical attempts were run
against a planned maximum of 200. They are outside the scientific count and
were not authorised retroactively; the affected test now uses synthetic data.

The nominal-integral campaign (`results/nominal_integral_v19a_20260907/`,
completion hash
`989221d9a07fea14716f67d9b3b4eeef28e01833ad0e049ea0fdfd9017c14f47`) executed
12 episodes and 2,733 physical steps against a ceiling of 3,000 on CPU in
float64 with one numerical thread. A first launcher failed before any episode
because a shell interpreted backticks in the code it forwarded; it consumed no
steps and was replaced by a launcher that passes code through standard input.
The runner verifies 47 source hashes, the historical references and the
frozen protocol before the first step, and the six R0 episodes equal the six
historical Z episodes on every shared field.

The four read-only analyses (`results/residual_analysis_20260907/`,
`results/nominal_static_review_20260907/`,
`results/nominal_integral_analysis_20260912/`, completion hash
`c218da12974312796df9426cdad18cf5a9561bff592d065640521144d94cc60d`, and
`results/integral_fault_static_review_20260912/`, completion hash
`5e368f12d85eb82cf57ec7a04e7b48c49d53cbd2269f7af12b7d65810d876edb`) use JSON
and the standard library or NumPy only, never import the simulator or the
policy, and each ran once. The torque balance closes within `7.1e-15 N m`, the
joint-error reconstruction within `1e-15 rad`, the acceleration decomposition
within `1.14e-13 rad/s^2` and the nominal barrier margin within
`3.2e-14`. These identities verify the recording and the algebra; they do not
validate the plant model or identify causal effects.

## v1.6.0 disagreement and operational failure evidence

v1.6 introduced no new episodes and no retraining. It is a preregistered
re-analysis of the retained v1.5 Phase-C gate-off arm, with the protocol,
statistic, threshold, decision rule and operating characteristics frozen and
committed before the association was computed.

### Preregistered operating characteristics

The procedure was simulated on synthetic nulls before the data were opened. The
simulator was verified against its own targets on 2,000 replicates per grid
point: empirical prevalence `0.0479` and `0.1837` against targets `0.048` and
`0.184`, empirical AUC `0.7008` in both scenarios against a target of `0.70`,
and latent within-seed correlation `0.0995` and `0.0990` against `0.100`.

Two corrections followed and are part of the record. The uniform 5th-percentile
bound was measured **anticonservative** under the composite null, with a
worst-case size of **7.8%** `[6.7%, 9.1%]` against a nominal 5%, attained with
`id_reference` on the boundary and `ood_compound` deep in the alternative. The
inflation is prevalence-driven: with the higher-prevalence scenario on the
boundary the size runs `0.044`-`0.063`, and with the 4.8%-event scenario on the
boundary it runs `0.056`-`0.078`. Critical quantiles were therefore recalibrated
per component on synthetic nulls only, selected on one seed and validated on a
disjoint one: **2.0th percentile for `id_reference`, 2.5th for `ood_compound`**,
validated worst-case size **3.65%** `[2.91%, 4.56%]`, conservative against the
nominal 5%.

Power was then recomputed under the calibrated rule. Joint power at the
preregistered target of AUC `0.70` and ICC `0.10` is **38.5%** against an 80%
goal, so the screen was executed and is reported as an explicitly **low-power
feasibility screen**:

| True AUC | Joint power | `id_reference` | `ood_compound` |
|---:|---:|---:|---:|
| 0.60 | 0.2% | 2.8% | 2.8% |
| 0.65 | 5.9% | 14.9% | 34.5% |
| 0.70 | 38.5% | 42.9% | 89.9% |
| 0.75 | 73.6% | 73.8% | 99.8% |
| 0.80 | 93.0% | 93.0% | 100.0% |

`id_reference` is the binding component throughout, with 24 events in 500
episodes; joint power tracks it closely because `ood_compound` rejects almost
certainly at moderate effect sizes.

### Result

| Scenario | AUC | Events | Lower bound | Upper bound |
|---|---:|---:|---:|---:|
| `id_reference` | 0.5200 | 24/500 | 0.3958 | 0.6391 |
| `ood_compound` | 0.4787 | 92/500 | 0.4085 | 0.5479 |

**Decision: Inconclusive.** Both point estimates sit essentially at chance.
`ood_compound`, the better-powered component, meets the component-negative
criterion on its own with an upper bound of `0.5479`. Precisely stated, the
**`ood_compound` component excludes an association above the preregistered
`AUC > 0.60` threshold, while the `id_reference` component does not**, so the
overall two-scenario decision remains Inconclusive. Neither component
establishes the absence of all weaker associations; weaker effects remain
plausible, particularly because `id_reference` is unresolved. The global verdict is Inconclusive rather than Negative only
because `id_reference`'s upper bound of `0.6391` does not fall below the
threshold, a direct consequence of its 24 events and the low power recorded
before the data were opened.

Per-cell AUCs show the dispersion those event counts imply: `id_reference`
0.643 / 0.612 / 0.447 / 0.716 / 0.470 and `ood_compound` 0.547 / 0.345 / 0.439 /
0.499 / 0.573.

### Secondary analyses, excluded from the decision by preregistration

| Analysis | `id_reference` | `ood_compound` |
|---|---:|---:|
| `uncertainty_ratio` predictor | 0.5016 | 0.4925 |
| `unsafe_episode` endpoint only | 0.5332 | 0.4787 |
| whole-episode predictor | 0.5851 | 0.4185 |
| pre-event truncated | 0.5082 | 0.4782 |

The whole-episode predictor gives a higher `id_reference` AUC than the fixed
window. This is **consistent with** duration and censoring contamination, since
whole-episode summaries partly encode how long an episode ran and that is itself
outcome-dependent, but a single paired comparison does not demonstrate the
mechanism and no such demonstration is claimed. The pre-event-truncated
sensitivity, at `0.5082` with no in-scope episode dropped, indicates the primary
estimate is not carried by post-failure observations.

The `motor_fault` onset-anchored secondary gives AUC `0.6531` over 471 included
and 29 excluded episodes (20 non-failure, 9 failure). It is reported
**exclusively as a selection-affected sensitivity**: its inclusion criterion,
surviving to raw step 45, is outcome-dependent, so the quantity is a conditional
post-onset estimand on a survivor subset and is not comparable with the primary
scenario AUCs. It cannot reopen the decision.

### Interpretation and scope

v1.6 closes as a preregistered **Inconclusive / no-go**. The conditional
intervention arm, which would have used disagreement to tighten the HOCBF margin
rather than to weaken the residual policy, is closed and was never executed; its
numerical parameters were never frozen.

The `ood_compound` component does not support an association above `AUC 0.60`;
the `id_reference` component remains unresolved, so the overall evidence is
Inconclusive. Nothing here establishes that no weak association exists, identifies
any causal relationship, or supports any deployment or actionability claim. The
analysis is observational and conditional on the five frozen training artifacts;
inference to other training runs would require new artifacts.

Retained evidence is under `results/uncertainty_gate_calibration/phase_r/`:
the derived landmark table and its manifest with SHA-256 bindings to the raw
source, the operating-characteristic and recalibration manifests, and the
analysis manifest carrying the primary result, every secondary and the low-power
label.

## v1.5.0 uncertainty-gate calibration evidence

v1.5 reused the five frozen v1.2 artifact pairs without retraining. Phase A
recorded 143,732 transitions from 1,000 episodes on disjoint seeds
`60000..60099`. All episodes qualified, no non-finite pair was excluded, and
the median within-episode Spearman correlation between ensemble disagreement
and exact residual-prediction error was **0.298**, with paired-bootstrap 95%
interval **[0.228, 0.356]**. This passed the frozen screening threshold of
`0.2` and authorized calibration.

Phase C ran 7,000 new episodes. The table reports pooled rates across five
frozen policies (500 episodes per cell):

| Condition | Scenario | Success | Unsafe episodes |
|---|---|---:|---:|
| A4c | ID reference | 39.2% | 68.0% |
| A4c | Compound OOD | 1.0% | 68.6% |
| A4c | Motor fault | 13.0% | 73.4% |
| A6c | ID reference | 45.2% | 6.2% |
| A6c | Compound OOD | 1.2% | 21.8% |
| A6c | Motor fault | 13.0% | 26.4% |
| A6c gate-off control | ID reference | 57.4% | 3.6% |
| A6c gate-off control | Compound OOD | 3.2% | 18.4% |
| A6c gate-off control | Motor fault | 23.2% | 24.0% |

Against A2, A4c success changed by -13.0 pp ID (95% CI
`[-19.4, -6.4]`), -5.0 pp OOD (`[-7.6, -2.8]`) and -10.6 pp under motor
fault (`[-14.8, -6.8]`). Against its otherwise identical gate-off control,
A6c changed by -12.2 pp ID (`[-20.0, -4.8]`), -2.0 pp OOD
(`[-4.0, -0.4]`) and -10.2 pp under motor fault (`[-16.2, -4.6]`). Both
preregistered acceptance decisions failed.

The independent verifier checked 7,000 outcome and safety rows, 5,500 gate
summaries, 609,865 raw transitions and 25 shard files. It recomputed gate
normalization, scale, residual, query-torque and candidate-torque identities,
and validated all retained hashes. The compact evidence is under
`results/uncertainty_gate_calibration/phase_c/`. The deterministic 106 MiB
raw gzip remains local because it exceeds GitHub's per-file Git limit and was
not authorized for separate public distribution. On a fresh clone, the Phase-C
verifier audits the complete compact record and reports
`raw_transitions_verified: false`; placing the original local gzip back in the
Phase-C directory enables the full 609,865-transition invariant audit.

## v1.4.0 quantified safety evidence

The v1.4 campaign reuses the frozen v1.2 checkpoints together with the v1.3
evaluation seeds `50000..50099` and scenarios. No policy, context encoder or
ensemble was retrained. This is a paired safety audit of the retained
controllers, not a new model-selection or generalization claim. Four
conditions were evaluated over five training seeds, three scenarios and 100
episodes per scenario, for 6,000 episodes.

The campaign isolates two filter effects on identical episode seeds:

- `A2_unfiltered` versus `A5_hocbf`: the same residual policy without and
  with required hard-HOCBF projection;
- `A6_prefilter` versus `A6_hocbf`: the same context-plus-gate stack
  immediately before and after required hard-HOCBF projection.

Earlier campaigns reported task success only. Success alone cannot separate a
controller that reaches the target through the safe set from one that reaches
it by traversing constraint violations, so v1.2 and v1.3 could not evaluate
the component whose only purpose is to prevent the second case.

### Measured safety and task outcomes

Values are means ± sample SD across five independently trained policies.

| Condition | Scenario | Unsafe episodes | Unsafe states | Success |
|---|---|---:|---:|---:|
| A2 unfiltered | ID reference | 72.6% ± 3.6 pp | 18.0% ± 2.6 pp | 52.2% ± 7.7 pp |
| A5 + HOCBF | ID reference | **4.0% ± 1.6 pp** | **0.6% ± 0.7 pp** | 47.6% ± 7.3 pp |
| A6 pre-filter | ID reference | 64.2% ± 0.4 pp | 16.6% ± 0.4 pp | 14.6% ± 1.3 pp |
| A6 + HOCBF | ID reference | **8.2% ± 0.4 pp** | **0.5% ± 0.0 pp** | 12.8% ± 0.8 pp |
| A2 unfiltered | Compound OOD | 67.8% ± 1.9 pp | 14.2% ± 1.4 pp | 6.0% ± 2.3 pp |
| A5 + HOCBF | Compound OOD | **19.6% ± 2.6 pp** | **1.1% ± 0.4 pp** | 3.0% ± 0.7 pp |
| A6 pre-filter | Compound OOD | 74.6% ± 1.1 pp | 20.7% ± 0.1 pp | 0.0% ± 0.0 pp |
| A6 + HOCBF | Compound OOD | **27.4% ± 0.5 pp** | **3.0% ± 0.0 pp** | 0.0% ± 0.0 pp |
| A2 unfiltered | Motor fault | 75.6% ± 3.3 pp | 17.3% ± 1.7 pp | 23.6% ± 1.9 pp |
| A5 + HOCBF | Motor fault | **21.6% ± 1.3 pp** | **4.5% ± 1.5 pp** | 16.4% ± 1.1 pp |
| A6 pre-filter | Motor fault | 71.6% ± 0.9 pp | 24.1% ± 0.6 pp | 3.6% ± 0.5 pp |
| A6 + HOCBF | Motor fault | **27.4% ± 1.1 pp** | **6.1% ± 0.2 pp** | 2.8% ± 0.4 pp |

### Paired filter effects

Differences are computed per trained model on identical episode seeds; the
spread is the sample SD across the five models.

| Pairing | Scenario | Unsafe episodes | Unsafe states | Violation integral | Success |
|---|---|---:|---:|---:|---:|
| A2 → A5 | ID reference | -68.6 ± 3.4 pp | -17.5 ± 1.6 pp | -0.126 ± 0.038 | -4.6 ± 2.7 pp |
| A2 → A5 | Compound OOD | -48.2 ± 2.2 pp | -12.9 ± 1.2 pp | -0.204 ± 0.040 | -3.0 ± 2.5 pp |
| A2 → A5 | Motor fault | -54.0 ± 4.0 pp | -12.7 ± 2.0 pp | -0.141 ± 0.012 | -7.2 ± 2.8 pp |
| A6 pre → A6 | ID reference | -56.0 ± 0.0 pp | -16.2 ± 0.3 pp | -0.187 ± 0.005 | -1.8 ± 0.8 pp |
| A6 pre → A6 | Compound OOD | -47.2 ± 1.3 pp | -17.6 ± 0.1 pp | -0.311 ± 0.013 | +0.0 ± 0.0 pp |
| A6 pre → A6 | Motor fault | -44.2 ± 0.4 pp | -18.3 ± 0.9 pp | -0.272 ± 0.037 | -0.8 ± 0.4 pp |

Across the 15 per-model paired bootstrap intervals of each pairing (5 seeds ×
3 scenarios, 10,000 draws each), the reduction in unsafe-episode rate and in
unsafe-state fraction excluded zero in **15/15** intervals for both pairings.
The paired success difference excluded zero in only 4/15 intervals for
A2 → A5 and in 0/15 for A6 pre → A6. The filter therefore removed the
majority of unsafe episodes at a task-success cost that was, for most trained
models, not distinguishable from zero.

Maximum joint-position excess fell by 0.161–0.717 rad and maximum
joint-velocity excess by 2.49–3.50 rad/s depending on pairing and scenario, so
the effect is a reduction in violation severity and not only in violation
count.

### Filter activity and residual violations

The hard-HOCBF runtime modified 39.8% (A5) and 18.6% (A6) of ID command
attempts, rising to 79.4% and 61.0% under compound OOD. It rejected 41/1,500
A5 episodes and 45/1,500 A6 episodes as infeasible; the infeasible command was
not executed and the episode was counted as unsuccessful. These counts match
the v1.3 campaign exactly, as required by the reuse of identical seeds and
frozen checkpoints.

Filtering did not eliminate physical violations: 226/1,500 A5 and 315/1,500
A6 episodes still entered unsafe states, against 1,080/1,500 and 1,052/1,500
unfiltered. The executed-command margin was non-negative throughout, so the
residual violations are not solver failures. They are the expected consequence
of a certificate defined on the nominal instantaneous command model while the
evaluated plant carries randomized parameters, actuator delay, injected faults
and discretization.

The A6 pre-filter stack was **less** safe than the plain residual policy under
compound OOD (74.6% versus 67.8% unsafe episodes, 20.7% versus 14.2% unsafe
states) while producing 0.0% success. The frozen uncertainty gate again
operated near minimum residual authority. This negative result is retained.

### Audit

The audit verified 6,000 episode rows, 6,000 diagnostic rows, 60 per-model
summaries and 180 paired comparisons; complete coverage of 60 condition ×
seed × scenario cells at exactly 100 episodes each; the full seed population
`50000..50099`; the state-observation invariant
(`state_observations == steps + 1`) and the command-attempt invariant on every
row; and finite values for all core diagnostics. All 20 referenced checkpoint
hashes were revalidated against the audited v1.3 evaluation manifest before
the first episode.

The 4,500 A2, A5 and A6 episodes that also exist in the v1.3 campaign
reproduced their retained v1.3 outcomes exactly: identical success and step
counts, with rewards matching to within 1e-12 and zero discrepancies. This
confirms that the frozen artifacts were reused without retraining and that the
filter comparison is genuinely paired.

The evaluation ran from commit
`29fa933717c921cab57fbe4023dd03b53267953a` on Ubuntu 24.04 under WSL2 with
Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1 and PyTorch 2.12.0+cu130. The
automated suite completed with 138 passing tests and Ruff reported no errors.

Retained evidence is under `results/quantified_safety/`. The campaign
quantifies the safety effect of the hard-HOCBF filter on the analytical planar
benchmark. It does not establish calibrated uncertainty, formal safety under
unmodelled dynamics, MuJoCo/Franka transfer or hardware behavior, and the
unfiltered conditions were never trained against the safety envelope, so their
violation rates characterize the baseline rather than demonstrating a defect
introduced by residual learning.

## v1.3.0 OOD and fault robustness evidence

The v1.3 campaign reuses the frozen v1.2 A2–A6 artifacts without retraining.
A0 and 25 learned policies were evaluated on seeds `50000..50099`, disjoint
from the v1.2 held-out set, in three paired scenarios: the v1.2
in-distribution randomization, compound OOD dynamics and abrupt joint-2 motor
authority loss to 55% from step 20. A1 is excluded because its selected policy
checkpoints were not retained.

| Condition | ID reference | Compound OOD | Motor fault |
|---|---:|---:|---:|
| A0 | 8.0% | 0.0% | 3.0% |
| A2 | 52.2% ± 7.7 pp | 6.0% ± 2.3 pp | 23.6% ± 1.9 pp |
| A3 | **62.4% ± 12.9 pp** | **11.6% ± 3.8 pp** | **32.6% ± 6.4 pp** |
| A4 | 14.4% ± 1.3 pp | 0.0% ± 0.0 pp | 3.2% ± 0.8 pp |
| A5 | 47.6% ± 7.3 pp | 3.0% ± 0.7 pp | 16.4% ± 1.1 pp |
| A6 | 12.8% ± 0.8 pp | 0.0% ± 0.0 pp | 2.8% ± 0.4 pp |

Values for learned conditions are means ± sample SD across five policies.
Relative to their paired ID references, mean OOD success fell by 46.2 pp for
A2, 50.8 pp for A3, 14.4 pp for A4, 44.6 pp for A5 and 12.8 pp for A6. Motor
loss reduced the same conditions by 28.6, 29.8, 11.2, 31.2 and 10.0 pp. A3
remained the strongest learned condition, but none of the frozen policies
demonstrated robust OOD performance.

The hard-HOCBF runtime rejected 41/1,500 A5 episodes and 45/1,500 A6 episodes
as infeasible. The final infeasible command was not executed and the episode
was counted as unsuccessful. Infeasibility increased from 16/1,000 ID
episodes to 37/1,000 OOD and 33/1,000 motor-fault episodes. These results do
not contradict the projection solver: its certificate is relative to the
nominal model and does not guarantee safety under unmodelled plant changes.

The audit verified exactly 7,800 episode rows, 1,500 gate rows, 3,000 stack
rows, 78 per-model summaries and 52 paired robustness comparisons. Every
condition/scenario group contains 100 unique seeds, all 2,600 motor-fault
episodes reached the injected fault, and independent recomputation matched
the stored success rates, deltas and sample standard deviations. All 20
checkpoint hashes matched the manifest. The evaluation ran from commit
`571ea22d858322815c43928bfeed87a784af78f3`; causal GRU, ensemble and HOCBF
inference used the frozen deterministic CPU paths.

Retained evidence is under `results/ood_fault_robustness/`. The campaign is
analytical-planar evidence only and does not establish calibrated uncertainty,
formal safety under OOD dynamics, MuJoCo/Franka transfer or hardware behavior.

## v1.2.0 planar ablation evidence

The A0–A6 campaign uses training seeds `0..4`, validation seeds
`20000..20029` and held-out seeds `40000..40099`. Each learned condition is
reported from five independently trained policies and 100 held-out episodes
per policy. Cross-model spreads below are sample standard deviations
(`ddof=1`).

| Condition | Controller | Held-out success |
|---|---|---:|
| A0 | Computed torque | 11.0% |
| A1 | Direct SAC | 6.0% ± 3.7 pp |
| A2 | Residual SAC | 56.4% ± 7.0 pp |
| A3 | Residual SAC + causal context | **64.2% ± 6.7 pp** |
| A4 | Residual SAC + uncertainty gate | 15.2% ± 1.6 pp |
| A5 | Residual SAC + hard HOCBF | 49.2% ± 7.9 pp |
| A6 | Context + uncertainty gate + hard HOCBF | 17.0% ± 2.3 pp |

A3 produced per-seed success rates of 73%, 66%, 57%, 67% and 58%. Its
policies and independently pretrained context encoders were produced from
commit `3068a858ae46d55a43705963ede6e0d72b66492d`.

A4 produced 13%, 17%, 14%, 16% and 16%. The uncertainty scale averaged
`0.102`, close to the frozen lower bound of `0.1`, and success fell by 41.2
percentage points relative to paired A2 episodes. The five ensemble artifacts
were produced and evaluated from commit
`22fde136682013990157b9a11d42b923d20afa3e`.

A5 produced 51%, 49%, 60%, 48% and 38%. Its mean paired difference from A2
was -7.2 percentage points. The HOCBF changed approximately 39.7% of command
attempts and explicitly aborted 15/500 episodes when the hard projection was
infeasible. These failures were counted as unsuccessful; no uncertified
fallback torque was executed.

A6 produced 18%, 13%, 18%, 19% and 17%. The learned context was active
(mean latent norm `3.94`), but the gate again operated near its lower bound
(mean scale `0.102`). Success fell by 47.2 percentage points relative to
paired A3 episodes. A5 and A6 were evaluated from commit
`c029558a464d4ece02188dd4c5f0486387252762`.

The A5/A6 audit verified 1,000 episode rows and 1,000 diagnostic rows, exact
held-out seed coverage, per-seed summaries, all referenced checkpoint hashes
and evaluation-manifest provenance. The v1.2 automated suite completed with
123 passing tests and Ruff reported no errors.

The retained evidence is under:

```text
results/planar_ablations/
├── A3_residual_sac_context/
├── A4_residual_sac_uncertainty_gate/
├── A5_residual_sac_hocbf/
└── A6_full_adaptive_stack/
```

These results complete the frozen v1.2 analytical planar matrix. They do not
establish OOD learned-policy, MuJoCo, Franka, hardware or sim-to-real
performance. The HOCBF certificate is relative to its nominal model.

## v1.1.0 five-seed residual-SAC evidence

The first completed learned-policy campaign was run from Git commit
`9f832614ce8b51c207873ff4861986ab72903115` (the verified v1.0.1 training stack) on Ubuntu 24.04 under WSL2
with Python 3.12.3 and PyTorch 2.12.0+cu130. The retained run manifests record `cuda_available: true` for all
five training runs.

### Protocol

- training seeds: `0, 1, 2, 3, 4`;
- 200,000 environment steps per training run;
- residual SAC, hidden layers 256×256, replay capacity 200,000 and batch size 256;
- start steps: 5,000; update cadence: one SAC update per environment step thereafter;
- domain randomization: ±15% mass, ±30% friction, ±15% motor gain, payload 0–1 kg and action delay 0–2 steps;
- validation: 30 fixed episodes every 25,000 steps beginning at seed `20000`;
- checkpoint selection: lexicographic `(success_rate, mean_return)` on validation only;
- held-out evaluation: seeds `40000..40099`, 100 episodes per selected policy;
- computed-torque baseline evaluated on the identical 100 held-out seeds;
- validation and held-out seed populations are disjoint.

Seed 0 was intentionally interrupted and resumed exactly from its 100,000-step training checkpoint; its retained
manifest records the resume source. The CUDA checkpoint-resume path had already been regression-tested in v1.0.1.

### Held-out results

| Training seed | Selected validation step | Held-out success | 95% Wilson interval | Mean reward | Mean final distance |
|---:|---:|---:|---:|---:|---:|
| 0 | 200k | 61/100 | 51.2–70.0% | -72.73 | 0.0863 m |
| 1 | 200k | 57/100 | 47.2–66.3% | -76.32 | 0.0859 m |
| 2 | 200k | 63/100 | 53.2–71.8% | -74.20 | 0.0803 m |
| 3 | 150k | 56/100 | 46.2–65.3% | -79.96 | 0.0986 m |
| 4 | 200k | 45/100 | 35.6–54.8% | -82.31 | 0.0928 m |

Across the five independently trained policies:

- mean held-out success rate: **56.4%**;
- sample standard deviation across training seeds: **7.0 percentage points**;
- observed seed range: **45% to 63%**;
- total policy successes: **282/500**.

The original generated `aggregate.json` also records a population standard deviation of 6.25 percentage points
(`ddof=0`). Release prose uses the sample standard deviation across independent training runs instead.

### Paired computed-torque comparison

The computed-torque controller achieved **11/100 = 11.0%** on the same held-out seeds, with Wilson 95% CI
**6.3–18.6%**.

| Training seed | Policy - baseline | Paired bootstrap 95% CI |
|---:|---:|---:|
| 0 | +50 pp | +37 to +62 pp |
| 1 | +46 pp | +35 to +57 pp |
| 2 | +52 pp | +40 to +63 pp |
| 3 | +45 pp | +34 to +56 pp |
| 4 | +34 pp | +22 to +46 pp |

Mean paired improvement: **+45.4 percentage points**. All five paired bootstrap 95% confidence intervals exclude
zero. This is a paired episode-seed comparison; it does not turn the 500 held-out policy episodes into 500
independent training replicates.

### Retained evidence

The release stores the evidence required to reconstruct the reported statistics under `artifacts/planar_sac_5seed/`:

```text
artifacts/planar_sac_5seed/
├── README.md
├── aggregate.json
├── baseline_heldout_40000.csv
├── checkpoint_sha256.txt
├── heldout_episodes.csv
├── paired_comparison.csv
├── result.json
├── run_manifest_seed_0.json
├── run_manifest_seed_1.json
├── run_manifest_seed_2.json
├── run_manifest_seed_3.json
├── run_manifest_seed_4.json
├── summary.csv
├── validation_seed_0.csv
├── validation_seed_1.csv
├── validation_seed_2.csv
├── validation_seed_3.csv
└── validation_seed_4.csv
```

The large model checkpoints themselves are not committed. Their SHA-256 fingerprints preserve the identity
of the exact models used to generate the retained evaluation records. The training manifests identify the
verified v1.0.1 training code commit used to generate the campaign (`9f832614ce8b51c207873ff4861986ab72903115`);
the v1.1.0 release adds the retained experimental evidence and documentation without rewriting that provenance.

This evidence supports a learned-policy claim only for residual SAC on the randomized analytical 2-DoF planar
benchmark. It does not establish the full ablation matrix required by `docs/experiments.md`, nor OOD learned-policy,
Franka/MuJoCo, hardware or sim-to-real performance.

## v1.0.1 automated suite

Executed under Ubuntu 24.04 WSL2 with PyTorch 2.12.0+cu130 and an NVIDIA GeForce RTX 4080 Laptop GPU.

Validation commands: `ruff check .`, `pytest`, and `git diff --check`.

Result: 72 passed and Ruff reported no errors.

The v1.0.1 regression suite additionally verifies that CUDA RNG states restored from a checkpoint are converted to CPU byte tensors before being passed to `torch.cuda.set_rng_state_all()`. This fixes checkpoint continuation on CUDA while preserving the existing CPU behaviour.

## v1.0 automated suite

Executed locally from the repository root:

```bash
pytest -q
```

Result before the v1.0 documentation freeze:

```text
71 passed
```

Additional static checks executed:

```bash
python -m compileall -q sarrl tests tools
git diff --check
```

Both completed successfully.

## Historical non-learned baselines

### Nominal computed-torque baseline

Command used for the original retained baseline:

```bash
python tools/evaluate_nominal.py --episodes 100 --seed 1000 \
  --output results/v0_1_nominal.csv
```

Measured on seeds 1000 through 1099:

```text
success: 100/100 = 100.0%
mean steps: 50.68
mean terminal distance: 0.04575 m
```

The exact per-episode target, success flag, steps, reward and terminal distance are retained in `results/v0_1_nominal.csv`.

### v0.9 robustness baseline campaign

`tools/run_planar_baselines.py` was executed on seeds 1000 through 1099. Raw episode records are retained in `results/v0_9_baselines.csv`; aggregate metrics and metadata are in `results/v0_9_baselines.json`.

| scenario | success | 95% Wilson interval | mean final distance |
|---|---:|---:|---:|
| nominal | 100/100 | 96.3–100.0% | 0.0455 m |
| ID randomisation | 8/100 | 4.1–15.0% | 0.3355 m |
| OOD dynamics | 0/100 | 0.0–3.7% | 0.8763 m |
| joint-2 motor fault | 1/100 | 0.2–5.4% | 0.8277 m |

These are deliberately **non-learned baseline** results. Their purpose is to establish that the otherwise strong nominal controller has a meaningful model-mismatch problem for residual learning to solve.

## SAC integration checks

### Early v0.1 smoke

A 300-step residual-SAC run was executed to verify replay sampling, updates, entropy-temperature changes, checkpoint save/load and deterministic evaluation. It achieved 0/5 success. This is not presented as a performance result because the training budget was intentionally negligible.

### v0.11 validation/resume smoke

A non-default 16×16 SAC network with replay capacity 80 was trained for 40 steps, checkpointed, reconstructed from `train_step40.pt`, and continued to step 55. The reconstructed session recovered mode, architecture, replay capacity and update cadence from the checkpoint rather than CLI defaults.

The smoke used only two validation episodes and a tiny training budget. Its success values are therefore integration evidence only, not a learned-policy claim.

### v0.12 sweep smoke

`tools/run_sac_sweep.py` was executed with two training seeds, 30 steps per seed, two validation episodes and three held-out episodes. The purpose was to verify orchestration and artifact generation:

```text
seed_0/
seed_1/
summary.csv
heldout_episodes.csv
aggregate.json
sweep_manifest.json
```

The resulting success rate is not retained as a scientific result because the residual policy remained effectively near the already-successful nominal controller and the episode count was tiny.

## Residual-dynamics tool smoke

After correcting the commanded-vs-applied torque semantics, a 96-sample / 10-step optimisation smoke completed successfully:

```text
initial ensemble MSE: 3935.738281
final ensemble MSE:   3796.406494
```

This verifies the data/training/checkpoint path. It is not an accuracy or generalisation claim.

## Incomplete long SAC probe

A larger randomized residual-SAC probe was attempted in the available execution environment but did not complete its requested budget before the execution limit. A partial checkpoint was inspected only as a debugging artifact and is **not included in the release or README results**.

This is why v1.0 reports no learned-policy headline metric. The repository instead ships the completed multi-seed protocol required to produce that evidence on suitable compute.

## CLI smoke audit before v1.0 freeze

The following command paths were executed with deliberately tiny budgets and temporary output directories:

- nominal evaluator;
- all four non-learned baseline scenarios;
- context-data collection and GRU training;
- residual-dynamics data collection and ensemble training;
- SAC training with validation/checkpointing;
- standalone checkpoint evaluation;
- composed runtime-stack evaluation with hard safety enabled.

All completed without runtime errors. These tiny-budget runs are integration checks only; their numerical outcomes are not retained as scientific results.

## Tooling limitation

The v1.0 verification environment did not provide MuJoCo or Gymnasium, and the v1.0 planar release has no dependency on either. MuJoCo arrived in v2.0 as an optional dependency of the planar plant (`pip install -e '.[mujoco]'`); its tests skip without it and run in the MuJoCo CI job. The later Panda integration checks are recorded above.
