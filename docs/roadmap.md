# Roadmap after v2.3

Each release addresses a defined question. New evaluation campaigns freeze
their implementations, metrics and decision rules before opening fresh seeds.
Diagnostic data remain separate from confirmation data. Negative findings
are retained and can change the scope of subsequent releases.

| Version | Scope | Completion evidence |
|---|---|---|
| v2.3.1 | Saved-data Panda demo, reference environment and documentation | A reproducible paired replay and an accessible installation example |
| v2.3.2 | Replay and documentation consolidation | Release-specific demo commands, installed-package checks and preserved controller evidence |
| v2.4 | Completed changing-latency diagnosis | Saved-trace diagnosis and a frozen 1,920-episode parameter-memory comparison; the candidate fails its age and per-joint tracking criteria |
| v2.5 | Completed Panda reference validation and shared scoring | Reference criteria passed on 4,000 fresh episodes; 30,000 saved-input timing cycles retained; separate 1,920-episode shared-scoring candidate failed its screen |
| v2.6 | Seven-joint safety filtering | QP formulation for joint position, velocity and torque constraints; numerical validation, explicit infeasibility and paired physical-violation/tracking measurements |
| v2.7 | Residual learning on an adaptive baseline | Planar comparison of fixed, adaptive and adaptive-plus-residual control, with at least five training seeds and separate held-out evaluation |
| v3.0, conditional | Integrated Panda stack | Residual learning justified by preceding results, seven-joint filter integration, component ablations and consolidated APIs |

The constant-delay cumulative bank remains the reference while changing
latency is unresolved. Innovation histories, parameter estimates and
information conditioning motivated the completed memory comparison.

The [saved-trace diagnosis](panda-latency-diagnosis.md) defines this first
stage of v2.4. Its retrospective fits and score resets are diagnostic;
the [candidate parameter-memory comparison](panda-estimator-memory.md)
holds score memory fixed and defines its decision rule before new episodes.
That comparison is complete. Joint memory improves final age selection from
83/160 to 99/160 against score-only memory, but fails its frozen screen.
The reference remains cumulative. No independent confirmation of this
candidate is scheduled on the strength of those results.

The reference validation passed its frozen constant-delay criteria and
saved-input timing check. A [separate shared-scoring comparison](panda-shared-scoring.md)
tested one causally acquired physical parameter vector against separate
candidate fits. It recovered 96/160 changing final ages against 79/160 for
score-only memory, but failed age, constant-age, success and tracking criteria.
It does not advance to confirmation. A separately calibrated estimate remains
a hypothesis requiring a new protocol and unused seeds. Multiple switches
and off-grid latency remain separate evaluations.

The first Panda safety increment targets joint constraints. General collision
avoidance requires a separate formulation and evaluation. Hardware and
sim-to-real are outside this sequence. Residual learning advances to the Panda
when a controlled comparison establishes where it adds value to adaptive
model-based control.
