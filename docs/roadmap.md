# Roadmap after v2.3

Each release addresses a defined question. New evaluation campaigns freeze
their implementations, metrics and decision rules before opening fresh seeds.
Diagnostic data remain separate from confirmation data. Negative findings
are retained and can change the scope of subsequent releases.

| Version | Scope | Completion evidence |
|---|---|---|
| v2.3.1 | Saved-data Panda demo, reference environment and documentation | A reproducible paired replay and an accessible installation example |
| v2.3.2 | Replay and documentation consolidation | Release-specific demo commands, installed-package checks and preserved controller evidence |
| v2.4 | Changing-latency diagnosis | Evidence distinguishing candidate parameter compensation, score persistence and insufficient trajectory excitation; a targeted comparison based on that diagnosis |
| v2.5 | Panda controller validation | Frozen paired evaluation on fresh seeds, with tracking, constraint measurements and per-cycle computational cost; separate constant, switching and off-grid latency comparisons |
| v2.6 | Seven-joint safety filtering | QP formulation for joint position, velocity and torque constraints; numerical validation, explicit infeasibility and paired physical-violation/tracking measurements |
| v2.7 | Residual learning on an adaptive baseline | Planar comparison of fixed, adaptive and adaptive-plus-residual control, with at least five training seeds and separate held-out evaluation |
| v3.0, conditional | Integrated Panda stack | Residual learning justified by preceding results, seven-joint filter integration, component ablations and consolidated APIs |

The constant-delay cumulative bank remains the reference while changing
latency is unresolved. Another memory length is not the next default experiment:
first inspect innovation histories, parameter estimates and information
conditioning in the retained traces.

The [saved-trace diagnosis](panda-latency-diagnosis.md) defines this first
stage of v2.4. Its retrospective fits and score resets are diagnostic;
the [candidate parameter-memory comparison](panda-estimator-memory.md)
holds score memory fixed and defines its decision rule before new episodes.

The first Panda safety increment targets joint constraints. General collision
avoidance requires a separate formulation and evaluation. Hardware and
sim-to-real are outside this sequence. Residual learning advances to the Panda
when a controlled comparison establishes where it adds value to adaptive
model-based control.
