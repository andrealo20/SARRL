# Changelog

This file records implemented release increments. Performance evidence is kept separately in `docs/verification.md` and requires retained raw artifacts.

## v1.6.0 — disagreement and operational failure

- Tested the link v1.5 assumed but never measured: Phase A validated `disagreement -> model prediction error`, while Phase C acted on `disagreement -> operational failure`.
- Introduced no new episodes and no retraining; the analysis is a preregistered re-analysis of the retained v1.5 Phase-C gate-off arm, frozen and committed before the association was computed.
- Used the `A6c_gate_off_control` arm — 1,500 episodes in 15 cells of exactly 100 over 100 shared episode seeds — because the gated arms would condition on gate-induced trajectory changes.
- Fixed the predictor to the median disagreement over raw rows `step = 0..24`, a uniform window that keeps exposure identical across outcomes, since 19.1% of unsafe, 32.1% of safe and 100% of aborted episodes end before the horizon.
- Defined the endpoint as `operational_failure = unsafe_episode OR safety_infeasible`, 236 events in 1,500, because a HOCBF abort is an operational failure rather than a success.
- Measured and corrected an **anticonservative decision rule** before opening the data: the uniform 5th-percentile bound had a worst-case composite-null size of 7.8% [6.7%, 9.1%] against a nominal 5%, recalibrated per component on synthetic nulls to the 2.0th and 2.5th percentiles and validated at 3.65% [2.91%, 4.56%].
- Recomputed power under the calibrated rule: 38.5% joint power at AUC 0.70 against an 80% target, so the screen is reported as an explicitly low-power feasibility screen.
- Measured `id_reference` AUC **0.5200** (24/500 events, bounds [0.3958, 0.6391]) and `ood_compound` AUC **0.4787** (92/500, [0.4085, 0.5479]).
- **Decision: Inconclusive.** The result excludes a useful association at the preregistered `AUC > 0.60` threshold; it does not demonstrate the absence of any weaker association, and identifies no causal relationship.
- Closed the conditional intervention arm unexecuted; its numerical parameters were never frozen.
- Retained the derived landmark table with SHA-256 bindings to its raw source, so the primary analysis reproduces from the repository without the 106 MiB local transition file.

## v1.5.1 — provenance and packaging correction

- Normalized the five Phase-A collection-runtime references to the surviving source-equivalent public commit after private, non-scientific planning files were removed from history.
- Rebuilt the hash-bound calibration and Phase-C aggregate manifest without changing any numerical calibration value, episode result or acceptance decision.
- Made the Phase-C verifier portable: a fresh clone now verifies all compact retained evidence and explicitly reports that raw-transition verification was skipped when the local 106 MiB file is absent.
- Corrected the documentation and manifest to state that the oversized raw Phase-C transition table is local-only and was not published as a release asset.
- Kept the repository-wide ignore rules that prevent internal planning and agent files from being committed again.

## v1.5.0 — uncertainty-gate calibration

- Reused the five frozen v1.2 A2/A3 policies, context encoders and dynamics ensembles without retraining.
- Retained 143,732 Phase-A transitions over 1,000 episodes; ensemble disagreement correlated with exact residual-prediction error at median within-episode Spearman rho **0.298**, with paired-bootstrap 95% interval **[0.228, 0.356]**.
- Replaced the dimensional gate with a frozen per-ensemble reference scale derived from 200 episode medians; all source artifacts and calibration outputs are hash-bound.
- Evaluated 7,000 new Phase-C episodes across held-out, ID, compound-OOD and motor-fault populations, including an otherwise identical gate-off control.
- Measured A4c success differences versus A2 of **-13.0 pp** ID, **-5.0 pp** OOD and **-10.6 pp** under motor fault; every paired 95% interval excluded zero on the negative side.
- Measured A6c success differences versus gate-off of **-12.2 pp** ID, **-2.0 pp** OOD and **-10.2 pp** under motor fault; every paired 95% interval excluded zero on the negative side.
- Both preregistered non-inferiority and strict OOD-benefit decisions failed. The calibrated gate is therefore retained as a negative result, not promoted as a robustness improvement.
- Audited 7,000 outcome and safety rows, 5,500 gate summaries, 609,865 raw transitions and 25 shard files. Compact evidence is retained in Git; the 106 MiB raw transition table remains local and is not part of the public release.

## v1.4.0 — quantified safety

- Reused the frozen v1.2 checkpoints and the v1.3 seeds `50000..50099` and scenarios to run a 6,000-episode paired safety audit with no retraining.
- Added a safety-audit evaluator measuring unsafe-episode rate, unsafe-state fraction, boundary entries, violation severity and integral, command margins, intervention rate and HOCBF infeasibility alongside task success.
- Isolated two filter effects on identical episode seeds: `A2_unfiltered` versus `A5_hocbf`, and `A6_prefilter` versus `A6_hocbf`.
- Measured a reduction in unsafe episodes of **-68.6 pp** (A2 → A5) and **-56.0 pp** (A6 pre → A6) on the ID reference, with all 15 per-model paired bootstrap intervals excluding zero for both pairings in every scenario.
- Measured the paired task-success cost as -4.6 pp (A2 → A5) and -1.8 pp (A6 pre → A6) on the ID reference, with intervals excluding zero in only 4/15 and 0/15 per-model comparisons respectively.
- Retained the negative result that the frozen uncertainty gate leaves the A6 pre-filter stack less safe than the plain residual policy under compound OOD while producing 0.0% success.
- Retained that filtering reduces but does not eliminate physical violations, because the HOCBF certificate covers the nominal instantaneous command model only.
- Audited 6,000 episode and diagnostic rows, 60 summaries, 180 paired comparisons, complete seed coverage, all invariants and all 20 checkpoint hashes; the 4,500 episodes shared with v1.3 reproduced their retained outcomes exactly.
- Retained the manifest, raw rows, diagnostics, paired comparisons and aggregates under `results/quantified_safety/`.

## v1.3.0 — OOD and fault robustness

- Reused the frozen v1.2 A2–A6 policies without retraining and evaluated A0 plus 25 learned policies on 7,800 new paired episodes.
- Added compound OOD dynamics and abrupt joint-2 motor-loss scenarios alongside an in-distribution reference, using disjoint seeds `50000..50099`.
- Measured the strongest learned result with A3 context: 62.4% ± 12.9 pp ID, 11.6% ± 3.8 pp OOD and 32.6% ± 6.4 pp under motor loss.
- Retained the negative result that every learned family degraded sharply outside its training distribution; A4 and A6 reached 0% success under compound OOD dynamics.
- Recorded 86 explicit HOCBF-infeasible episodes across 3,000 A5/A6 evaluations; no uncertified fallback command was executed.
- Audited all episode, gate and stack rows, exact seed coverage, fault exposure, aggregate statistics and all 20 referenced checkpoint hashes.
- Retained the complete manifest, raw rows, diagnostics, paired bootstrap deltas and aggregate results under `results/ood_fault_robustness/`.

## v1.2.0 — complete planar ablation matrix

- Completed A0–A6 on the frozen randomized analytical planar benchmark with five training seeds and 100 held-out episodes per learned policy.
- Added audited A3 causal-context, A4 uncertainty-gate, A5 hard-HOCBF and A6 complete adaptive-stack evaluation paths with checkpoint hashes and source provenance.
- Measured held-out success of **64.2% ± 6.7 pp** for A3, **15.2% ± 1.6 pp** for A4, **49.2% ± 7.9 pp** for A5 and **17.0% ± 2.3 pp** for A6.
- Retained negative ablations: the frozen uncertainty gate operated near minimum residual authority, while A5 reported 15 explicit HOCBF-infeasible episodes rather than executing an uncertified fallback.
- Integrated causal context updates into the composed torque runtime so A6 genuinely combines context, ensemble uncertainty, gated residual authority and required HOCBF projection.
- Retained raw held-out rows, stack diagnostics, paired bootstrap comparisons, manifests and aggregate summaries under `results/planar_ablations/`.
- MuJoCo, Franka, OOD learned-policy, hardware and sim-to-real evidence remain future work.

## v1.1.0 — retained multi-seed residual-SAC evidence

- Completed five independent 200k-step residual-SAC training runs under planar domain randomization.
- Retained 500 held-out policy episodes and the matching computed-torque baseline evaluation.
- Measured **56.4% ± 7.0 pp** held-out success across training seeds (mean ± sample SD), versus **11.0%** for computed torque on identical held-out episode seeds.
- Mean paired success-rate improvement was **+45.4 pp**; all five paired bootstrap 95% confidence intervals excluded zero.
- Added raw validation curves, held-out episode records, run manifests, checkpoint SHA-256 fingerprints and machine-readable result summaries under `artifacts/planar_sac_5seed/`.
- Updated release documentation to distinguish retained planar learned-policy evidence from still-unimplemented MuJoCo/Franka transfer.
- This is retained method-specific planar evidence; the full ablation study and MuJoCo/Franka transfer remain pending.

## v1.0.1 — CUDA checkpoint restore compatibility

- Fixed CUDA RNG restoration when training checkpoints are loaded onto a CUDA device.
- CUDA RNG states are explicitly converted back to CPU `ByteTensor` objects before calling `torch.cuda.set_rng_state_all()`.
- Added a regression test for CUDA-remapped RNG checkpoint state that also runs on CPU-only CI.
- Updated Ruff-clean typing/import formatting for the current development toolchain.
- Automated suite increased from 71 to 72 tests.

## v1.0.0 — verified planar research stack

- Promoted the planar stack to the first stable research release.
- Rewrote README and technical documentation to match the implemented architecture and current evidence.
- Documented exact boundaries between implemented, tested, experimentally measured and planned components.
- Added release/version consistency testing and final audit/packaging checks.
- M10 MuJoCo/Franka remains intentionally planned rather than being shipped untested.

## v0.12.0 — experiment provenance, multi-seed orchestration and residual-model semantics

- Every SAC training run writes a machine-readable manifest with Git commit, runtime/library versions, agent/environment configuration and validation protocol.
- Added `tools/run_sac_sweep.py` for reproducible multi-seed training plus a disjoint held-out seed set, raw held-out episode CSVs and cross-model aggregate statistics.
- Added retained per-episode deterministic policy evaluation records and validation/held-out seed-range guards.
- Fixed residual-dynamics data generation so the model input is the **commanded** torque while the target contains motor-gain degradation; using already-degraded torque had hidden actuator-gain error from the learned residual.
- Updated the package version source, which had remained stale at `0.1.0`.

## v0.11.0 — reproducible validation and exact session reconstruction

- Deterministic SAC actions no longer sample internally or advance PyTorch RNG state.
- Added fixed-seed deterministic policy evaluation as a library primitive.
- Training selects `best.pt` on a dedicated validation seed set, lexicographically by success rate then mean return; held-out evaluation remains separate.
- Training checkpoint format v2 records Python/NumPy/PyTorch RNGs, replay state, full environment constructor configuration, delay queue and trainer update cadence.
- `load_training_session()` reconstructs non-default SAC architecture, replay capacity and randomized/faulted environment directly from the checkpoint.

## v0.10.0 — architecture-safe SAC loading and configurable training

- `SACAgent.from_checkpoint()` reconstructs observation/action dimensions and non-default hidden architecture directly from the checkpoint.
- Evaluation tools no longer assume the default SAC width.
- Trainer exposes hidden layer sizes, update frequency and replay capacity.

## v0.9.0 — reproducible evaluation protocol

- Added retained per-episode result structures, Wilson confidence intervals, paired bootstrap comparison and Git provenance.
- Added controlled computed-torque campaigns for nominal, ID-randomized, OOD-dynamics and motor-fault scenarios.
- Retained the raw v0.9 baseline CSV and aggregate JSON.

## v0.8.0 — integrated runtime control stack

- Composed physics baseline, residual policy, optional residual-dynamics uncertainty gate and hard safety projection behind one runtime command interface.
- Made uncertified safety failure explicit through `executable` and `safety_certified` semantics.

## v0.7.0 — full off-policy training checkpoints

- Added replay-buffer and environment state serialization.
- Preserved stochastic continuation state rather than saving network weights alone.
- Added exact continuation regression tests.

## v0.6.0 — learned residual dynamics and epistemic uncertainty

- Added bootstrap ensemble of residual acceleration networks.
- Added uncertainty estimation from ensemble disagreement.
- Added a bounded uncertainty gate that reduces residual authority without claiming safety certification.

## v0.5.0 — hard HOCBF safety projection

- Added relative-degree-two joint and Cartesian obstacle barrier constraints.
- Added one-step velocity and torque constraints.
- Implemented exact 2-D Euclidean projection by active-set enumeration.
- Hard infeasibility is reported explicitly; no slack variable is hidden behind a “safe” claim.

## v0.4.0 — causal dynamics-context adaptation

- Added GRU context encoder from transition history.
- Added supervised auxiliary physical-context head and runtime latent wrapper.
- Kept ground-truth dynamics parameters out of the runtime policy input.

## v0.3.0 — controlled model mismatch and faults

- Added payload dynamics, mass/friction/motor-gain randomisation, sensor noise and actuator delay.
- Added abrupt motor-gain/payload fault injection.
- Added deterministic environment checkpoint state for delayed execution.

## v0.2.0 — constrained nonlinear MPC

- Added direct-shooting nonlinear MPC through the analytical RK4 plant.
- Added torque, joint-position and velocity constraints, warm start and optimisation regression tests.

## v0.1.0 — analytical/RL foundation

- Added analytical 2-DoF dynamics, RK4, kinematics and computed-torque baseline.
- Added direct/residual reaching environment.
- Added from-scratch Soft Actor-Critic, replay buffer, initial checkpointing and CI.
- Retained the 100/100 nominal computed-torque sanity baseline.
