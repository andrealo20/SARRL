# Experiment protocol

## Seed populations

Three seed populations have different roles and must remain disjoint:

```text
training seeds   independent optimisation randomness
validation seeds checkpoint selection
held-out seeds   final evaluation only
```

The default examples use validation from 20000 and held-out evaluation from 40000.

Formally,

```math
\mathcal S_{\mathrm{train}} \cap \mathcal S_{\mathrm{validation}} = \varnothing, \qquad
\mathcal S_{\mathrm{train}} \cap \mathcal S_{\mathrm{test}} = \varnothing, \qquad
\mathcal S_{\mathrm{validation}} \cap \mathcal S_{\mathrm{test}} = \varnothing
```

The held-out population must not influence checkpoint selection. For learned-policy comparisons, SARRL
reports variation across **independently trained models**, not only binomial uncertainty across episodes
from a single model.

Every training run writes a machine-readable manifest containing: Git commit, Python version, library
versions, device information, agent configuration, environment configuration, domain-randomization
parameters, validation protocol and training configuration.

## Model-level reporting

One policy evaluated for many episodes does not measure training instability. Final learned results should
therefore train at least five independent models and report, at minimum:

- held-out success rate for each training seed;
- mean and standard deviation across training seeds;
- minimum and maximum success rate;
- mean return across models;
- retained raw per-episode results.

Episode-level Wilson intervals may also be reported for each model, but they do not replace cross-seed
variation.

## Checkpoint selection

`tools/train_sac.py` evaluates deterministic policy checkpoints on a fixed validation set. `best.pt` is
updated by `(success_rate, mean_return)` in lexicographic order.

Validation uses a separate environment instance. Deterministic SAC inference does not sample internally, so
validation does not advance the policy RNG and therefore does not change the subsequent stochastic training
trajectory.

## Multi-seed campaign

`tools/run_sac_sweep.py` runs independent training seeds and evaluates the selected checkpoint on the same
held-out environment seeds for a paired comparison. It writes one run directory per training seed, a
per-run manifest, a sweep manifest, per-model summary CSV, raw held-out episode CSV and cross-model
aggregate JSON. The runner rejects overlapping validation and held-out ranges.

## Baseline scenarios

The planar baseline campaign includes: nominal dynamics; in-distribution identification uncertainty;
stronger OOD dynamics mismatch; abrupt joint-2 motor degradation. These scenarios establish the
model-mismatch gap before learned compensation is credited with improvement.

## Retained v1.1.0 method-specific campaign

The v1.1.0 release retains a completed five-seed residual-SAC campaign following the seed-separation and
model-level reporting rules above. It used residual SAC with hidden layers 256×256, batch size 256, replay
capacity 200,000, 5,000 initial random steps and one SAC update per environment step thereafter. Validation
used 30 fixed episodes every 25,000 training steps starting at seed 20000; final evaluation used the 100
held-out episodes per policy at seeds 40000–40099, which were never used for checkpoint selection. It is a
controlled method-specific result, not the complete comparative study listed below. Raw evaluation evidence
and provenance are stored in `artifacts/planar_sac_5seed/` (see [`docs/verification.md`](verification.md)).

## Required ablations for a learned headline result

A full final study should compare at least:

```text
computed torque
nonlinear MPC
direct SAC
residual SAC
residual SAC + randomisation
residual SAC + causal context
residual SAC + uncertainty gate
full stack + hard safety projection
```

Useful metrics include success rate, return, terminal distance, successful-step count, peak speed, peak
command torque, safety intervention magnitude and hard-safety infeasibility rate.

## OOD protocol

OOD parameters must lie outside the training randomisation range. For example, if training payload is
sampled from 0 to 1.0 kg, an OOD payload test should use a fixed value above 1.0 kg rather than another
sample from the same interval.

## Negative results

Incomplete or unsuccessful training probes are not promoted to headline metrics. They may be retained as
debugging evidence, but only controlled completed campaigns belong in the main result table.

## Commands

### Nominal computed-torque baseline

```bash
python tools/evaluate_nominal.py \
  --episodes 100 \
  --seed 1000 \
  --output results/nominal.csv
```

### Train residual SAC

```bash
python tools/train_sac.py \
  --mode residual \
  --randomize \
  --steps 200000 \
  --seed 0 \
  --output results/residual_seed0
```

Training uses a dedicated fixed validation set for model selection.

### Resume an exact training session

```bash
python tools/train_sac.py \
  --steps 400000 \
  --resume results/residual_seed0/training_final.pt \
  --output results/residual_seed0
```

The training checkpoint reconstructs more than network weights. It includes replay state, environment
state, architecture, optimization state and RNG state required for reproducible off-policy continuation.

### Held-out policy evaluation

```bash
python tools/evaluate.py \
  results/residual_seed0/best.pt \
  --mode residual \
  --episodes 100 \
  --seed 40000
```

### Five-seed campaign

```bash
python tools/run_sac_sweep.py \
  --seeds 0 1 2 3 4 \
  --mode residual \
  --randomize \
  --steps 200000 \
  --validation-seed 20000 \
  --heldout-seed 40000 \
  --heldout-episodes 100 \
  --output results/residual_sweep
```

The sweep runner refuses overlapping validation and held-out seed ranges. Generated outputs include
`sweep_manifest.json`, `summary.csv`, `heldout_episodes.csv`, `aggregate.json` and one `seed_*/` directory
per training seed.

### Train the context encoder

```bash
python tools/train_context.py \
  --samples 2000 \
  --history 16 \
  --steps 1500 \
  --output results/context/context.pt
```

### Train a residual-dynamics ensemble

```bash
python tools/train_residual_dynamics.py \
  --samples 10000 \
  --steps 2000 \
  --seed 0 \
  --device cpu \
  --output results/residual_dynamics/ensemble.pt
```

The trainer also retains `ensemble.npz` and `ensemble_manifest.json`, including
the dataset seed range, training configuration, Git commit and checkpoint
SHA-256 required by the A4 evaluator.

### Evaluate the composed runtime stack

```bash
python tools/evaluate_stack.py \
  results/residual_seed0/best.pt \
  --episodes 100 \
  --randomize \
  --safety
```

### Evaluate A4: Residual SAC + uncertainty gate

A4 reuses the five retained A2 policy checkpoints and pairs each training seed
with an independently trained residual-dynamics ensemble. The runner verifies
the A2 checkpoint SHA-256 values before evaluation and retains both outcome and
gate-diagnostic rows.

```bash
python tools/run_planar_ablations.py \
  --execute A4 \
  --a4-policy-checkpoints \
    /path/to/seed_0/best.pt /path/to/seed_1/best.pt \
    /path/to/seed_2/best.pt /path/to/seed_3/best.pt \
    /path/to/seed_4/best.pt \
  --a4-ensemble-checkpoints \
    /path/to/ensemble_seed_0/ensemble.pt /path/to/ensemble_seed_1/ensemble.pt \
    /path/to/ensemble_seed_2/ensemble.pt /path/to/ensemble_seed_3/ensemble.pt \
    /path/to/ensemble_seed_4/ensemble.pt
```

The default gate is `max(0.1, 1 / (1 + 4 ||uncertainty||))`. It is a
robustness heuristic, not a safety certificate. A4 does not enable context or
HOCBF filtering. Generated evidence is stored under
`A4_residual_sac_uncertainty_gate/` as an evaluation manifest, raw held-out
episodes, gate diagnostics, paired A4-vs-A2 bootstrap comparisons, per-seed
summary and cross-model aggregate.

Alternatively, omit `--a4-ensemble-checkpoints` and add `--confirm-training`.
The runner then prepares one provenance-checked CPU ensemble per seed using the
frozen 10,000-sample / 2,000-step protocol before evaluation.

### Evaluate A5: Residual SAC + HOCBF

A5 reuses the five retained A2 policies. The hard HOCBF projection enforces
the frozen joint, velocity and torque constraints relative to the nominal
planar model. The reaching benchmark has no obstacle constraints. An
infeasible projection aborts the episode and counts as unsuccessful; no
uncertified fallback command is executed.

```bash
python tools/run_planar_ablations.py \
  --execute A5 \
  --a5-policy-checkpoints \
    /path/to/seed_0/best.pt /path/to/seed_1/best.pt \
    /path/to/seed_2/best.pt /path/to/seed_3/best.pt \
    /path/to/seed_4/best.pt
```

### Evaluate A6: Full adaptive stack

A6 composes the retained A3 context-conditioned policies and context encoders,
the retained A4 ensembles and uncertainty gate, and the A5 hard HOCBF. The
context encoder receives the normalized raw residual action proposed by the
policy, while the plant receives the baseline plus gated residual after HOCBF
projection. Runtime context inference remains causal and CPU-only.

```bash
python tools/run_planar_ablations.py \
  --execute A6 \
  --a6-policy-checkpoints \
    /path/to/a3/seed_0/best.pt /path/to/a3/seed_1/best.pt \
    /path/to/a3/seed_2/best.pt /path/to/a3/seed_3/best.pt \
    /path/to/a3/seed_4/best.pt \
  --a6-context-checkpoints \
    /path/to/context_seed_0/context.pt /path/to/context_seed_1/context.pt \
    /path/to/context_seed_2/context.pt /path/to/context_seed_3/context.pt \
    /path/to/context_seed_4/context.pt \
  --a6-ensemble-checkpoints \
    /path/to/ensemble_seed_0/ensemble.pt /path/to/ensemble_seed_1/ensemble.pt \
    /path/to/ensemble_seed_2/ensemble.pt /path/to/ensemble_seed_3/ensemble.pt \
    /path/to/ensemble_seed_4/ensemble.pt
```

Both conditions retain raw episode rows, per-episode stack diagnostics, paired
bootstrap comparisons, per-seed summaries and aggregate metrics. The HOCBF
certificate is model-relative and is not a hardware guarantee.

## v1.3 OOD and fault robustness

v1.3 reuses the frozen v1.2 A2–A6 artifacts without retraining. A0 and each
retained learned-policy family are evaluated on the same new episode seeds
`50000..50099` in three paired scenarios:

- the v1.2 in-distribution randomization as reference;
- compound OOD dynamics with mass ±30%, friction ±50%, motor gain ±25%,
  payload 1.25–1.75 kg and delay up to three steps;
- the ID distribution with abrupt joint-2 motor authority reduced to 55% at
  step 20.

The OOD payload is always outside the 0–1 kg training range. A1 is excluded
because its selected policy checkpoints were not retained; evaluation CSVs
alone cannot reconstruct its policy. Scenario differences use paired episode
seeds and 10,000-draw paired bootstrap intervals. Cross-policy spread remains
the sample standard deviation across the five training seeds.

The campaign runner is `tools/run_planar_v13.py`. It verifies all A2/A3 policy,
context and ensemble hashes before writing raw episodes, gate/stack diagnostics,
per-model summaries, paired robustness deltas and aggregate results.

The completed evidence is retained under `results/ood_fault_robustness/`:

```text
evaluation_manifest.json
heldout_episodes.csv
gate_diagnostics.csv
stack_diagnostics.csv
summary.csv
robustness_deltas.csv
aggregate.json
```

The campaign completed 7,800/7,800 episodes. A3 produced the strongest OOD
and fault results, but every learned condition degraded relative to its paired
ID reference. Full audited metrics and limitations are in
`docs/verification.md`.

## v1.4 quantified safety

v1.4 reuses the frozen v1.2 checkpoints and the v1.3 evaluation seeds and
scenarios. Reusing seeds `50000..50099` is deliberate: this is a paired safety
audit of the retained controllers, not a new model-selection or generalization
claim. No policy, context encoder or ensemble is retrained.

The campaign isolates two filter effects:

- `A2_unfiltered` versus `A5_hocbf`: the same residual policy without and with
  required hard-HOCBF projection;
- `A6_prefilter` versus `A6_hocbf`: the same context-plus-gate stack immediately
  before and after required hard-HOCBF projection.

Each of the four conditions uses five training seeds, three scenarios and 100
episodes per scenario, for 6,000 episodes. Every trajectory records the initial
state and every executed transition. Metrics include unsafe-episode rate,
unsafe-state fraction, boundary-entry count, maximum joint-position and
joint-velocity excess, normalized violation mean/maximum/integral, candidate
constraint violations, executed-command margin, intervention rate/magnitude,
HOCBF infeasibility and task success.

Filter effects are computed per trained model from identical episode seeds with
10,000-draw paired bootstrap intervals. Cross-model spread is the sample
standard deviation over the five training seeds. The runner is
`tools/run_planar_v14.py`; official output is written to
`results/quantified_safety/`. By default the runner reads the audited v1.3
evaluation manifest as its checkpoint inventory and revalidates every artifact
hash before evaluation.

The HOCBF certificate covers the nominal instantaneous command model only.
Randomized plant parameters, actuator delay, injected faults, discretization
and hardware are outside that guarantee. Physical state violations are
therefore measured independently from command-level certificate margins.

## v1.5 uncertainty-gate calibration

v1.5 reuses the five frozen v1.2 A2/A3 policy, context and ensemble artifact
pairs without retraining. Phase A tests whether ensemble disagreement is an
informative signal on a disjoint ID-reference population (`60000..60099`). It
runs A2 and A3 for each matched training/ensemble seed, for 10 cells and 1,000
episodes. At every transition it records ensemble disagreement and the exact
pre-RK4 residual-acceleration prediction error, keeping commanded, delayed,
actuator-scaled and plant-input torque distinct.

The primary statistic is the median across the 10 cells of their median
within-episode Spearman correlations. Episodes require at least 10 finite
pairs; constant variables are retained with rho zero. The analysis uses one
global common qualifying seed set (minimum 90) and a 10,000-draw paired
episode-seed percentile bootstrap with seed `150000`, conditional on the five
frozen artifact pairs. Phase B proceeds only when the 95% lower bound is at
least `0.2`; the gate is retired only when the upper bound is below `0.2`.

Phase A retained 143,732 transitions from 1,000/1,000 qualifying episodes with
no non-finite exclusions. The target median rho was `0.2976`, with 95%
interval `[0.2283, 0.3557]`, so the frozen rule returned `proceed_phase_b`.

Phase B defines the dimensionless gate
`max(0.1, 1 / (1 + ||u|| / u_ref))`. For each ensemble, `u_ref` is the median
of 200 equally weighted episode-median disagreement values: 100 A2 and 100 A3,
each independently required to have at least 10 finite disagreement values.
The five frozen values are `4.1971`, `4.4015`, `5.7418`, `4.0800` and `5.2842`
rad/s^2. The canonical calibration artifact records all source and output
hashes; legacy v1.2-v1.4 runners retain their original dimensional gate.

Phase C evaluates new A4c and A6c conditions on held-out seeds
`40000..40099` and the v1.4 ID/OOD/fault safety protocol on
`50000..50099`. It also evaluates `A6c_gate_off_control`, which performs the
same ensemble inference and HOCBF projection but forces residual scale one,
plus an explicit A3 safety comparator. For A4c versus A2 and A6c versus its
gate-off control, a fixed-model paired bootstrap (10,000 draws, seed `150001`)
requires in every scenario a success lower bound of at least `-0.05` and an
unsafe-episode upper bound of at most `+0.05`. The preregistered strict-benefit
endpoint additionally requires the compound-OOD success lower bound above
zero. A6c versus A3 is reported separately as a total effect. Distribution
shift is described with per-cell two-sample KS distances on episode-median
normalized disagreement, without a binary KS threshold.

Phase C completed all 7,000 episodes. A4c lost `13.0`, `5.0` and `10.6`
percentage points of success versus A2 on ID, compound OOD and motor fault;
the respective 95% intervals were `[-19.4, -6.4]`, `[-7.6, -2.8]` and
`[-14.8, -6.8]`. A6c lost `12.2`, `2.0` and `10.2` points versus its gate-off
control, with intervals `[-20.0, -4.8]`, `[-4.0, -0.4]` and
`[-16.2, -4.6]`. Neither stack passed non-inferiority or the strict OOD
benefit endpoint. The calibrated gate is retained as a negative result.

## v1.6 disagreement and operational failure

v1.6 asks whether ensemble disagreement carries information about operational
failure, a link v1.5 assumed but never tested: Phase A validated
`disagreement -> model prediction error`, while Phase C acted on
`disagreement -> operational failure`. No new episodes and no retraining are
involved; the analysis is a preregistered re-analysis of retained v1.5 Phase-C
evidence, frozen before the association was computed.

The arm is `A6c_gate_off_control` on the safety population: 1,500 episodes in 15
cells of exactly 100, over 100 episode seeds shared across scenarios and cells.
The gated arms are excluded from the primary analysis because there the gate's
action alters the trajectory, which alters both subsequent disagreement and the
outcome. In the gate-off control the ensemble is queried and its disagreement
recorded while the policy retains full authority, so disagreement is an
observation rather than a cause of the trajectory. This avoids conditioning on
gate-induced trajectory changes; it does not remove confounding, and the
analysis is reported as observational. No covariate adjustment is performed and
no claim that disagreement adds information beyond state or scenario difficulty
is admissible.

Exposure is outcome-dependent: 19.1% of unsafe, 32.1% of safe non-infeasible and
100% of aborted episodes end before the 250-step horizon. The predictor is
therefore the median `uncertainty_norm` over a **fixed window of raw rows
`step = 0..24`** (25 transitions, 0.5 s at `dt = 0.02`, the first 10% of the
horizon), identical for every episode so that exposure does not vary with
outcome. The window is not uniformly pre-outcome and is not claimed to be: 5 of
18 `id_reference` and 15 of 92 `ood_compound` unsafe episodes have their first
unsafe observation inside it. The estimand is a fixed-window association that
permits early post-failure observations; truncating each episode at its own
first failure would restore the outcome-dependent exposure the window exists to
remove. A derived per-episode table is retained so the analysis reproduces from
the repository without the 106 MiB raw transition file.

The endpoint is `operational_failure = unsafe_episode OR safety_infeasible`,
236 events in 1,500 episodes. The composite is used because the HOCBF can abort
rather than violate, and an abort is an operational failure the filter exists to
prevent, not a success. All six additional composite events are `id_reference`;
the fault and OOD aborts were already unsafe.

The statistic is the per-scenario AUC over the five seeds pooled, with a
clustered bootstrap of 10,000 draws that resamples the 100 shared episode seeds
into one joint index applied identically across scenarios, the five artifacts
held fixed. The effective independent unit is the episode seed, not the episode.
The decision is an intersection-union test on marginal one-sided lower bounds
against a threshold of `AUC = 0.60`, with the boundary value belonging to the
null: Positive requires the lower bound above `0.60` in both primary scenarios,
Negative requires both upper bounds at or below it, anything else is
Inconclusive.

The primary scenarios are `id_reference` and `ood_compound`, where the
perturbation is present from t = 0 and the window is representative of the
regime being scored. `motor_fault` is a prespecified secondary: the fault
activates at step 20, so the primary window contains at most four post-onset
predictor observations. It is analysed on its own onset-anchored window, raw
rows `21..45`, conditional on surviving to step 45. Because that inclusion
criterion is outcome-dependent, its AUC is a conditional post-onset estimand and
is reported exclusively as a selection-affected sensitivity; it is never ranked
against or compared with the primary AUCs and cannot change the decision.

Operating characteristics were simulated on synthetic nulls before the
association was read. Two corrections were required and are recorded in the
retained evidence. First, the uniform 5th-percentile bound proved
anticonservative under the composite null, with a worst-case size of 7.8%
[6.7%, 9.1%] against a nominal 5%: the size of an intersection-union test is the
supremum over the null, attained when one component sits on the boundary while
the other lies deep in the alternative, and symmetric boundary cells are
trivially conservative. Critical quantiles were therefore recalibrated
separately per component, since the inflation is prevalence-driven and the two
scenarios carry 4.8% and 18.4% event rates. Selection and independent validation
used disjoint synthetic seeds; the frozen values are the 2.0th percentile for
`id_reference` and the 2.5th for `ood_compound`, validated at a worst-case size
of 3.65% [2.91%, 4.56%], conservative rather than anticonservative. Second,
power was recomputed under the calibrated rule: joint power at the preregistered
target of AUC 0.70 and ICC 0.10 is 38.5% against an 80% goal, so v1.6-R is
executed and reported as an explicitly **low-power feasibility screen** in which
a non-rejection is Inconclusive and never evidence of absence.

## v1.7 safety-aware policy training

v1.4 measured a paired task-success cost of 4.6 pp when the hard HOCBF filter
is applied at evaluation to a policy that never saw it during training. v1.7
asks whether training through the filter removes that cost: a policy optimised
against the projected command should learn to propose commands the filter
accepts, instead of commands it must correct.

Two conditions share architecture, optimiser, randomisation, budget,
validation seeds and checkpoint-selection rule. `C0_posthoc_hocbf` trains
unfiltered and is evaluated through the HOCBF, as in v1.4. `C1_inloop_hocbf`
trains inside `SafetyProjectedEnv`, which projects every proposed command
through the same filter before the plant step; an infeasible projection
terminates the training episode with a fixed terminal penalty of `-500`,
chosen below the worst feasible episode return so that an abort is never
preferable to any feasible outcome. No other reward shaping is applied.
Validation runs through the filter in both conditions.

Five optimisation seeds `20..24` are paired across conditions, 200,000
action decisions each, 5,000 random warm-up decisions, replay capacity
200,000, one update per decision. Validation every 25,000 decisions on 30
shared episodes (seeds `630000..630029`) selects the checkpoint by success
rate, then mean reward, then earliest step. An engineering smoke run had
opened seeds `10` and `610000..`, so the official ranges were moved before the
source freeze and no official seed was opened before it.

The ten selected checkpoints are evaluated on fresh paired episodes:
`id_reference` 500 episodes from seed `620000`, `ood_compound` 100 from
`621000`, `motor_fault` 100 from `622000`, 7,000 in total. The primary estimand
is the ID success difference `C1 - C0`. A two-level paired bootstrap of 20,000
replicates resamples the five seed pairs, then the paired episode seeds within
each pair. `go` required a gain of at least `+3 pp` with a positive lower
bound, at least four non-negative seed effects, and ID unsafe-episode and
abort rates within `+2 pp` of C0. `no_go` follows a non-positive point
estimate or a safety breach.

Result: **no_go**. ID success fell from 39.72% to 22.44%, `-17.28 pp`
`[-23.60, -9.80]`, with every seed effect negative. Safety did not change
(unsafe 6.36% against 6.44%, aborts 4.20% against 4.16%); the filter
intervened less often on the in-loop policy (27.4% of steps against 36.4%) and
corrected less (2.57 against 3.72 N m), so the policy did learn to propose
acceptable commands, at the price of reaching the target less often. The
training-replay diagnostic retained for the follow-up showed that in C1 the
160 abort transitions in one million carried 31.39% of the critic's squared
error, which motivated v1.8. Replay distributions differ between arms, so that
share is descriptive.

Commands, per condition and seed, then inventory, evaluation and aggregation:

```bash
python tools/run_planar_v17.py train-shard --condition C1_inloop_hocbf --training-seed 20
python tools/run_planar_v17.py build-inventory
python tools/run_planar_v17.py evaluate-shard --training-seed 20
python tools/run_planar_v17.py aggregate-evaluation
```

Retained evidence is under `results/safety_aware_training/`: the checkpoint
inventory, per-seed training manifests, selections and validation curves,
`evaluation/{aggregate,decision}.json`,
`evaluation/{episodes,safety_diagnostics,summary,paired_comparisons}.csv`
and the evaluation manifest with output hashes. Checkpoints are hash-bound in
the inventory and remain local.

## v1.8 terminal-penalty ablation

v1.8 isolates one candidate cause of the v1.7 loss: the `-500` terminal
penalty. Two in-loop arms differ only in the training penalty,
`P0_inloop_reference` at `-500` and `P1_inloop_half_penalty` at `-250`.
Validation uses `-500` in both arms so that checkpoint selection is common;
held-out evaluation assigns no artificial abort reward. Training replays the
raw normalised policy action rather than the projected command.

Five paired seeds `30..34`, 200,000 decisions each, checkpoints every 50,000
decisions, validation every 25,000 on 30 episodes from seed `640000`. Held-out
evaluation per model: 500 ID episodes from `650000`, 100 OOD from `651000`,
100 fault from `652000`, 7,000 in total. The bootstrap is crossed and paired,
20,000 replicates, seeds `180000..180002`, resampling training-seed pairs and
shared episode seeds. `advance` required an ID gain of at least `+3 pp` with a
positive lower bound, at least four positive seed pairs, ID unsafe and abort
upper bounds within `+2 pp` and `+1 pp`, no pointwise veto and no increase of
the late-training abort rate above `+1 pp`.

Result: **inconclusive**. ID success 23.48% (P0) against 28.00% (P1),
`+4.52 pp` `[-0.88, +10.40]`; seed effects `+1.8, +14.0, +7.8, -0.4, -0.6` pp.
OOD `+0.8 pp` `[-0.4, +2.4]`, fault `+3.0 pp` `[-2.0, +8.4]`. ID unsafe 8.68%
against 7.68%, ID aborts 4.76% against 4.84%, late-training abort difference
`+0.55 pp`. No veto fired, which does not establish safety equivalence. The
critic diagnostic on each arm's own replay gives aggregate RMSE 2.632 (P0)
against 1.667 (P1) with abort shares of 31.39% and 11.03%; the replays differ,
so no mediation claim is made. Absolute rates are not comparable with v1.7,
whose campaign used different seeds.

The campaign was interrupted once: training stopped advancing while processes
stayed alive and GPU memory stayed allocated, coinciding with Windows
`nvlddmkm` driver errors. After a WSL restart the P0 arm resumed from the
retained 100k checkpoints (seed 33 from 150k). The 200k budget per model is
preserved; the recomputed segments are not bit-identical to the lost ones, and
the workflow log was rewritten at the restart while training logs are
append-only.

```bash
python tools/run_planar_v18.py --output results/penalty_ablation run-all
```

Retained evidence is under `results/penalty_ablation/`: `campaign.json`
with the frozen protocol and runtime, `checkpoint_inventory.json`,
`aggregate.json`, `summary.csv`, `episodes.csv`, `safety_diagnostics.csv`,
`critic_diagnostics.json`, per-seed training manifests, selections, validation
curves and logs, and the `complete.json` / `workflow_complete.json` chain. The
evaluation shards duplicate the top-level CSVs and remain local, as do the
checkpoints.

## Failure diagnosis after v1.8

Three releases in a row (v1.5, v1.7, v1.8) had measured how a change to the
learned component moves success, without establishing why the frozen policies
fail. The diagnosis below is exploratory: it runs on six fixed initial
conditions, two per scenario, with seeds `9801800..9801801` (ID),
`9801900..9801901` (OOD) and `9802000..9802001` (fault), outside every
official and smoke range. Counts on six cases do not estimate population
rates, and the cases, once observed, informed the hypotheses tested on them.

### Official failure classification

The 7,000 official v1.8 episodes split into 3,473 ID timeouts, 240 ID aborts
and 1,287 ID successes across both arms. Exactly one ID timeout ends within
5 cm of the target, and the median final distance of ID timeouts is 0.16 to
0.18 m. Final velocity alone therefore does not explain the timeouts; the
policies stop short of the target. The classification is in
`results/penalty_ablation/failure_audit/official_failure_summary.json`, with
the 60-episode trajectory audit, its protocol, integrity checks over 208
retained hashes and manifest alongside.

### Null-residual control

Sixty episodes replay the ten frozen v1.8 checkpoints on the six cases (arm L)
and six episodes run the same nominal controller and filter with the residual
fixed at zero (arm Z), sharing initial state, plant parameters and RNG. The
actor runs on CUDA in float32 because the historical first actions reproduce
exactly on CUDA in 60/60 cases and on CPU in only 8/60; the plant and filter
run on CPU in float64. All 60 L episodes reproduce the retained audit rows
within `1e-10`.

Z yields no success: five timeouts and one abort. On ID `9801800` the bare
nominal stops at 0.594 m while all ten policies end between 0.060 and
0.353 m. On fault `9802001` Z stops at 0.436 m while four policies succeed.
In all 17 stop tails (50 transitions with low velocity) HOCBF projection,
nominal saturation and plant clipping are exactly zero. In the two Z tails the
net torque is below `1.4e-3 N m` while the applied torque balances the real
load: on the ID case the applied torque is `(18.45, 9.00) N m`, the real load
`(18.45, 9.00) N m` and the nominal load `(11.28, 4.35) N m`. The torque
balance of seven recorded terms closes within `7.1e-15 N m`.

```bash
python tools/run_residual_diagnosis.py plan --protocol <protocol.md> --amendment <amendment.md>
python tools/run_residual_diagnosis.py run --protocol <protocol.md> --amendment <amendment.md> --output results/residual_diagnosis_20260905
```

The runner freezes the protocol and amendment by hash into its manifest and
refuses to run if any source, checkpoint or retained reference changed.

### Static review of the nominal controller

`ComputedTorqueController` forms the virtual acceleration `Kp e - Kd qd`
with `Kp = 36 I`, `Kd = 12 I`, maps it through the nominal inverse dynamics
with Coriolis, gravity and friction, and saturates at 40 N m. It holds no
integral state and no bias estimate. The environment randomises link masses,
friction, payload, motor gain and delay at reset without updating the
controller model; the motor fault scales the gain and payload of the plant
only. With `M_n` the nominal mass matrix, `l_n` and `l_r` the nominal and real
load terms, `G` the motor gains and `d - u` the delay difference,

```text
M_n Kp e = (l_r - l_n) + (G^-1 - I) l_r + M_n Kd qd + G^-1 net - (d - u)
```

holds identically along the recorded states and reproduces the joint error on
all 100 tail states within `1e-15 rad`. The stop is the equilibrium of a
fixed-model PD under a persistent disturbance. On `9801800` the load mismatch
dominates; on `9802001` the gain term dominates joint 2. The kinematic
references are exact within `3.7e-16 m`. No algebraic defect was found.

### Integral nominal candidate

One candidate was frozen before evaluation: an integral state `z` in the
virtual acceleration with `Ki = 36 I`, back-calculation anti-windup
`Kaw = 4` on the difference between the sent and the raw command, and a
per-component limit of `36 rad/s^2`. `IntegralNominalController` keeps the
pre-command data in a transaction that the caller commits after the filter
and plant step, or aborts when the filter rejects the command, so that the
integral never advances on a command that was not executed. The anti-windup
term is decomposed into nominal clipping, HOCBF correction and send clipping,
and the decomposition is checked at every commit.

Twelve episodes compare the frozen nominal (R0) with the candidate (I1) on the
six cases. I1 succeeds on both motivating stops (`9801800` in 172 steps,
`9802001` in 153) and on `9801801`, where R0 aborts. On fault `9802000`,
unsafe already under R0, I1 raises the maximum normalised violation from
0.3267 to 0.4027 and drives joint 2 to `4.278 rad` against the `3.05 rad`
limit. The preregistered order of decisions puts the safety veto first:
**no_go_safety**. The candidate is not promoted.

```bash
python tools/run_nominal_integral.py plan --protocol <protocol.md>
python tools/run_nominal_integral.py run --protocol <protocol.md> --output results/nominal_integral_v19a_20260907
```

### Reconstruction of the fault trajectory

The I1 trajectory on `9802000` was reconstructed algebraically from the
recorded states, with no new simulation. The real acceleration minus the
filter's nominal prediction decomposes exactly along the trajectory into

```text
qdd_real - qdd_nominal(h) =
    M_r^-1 (u - h) + M_r^-1 (d - u) + M_r^-1 (a - d)
  + (M_r^-1 - M_n^-1)(h - l_n) + M_r^-1 (l_n - l_r)
```

with `h` the filtered command, `u` the sent command, `d` the delayed command
and `a` the applied torque. In the 12 commands before joint 2 crossed its
limit, the nominal model predicted a mean braking of `-25.57 rad/s^2` and the
plant produced `-0.13 rad/s^2`. The difference splits into `+13.98` from the
mass matrices, `+6.27` from the faulted motor gain, `+2.75` from the delay and
`+2.45 rad/s^2` from the load terms; clipping contributes nothing. All 250
filtered commands satisfy the nominal HOCBF inequalities, so the filter never
intervened in the final 54 commands: the certificate was valid for a model the
plant no longer matched. During the same phase the integral kept a positive
torque contribution for 30 commands after the joint had passed its target,
because back-calculation compares the sent command with the raw one and
cannot see the lost motor gain. The decomposition is an identity along one
observed trajectory, not a causal attribution.

The diagnosis points at the shared fixed model rather than at the learned
residual, the reward or the penalty. Both the nominal controller and the HOCBF
certificate use it, and both were wrong by the same amount on the case that
failed. Compact outputs of the four analyses are retained under
`results/residual_analysis_20260907/`, `results/nominal_static_review_20260907/`,
`results/nominal_integral_analysis_20260912/` and
`results/integral_fault_static_review_20260912/`; the full per-episode
transition records of the two campaigns remain local.

## v1.9 adaptive nominal control

The diagnosis after v1.8 located the dominant failure mode in the fixed model
shared by the computed-torque nominal and the HOCBF certificate. v1.9 asks
whether identifying the plant online, and giving the identified model to both
the controller and the filter, recovers task success without weakening the
safety envelope. No learned residual is involved and nothing is trained: the
study compares two model-based controllers behind the same filter.

### Controller under test

`AdaptiveNominalController` identifies the plant in command coordinates.
Dividing the joint-i torque equation by its motor gain leaves a relation that
is linear in seven parameters per joint (two link masses, two inertias,
payload, viscous and Coulomb friction, all scaled by `1 / g_i`) and whose
left-hand side is the command the controller sent. A recursive least-squares
estimator per joint runs on the finite-difference acceleration of the measured
velocity, with the regressor at the midpoint state, a random-walk covariance
term (`process_noise = 0.02` of the prior) so that an in-episode change is
tracked, and a physical box on the estimates. One estimator per candidate
actuator lag `0..3` runs in parallel; the lag with the smallest squared
innovation accumulated over the episode supplies the parameters. The control
law is the same PD in the virtual acceleration as the fixed nominal
(`Kp = 36 I`, `Kd = 12 I`), mapped through the identified model.

The HOCBF filter receives the identified model through a command-space view:
the mass matrix has row i scaled by `1 / g_i`, so the affine acceleration map
the filter linearises is the one from command to acceleration. Because the
controller knows the commands still queued in the actuator and the identified
lag, the state at which the new command will act is predicted by integrating
the identified model over those commands, once the lag has at least ten
updates behind it, and both the control law and the certificate are evaluated
at that predicted state. The configuration is `AdaptiveNominalConfig()` with
library defaults and `compensate_delay = True`; no parameter is changed after
this section is committed.

The estimator consumes only joint positions, velocities and its own sent
commands. Plant parameters, motor gains and the true delay are read from the
environment for diagnostics only, after the episode, and never enter the
control path. Both arms receive the same input: the benchmark has provided
full-state, noise-free feedback to every controller since v1.2, and the
adaptive arm inherits that convention rather than a separate measured-state
interface. The study therefore says nothing about sensor noise.

### Design

Four arms run on identical episode seeds, plant draws and targets:

- `fixed`: the frozen computed-torque nominal, unfiltered;
- `fixed_hocbf`: the same nominal behind the HOCBF on the nominal model;
- `adaptive`: the adaptive nominal, unfiltered;
- `adaptive_hocbf`: the adaptive nominal behind the HOCBF on the identified
  model, with delay compensation.

Episodes fall into two blocks in each of the three v1.3 scenarios
(`id_reference`, `ood_compound`, `motor_fault`).

The **decision block** runs the two filtered arms on seeds `50200..52099`,
1,900 paired episodes per scenario. These seeds have never been used as
episode seeds by any retained artifact: `tools/scan_seed_usage.py` reads the
committed CSV, JSON and JSON-lines blobs of `HEAD`, recognises any column or
key whose name contains `seed`, expands `*_start` keys into ranges with their
sibling end or count, and finds no value in `50200..52099`. Step counters and
rewards are not seeds. The runner repeats that scan against the committed
tree, refuses to start on a hit, and records the result in the manifest. None
of these seeds was opened by the pilot, which used `9801800..9802001` and
`9803000..9803219`. Seed `50100` was executed once by a unit test of the
runner before the protocol was sealed, so the block starts at `50200` and
`50100..50199` stay unused; the test suite now runs the runner's cell
function on a pilot seed only.

The **reproduction block** runs all four arms on seeds `50000..50099`, the
v1.3 and v1.4 evaluation seeds, and carries no decision weight. It links the
campaign to retained evidence: the unfiltered `fixed` arm must reproduce the
retained `A0_computed_torque` rows of `results/ood_fault_robustness/heldout_episodes.csv`
on success and final distance, and the analysis records how many of the 300
rows match. The reference file is among the frozen paths, its SHA-256 is
recorded in the manifest, and the analysis refuses to run if it is missing
or does not hold exactly those 300 rows. Those seeds were opened by earlier
campaigns after which the adaptive controller was designed, so they do not
enter the decision.

The campaign totals 12,600 episodes.

The primary contrast is `adaptive_hocbf` minus `fixed_hocbf`. The two
unfiltered arms are descriptive: they separate the effect of the model from
the effect of the filter but carry no decision weight.

### Endpoints and decision rule

All contrasts are computed on the decision block, paired over its 1,900
episode seeds, with a 10,000-draw percentile bootstrap seeded at `190000`.
The rule is applied in this order:

1. **Safety veto.** For each of the three scenarios, the paired difference in
   unsafe-episode rate (adaptive minus fixed) triggers the veto if its 95%
   lower bound is above zero, or if its point value exceeds `+3 pp`. Any
   scenario vetoed yields `no_go_safety`, whatever the task result.
2. **Primary endpoint.** The paired success difference must be at least
   `+20 pp` with a 95% lower bound above zero in both `id_reference` and
   `motor_fault`.
3. **Non-inferiority.** In every scenario the unsafe-episode difference must
   have a 95% upper bound at or below `+3 pp`.
4. `go` requires 2 and 3 together. Anything else without a veto is
   `inconclusive`, and the decision file names which of the two failed.

The margins are sized to the precision the sample affords. At the
unsafe-episode rates the fixed filtered arm showed in the pilot (roughly
5..15%), 1,900 paired episodes give a 95% interval half-width of about
1.4..2.3 pp, so two arms with identical rates show non-inferiority at `+3 pp`
in most draws, a worsening of about 3 pp or more is detected by the veto, and
a worsening between those sizes leaves the safety claim open and the decision
`inconclusive`. A `go` therefore supports the claim "recovers success without
weakening the safety envelope by more than 3 pp in any scenario"; it does not
exclude a smaller weakening.

Secondary, reported without decision weight: the success contrast under
`ood_compound`; abort rates and their paired differences; intervention
fractions; median final distances; maximum normalised violations; the
fraction of adaptive episodes whose selected lag equals the true delay; and
the per-joint RMS prediction error of the final estimate on one preregistered
bank of 50 probe states shared by every episode (drawn once from seed
`190001`), reported per episode in `episodes.csv` and as a per-cell mean in
`decision.json`.

### What the pilot showed and what it did not

The hypothesis was formed on the six diagnostic cases and refined on 60 fresh
seeds outside every official range. On those 60 seeds `adaptive_hocbf` with
delay compensation reached the target in 20/20 ID, 16/20 OOD and 19/20 fault
episodes against 4/20, 0/20 and 0/20 for `fixed_hocbf`, with 0, 3 and 1
unsafe episodes against 1, 2 and 3. Two of the OOD unsafe episodes exceeded
the envelope by less than `1e-4` in normalised units. Delay compensation was
added after tracing the pilot's worst OOD abort to a three-step queue; a
payload prior and a convergence gate on the filter model were also tried and
discarded. Those 60 seeds are therefore not evidence and are not reused.

The OOD result is declared uncertain a priori: the pilot produced aborts
there, and OOD carries the largest payload and the longest delays. Targets
that lie close to a joint limit remain unreachable through the filter for
either controller. The kinematic reference does not choose the joint
representation inside the limits, so a target whose shorter angular path
crosses a limit is blocked by the filter. The lag is identified but the
nominal PD is not retuned for it, and the prediction relies on the identified
model being adequate over at most three steps.

### Commands and retained evidence

```bash
python tools/run_adaptive_campaign.py --workers 4
```

The output path is fixed to `results/adaptive_nominal_v19`. The protocol is
sealed in two commits: the freeze commit fixes this section and the code, and
a sealing commit writes the freeze commit's full hash to `docs/v19_seal.json`,
which lies outside the compared paths so that the seal does not invalidate
itself. Before the first episode the runner checks that the seal names a
40-character commit id that exists and is an ancestor of `HEAD`, that the
seal file is committed and unmodified, that `sarrl/`, `tools/`, `tests/`,
`docs/experiments.md`, `pyproject.toml` and the reproduction reference at
`HEAD` are identical to the sealed commit and carry no modification or
untracked file, repeats the seed scan against the committed tree, and writes
`manifest.json` with the protocol, the commit, the tree hashes of the frozen
paths, the seed scan, the reference hash, the runtime and an execution
fingerprint (interpreter, platform, installed distributions, worker count).
Every completed cell is appended to `journal.jsonl` as one flushed record, in
completion order, so that nothing finished is lost and nothing is executed
twice. An interrupted run resumes only if the existing manifest carries the
same protocol, source tree and execution fingerprint; it re-validates the
journal against the planned cells and runs the missing ones; each session is
timestamped in the manifest; a run with `complete.json` is never repeated.
At the end the canonical ordered `episodes.jsonl` is assembled from the
journal and reloaded for the analysis, which checks that it holds exactly the
planned cells and writes `episodes.csv`, `decision.json` with the full
analysis and the reproduction check, and `complete.json` hashing every
output. All of these are retained.
