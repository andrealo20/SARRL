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
`step = 0..24`** — 25 transitions, 0.5 s at `dt = 0.02`, the first 10% of the
horizon — identical for every episode so that exposure does not vary with
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
of 3.65% [2.91%, 4.56%] — conservative rather than anticonservative. Second,
power was recomputed under the calibrated rule: joint power at the preregistered
target of AUC 0.70 and ICC 0.10 is 38.5% against an 80% goal, so v1.6-R is
executed and reported as an explicitly **low-power feasibility screen** in which
a non-rejection is Inconclusive and never evidence of absence.
