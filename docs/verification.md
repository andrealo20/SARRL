# Verification record

This file distinguishes checks that were actually executed from features that are only implemented or
planned.

## Test coverage checklist

The automated regression suite is executed across Python 3.10, 3.11 and 3.12.
The exact test count evolves with the codebase; release-specific historical counts are retained below. The CI pipeline runs `ruff check .`, `python -m compileall -q sarrl tests
tools` and `pytest`. Important regression and numerical tests cover:

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

The verification environment did not provide MuJoCo or Gymnasium. The v1.0 planar release has no dependency on either. M10 Franka/MuJoCo transfer remains explicitly unimplemented until it can be exercised and tested rather than added as dead code.
