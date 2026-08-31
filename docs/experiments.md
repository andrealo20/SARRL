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
  --output results/residual_dynamics/ensemble.pt
```

### Evaluate the composed runtime stack

```bash
python tools/evaluate_stack.py \
  results/residual_seed0/best.pt \
  --episodes 100 \
  --randomize \
  --safety
```
