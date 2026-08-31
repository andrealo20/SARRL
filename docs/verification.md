# Verification record

This file distinguishes checks that were actually executed from features that are only implemented or planned.

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

The test suite covers:

- rigid-body dynamics invariants;
- kinematics and finite-difference Jacobians;
- numerical integration;
- computed-torque convergence;
- nonlinear MPC constraints and optimisation behaviour;
- deterministic environment randomisation, noise, delay and faults;
- replay sampling and state restoration;
- SAC Bellman targets, actor log probabilities, entropy tuning and target networks;
- deterministic SAC inference without RNG consumption;
- architecture-safe checkpoint reconstruction;
- exact training-session continuation including replay/environment/RNG state;
- GRU context causality, training and checkpointing;
- exact 2-D polytope projection and HOCBF safety semantics;
- residual-dynamics targets, learning, checkpointing and uncertainty gating;
- motor-gain mismatch visibility in residual-model targets;
- integrated runtime-stack behaviour;
- Wilson intervals, paired bootstrap and validation/test seed separation.

## Nominal computed-torque baseline

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

## v0.9 robustness baseline campaign

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
