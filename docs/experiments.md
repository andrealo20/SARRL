# Experiment protocol

## Seed populations

Three seed populations have different roles and must remain disjoint:

```text
training seeds   independent optimisation randomness
validation seeds checkpoint selection
held-out seeds   final evaluation only
```

The default examples use validation from 20000 and held-out evaluation from 40000.

## Model-level reporting

One policy evaluated for many episodes does not measure training instability. Final learned results should therefore train at least five independent models and report, at minimum:

- held-out success rate for each training seed;
- mean and standard deviation across training seeds;
- minimum and maximum success rate;
- mean return across models;
- retained raw per-episode results.

Episode-level Wilson intervals may also be reported for each model, but they do not replace cross-seed variation.

## Checkpoint selection

`tools/train_sac.py` evaluates deterministic policy checkpoints on a fixed validation set. `best.pt` is updated by:

```text
(success_rate, mean_return)
```

in lexicographic order.

Validation uses a separate environment instance. Deterministic SAC inference does not sample internally, so validation does not advance the policy RNG and therefore does not change the subsequent stochastic training trajectory.

## Multi-seed campaign

`tools/run_sac_sweep.py` runs independent training seeds and evaluates the selected checkpoint on the same held-out environment seeds for a paired comparison.

It writes:

- one run directory per training seed;
- a per-run manifest;
- a sweep manifest;
- per-model summary CSV;
- raw held-out episode CSV;
- cross-model aggregate JSON.

The runner rejects overlapping validation and held-out ranges.

## Baseline scenarios

The planar baseline campaign includes:

1. nominal dynamics;
2. in-distribution identification uncertainty;
3. stronger OOD dynamics mismatch;
4. abrupt joint-2 motor degradation.

These scenarios establish the model-mismatch gap before learned compensation is credited with improvement.

## Retained v1.1.0 method-specific campaign

The v1.1.0 release retains a completed five-seed residual-SAC campaign following the seed-separation and
model-level reporting rules above. It is a controlled method-specific result, not the complete comparative study
listed below. Raw evaluation evidence and provenance are stored in `artifacts/planar_sac_5seed/`.

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

Useful metrics include success rate, return, terminal distance, successful-step count, peak speed, peak command torque, safety intervention magnitude and hard-safety infeasibility rate.

## OOD protocol

OOD parameters must lie outside the training randomisation range. For example, if training payload is sampled from 0 to 1.0 kg, an OOD payload test should use a fixed value above 1.0 kg rather than another sample from the same interval.

## Negative results

Incomplete or unsuccessful training probes are not promoted to headline metrics. They may be retained as debugging evidence, but only controlled completed campaigns belong in the main result table.
