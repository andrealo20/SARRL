# Planar residual-SAC five-seed evidence

This directory contains the retained evidence for the SARRL v1.1.0 method-specific learned-policy result.

## Result

Five independently trained residual-SAC policies achieved held-out success rates of **61%, 57%, 63%, 56% and
45%** on 100 episodes each, giving **56.4% ± 7.0 percentage points** (mean ± sample SD across training seeds).
The computed-torque baseline achieved **11.0%** on the identical held-out episode seeds. Mean paired improvement
was **+45.4 pp**, and every per-policy paired bootstrap 95% confidence interval excluded zero.

## Protocol

- source training commit: `9f832614ce8b51c207873ff4861986ab72903115` (`v1.0.1`);
- training seeds: `0..4`;
- 200,000 steps per training run;
- validation seed start: `20000`, 30 episodes per validation checkpoint;
- held-out seeds: `40000..40099`;
- 100 held-out episodes per selected policy;
- computed-torque baseline evaluated on the same held-out seeds.
- validation-selected checkpoints came from steps `200k, 200k, 200k, 150k, 200k` for training seeds `0..4`.

The held-out set was not used for checkpoint selection. Seed 0 was resumed from its 100,000-step checkpoint; the
run manifest records that resume source.

## Files

- `summary.csv`: per-training-seed held-out metrics and Wilson intervals.
- `heldout_episodes.csv`: all 500 policy held-out episodes.
- `baseline_heldout_40000.csv`: paired computed-torque baseline episodes.
- `paired_comparison.csv`: per-policy paired success differences and bootstrap intervals.
- `validation_seed_*.csv`: checkpoint-selection learning curves.
- `run_manifest_seed_*.json`: runtime and training provenance.
- `checkpoint_sha256.txt`: SHA-256 fingerprints of the evaluated `best.pt` checkpoints.
- `result.json`: release-level summary using sample SD across training runs.
- `aggregate.json`: original generated cross-model aggregate (population SD, `ddof=0`).

The model checkpoint binaries are intentionally not committed because of their size. This bundle supports the
retained randomized planar residual-SAC result only; it is not evidence for Franka/MuJoCo, hardware or sim-to-real
performance.
