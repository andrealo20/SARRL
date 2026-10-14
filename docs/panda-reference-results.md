# Panda reference validation and timing

The frozen [validation protocol](panda-reference-validation.md) reproduced
all 1,920 v2.4 records before running 4,000 fresh paired episodes. Fifty
seeds crossed two posture ranges, clean/noisy measurements and five constant
delays from 0 to 40 ms. Source commit `09858a7` is preserved in history.

| Controller | Task successes | Correct final ages | Measured unsafe episodes |
|---|---:|---:|---:|
| Fixed model | 0/1,000 | Not estimated | 0 |
| Original adaptive | 934/1,000 | Not estimated | 0 |
| Cumulative bank | 1,000/1,000 | 1,000/1,000 | 0 |
| Known-age diagnostic | 1,000/1,000 | 1,000/1,000, supplied | 0 |

The reference passed every aggregate and per-cell criterion, including each
joint's tracking comparison with the privileged diagnostic. The audit checked
4,000 records, 80 first-seed traces and 60 causal estimator replays. Fresh
episodes and reproduction used 8,976,000 physical substeps and zero virtual
prediction substeps. Completion hashes retain the records and decision in
[`results/panda_reference_validation`](../results/panda_reference_validation).

The table below reports paired mean bank-minus-reference differences in
mrad, with 95% percentile intervals from 5,000 complete-seed-block resamples.
Each resample preserves the twenty conditions belonging to a seed. These
intervals describe the fifty task/parameter draws; the engineering gate is
reported separately and does not establish general safety or population
noninferiority.

| Metric | Against original adaptive | Against known-age diagnostic |
|---|---:|---:|
| Final maximum joint error | -3.567211 [-3.904988, -3.240576] | -0.000001 [-0.000003, 0] |
| Full-episode RMS, mean across seven joints | -2.265941 [-2.460985, -2.063037] | 0.000055 [0, 0.000165] |

## Saved-input control-cycle timing

Profiling replayed the twenty retained bank traces sequentially after one
discarded warmup pass. Five measured passes retained 30,000 durations on an
Intel Core i9-13905H under WSL2, Python 3.12.3, NumPy 2.5.2 and MuJoCo
3.13.0, with one process and one numerical thread.

| Statistic | Duration |
|---|---:|
| p50 | 0.976287 ms |
| p95 | 1.378157 ms |
| p99 | 1.827530 ms |
| Maximum | 3.928535 ms |
| Cycles exceeding the 10 ms period | 0/30,000 |

The measured cycle includes state preparation, reference evaluation, control
and the causal estimator update. Plant stepping, construction, I/O and
diagnostic serialization are excluded. This host/workload check passed;
desktop scheduling and the omitted costs prevent a hard real-time or hardware
claim. Timing opened no seeds and added no physical steps. The timing audit
verified all 30,000 retained durations and their aggregates in
[`results/panda_reference_profile`](../results/panda_reference_profile).
