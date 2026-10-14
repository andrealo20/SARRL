# Shared-parameter scoring results

Source `1a47469` reproduced all 4,000 constant-delay validation records before
opening twenty fresh paired seeds. The [frozen protocol](panda-shared-scoring.md)
crossed sixteen conditions and six arms, retaining 1,920 episodes. The scoring
vector was acquired at update 50 in all 320 candidate episodes; none failed
the acquisition rule. All five control estimators remained cumulative.

| Controller | Successes | Correct constant final ages | Correct changing final ages | Unsafe episodes |
|---|---:|---:|---:|---:|
| Fixed model | 0/320 | Not estimated | Not estimated | 0 |
| Original adaptive | 272/320 | Not estimated | Not estimated | 0 |
| Cumulative bank | 320/320 | 160/160 | 0/160 | 0 |
| Known-age diagnostic | 320/320 | 160/160, supplied | 160/160, supplied | 0 |
| Score-only memory | 317/320 | 159/160 | 79/160 | 0 |
| Shared-parameter scoring | 316/320 | 155/160 | 96/160 | 0 |

The candidate fails the exploratory screen and does not advance to
confirmation. Changing-age correctness gains 10.625 percentage points over
score-only memory, below the required twenty points. The clean decreasing
cells gain zero points from cold startup and thirty points from warm startup;
both cells were required to gain at least twenty. Constant-age correctness,
success counts and per-joint tracking criteria also fail.

## Every condition

All entries use twenty paired seeds. Correct age means the final selected
command age, not task success. Acquisition occurred at update 50 in every cell.

| Condition | Cumulative age | Score-only age | Shared age | Shared success |
|---|---:|---:|---:|---:|
| cold_clean_fixed10 | 20/20 | 20/20 | 20/20 | 20/20 |
| cold_clean_fixed40 | 20/20 | 20/20 | 20/20 | 20/20 |
| cold_clean_up | 0/20 | 2/20 | 2/20 | 20/20 |
| cold_clean_down | 0/20 | 0/20 | 0/20 | 20/20 |
| cold_noise_fixed10 | 20/20 | 20/20 | 20/20 | 20/20 |
| cold_noise_fixed40 | 20/20 | 20/20 | 20/20 | 20/20 |
| cold_noise_up | 0/20 | 20/20 | 20/20 | 20/20 |
| cold_noise_down | 0/20 | 16/20 | 20/20 | 18/20 |
| warm_clean_fixed10 | 20/20 | 19/20 | 16/20 | 20/20 |
| warm_clean_fixed40 | 20/20 | 20/20 | 19/20 | 20/20 |
| warm_clean_up | 0/20 | 1/20 | 8/20 | 20/20 |
| warm_clean_down | 0/20 | 0/20 | 6/20 | 19/20 |
| warm_noise_fixed10 | 20/20 | 20/20 | 20/20 | 20/20 |
| warm_noise_fixed40 | 20/20 | 20/20 | 20/20 | 20/20 |
| warm_noise_up | 0/20 | 20/20 | 20/20 | 20/20 |
| warm_noise_down | 0/20 | 20/20 | 20/20 | 19/20 |

The decision records 74 failed comparisons: 51 per-joint post-switch RMS
comparisons, thirteen final-error comparisons, four success-count comparisons,
four constant-age comparisons and two age-gain criteria. These counts include
separate comparisons against both references and are not independent trials.
The largest RMS increase is 5.014908 mrad at joint six in `cold_noise_down`
against the cumulative bank. The largest mean final maximum-error increase
is 6.095071 mrad in the same cell, against a 0.1 mrad tolerance. Four candidate
episodes miss the final joint-error threshold without measured physical
violations.

## Interpretation and retained evidence

Using one causally acquired vector improves some age selections but this
acquisition rule is insufficient for the declared grid. Full information rank
and stable early selections do not establish parameter accuracy. This result
does not rule out a separately calibrated physical estimate; such an estimate
requires its own protocol and untouched seeds. No acquisition length is tuned
on this block, and the cumulative bank remains the reference.

The [retained directory](../results/panda_shared_scoring_pilot) contains all
records, 96 first-seed traces, acquisition diagnostics, per-transition
innovations, reproduction and decision reports, and completion hashes. New
episodes use 2,976,000 physical substeps including warmup; reproduction uses
6,000,000. Prediction adds no virtual substeps. The audit verified all records,
96 traces and 80 causal estimator replays without new physical episodes.
