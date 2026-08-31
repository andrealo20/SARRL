# SARRL v1.2 planar ablation evidence

Canonical retained evidence for the randomized planar ablation study.

## Protocol

- training seeds: 0–4
- training budget: 200,000 environment steps per learned policy
- validation: 30 episodes, seeds 20000–20029
- held-out evaluation: 100 episodes, seeds 40000–40099
- domain randomization:
  - mass: ±15%
  - friction: ±30%
  - motor gain: ±15%
  - payload: 0–1 kg
  - action delay: 0–2 steps
- multi-seed spread: sample standard deviation (`ddof=1`)
- episode success intervals: Wilson 95%
- paired comparisons: paired bootstrap 95%

## Current conditions

| Condition | Controller | Held-out success |
|---|---|---:|
| A0 | Computed torque | 11.0% |
| A1 | Direct SAC | 6.0% ± 3.7 pp |
| A2 | Residual SAC | 56.4% ± 7.0 pp |

Residual SAC improved over Direct SAC by **+50.4 percentage points**
on average across the five training seeds. All five paired-bootstrap
95% confidence intervals for A2 minus A1 excluded zero.

Direct SAC did not demonstrate an improvement over the computed-torque
baseline: its mean difference was -5.0 percentage points, and the
paired confidence interval excluded zero for only one of the five
training seeds.

A2 is retained from the v1.1.0 evidence campaign and was not retrained
for this comparison. See `provenance.json`.

Model checkpoints are intentionally not retained in this directory.
