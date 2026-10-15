# Adaptive residual comparison results

The bounded residual fails the [frozen comparison](adaptive-residual-comparison.md).
On the equal-weight ID/motor-fault endpoint, the five-policy mean success is
65.9%, against 90.5% for the adaptive reference. The paired difference is
minus 24.6 percentage points, with a crossed-bootstrap 95% interval of
[-32.2, -17.3] points. All five policies have negative primary differences.
The adaptive reference remains unchanged; this residual does not advance to Panda.

Source `b41e167c2a609849c01031010e5710aa4801670f` completed the entire
registered sequence: 300 historical records, five fixed-budget training
runs and 2,100 fresh held-out episodes. No model was discarded or selected
using held-out results. The preceding Panda filter failed its separate screen.

## Every scenario and policy

Each policy and baseline has 100 paired task seeds per scenario. The learned
mean weights the five models equally. Success, unsafe episodes and aborts
are separate endpoints; unsafe events can occur in successful tasks.

| Scenario | Fixed success | Adaptive success | Residual mean success | Adaptive unsafe | Residual mean unsafe | Adaptive abort | Residual mean abort |
|---|---:|---:|---:|---:|---:|---:|---:|
| ID | 14% | 90% | 62.0% | 1% | 1.8% | 7% | 6.4% |
| Compound OOD | 1% | 75% | 66.6% | 15% | 11.6% | 6% | 7.0% |
| Motor fault | 4% | 91% | 69.8% | 0% | 0.8% | 4% | 5.0% |

| Training seed | ID success | OOD success | Fault success | Primary success difference |
|---|---:|---:|---:|---:|
| 9806000 | 65/100 | 68/100 | 70/100 | -23.0 pp |
| 9806001 | 72/100 | 64/100 | 78/100 | -15.5 pp |
| 9806002 | 54/100 | 61/100 | 61/100 | -33.0 pp |
| 9806003 | 55/100 | 67/100 | 66/100 | -30.0 pp |
| 9806004 | 64/100 | 73/100 | 74/100 | -21.5 pp |

| Scenario | Mean success ± model SD | Model range | Mean episode return |
|---|---:|---:|---:|
| ID | 62.0% ± 7.5 pp | 54% to 72% | -116.62 |
| Compound OOD | 66.6% ± 4.5 pp | 61% to 73% | -184.90 |
| Motor fault | 69.8% ± 6.6 pp | 61% to 78% | -120.85 |

SD is the sample standard deviation across the five training models, not a
confidence interval over individual episodes. Returns average each model's
100 episodes and then the five models equally.

The residual improves success over the descriptive fixed baseline, but loses
against the adaptive controller that it augments. Earlier fixed-baseline
learning results therefore do not establish a benefit on this stronger stack.
These data reject this recipe and budget; they do not establish that every
possible learned correction would fail. No post-result tuning or replacement
training seed is included in this release.

## Paired contrasts and decision

All values below are residual minus adaptive, in percentage points. Intervals
use 5,000 crossed bootstrap draws over five independent models and 100 whole
task blocks. Each task block retains all three scenarios and all endpoints.
The 1,500 learned-policy episodes are not independent training replications.

| Scenario | Success difference [95% interval] | Unsafe difference [95% interval] | Abort difference [95% interval] |
|---|---:|---:|---:|
| ID | -28.0 [-36.8, -19.6] | +0.8 [-0.8, +3.0] | -0.6 [-3.4, +1.8] |
| Compound OOD | -8.4 [-14.8, -2.6] | -3.4 [-8.2, +0.8] | +1.0 [-1.8, +4.4] |
| Motor fault | -21.2 [-29.2, -13.8] | +0.8 [0.0, +2.6] | +1.0 [-0.4, +3.4] |

Seven registered checks fail: success noninferiority in all three scenarios,
the unsafe upper bound in ID and motor fault, the primary success gain and
the requirement for at least four nonnegative model-level primary differences.
The OOD unsafe bound and every abort point-increase bound pass. Abort
confidence intervals are descriptive; the registered abort rule uses the
point increase. Passing some checks does not override the failed screen.

## Training and historical reproduction

Each final model received 200,000 environment calls and 195,000 optimizer
updates. Across all five runs, 999,639 calls executed physical control and
361 were abort transitions. Checkpoints include replay, environment,
optimizers and random states. Only the final policies were evaluated, with
an 8 N m per-joint residual bound and the original covariance margin.

The historical replay matches 300/300 records under the portability bound
fixed before fresh training (`rtol=1e-7`, `atol=1e-9`). It matches 299/300
under the original stricter bound (`rtol=1e-9`, `atol=1e-10`). The five strict
floating-field differences in ID seed `56212` remain published; discrete
values, types and metadata must match exactly. The
[protocol amendment](adaptive-residual-comparison.md#historical-portability-amendment-15-october-2026)
describes the observed drift and its limits. The acceptance criteria did not change.

## Evidence and audit scope

The [retained directory](../results/planar_residual_v26) contains every held-out
record, both historical comparisons, all recalculated historical records,
21 first-seed traces, model hashes, training summaries and completion hashes.
The recorded runtime is Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1,
MuJoCo 3.14.0 and PyTorch 2.12.0+cpu, with two workers and one numerical
thread per worker. The plant envelope is sampled at control boundaries;
this study adds no guarantee about unobserved physical substep extrema.

The public audit recomputes pairing, statistics, the decision, historical
comparisons and trace bounds from the frozen source. It reports
`policy_inference_verified: false`. The full local audit additionally requires
the ignored model/checkpoint files and checks their hashes, actor actions and
causal controller/estimator replays. Public statistics alone do not establish
policy inference. See the [release guide](v2.6.md) for audit commands.

The full local audit passed all 2,100 records, 300 historical records,
21 traces, 2,416 policy-action checks and 21 causal stack replays. Its output
reports `policy_inference_verified: true`. This local result does not change
the narrower inference-free scope of the public audit.
