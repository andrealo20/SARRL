# Panda joint filter on the selected delay

The [v2.6 joint filter](panda-joint-filter.md) failed its screen on fast moves:
107/120 aborts and 20/120 unsafe episodes. The
[saved-data diagnosis](control-failure-results.md) separates two mechanisms.
All 107 aborts are `pending_command_violation`, raised before any QP because
the constraints are built for every constant age 0..4 at once; in the 0 ms and
20 ms first-seed traces a construction limited to the recorded delay would have
reached the QP. All 20 violations are at 40 ms, after an abort, under commands
sent before braking latched, and there the recorded-delay construction still
rejects.

This mechanism test changes one thing: the filter is built only over the ages
the cumulative delay bank cannot yet tell apart. Everything else is the v2.6
filter. The draft protocol went through three rounds of independent
adversarial review before the freeze. Source, predictions and decision rules
were frozen at commit `1812d9d7958e02c468409826cb0976e497b430fa`.

## The change

At control step `k`, the number of transitions the bank has observed, the
filter uses

`L_k = { a in 0..4 : score_a == min_j score_j }`,

where `score_a` is the bank's accumulated pre-update innovation for age `a`.
At step 0 every score is zero and `L_0 = {0..4}`. While the true age is at
least `k`, the ages from `k` upward have only been aligned with the known zero
startup queue, so their estimators and scores are equal bit for bit, also with
sensor noise. Any later score gap removes an age. This is certainty-equivalent
filtering on the selected age plus the exact startup tie, not a statistical
exclusion, and the tie relies on the command queue starting at zero.
Non-finite scores or an empty set latch an abort with reason
`invalid_age_set`.

Unchanged: the cumulative bank and its five estimators, the frozen
estimated-model coefficients, the affine prefix integration of the sent FIFO,
barrier gains, reserves, the SLSQP projection and its independent checks, the
latched abort and the braking fallback through the unchanged FIFO, and every
task, interface and measurement definition. The filter receives no plant
object, true parameter, applied age, delivered command or task phase. Because
`L_k` is a subset of `{0..4}`, the candidate equals the v2.6 filter until the
v2.6 filter's first projected or rejected step; slow cells, where v2.6 never
projected, carry no evidence for the candidate.

## Arms, grid and reproduction

Three arms on the same paired tasks: `cumulative` (cumulative alignment),
`robust_filter` (the v2.6 filter over ages 0..4) and `selected_filter` (the
filter over `L_k`). Twelve v2.6 conditions: ordinary 2 s or fast 0.25 s moves,
clean or noisy measurements, constant delays 0, 20 and 40 ms. Twenty fresh
seeds `9806200..9806219` from the reserved block `9806200..9806299`; 720
episodes and 1,080,000 physical substeps.

Before the new seeds are opened, the v2.8 entry point reproduces all 480 v2.6
joint-filter records with their original arm names, under the existing
record comparison (relative tolerance `1e-9`, identical fields). The bank,
plant and interface files are hash-checked against the v2.5 reference, which
v2.6 reproduced in full. Any difference stops the run.

```bash
python -m tools.run_panda_selected_filter --workers 4 --reproduce-only
python -m tools.run_panda_selected_filter --workers 4 --resume
python -m tools.audit_panda_selected_filter
```

Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1, MuJoCo 3.13.0, one numerical thread
per worker; the shared cell executor allows at most four workers.

## Decisions fixed before computation

Predictions, counted on the fast cells with twenty paired tasks each:

- A, 20 ms: in each of the two 20 ms fast cells, `selected_filter` aborts are
  strictly fewer than `robust_filter` aborts.
- B, 40 ms: in each of the two 40 ms fast cells, `selected_filter` aborts are at
  least the `robust_filter` aborts minus 2, and at least one 40 ms fast
  `selected_filter` episode is physically unsafe.
- C, 0 ms risk: materialises if any 0 ms fast `selected_filter` episode is
  physically unsafe, since from step 1 the set is `{0}` and no pending check
  remains.

The 0 ms abort reduction is structural and is reported, not counted. If A and
B hold, the 20 ms aborts are reduced by narrowing the age set and the 40 ms
failure is not; its cause is left to a later study. Other combinations are
reported with the same labels.

Answer to the research question, `selected_filter` against `robust_filter`:
`aborts_reduced` is prediction A; `no_new_violations` holds if no paired task
in any cell is unsafe under `selected_filter` and safe under `robust_filter`;
`no_success_loss` holds if no paired task succeeds under `robust_filter` and
fails under `selected_filter`. Reverse counts are reported with them.

Advancement keeps the four [v2.6 criteria](panda-joint-filter.md) for
`selected_filter` against `cumulative`; prediction B expects the zero-unsafe
criterion to fail. Zero unsafe episodes in 240 would bound the per-episode rate
below about 1.25% under this task distribution only. A confirmation would need
its own frozen protocol on a new block above `9806299`.

Also reported: abort reasons and steps, the age set at every abort, paired
differences per cell, and whether the set ever excluded the true age, read
after the run and never used as an input. These are engineering results on
twenty paired tasks per cell, not a safety certificate. No gain, reserve, task
speed or fallback is tuned on these seeds.

## Development exposure

All on used seeds: the v2.6 fast-move diagnoses, the 20 bank traces of the
v2.5 reference validation, the delay scores of the 24 v2.6 filter traces and
the selection histories of all 480 v2.6 records. In every v2.6 record the
selection was `min(k + 1, true_age)` at step `k`; in the 24 traces the true age
was always in the tied set and the set became the true age alone at step
`a + 1` for a true age `a < 4` and at step 4 for a true age of 4. No candidate
episode was run before the freeze.

## Scope

Constant on-grid delays with a zero-initialised queue. Changing latency,
off-grid delays, collision avoidance and hardware are outside. The estimated
model is not a physical certificate, and the in-flight commands and the
fallback are untouched by design.
