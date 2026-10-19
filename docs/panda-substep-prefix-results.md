# Held substeps for the shared-parameter prefix: result

The [substep phase](panda-substep-prefix-methods.md) ran once from the clean
freeze `290dc05`, in 7 minutes 32 seconds on the pinned Panda runtime. Output:
`results/panda_substep_prefix/assessment.json.gz`, SHA256
`9643ea2b4d3e31bc268c6ffc82d7bb87026c68458891a445112a4552138569a3`. No physical
step, seed or task was used.

## Checks

Before any substep result existed, one substep reproduced the retained
shared-parameter report field for field in all sixteen mass cells. The
auditor then checked all 1,396 substep records, including one shared send per
held period and exact-rational Taylor endpoints with the 2.5 ms length. A
complete `--audit` replay recomputed the report and found it equal.

## Held periods verified

| Lag | One substep | Four substeps |
|---|---:|---:|
| 0 | 47 | 48 |
| 1 | 55 | 64 |
| 2 | 59 | 67 |
| 3 | 60 | 69 |
| 4 | 60 | 69 |
| Total | 281 | 317 |

Thirty-six branches gain one held period, 44 are unchanged and none
regresses. Sixteen branches reach three periods, 51 reach four and 13 reach
five. None completes six. Seventy-seven branches stop with
`whole_interval_bounds_not_established` and three with
`state_region_not_covered`, where the hull of the propagated state leaves the
domain of the coefficient enclosures. The failing substep is spread over all
four positions (17, 18, 25 and 20 branches).

## Decision

Zero of eighty branches complete the six held periods. Under the rule fixed
before the run, the shared-parameter prefix line stops, and the current chain
of numerical changes is prepared as a release of methods and negative
results.

## Reported without decision weight

At the last accepted endpoint, the largest velocity residual radius is 30 to
800 times the largest generator radius (median 55), against 8 to 50 (median
13.5) with one substep. Shorter substeps let the branches go further, but the
independent residual grows faster than the shared first-order part. This is
consistent with the Jacobian variation term, which does not shrink with the
substep length, becoming the limit once the Taylor term is reduced; the split
was not repeated on this report, so that attribution is not measured here.

## What this does not show

The result rejects this representation with four held substeps on this
fixture. It does not show that no enclosure of the normalized model exists,
and it leaves the terminal, sensor, runtime and simulator conditions as open
as before. It says nothing about the published Panda reference controller.
