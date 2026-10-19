# Origin of the shared-parameter remainders: result

The [split](panda-remainder-split-methods.md) ran once from the clean commit
`cc2351fe`, in 34 seconds on the pinned Panda runtime. Output:
`results/panda_remainder_split/split.json.gz`, SHA256
`a5af0cccb2dc0a97fc3ca6b5b1175f3283f520a3fd6c15a0291edf0ec3e4e4c7`. No physical
step, seed or task was used.

## Checks

All 281 recomputed accelerations and all 201 recomputed feedback sends equal
the retained report exactly. Every delivered torque matches the send queued
`lag` intervals earlier, and no injected rounding radius falls below the
tolerance. The split reproduces the ratio between the last velocity residual
and the generator radius stated for the screen: 8.27 to 50.04, median 13.48.

## Shares at the last accepted endpoint

Median over the eighty branches, share of the summed residual radius:

| Origin | Velocity | Position |
|---|---:|---:|
| Taylor remainder | 0.656 | 0.636 |
| Jacobian variation, acceleration map | 0.344 | 0.364 |
| Jacobian variation, feedback map | 1.3e-5 | 7.3e-6 |
| Rounding | 4.3e-8 | 5.1e-8 |

The Taylor share of velocity lies between 0.555 and 0.805 in every branch.
Its median is 0.704 at lag zero and 0.639 to 0.665 at lags one to four.

## Decision

The frozen metric is 0.656, above the 0.5 threshold. The held-flow Taylor
remainder is the main source of the residual that stops every branch, and one
final phase on that remainder is justified under its own freeze.

## Reported without decision weight

From the second interval onwards, 39% to 66% of each new velocity radius
(median 57%) is inherited from residuals present at the start of the interval;
the rest is injected in that interval. The summed velocity radius grows by a
factor of 6.2 to 63 (median 28.6) from the first to the last accepted interval.

The velocity Taylor term `dt^2 J_jerk / 2` is also compared with the bound
`dt` times the width of the acceleration tube of the same interval, which the
Picard enclosure already implies. Over all joints of all accepted intervals
the ratio has median 0.36, and the tube bound would be tighter in only 8.8% of
them. A tighter jerk bound alone therefore has little room; the remainder
mainly reflects the length of the held interval. This comparison is computed
from the retained report and is descriptive.

## What this does not show

The attribution is conditional on the retained Jacobian enclosures. A smaller
Taylor remainder narrows the hulls and therefore those enclosures too, so the
gain of a new design cannot be read off these shares. The split does not show
that any branch will complete six intervals, and it changes none of the open
terminal, sensor, runtime or simulator conditions.
