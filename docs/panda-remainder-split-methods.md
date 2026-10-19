# Origin of the shared-parameter remainders

The [shared-parameter screen](panda-mean-value-results.md) verifies 281 of 480
intervals and completes none of its eighty branches. At the last accepted
endpoints the independent velocity residual is 8 to 50 times wider than the
shared generator radius. This diagnostic asks where that residual comes from.
It reads the retained report and changes no controller, enclosure, gain,
physical box, budget, simulator or task protocol. It uses no physical steps
and no seeds.

## Bookkeeping

The tool reloads `results/panda_mean_value_prefix/assessment.json.gz` (SHA256
`18746bc56a77f70bde364ec068f866b6c9316f57a09ea0df24c52ec3478c19d6`). For every
accepted interval it rebuilds the recorded start state, delivered torque and
parameter coordinates, and calls the unchanged acceleration and feedback maps
of the freeze `f21fa2cd59d533b1b335c168fb366644e8bc7c67`. Both outputs must
equal the recorded acceleration and sent torque exactly. This shows that the
inputs are the ones used by the screen and yields the Jacobian enclosure
`J(D)` of each map.

Each mean-value step adds two residual terms:

```text
J(D) r               propagation of residuals already present
(J(D) - J0) G eps    Jacobian variation over the hull
```

and each held-flow endpoint adds the Taylor terms `dt^2 J_jerk / 2` to
velocity and `dt^3 J_jerk / 6` to position. Residuals are symmetric, so their
radii combine linearly: the radius of `J(D) r` is `|J(D)| rho` and that of the
variation term is `rad(J(D))` times the generator radius. Radii are carried
through the FIFO and the endpoint formula separately for four origins:

| Origin | Injected by |
|---|---|
| `jacobian_acceleration` | variation term of the acceleration map |
| `jacobian_feedback` | variation term of the feedback map |
| `taylor` | held-flow Taylor remainder |
| `rounding` | the recorded radius minus the reconstructed one, at every step |

Propagation is not an origin: it carries and amplifies the four origins. The
per-step split between inherited and newly injected radius is reported too.

The attribution is conditional on the realized enclosures `J(D)`. A wider
residual widens the hull and therefore `J(D)` itself; that coupling is not
attributed to any origin. The result describes the retained enclosures and
is not a new bound.

## Validity check

The decomposition is valid only if both recomputed maps equal the record
exactly at every interval and every injected rounding radius is at least
`-1e-9` times the recorded radius. If either fails, no decision is drawn
and only bookkeeping errors may be corrected.

## Decision fixed before computation

Primary metric, per branch: the share of the velocity residual radius,
summed over the seven joints at the last accepted endpoint, whose origin is
`taylor`.

- If the median share over the eighty branches is at least 0.5, the held-flow
  remainder is the bottleneck. One final phase on that remainder (finer
  substeps or a tighter jerk bound) is justified, with its own freeze.
- Otherwise the shared-parameter prefix line stops. The current chain of
  numerical changes is prepared as a release of methods and negative results.

Neither outcome opens task seeds or a task comparison. The threshold does not
change after the result. Reported without decision weight: the same shares
for position, the split between the acceleration and feedback maps, the
rounding share, inherited against injected radius per step and the growth
of the total radius along each branch.

## Command

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m tools.split_panda_mean_value_remainder \
  --output results/panda_remainder_split/split.json.gz
```

Runtime: the pinned Panda environment, Python 3.12.3, NumPy 2.5.2, SciPy
1.18.1, MuJoCo 3.13.0 and python-flint 0.9.0.
