# Planar residual-learning comparison

This second v2.6 study follows evaluation of the Panda filter decision. It
tests whether a bounded learned residual adds value to the identified planar
nominal with the covariance margin retained from v2.2. The Panda candidate
failed its screen and remains experimental; it is not used in this study.

## Controlled stack

The descriptive fixed arm uses the unchanged fixed nominal and nominal HOCBF.
The adaptive reference uses the original identified nominal, causal delay and
actuator-time-constant bank, delay compensation and covariance-margin HOCBF
with `kappa=0.25`. The treatment adds a residual policy to that same adaptive
reference before projection. Primary comparisons are treatment minus adaptive;
the fixed arm supplies context and does not determine advancement.

Each action is clipped to `[-1,1]` and contributes at most 8 N m per joint.
The policy observes the existing eight-dimensional measured-state/target/error
observation, with no parameter, applied-delay, fault or estimator-state input.
The estimator observes the actual projected sent command. A zero residual
must match the retained adaptive/covariance stack. The new training wrapper
preserves the existing plant, estimator, reward, cached measurement interface
and hard-projection behavior; only the certificate gains the existing margin.

The plant uses the v2.2 options: MuJoCo, measured noise `1e-3`, armature range
`0.02..0.08`, actuator time constants `0.01..0.05` s and estimator grid
`{0,0.01,0.03,0.06}` s. Domain randomization and ID/OOD/motor-fault scenarios
remain the original v1.3 definitions. Each episode has at most 250 control
steps. Evaluation success, abort and physical-envelope definitions remain
those of the retained planar evaluator. Its physical envelope is observed at
control boundaries; this study adds no claim about unobserved substep extrema.
The covariance margin is empirical and is not calibrated population uncertainty.

## Reproduction and fixed training

Commit implementation, tests, this protocol and the reserved registry before
execution. Reproduce the first hundred v2.2 covariance-margin decision seeds
`56200..56299` across all three scenarios, 300 records, before opening a new
training seed. Use historical NumPy 2.4.6 and MuJoCo 3.14.0 in a separate
Python 3.12 environment, with PyTorch 2.12.0+cu130 on CPU. The original records
used Python 3.11.15; reproduction compares every retained field with the
declared `rtol=1e-9`, `atol=1e-10` rather than claiming an identical host.
The same declared libraries run the new planar training and evaluation.

Five independent training seeds are `9806000..9806004`, inside the reserved
`9806000..9806099` block. Every model receives 200,000 environment calls,
including abort transitions; executed physical-control calls and abort calls
are recorded separately. The first 5,000 actions are random. SAC uses its
unchanged defaults, two hidden layers of 256 units, batch size 256, replay
capacity 200,000 and one update per subsequent call. Each completed policy
therefore has 195,000 optimizer updates. Train only on randomized ID tasks.

Only the final policy is evaluated. There is no validation-based checkpoint
selection, early stopping, hyperparameter sweep or exclusion of a training
seed. All five models must complete and pass finite-parameter checks before
the held-out block is opened. One process/numerical thread per model and at
most five workers are permitted; CUDA is disabled. An interruption resumes
the same source/runtime, policy/optimizer, replay, environment, margin counters
and every RNG from atomic checkpoints saved every 25,000 calls. Resume parity
is tested against uninterrupted training. Existing checkpoints are preserved.
The cell executor uses at most four workers for historical reproduction and
held-out evaluation; training permits five. The manifest records both phase
limits separately from the requested maximum. Worker count does not change
the seed/model budget or pairing.

## Held-out grid and uncertainty unit

One hundred untouched task seeds, `9806100..9806199`, cross ID, compound OOD
and motor fault. Seven arms, two baselines and five trained policies, yield
2,100 records. All arms share task and parameter draws and the cached sensor
stream per task/scenario. None of these seeds is used for training or selection.
Retain every record, model hash, first-seed action/state trace and decision.

Paired contrasts average all five policies equally. The 95% percentile
bootstrap independently resamples the five training-model indices and one
hundred complete task-seed blocks, preserving all three scenarios within
each task. The same sampled indices apply to success, unsafe and abort
endpoints. Use 5,000 draws and generator seed `260000`, also registered for
statistical unit tests. The dependence from shared tasks and baselines is
retained; 1,500 learned-policy episodes are not treated as independent trials.
Five training models limit the precision of the model-variation estimate.

All criteria must pass to advance toward Panda integration:

1. Across ID and motor fault with equal scenario weights, mean success gain
   is at least three percentage points and its 95% lower bound is above zero.
2. Every scenario has success-difference lower bound at least minus three
   percentage points and unsafe-difference upper bound at most plus one point.
3. Every scenario has mean abort-fraction increase at most three points.
4. At least four of five policies have nonnegative ID/fault mean success gains.

All outcomes, intervals and failures are reported. A failed screen retains
the adaptive reference and prevents advancing this residual candidate. No
criterion, budget, model choice or action bound changes after opening seeds.

## Run and audit

From the frozen checkout with the recorded runtime:

```bash
python -m tools.run_planar_residual --workers 5 --reproduce-only
python -m tools.run_planar_residual --workers 5 --resume --train-only
python -m tools.run_planar_residual --workers 5 --resume
python -m tools.audit_planar_residual \
  --models results/planar_residual_v26/training
```

The canonical directory is `results/planar_residual_v26`. Completed output
is never replaced. Source hashes include every tracked package/tool/test
Python file plus protocol, registry and reference evidence. Model/checkpoint
tensors remain local and ignored; their hashes and compact training summaries
are retained. The dedicated runner restores this wrapper's checkpoints.

The public audit without `--models` verifies source/data hashes, complete
pairing, endpoint definitions, bounded trace actions and recomputed crossed
bootstrap/decision. It explicitly reports that policy inference was not
verified. The full local audit also checks all five checkpoint hashes,
policy actions and causal nominal/filter/estimator replays. Reproducing
inference elsewhere requires the recorded local policies or retraining the
frozen recipe; public statistics auditing alone does not prove inference.
