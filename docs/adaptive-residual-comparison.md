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
training seed. Use Python 3.11.15, NumPy 2.4.6, MuJoCo 3.14.0, SciPy 1.17.1
and PyTorch 2.12.0+cpu in a separate environment. The CPU-only installation
is fixed before training and avoids loading unused accelerator libraries.
Historical reproduction compares every floating
field with `rtol=1e-7`, `atol=1e-9`; discrete values and their types must match
exactly. It also retains the stricter `rtol=1e-9`, `atol=1e-10` comparison as a
separate diagnostic. No identical-host or bitwise reproduction claim is made.
The same declared libraries run the new planar training and evaluation.

### Historical portability amendment, 15 October 2026

Before opening any new training or evaluation seed, the historical preflight
found an incorrect provenance label and one numerically sensitive episode.
The label is restored to the retained `official` value. With that metadata
corrected, 299 of 300 records meet the original floating tolerance. The five
remaining differences occur in `id_reference`, seed `56212`: final distance
differs by 1.96e-10 m, maximum speed by 2.10e-8 rad/s, and estimated parameters,
mean margin and prediction RMS by at most 2.37e-8 in their respective units.
Python 3.11.15 and alternative BLAS/SIMD checks do not recover the original
tolerance on this host. These trials do not identify a single underlying cause.

The revised historical bound is fixed before fresh seeds. All 300 recalculated
records are published in `reproduction_records.json`, alongside both comparisons,
so the original strict failure remains visible and independently auditable.
Success, unsafe flags, counters, seed identities and metadata retain exact
comparison. Nonfinite values fail. This amendment changes only historical
floating-value comparison; the training budget, held-out grid, confidence
intervals and every advancement criterion below remain unchanged.

### Fixed training budget

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

Create a separate environment using Python 3.11.15:

```bash
python3.11 -m venv ../sarrl-residual-env
source ../sarrl-residual-env/bin/activate
python -m pip install 'torch==2.12.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[mujoco]' 'numpy==2.4.6' 'scipy==1.17.1' 'mujoco==3.14.0'
```

From the frozen checkout with that runtime:

```bash
python -m tools.run_planar_residual --workers 2 --reproduce-only
python -m tools.run_planar_residual --workers 2 --resume --train-only
python -m tools.run_planar_residual --workers 2 --resume
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
bootstrap/decision. It recomputes both historical comparisons from the public
reproduction records. It explicitly reports that policy inference was not
verified. The full local audit also checks all five checkpoint hashes,
policy actions and causal nominal/filter/estimator replays. Reproducing
inference elsewhere requires the recorded local policies or retraining the
frozen recipe; public statistics auditing alone does not prove inference.
