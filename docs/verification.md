# v0.1 verification record

This file records only checks actually executed for v0.1.

## Automated tests

`pytest -q` on the analytical/PyTorch core:

```text
28 passed
```

Coverage includes dynamics invariants, kinematics, controller convergence, environment contracts, reproducible replay sampling, SAC Bellman targets, target-network updates, squashed-policy log probabilities, checkpoint round trips and two deliberate sabotage tests.

## Nominal controller baseline

Command:

```bash
python tools/evaluate_nominal.py --episodes 100 --seed 1000 \
  --output results/v0_1_nominal.csv
```

Measured on seeds 1000 through 1099:

```text
success: 100/100 = 100.0%
mean steps: 50.68
mean terminal distance: 0.04575 m
```

The exact per-episode target, success flag, steps, reward and terminal distance are retained in `results/v0_1_nominal.csv`.

## SAC integration smoke test

A 300-step residual-SAC run was executed to verify that replay sampling, gradient updates, entropy-temperature updates, checkpoint save/load and deterministic evaluation complete without runtime errors.

It achieved 0/5 success in a five-episode evaluation. This is intentionally **not** a performance result: 300 environment steps are far below a meaningful SAC training budget. No convergence claim is made in v0.1.

## Tooling limitation

The execution environment used for this verification did not provide MuJoCo or Gymnasium, which is consistent with the v0.1 design: the analytical core has no dependency on either. MuJoCo becomes an optional integration dependency at M10.

## v0.9 robustness baselines

`tools/run_planar_baselines.py` was executed on seeds 1000--1099 for four
non-learned computed-torque scenarios. Raw episode records are retained in
`results/v0_9_baselines.csv`; aggregate Wilson intervals and metadata are in
`results/v0_9_baselines.json`.

| scenario | success | 95% Wilson interval |
|---|---:|---:|
| nominal | 100/100 | 96.3--100.0% |
| ID randomization | 8/100 | 4.1--15.0% |
| OOD dynamics | 0/100 | 0.0--3.7% |
| joint-2 motor fault | 1/100 | 0.2--5.4% |

These are deliberately baseline results, not learned-policy claims. The large
nominal-to-mismatch gap is the experimental motivation for residual learning,
context adaptation and uncertainty-aware control.
