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
