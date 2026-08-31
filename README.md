# SARRL

[![CI](https://github.com/andrealo20/SARRL/actions/workflows/ci.yml/badge.svg)](https://github.com/andrealo20/SARRL/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![tests](https://img.shields.io/badge/tests-71%20passing-brightgreen.svg)](tests/)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Safe Adaptive Residual Reinforcement Learning for Robotic Manipulation**

SARRL is a research-oriented robotics and reinforcement-learning stack for control under model mismatch. It combines a physics controller, from-scratch Soft Actor-Critic, causal dynamics-context estimation, learned residual dynamics with epistemic uncertainty, controlled fault injection and a hard high-order Control Barrier Function safety projection.

The current **v1.0 planar release** is deliberately self-contained: its reference plant is an analytical 2-DoF arm, so the mathematics, learning code and safety layer can be tested without MuJoCo, Gymnasium or a black-box dynamics engine. Franka/MuJoCo transfer remains the next major milestone rather than an unverified feature claim.

> **Status:** the complete planar stack is implemented and the current automated suite passes **71/71 tests**. Measured non-learned baselines are retained in `results/`. No multi-seed learned-policy success claim is made yet; the repository includes the validation/held-out protocol and sweep runner required to produce one reproducibly.

## Core idea

Direct RL asks a policy to rediscover the whole controller:

```text
tau = pi(s)
```

SARRL instead starts from a competent physics baseline and learns only bounded correction:

```text
tau_candidate = tau_nominal + residual_limit * a_RL
```

The full runtime can then reduce learned authority when model uncertainty is high and project the command onto hard safety constraints:

```text
                    target
                      |
                      v
               inverse kinematics
                      |
                      v
state ----------> physics controller ----------> tau_nom
  |                                             |
  |                                             +----------+
  |                                                        |
  +--> causal history --> GRU context --> policy --> residual torque
  |                                                        |
  +--> residual-dynamics ensemble --> epistemic uncertainty |
                                                           v
                                                uncertainty gate
                                                           |
                                                           v
                                                   candidate torque
                                                           |
                                                           v
                                                hard HOCBF projection
                                                           |
                                                           v
                                      delay / motor gain / fault / plant
                                                           |
                                                           +----> next state
```

The pieces are intentionally separable. Every layer can be tested or ablated without requiring the others.

## What is implemented

### Analytical robotics core

- exact 2-link rigid-body `M(q)`, `C(q, qd)` and `g(q)`;
- viscous + smoothed Coulomb friction;
- endpoint payload mass;
- forward and inverse dynamics;
- RK4 state integration;
- forward/inverse kinematics;
- analytical end-effector Jacobian and `Jdot(q,qd) qd`;
- computed-torque feedback linearisation;
- nonlinear constrained MPC by direct shooting and SLSQP.

### Reinforcement learning

Soft Actor-Critic is implemented directly in PyTorch, without SB3 or RLlib:

- tanh-squashed reparameterised Gaussian actor;
- stable exact change-of-variables log probability;
- twin critics and target critics;
- Polyak target updates;
- replay buffer with reproducible RNG;
- automatic entropy-temperature tuning;
- direct-torque and residual-torque environments;
- deterministic action path that does **not** consume stochastic RNG;
- architecture-safe checkpoints;
- exact off-policy session reconstruction, including replay and environment state.

### Robustness and adaptation

The analytical environment supports controlled variation of:

- link mass/inertia;
- friction;
- payload;
- motor gain;
- sensor noise;
- actuator-command delay;
- abrupt in-episode actuator/payload faults.

A causal GRU context encoder consumes only transition history:

```text
(obs_t, action_t, obs_{t+1} - obs_t)
```

and produces a latent context for adaptive policies. Ground-truth physical parameters are optional auxiliary labels for training/diagnostics, not runtime policy inputs.

### Learned residual dynamics and uncertainty

An ensemble learns the acceleration error relative to the nominal rigid-body model:

```text
qdd_real = qdd_nominal(state, commanded_tau) + Delta_qdd_learned
```

Each member receives the **commanded** torque, so motor-gain degradation remains visible in the residual target. Bootstrap training makes ensemble disagreement usable as an epistemic-uncertainty signal. An uncertainty gate can reduce residual-policy authority when the models disagree.

This gate is explicitly a robustness heuristic, **not a safety certificate**.

### Hard safety projection

The planar safety layer builds affine torque constraints from:

- joint-position HO-CBFs;
- one-step joint-velocity bounds;
- circular Cartesian obstacle HO-CBFs;
- actuator torque limits.

The command is projected onto the resulting 2-D polytope by exhaustive active-set enumeration. There is no hidden slack variable: if the hard constraint set is infeasible, the filter reports failure explicitly. `require_safety=True` therefore has no uncertified fallback command.

## Verification

The current local release passes:

```text
71 passed
```

The suite includes numerical identities, regression tests and deliberate sabotage tests. Important checks include:

- `M(q)` symmetry and positive definiteness;
- finite-difference verification that `Mdot - 2C` is skew-symmetric;
- forward/inverse dynamics round trip;
- analytical Jacobian vs finite differences;
- RK4 energy behaviour in the conservative plant;
- computed-torque closed-loop convergence;
- MPC feasibility, constraints, warm-start and objective improvement;
- deterministic seeded randomisation, sensor noise, delay and faults;
- SAC Bellman targets, target updates, tanh log-probability and entropy tuning;
- deterministic SAC inference does not advance PyTorch RNG;
- checkpoint reconstruction with non-default neural architecture;
- exact replay/environment/RNG training continuation;
- context causality and checkpoint round trip;
- exact 2-D safety projection and hard-constraint behaviour;
- residual-dynamics training/checkpoint/uncertainty behaviour;
- motor-gain mismatch remains visible in learned residual targets;
- runtime-stack execution and safety-failure semantics;
- Wilson intervals, paired bootstrap and validation/test seed separation.

See [`docs/verification.md`](docs/verification.md) for checks actually executed and retained empirical evidence.

## Retained baseline results

These results are committed as raw CSV/JSON artifacts; they are **not learned-policy results**.

| Scenario | Success | 95% Wilson interval | Purpose |
|---|---:|---:|---|
| nominal computed torque | 100/100 | 96.3–100.0% | plant/controller sanity baseline |
| in-distribution dynamics randomisation | 8/100 | 4.1–15.0% | exposes model-mismatch gap |
| OOD dynamics | 0/100 | 0.0–3.7% | stress baseline |
| joint-2 motor fault | 1/100 | 0.2–5.4% | fault-recovery baseline |

Raw data:

```text
results/v0_1_nominal.csv
results/v0_9_baselines.csv
results/v0_9_baselines.json
```

The large drop from nominal control under mismatch is the experimental motivation for residual learning, context adaptation and uncertainty-aware control.

## Install

```bash
python -m pip install -e .
pytest -q
```

Development tools:

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest -q
```

The planar core requires only NumPy, SciPy and PyTorch.

## Quick experiments

### Reproduce the nominal baseline

```bash
python tools/evaluate_nominal.py \
  --episodes 100 \
  --seed 1000 \
  --output results/nominal.csv
```

### Train residual SAC under model randomisation

```bash
python tools/train_sac.py \
  --mode residual \
  --randomize \
  --steps 200000 \
  --seed 0 \
  --output results/residual_seed0
```

Training uses a separate fixed validation set for `best.pt`. A full off-policy checkpoint is also saved so training can resume without reconstructing hidden state from CLI defaults.

```bash
python tools/train_sac.py \
  --steps 400000 \
  --resume results/residual_seed0/training_final.pt \
  --output results/residual_seed0
```

### Held-out evaluation

```bash
python tools/evaluate.py \
  results/residual_seed0/best.pt \
  --mode residual \
  --episodes 100 \
  --seed 40000
```

### Five-seed campaign

```bash
python tools/run_sac_sweep.py \
  --seeds 0 1 2 3 4 \
  --mode residual \
  --randomize \
  --steps 200000 \
  --validation-seed 20000 \
  --heldout-seed 40000 \
  --heldout-episodes 100 \
  --output results/residual_sweep
```

The runner refuses overlapping validation and held-out seed ranges and writes:

```text
sweep_manifest.json
summary.csv
heldout_episodes.csv
aggregate.json
seed_0/ ... seed_4/
```

### Train the context encoder

```bash
python tools/train_context.py \
  --samples 2000 \
  --history 16 \
  --steps 1500 \
  --output results/context/context.pt
```

### Train residual dynamics ensemble

```bash
python tools/train_residual_dynamics.py \
  --samples 10000 \
  --steps 2000 \
  --output results/residual_dynamics/ensemble.pt
```

### Evaluate the composed runtime stack

```bash
python tools/evaluate_stack.py \
  results/residual_seed0/best.pt \
  --episodes 100 \
  --randomize \
  --safety
```

## Experimental discipline

SARRL separates three seed populations:

```text
training seeds     -> independent policy optimisation runs
validation seeds   -> checkpoint selection only
held-out seeds     -> final reported evaluation only
```

The held-out set never selects checkpoints. Final learned comparisons should use at least five independent training seeds and report variation across **models**, not only binomial uncertainty across episodes from one model.

Every training run writes `run_manifest.json` with the Git commit, Python/library versions and actual agent/environment/trainer configuration. Multi-seed campaigns write an additional sweep manifest and raw held-out episode records.

## Milestones

| Milestone | Scope | Status |
|---|---|---|
| M0 | analytical 2-DoF rigid-body dynamics | implemented + tested |
| M1 | computed-torque baseline | implemented + measured |
| M2 | constrained nonlinear MPC | implemented + tested |
| M3 | SAC from scratch | implemented + tested |
| M4 | direct-torque RL path | implemented; full campaign pending |
| M5 | residual SAC path | implemented; full campaign pending |
| M6 | domain randomisation + OOD protocol | implemented + baseline measured |
| M7 | causal GRU dynamics context | implemented + tested |
| M8 | actuator/payload fault injection | implemented + baseline measured |
| M9 | hard HOCBF safety projection | implemented + tested |
| **M10** | **MuJoCo + Franka Panda transfer** | **planned / not claimed** |
| M11 | learned residual dynamics | implemented + tested |
| M12 | ensemble epistemic uncertainty | implemented + tested |

## Repository layout

```text
sarrl/
  adaptation/     causal GRU context estimation
  controllers/    computed torque and nonlinear MPC
  dynamics/       analytical planar-arm model
  envs/           randomized/faulted reaching environment
  evaluation/     statistics, provenance and fixed-seed protocols
  models/         residual dynamics ensemble and uncertainty gate
  rl/             SAC, networks, replay and full training checkpoints
  runtime/        composed controller stack
  safety/         hard HOCBF projection
  utils/          seeding and lightweight Box space

tools/
  train_sac.py
  run_sac_sweep.py
  evaluate.py
  evaluate_stack.py
  train_context.py
  train_residual_dynamics.py
  run_planar_baselines.py

tests/            71 automated tests
results/          retained raw evidence only
docs/             design, mathematics, verification, experiments, changelog
```

## Current limitations

- The reference release is planar 2-DoF, not yet a 7-DoF Franka manipulation benchmark.
- No hardware or sim-to-real claim is made.
- No learned-policy headline metric is reported until a completed multi-seed campaign is retained in the repository.
- The SciPy/SLSQP MPC is a reference constrained controller, not a hard real-time solver.
- The HOCBF guarantee is model-relative: uncertainty in the dynamics model remains relevant even when the mathematical projection is exact.
- The uncertainty gate is heuristic and must not be interpreted as a safety guarantee.

These limitations are intentional boundaries of the current evidence, not hidden assumptions.

## Documentation

- [`docs/design.md`](docs/design.md) — architecture and design decisions
- [`docs/mathematics.md`](docs/mathematics.md) — equations and control formulation
- [`docs/experiments.md`](docs/experiments.md) — training/evaluation protocol
- [`docs/verification.md`](docs/verification.md) — executed checks and retained results
- [`docs/changelog.md`](docs/changelog.md) — release history

## License

MIT.
