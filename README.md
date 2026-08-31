# SARRL

**Safe Adaptive Residual Reinforcement Learning for Robotic Manipulation**

SARRL is a research-oriented robotics/control project that combines an analytical model-based controller with reinforcement learning that only learns the residual correction. The long-term target is a context-adaptive, safety-filtered controller for manipulation under payload, friction, delay and actuator uncertainty.

> **Current status — v0.1:** the analytical 2-DoF core, computed-torque baseline, from-scratch Soft Actor-Critic and residual-control environment are implemented and tested. No learned-performance claim is made yet; training results will be added only after controlled multi-seed experiments.

## Why residual RL?

Instead of asking a policy to rediscover rigid-body control from scratch,

```text
tau = pi(s),
```

SARRL starts from a physics controller and learns only its correction:

```text
tau = tau_nominal + Delta_tau_RL.
```

The hypothesis is that this improves sample efficiency and robustness when the nominal model is useful but imperfect.

## Architecture

```text
Cartesian target
      |
      v
inverse kinematics
      |
      v
computed-torque baseline ---- tau_nom
      |                           |
state + target                    +----+
      |                                |
      v                                v
from-scratch SAC policy ---- residual torque
                     \          /
                      v        v
                    candidate torque
                         |
                         v
                  actuator limits
                         |
                         v
                  analytical plant
```

Later releases add MPC, online dynamics context, CBF/QP projection, MuJoCo/Franka and learned residual dynamics.

## Implemented in v0.1

- analytical two-link `M(q)`, `C(q,qd)`, `g(q)` and friction;
- forward and inverse dynamics;
- RK4 integration;
- forward/inverse kinematics and analytical Jacobian;
- computed-torque control with angle wrapping;
- direct-torque and residual-torque reaching environments;
- controlled mass/friction/motor-gain randomization hooks;
- Soft Actor-Critic implemented directly in PyTorch:
  - tanh-squashed Gaussian actor,
  - exact log-Jacobian correction,
  - twin critics,
  - target critics,
  - Polyak averaging,
  - replay buffer,
  - automatic entropy-temperature tuning;
- deterministic seeding and complete SAC checkpoints;
- **28 automated tests** covering numerical, controller, environment, replay and RL invariants;
- GitHub Actions CI.

## Quick start

```bash
python -m pip install -e .
pytest
```

Short training smoke run:

```bash
python tools/train_sac.py --mode residual --steps 20000 --seed 0 --output results/smoke
```

A real experiment should use a materially larger budget and multiple seeds; the smoke command is only for integration testing.

Evaluate a checkpoint deterministically:

```bash
python tools/evaluate.py results/smoke/final.pt --mode residual --episodes 100
```

## Validation philosophy

The plant is tested using properties that are difficult for the same bug to satisfy accidentally:

- `M(q)` symmetry and positive definiteness;
- finite-difference check that `Mdot - 2C` is skew-symmetric;
- inverse-dynamics / forward-dynamics round trip;
- analytical Jacobian vs finite differences;
- RK4 energy drift in a conservative configuration;
- computed-torque convergence on a fixed target;
- deterministic seeded environment trajectories;
- SAC terminal Bellman target, target-network update and checkpoint round trip;
- squashed-policy log probability consistency near action bounds.

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| M0 | 2-DoF analytical dynamics | implemented |
| M1 | computed-torque baseline | implemented |
| M2 | constrained MPC | next |
| M3 | SAC from scratch | implemented |
| M4 | direct-torque RL baseline | implemented, experiment pending |
| M5 | residual SAC | implemented, experiment pending |
| M6 | domain randomization + OOD study | planned |
| M7 | GRU online dynamics context | planned |
| M8 | fault adaptation | planned |
| M9 | hard CBF/QP safety projection | planned |
| M10 | MuJoCo + Franka Panda | planned |
| M11 | learned residual dynamics | planned |
| M12 | uncertainty ensemble | planned |

See `docs/design.md` and `docs/mathematics.md` for the experimental rules and equations.
