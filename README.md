# SARRL

[![CI](https://github.com/andrealo20/SARRL/actions/workflows/ci.yml/badge.svg)](https://github.com/andrealo20/SARRL/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![version](https://img.shields.io/badge/version-1.3.0-6f42c1.svg)](docs/changelog.md)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

![SARRL banner](assets/banner.png)

**SARRL** (Safe Adaptive Residual Reinforcement Learning) is a research stack for robotic control under model mismatch. It combines model-based control with bounded residual reinforcement learning, learned dynamics context, uncertainty-aware policy authority and HOCBF safety filtering.

The current reference platform is a reproducible analytical **2-DoF planar arm**. MuJoCo, Franka Panda, hardware and sim-to-real results are not currently claimed.

## Results

On the randomized planar held-out benchmark:

| Controller | v1.2 held-out | v1.3 compound OOD | v1.3 motor fault |
|---|---:|---:|---:|
| Computed torque | 11.0% | 0.0% | 3.0% |
| Direct SAC | 6.0% ± 3.7 pp | — | — |
| Residual SAC | 56.4% ± 7.0 pp | 6.0% ± 2.3 pp | 23.6% ± 1.9 pp |
| Residual SAC + learned context | **64.2% ± 6.7 pp** | **11.6% ± 3.8 pp** | **32.6% ± 6.4 pp** |
| Residual SAC + uncertainty gate | 15.2% ± 1.6 pp | 0.0% | 3.2% ± 0.8 pp |
| Residual SAC + HOCBF | 49.2% ± 7.9 pp | 3.0% ± 0.7 pp | 16.4% ± 1.1 pp |
| Full adaptive stack | 17.0% ± 2.3 pp | 0.0% | 2.8% ± 0.4 pp |

The v1.2 campaign established the benefit of residual learning over Direct SAC. The v1.3 campaign reused the frozen policies on new paired seeds: every learned controller degraded sharply under compound OOD dynamics and abrupt motor loss. Learned context retained the highest success, but did not solve robustness. The gate stayed near minimum authority, and hard-HOCBF stacks explicitly rejected 86/3,000 episodes when projection became infeasible.

Learned-policy values are means ± sample SD across five independently trained policies, with 100 episodes per policy and scenario. Evidence remains limited to the analytical planar benchmark.

See [`docs/experiments.md`](docs/experiments.md) and [`docs/verification.md`](docs/verification.md) for the protocol, retained evidence and provenance.

## How it works

SARRL starts from a physics-based controller and lets SAC learn only a bounded correction:

```math
\tau_{candidate}=\tau_{nominal}+\tau_{residual}
```

This narrows the learning problem to compensating for model mismatch. The repository also includes domain randomization, fault injection, a causal GRU context encoder, residual-dynamics ensembles, uncertainty gating and hard HOCBF projection. Components are modular and independently testable.

## Quick start

```bash
git clone https://github.com/andrealo20/SARRL.git
cd SARRL
python -m pip install -e '.[dev]'
pytest -q
```

The planar stack requires NumPy, SciPy and PyTorch; MuJoCo and Gymnasium are not required. Training and evaluation commands are documented in [`docs/experiments.md`](docs/experiments.md).

## Repository guide

- `sarrl/` — dynamics, controllers, RL, adaptation and safety
- `tools/` — training, evaluation and sweep commands
- `tests/` — automated verification
- `artifacts/` and `results/` — retained experimental evidence
- `docs/` — [design](docs/design.md), [mathematics](docs/mathematics.md), [experiments](docs/experiments.md), [verification](docs/verification.md) and [changelog](docs/changelog.md)

## Limitations

The v1.3 OOD/fault campaign is complete, but quantified-safety, MuJoCo, Franka, hardware and sim-to-real campaigns remain future work. HOCBF guarantees are model-relative, ensemble disagreement is not calibrated probability, and the uncertainty gate is not a formal safety certificate.

## License and citation

Released under the [MIT License](LICENSE) by [Andrea Loroni](https://github.com/andrealo20).

```bibtex
@software{loroni_sarrl_2026,
  author  = {Andrea Loroni},
  title   = {SARRL: Safe Adaptive Residual Reinforcement Learning for Robotic Manipulation},
  year    = {2026},
  version = {1.3.0},
  url     = {https://github.com/andrealo20/SARRL}
}
```
