# Design and milestone discipline

SARRL is intentionally developed bottom-up. No result from a later layer is trusted until the lower layer has invariant tests and an independent failure mode.

## v0.1 implemented

- M0: analytical two-link rigid-body plant and RK4 integrator.
- M1: computed-torque baseline.
- M3 core: Soft Actor-Critic from scratch in PyTorch.
- M4 environment: direct torque RL baseline.
- M5 architecture: residual SAC action path on top of the nominal controller.
- deterministic seeding, replay buffer, checkpointing and CI.

The code exists for M4/M5, but v0.1 deliberately makes no convergence or success-rate claim until training experiments are run and archived.

## Planned

- M2: constrained MPC baseline.
- M6: controlled domain randomization and OOD protocol.
- M7: GRU dynamics-context encoder.
- M8: actuator/payload fault injection and adaptation-time metrics.
- M9: hard CBF/QP safety projection, with soft feasibility reported separately.
- M10: MuJoCo Franka Panda transfer.
- M11: learned residual dynamics.
- M12: ensemble epistemic uncertainty and uncertainty-aware control.

## Scientific rules

1. Held-out evaluation seeds never select checkpoints.
2. Final comparisons use at least five independent training seeds.
3. A failed training run is first treated as a debugging signal, not as evidence that an algorithm cannot solve the task.
4. No README metric is added without a retained raw artifact that reproduces it.
5. Softened constraints are never described as hard safety guarantees.
