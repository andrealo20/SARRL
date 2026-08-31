# Changelog

This file records implemented release increments. Performance evidence is kept separately in `docs/verification.md` and requires retained raw artifacts.

## v1.0.1 — CUDA checkpoint restore compatibility

- Fixed CUDA RNG restoration when training checkpoints are loaded onto a CUDA device.
- CUDA RNG states are explicitly converted back to CPU `ByteTensor` objects before calling `torch.cuda.set_rng_state_all()`.
- Added a regression test for CUDA-remapped RNG checkpoint state that also runs on CPU-only CI.
- Updated Ruff-clean typing/import formatting for the current development toolchain.
- Automated suite increased from 71 to 72 tests.

## v1.0.0 — verified planar research stack

- Promoted the planar stack to the first stable research release.
- Rewrote README and technical documentation to match the implemented architecture and current evidence.
- Documented exact boundaries between implemented, tested, experimentally measured and planned components.
- Added release/version consistency testing and final audit/packaging checks.
- M10 MuJoCo/Franka remains intentionally planned rather than being shipped untested.

## v0.12.0 — experiment provenance, multi-seed orchestration and residual-model semantics

- Every SAC training run writes a machine-readable manifest with Git commit, runtime/library versions, agent/environment configuration and validation protocol.
- Added `tools/run_sac_sweep.py` for reproducible multi-seed training plus a disjoint held-out seed set, raw held-out episode CSVs and cross-model aggregate statistics.
- Added retained per-episode deterministic policy evaluation records and validation/held-out seed-range guards.
- Fixed residual-dynamics data generation so the model input is the **commanded** torque while the target contains motor-gain degradation; using already-degraded torque had hidden actuator-gain error from the learned residual.
- Updated the package version source, which had remained stale at `0.1.0`.

## v0.11.0 — reproducible validation and exact session reconstruction

- Deterministic SAC actions no longer sample internally or advance PyTorch RNG state.
- Added fixed-seed deterministic policy evaluation as a library primitive.
- Training selects `best.pt` on a dedicated validation seed set, lexicographically by success rate then mean return; held-out evaluation remains separate.
- Training checkpoint format v2 records Python/NumPy/PyTorch RNGs, replay state, full environment constructor configuration, delay queue and trainer update cadence.
- `load_training_session()` reconstructs non-default SAC architecture, replay capacity and randomized/faulted environment directly from the checkpoint.

## v0.10.0 — architecture-safe SAC loading and configurable training

- `SACAgent.from_checkpoint()` reconstructs observation/action dimensions and non-default hidden architecture directly from the checkpoint.
- Evaluation tools no longer assume the default SAC width.
- Trainer exposes hidden layer sizes, update frequency and replay capacity.

## v0.9.0 — reproducible evaluation protocol

- Added retained per-episode result structures, Wilson confidence intervals, paired bootstrap comparison and Git provenance.
- Added controlled computed-torque campaigns for nominal, ID-randomized, OOD-dynamics and motor-fault scenarios.
- Retained the raw v0.9 baseline CSV and aggregate JSON.

## v0.8.0 — integrated runtime control stack

- Composed physics baseline, residual policy, optional residual-dynamics uncertainty gate and hard safety projection behind one runtime command interface.
- Made uncertified safety failure explicit through `executable` and `safety_certified` semantics.

## v0.7.0 — full off-policy training checkpoints

- Added replay-buffer and environment state serialization.
- Preserved stochastic continuation state rather than saving network weights alone.
- Added exact continuation regression tests.

## v0.6.0 — learned residual dynamics and epistemic uncertainty

- Added bootstrap ensemble of residual acceleration networks.
- Added uncertainty estimation from ensemble disagreement.
- Added a bounded uncertainty gate that reduces residual authority without claiming safety certification.

## v0.5.0 — hard HOCBF safety projection

- Added relative-degree-two joint and Cartesian obstacle barrier constraints.
- Added one-step velocity and torque constraints.
- Implemented exact 2-D Euclidean projection by active-set enumeration.
- Hard infeasibility is reported explicitly; no slack variable is hidden behind a “safe” claim.

## v0.4.0 — causal dynamics-context adaptation

- Added GRU context encoder from transition history.
- Added supervised auxiliary physical-context head and runtime latent wrapper.
- Kept ground-truth dynamics parameters out of the runtime policy input.

## v0.3.0 — controlled model mismatch and faults

- Added payload dynamics, mass/friction/motor-gain randomisation, sensor noise and actuator delay.
- Added abrupt motor-gain/payload fault injection.
- Added deterministic environment checkpoint state for delayed execution.

## v0.2.0 — constrained nonlinear MPC

- Added direct-shooting nonlinear MPC through the analytical RK4 plant.
- Added torque, joint-position and velocity constraints, warm start and optimisation regression tests.

## v0.1.0 — analytical/RL foundation

- Added analytical 2-DoF dynamics, RK4, kinematics and computed-torque baseline.
- Added direct/residual reaching environment.
- Added from-scratch Soft Actor-Critic, replay buffer, initial checkpointing and CI.
- Retained the 100/100 nominal computed-torque sanity baseline.
