# Changelog

This file records implemented and verified release increments. Performance claims are kept separately in `docs/verification.md` and require retained raw artifacts.

## v0.12.0 — experiment provenance, multi-seed orchestration and residual-model semantics

- Every SAC training run writes a machine-readable manifest with Git commit, runtime/library versions, agent/environment configuration and validation protocol.
- Added `tools/run_sac_sweep.py` for reproducible multi-seed training plus a disjoint held-out seed set, raw held-out episode CSVs and cross-model aggregate statistics.
- Added retained per-episode deterministic policy evaluation records and validation/held-out seed-range guards.
- Fixed residual-dynamics data generation so the model input is the **commanded** torque while the target contains motor-gain degradation; using already-degraded torque had hidden actuator-gain error from the learned residual.
- Updated the package version source, which had remained stale at `0.1.0`.

## v0.11.0 — reproducible validation and exact session reconstruction

- Deterministic SAC actions no longer sample internally or advance PyTorch RNG state.
- Added fixed-seed deterministic policy evaluation as a library primitive.
- Training can select `best.pt` on a dedicated validation seed set, lexicographically by success rate then mean return; held-out evaluation remains separate.
- Training checkpoint format v2 records Python/NumPy/PyTorch RNGs, replay state, full environment constructor configuration, delay queue and trainer update cadence.
- `load_training_session()` reconstructs non-default SAC architecture, replay capacity and randomized/faulted environment directly from the checkpoint.
- Added environment/replay reconstruction helpers and regression tests for configuration mismatch and exact RNG continuation.

## v0.10.0 — architecture-safe SAC loading and configurable training

- `SACAgent.from_checkpoint()` reconstructs observation/action dimensions and non-default hidden architecture directly from the checkpoint.
- `tools/evaluate.py` and `tools/evaluate_stack.py` no longer assume the default SAC network width.
- `tools/train_sac.py` exposes hidden layer sizes, update frequency and replay capacity with input validation.
- Added a regression test proving exact deterministic-action recovery for a non-default network architecture.

