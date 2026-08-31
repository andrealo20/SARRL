# Changelog

This file records implemented and verified release increments. Performance claims are kept separately in `docs/verification.md` and require retained raw artifacts.

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

