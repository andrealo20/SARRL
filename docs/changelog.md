# Changelog

This file records implemented and verified release increments. Performance claims are kept separately in `docs/verification.md` and require retained raw artifacts.

## v0.10.0 — architecture-safe SAC loading and configurable training

- `SACAgent.from_checkpoint()` reconstructs observation/action dimensions and non-default hidden architecture directly from the checkpoint.
- `tools/evaluate.py` and `tools/evaluate_stack.py` no longer assume the default SAC network width.
- `tools/train_sac.py` exposes hidden layer sizes, update frequency and replay capacity with input validation.
- Added a regression test proving exact deterministic-action recovery for a non-default network architecture.

