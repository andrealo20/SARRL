# v1.5 Phase A runbook

This is the continuity record for the uncertainty-gate signal screen. The
protocol itself is frozen in `PLAN.md`; this file records how to execute and
resume it without changing that protocol.

Current status: Phase A completed and independently verified on 2026-09-02.
The decision was `proceed_phase_b`; retained results are documented in
`docs/v15-phase-a-results.md`.

## Frozen campaign

- Branch: `feature/v1.5-uncertainty-gate-calibration`
- Five frozen artifact pairings: training/ensemble seeds `0..4`
- Per pairing: 100 A2 plus 100 A3 episodes on seeds `60000..60099`
- Total: 10 cells and 1,000 episodes
- Accelerator: CUDA for SAC policy and ensemble inference; the causal context
  encoder remains on its deterministic CPU-only runtime path
- Output: `results/uncertainty_gate_calibration/phase_a/`

Each seed is an independent resumable shard. A shard is valid only when its
`complete.json` exists and every recorded SHA-256 verifies. Re-running a valid
shard verifies and skips it. An interrupted shard is overwritten from its
first episode, so partially written transition data is never treated as final.

## Official WSL commands

Run these five commands concurrently from the repository root:

```bash
for seed in 0 1 2 3 4; do
  /home/andrea/projects/SARRL/.venv/bin/python tools/run_planar_v15_phase_a.py \
    --training-seed "$seed" --device cuda &
done
wait
```

After all five `complete.json` files exist, aggregate and apply the frozen
screening rule:

```bash
/home/andrea/projects/SARRL/.venv/bin/python \
  tools/run_planar_v15_phase_a.py --aggregate --device cuda
```

The final decision is in `phase_a/decision.json`. `proceed_phase_b` authorizes
implementation of the already specified Phase B. `retire_gate` records a
negative v1.5 result. `inconclusive` stops the release without changing the
gate. Never start Phase B before this decision exists.

## Phase B calibration

After a verified `proceed_phase_b` decision, build and then independently
verify the versioned calibration artifact:

```bash
/home/andrea/projects/SARRL/.venv/bin/python \
  tools/build_planar_v15_calibration.py
/home/andrea/projects/SARRL/.venv/bin/python \
  tools/build_planar_v15_calibration.py --verify
```

This produces canonical `calibration.json`, `u_ref.csv` and
`sensitivity.csv` files. Phase C must hash and verify `calibration.json` before
its first episode and select the `u_ref` matching each ensemble seed.

## Verification before handoff

```bash
/home/andrea/projects/SARRL/.venv/bin/python -m ruff check sarrl tests tools
/home/andrea/projects/SARRL/.venv/bin/python -m pytest -q
git status --short --branch
```

Do not stage or delete ignored `.pt` or `.npz` artifacts. Preserve all retained
v1.2-v1.4 evidence unchanged.
