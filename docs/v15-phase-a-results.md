# v1.5 Phase A result

Phase A passed its preregistered heuristic screen on 2026-09-02 and authorizes
Phase B. This is evidence that frozen ensemble disagreement contains useful
rank information on the calibration population; it is not yet evidence that a
calibrated gate improves control outcomes.

## Result

- Frozen artifact pairs: 5
- Cells: 10 (A2 and A3 for each paired training/ensemble seed)
- Episodes: 1,000 total; 1,000 qualified
- Global common episode seeds: all 100, `60000..60099`
- Retained transitions: 143,732
- Non-finite transition exclusions: 0
- Zero-variance episodes: 0
- Median of the 10 cell median within-episode Spearman correlations: `0.2975714414299495`
- Paired episode-seed bootstrap 95% interval: `[0.22826571506750873, 0.35569407508291667]`
- Frozen threshold: `0.2`
- Decision: `proceed_phase_b`

The lower interval bound exceeds the threshold. The association is moderate
and heterogeneous: individual cell medians range from `0.1528` to `0.4448`.
The conclusion is therefore deliberately scoped to the five frozen artifact
pairs and does not claim generalization to unseen training runs.

## Audit

`tools/verify_planar_v15_phase_a.py` independently verifies every retained
hash and every transition-level dynamics, target, uncertainty-norm and
prediction-error invariant, then reproduces all episode summaries and the
10,000-draw decision from the raw table. The audit passed for all 143,732
transitions.

The canonical raw CSV is retained as deterministic `transitions.csv.gz` to
stay below GitHub's single-file limit. Its decompressed bytes hash identically
to the uncompressed `transitions.csv` recorded in the evaluation manifest.
