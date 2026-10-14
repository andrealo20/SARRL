# Panda shared-parameter delay scoring

The previous parameter-memory candidate failed its frozen screen. This
comparison tests candidate-specific parameter compensation by changing only
the physical parameter vector used to score command-age hypotheses. The
five original bounded cumulative RLS estimators continue to update and
provide control parameters. Score memory remains exponential with effective
size fifty; prediction is disabled.

## Causal acquisition

Before acquisition, the candidate follows the original score-only bank
exactly. After at least fifty observed transitions, twenty-five consecutive
unchanged selected ages and full rank eight in the selected estimator's
data information, it copies that estimator's parameter vector. The copy
is retained for scoring every age, without later fitting or score reset.
It is acquired at the end of a transition and first used in the next one.
If acquisition never occurs, the original score-only rule continues and
the failure is reported.

This rule receives measured pre/post positions and velocities plus sent
commands. It receives no applied age, true parameter, delivered torque,
switch time or task-phase signal. Bootstrap and stability lengths are
fixed at 50 and 25, without a length sweep. Full information rank is an
acquisition heuristic, not a bound on parameter error or a confidence claim.
The scoring copy assumes fixed physical parameters within an episode.

Using one vector prevents each hypothesis from reducing its score through
its own fit after acquisition. Control still uses the selected candidate's
original estimator. Changing score ranking can therefore affect tracking
through both alignment and selection of inherited control estimates.
The comparison records that consequence rather than freezing the control
model as an additional intervention.

## Frozen comparison

Twenty fresh paired seeds, `9805600` through `9805619`, cross the existing
sixteen cold/warm, clean/noisy, fixed-10/fixed-40/increasing/decreasing
latency conditions. Six arms give 1,920 episodes: fixed, original adaptive,
cumulative bank, known-age diagnostic, score-only bank and shared-scoring
bank. Task draws, physical parameters, transport, measurements, priors,
gains and parameter bounds remain unchanged. Only the known-age diagnostic
receives applied age.

Commit the implementation, tests, this protocol and the reserved registry
before execution. The constant-delay reference validation must complete
and pass its frozen gate. Reproduce all 4,000 validation records before
opening these new seeds. The run uses the recorded NumPy/MuJoCo versions,
four workers and one numerical thread per worker. The canonical output
allows interrupted execution to resume with the same commit/runtime;
completed output is never replaced.

```bash
python -m tools.run_panda_shared_scoring --workers 4 --reproduce-only
python -m tools.run_panda_shared_scoring --workers 4 --resume
python -m tools.audit_panda_shared_scoring
```

Retained evidence includes every episode, 96 first-seed traces, acquisition
times/vectors, per-transition pre-update innovations, source/input hashes,
reproduction, summaries and the decision. The run uses 6,000,000 physical
substeps for reproduction and 2,976,000 for new episodes including warmup.
Prediction adds zero virtual substeps.

## Exploratory decision rule

The comparison retains the v2.4 screen:

1. Final-age correctness gains at least twenty percentage points against
   score-only memory overall in changing conditions, and separately in
   each cold/warm clean decreasing-delay cell.
2. Each constant cell has no fewer correct final ages than either the
   cumulative or score-only reference.
3. Each cell has no fewer task successes than either reference, and zero
   candidate physically unsafe episodes.
4. Each joint's paired mean RMS after 1 s and mean final maximum joint error
   increase by at most 0.1 mrad against each reference in every cell.

All criteria must pass to advance to independent confirmation. They do not
change the default controller. Record all cells, acquisition failures and
tracking regressions. A failed screen retains the reference; further
controller designs require a separate protocol and untouched seeds.
