# Panda joint-filter results

Source `6891dd2` reproduced all 4,000 v2.5 validation records before opening
twenty new paired seeds. The [frozen protocol](panda-joint-filter.md) retained
480 episodes across ordinary/fast moves, clean/noisy measurements and constant
delays of 0, 20 and 40 ms. The candidate does not pass its exploratory screen.

| Regime | Cumulative successes | Filtered successes | Cumulative unsafe | Filtered unsafe | Filtered aborts |
|---|---:|---:|---:|---:|---:|
| Ordinary moves | 120/120 | 120/120 | 0/120 | 0/120 | 0/120 |
| Fast moves | 16/120 | 13/120 | 104/120 | 20/120 | 107/120 |

Fast physical violations decrease by seventy percentage points, but twenty
candidate episodes remain unsafe at 40 ms and the two zero-delay cells lose
task successes. All four failures are retained in the decision. The candidate
does not advance to confirmation or replace the cumulative reference.
An abort excludes controller success even if the plant subsequently reaches
the target. The fixed physical horizon continues under bounded model braking.

## Every cell

Each entry contains twenty paired tasks. Ordinary cells all retain twenty
successes per arm, zero unsafe episodes and zero candidate aborts.

| Fast condition | Cumulative success | Filtered success | Cumulative unsafe | Filtered unsafe | Filtered abort |
|---|---:|---:|---:|---:|---:|
| clean, 0 ms | 4 | 2 | 16 | 0 | 18 |
| clean, 20 ms | 2 | 2 | 18 | 0 | 18 |
| clean, 40 ms | 2 | 3 | 18 | 10 | 17 |
| noisy, 0 ms | 4 | 2 | 16 | 0 | 18 |
| noisy, 20 ms | 2 | 2 | 18 | 0 | 18 |
| noisy, 40 ms | 2 | 2 | 18 | 10 | 18 |

The predicted-model checks and physical outcomes describe different quantities.
An optimal feasible QP does not certify the actual plant with estimated
parameters, frozen prediction coefficients and commands already in flight.
The retained reduction does not establish safety; frequent braking aborts and
remaining physical violations prevent promotion. No reserve, gain, task speed
or fallback is tuned on these seeds.

## Audit and provenance

The [retained directory](../results/panda_joint_filter_pilot) includes all
records, 24 first-seed traces, every physical substep in those traces,
reproduction, summaries, decision and completion hashes. The audit verified
480 records, 24 causal controller replays, 36,000 recorded physical substeps
and 1,846 independently checked feasible projections. Its primal/dual and
polynomial-interval checks reconstruct the causal inputs; braking is never
accepted as predicted-feasible. The production run used 6,720,000 physical
substeps including reproduction. Auditing opens no new physical episode.

This concludes the first study in the combined v2.6 scope. The separately
frozen planar residual-learning comparison follows with its own seeds and
criteria; this filter result adds no seven-joint residual policy.
