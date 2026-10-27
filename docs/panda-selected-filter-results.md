# Panda selected-delay filter results

Source `2cceedb`, which seals the freeze `1812d9d`, reproduced all 480 v2.6
joint-filter records through the new entry point with no mismatch before
opening twenty new paired seeds. The [frozen protocol](panda-selected-filter.md)
retained 720 episodes on three arms. All three predictions came out as written:
narrowing the age set removes some 20 ms aborts and leaves the 40 ms failure
in place. The narrowing also adds seven physical violations on tasks the v2.6
filter kept safe, so the candidate does not advance.

| Fast moves, 120 tasks | Successes | Unsafe | Aborts |
|---|---:|---:|---:|
| `cumulative` | 28 | 92 | 0 |
| `robust_filter` (v2.6, ages 0..4) | 24 | 16 | 96 |
| `selected_filter` (tied ages) | 36 | 23 | 84 |

The ordinary-move cells keep 20/20 successes on every arm, with no abort and
no unsafe episode. An abort excludes controller success even if the plant
later reaches the target.

## Every fast cell

Twenty paired tasks per cell, aborts and unsafe episodes as robust / selected.

| Condition | Robust aborts | Selected aborts | Robust unsafe | Selected unsafe | Robust success | Selected success |
|---|---:|---:|---:|---:|---:|---:|
| clean, 0 ms | 16 | 12 | 0 | 0 | 4 | 8 |
| clean, 20 ms | 16 | 14 | 0 | 3 | 4 | 6 |
| clean, 40 ms | 16 | 16 | 8 | 9 | 4 | 4 |
| noisy, 0 ms | 16 | 12 | 0 | 1 | 4 | 8 |
| noisy, 20 ms | 16 | 14 | 0 | 2 | 4 | 6 |
| noisy, 40 ms | 16 | 16 | 8 | 8 | 4 | 4 |

## Decisions

| Rule fixed before computation | Outcome |
|---|---|
| A: fewer selected aborts in each 20 ms fast cell | Holds, 16 to 14 in both |
| B: 40 ms aborts within two and a 40 ms selected episode unsafe | Holds, 16 and 16, with 17 unsafe |
| C: a 0 ms fast selected episode unsafe | Materialised, one noisy episode |
| `aborts_reduced` | Holds (prediction A) |
| `no_new_violations` | Fails: 7 tasks unsafe under selected and safe under robust, 0 in the other direction |
| `no_success_loss` | Holds: 0 tasks lost, 12 gained |
| Advancement against `cumulative` | Fails: zero-unsafe criterion in five fast cells, success count in both 0 ms fast cells |

The 0 ms abort reduction, 16 to 12 in both cells, is structural and not
counted. The tied set never excluded the true delay in any of the 240
selected episodes. Fast unsafe episodes fall from 92 to 23 against the
cumulative reference, a reduction of 57.5 percentage points, but the
zero-unsafe criterion fails and the v2.6 filter keeps the lower count.

## The seven new violations

All seven are joint-velocity excesses, from 0.021 to 0.328 rad/s, with no
position excess and no contact. In each the selected filter aborts later than
the v2.6 filter on the same task: at control steps 8 to 13 against 6 to 10.

| Condition | Seed | Selected abort, age set | Robust abort | Velocity excess, rad/s |
|---|---:|---|---:|---:|
| clean, 20 ms | 9806208 | 12, {2} | 7 | 0.328 |
| clean, 20 ms | 9806213 | 10, {2} | 6 | 0.043 |
| clean, 20 ms | 9806215 | 8, {2} | 6 | 0.067 |
| clean, 40 ms | 9806211 | 11, {4} | 10 | 0.021 |
| noisy, 0 ms | 9806208 | 13, {0} | 6 | 0.279 |
| noisy, 20 ms | 9806208 | 12, {2} | 7 | 0.317 |
| noisy, 20 ms | 9806215 | 8, {2} | 6 | 0.071 |

At 0 ms the selected aborts carry a different reason,
`outside_prediction_domain`: with the set `{0}` there is no pending command to
check, and the filter stops because the measured state is already outside the
barrier domain. The commands it had accepted moved the plant further than the
estimated model predicted. The v2.6 aborts on pending commands were early
enough to brake from lower speeds; narrowing the set postpones braking into
faster states. This reading comes from the episode records, abort steps and
reasons; it was not tested as a separate hypothesis on the trajectories.

The test separates the two v2.6 failure causes as intended. Part of the 20 ms
abort count came from building constraints over every age, and the 40 ms
failure does not depend on that construction. The zero-delay violation,
without any command in flight, points at the estimated model and the velocity
reserve as well as the queue. These are engineering results on twenty paired
tasks per cell. No gain, reserve, task speed or fallback was tuned on these
seeds.

## Audit and provenance

The [retained directory](../results/panda_selected_filter_pilot) holds the
reproduction report, all 720 records, 36 first-seed traces with every physical
substep, summaries, decision and completion hashes. The audit verified 720
records, 36 causal controller replays with the age set recomputed from the
recorded scores at every step, 54,000 physical substeps and 4,902
independently checked feasible projections. The run used 1,800,000 physical
substeps including the reproduction. Auditing opens no physical episode.

Audit from a detached worktree at the source commit, with Python 3.12.3, NumPy
2.5.2, SciPy 1.18.1 and MuJoCo 3.13.0:

```bash
git worktree add --detach ../SARRL-selected-audit 2cceedb6d78ed00854add86d42de418c8c0d5865
cd ../SARRL-selected-audit
python -m tools.audit_panda_selected_filter --input <checkout>/results/panda_selected_filter_pilot
```

The auditor compares regenerated sent commands with delivered torques exactly,
so a host whose floating point differs in the last bit can fail it; the v2.6
portable auditor exists for that case and was not extended here. The audit
passed on the production host.

The block `9806200..9806299` is now used. A confirmation of any successor
design needs its own frozen protocol and a new block.
