# Position extension to the saved prediction-gap diagnosis

The initial [velocity diagnosis](panda-prediction-gap-methods.md), frozen at
`b66f3ba8ae6db4ba0ba07cfe989aa143a09f395b`, remains intact under
`results/panda_prediction_gap_v26`. Its valid velocity identities revealed
that the two zero-delay regressions first violate position limits at about
2.7 s, with no velocity-limit excess. This extension follows that observation.
It is post-hoc and preserves the same 26-episode selection, input identities,
physical-step prohibition, model replacements and two focus cases.

Retain position as well as velocity for the physical, witness and
recorded-command-model trajectories. At each physical substep, verify

```text
q_physical - q_witness = initial position offset
                      + elapsed time * initial velocity offset
                      + accumulated position-transition discrepancy
                      + accumulated position response to command differences.
```

The position-transition discrepancy propagates the accumulated velocity
discrepancy and adds the difference between the physical position increment
and the frozen constant-acceleration increment from that physical start state.
The command term propagates its corresponding velocity term and adds
`0.5 * substep^2 * B * (u_recorded - u_witness)`. Require closure within
`1e-9 rad`. This includes the physical integrator's position update convention;
the discrepancy is not purely a parameter error. Retain first-violation
position/velocity joint indices and all identity components.

Freeze this extension before retaining its separate canonical output,
`results/panda_prediction_gap_position_v26`. Preserve the initial output and
its source for a separate audit. The current CLI writes the extended output;
auditing the initial output requires its original frozen checkout. Both
calculations are descriptive and introduce no episode, physics call or
controller change.
