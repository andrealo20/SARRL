# Panda backup integration results

The causal supervisor and independent audit complete the
[frozen used-task protocol](panda-backup-integration-methods.md), implemented
at `9fa145b9cb1590b84b9db8749fba3332c7f39a0e`. The controller still violates
physical limits and does not advance to a fresh comparison in its present
form. The cumulative bank remains the reference.

All 64 original records reproduce before candidate execution. The 32 candidate
episodes retain 48,000 physical substeps and complete traces. Including the
96,000 reproduction calls, the canonical budget is 144,000 retained physical
calls on used tasks. The [evidence directory](../results/panda_backup_integration_v26)
preserves every result, source/input hash, timing sample and audit total.

## Outcomes include two new failure cases

The selection contains twelve first-seed conditions and every task with an
original filtered violation. It deliberately overrepresents known failures.
The following counts describe those selected tasks; they are not population
estimates or independent confirmation.

| Controller on the same 32 tasks | Successes | Unsafe episodes | Aborted episodes |
|---|---:|---:|---:|
| Original cumulative alignment | 6 | 26 | 0 |
| Original joint filter | 6 | 20 | 26 |
| Retained-backup candidate | 6 | 18 | 26 |

Four original unsafe cases no longer violate: clean/noisy 40 ms tasks at
seeds `9805807` and `9805809`. Two previously nonviolating filtered cases
become unsafe: clean/noisy zero-delay fast moves at seed `9805800`. A lower
aggregate unsafe count therefore includes a regression in another condition.
The candidate fails the zero-violation requirement even on these used tasks.

| Condition | Episodes | Successes | Unsafe | Aborts |
|---|---:|---:|---:|---:|
| Fast, clean, 0 ms | 1 | 0 | 1 | 1 |
| Fast, clean, 20 ms | 1 | 0 | 0 | 1 |
| Fast, clean, 40 ms | 11 | 0 | 8 | 11 |
| Fast, noisy, 0 ms | 1 | 0 | 1 | 1 |
| Fast, noisy, 20 ms | 1 | 0 | 0 | 1 |
| Fast, noisy, 40 ms | 11 | 0 | 8 | 11 |
| Ordinary, clean, 0 ms | 1 | 1 | 0 | 0 |
| Ordinary, clean, 20 ms | 1 | 1 | 0 | 0 |
| Ordinary, clean, 40 ms | 1 | 1 | 0 | 0 |
| Ordinary, noisy, 0 ms | 1 | 1 | 0 | 0 |
| Ordinary, noisy, 20 ms | 1 | 1 | 0 | 0 |
| Ordinary, noisy, 40 ms | 1 | 1 | 0 | 0 |

All six ordinary first-seed tasks succeed without violations or aborts.
All 26 selected fast tasks abort: twenty at a predicted pending-command
violation and six at proposal-witness rejection. There are no emergency
initializations. The ordinary observations do not establish performance on a
new task distribution.

## Accepted model witnesses do not bound the physical plant

A post-hoc saved-state review locates sixteen first violations in intervals
still executing pre-abort commands at 40 ms delay. The two zero-delay
violations first occur under the retained fallback. No first violation
precedes abort. The existing interval/FIFO timing calculation assigns a
boundary sample to the command that generated it.

The independent audit accepts the recorded frozen-model witnesses while
physical violations remain. These witnesses therefore cannot be treated as
calibrated bounds for the nonlinear plant and its evolving estimated dynamics.
The zero-delay failures also prevent attributing every failure to transport
delay. This evidence does not isolate a unique model error or establish that
another damping gain or horizon would resolve it. No such tuning is performed.

Reproduce the descriptive paired counts, event order and timing from the
completed files without physical stepping:

```bash
python -m tools.diagnose_panda_backup
```

## Timing includes the complete control calculation

Across 9,600 recorded command-plus-estimator cycles, median computation is
1.093999 ms, p99 is 7.105068 ms and maximum is 75.883577 ms. Twelve cycles
exceed 10 ms. The median includes latched fallback cycles, which skip nominal
and witness recomputation; it is not the median cost of active filtering.
Controller construction, plant integration, interface bookkeeping and output
serialization remain outside these timings. Every cycle and construction time
is retained. The earlier verifier-only timing cannot substitute for this check,
and this run does not establish deadline compliance.

## Audit and next work

The saved-state audit checks 32 causal controller replays, 9,600 control
cycles, 48,000 physical substep records, 2,078 witnesses and all 10,390 delay
cases. Independent sequential predictions cover 429,209 backup intervals;
2,046 feasible QP projections are checked separately. It verifies noise,
references, all five RLS models, sent-command ownership, unchanged FIFO,
retained context, physical outcomes and tracking RMS with zero physical
integration calls. The audit does not replay a replacement nonlinear plant.

The next diagnostic question is how the accepted frozen prediction diverges
from the physical trajectory, including the two zero-delay regressions.
Any revised design must define a defensible treatment of prediction error and
measure its full control cost. Preserve these failed outcomes and the original
advancement criteria before considering a new frozen comparison. No fresh
seed block, policy training, default-controller change or release follows
from this integration result.
