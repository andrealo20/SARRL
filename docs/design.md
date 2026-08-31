# SARRL design

SARRL is developed bottom-up. Later layers are not trusted until lower layers have invariant tests and a separate failure mode.

## Layered architecture

1. **Analytical plant** — deterministic 2-DoF rigid-body dynamics, friction and payload.
2. **Nominal control** — computed torque and constrained nonlinear MPC.
3. **Learning** — direct or residual Soft Actor-Critic implemented in PyTorch.
4. **Robustness** — domain randomisation, delay and explicit fault injection.
5. **Adaptation** — causal GRU context inferred only from transition history.
6. **Learned dynamics** — residual acceleration ensemble with bootstrap disagreement.
7. **Runtime gating** — uncertainty may reduce residual-policy authority.
8. **Safety** — exact 2-D hard HOCBF projection with explicit infeasibility.
9. **Evaluation** — disjoint training, validation and held-out seed populations with retained raw artifacts.

Each layer has a narrow API so it can be ablated independently.

## Why an analytical 2-DoF plant first

The planar system is not intended as the final manipulation benchmark. It is the reference system on which equations can be audited directly. A black-box simulator would make it harder to determine whether a failed learning result came from the algorithm, the dynamics interface or the simulator.

The planned Franka/MuJoCo transfer is therefore downstream of the verified planar stack.

## Nominal controller

Residual learning only has value if the baseline is competent. Computed torque is therefore deliberately strong under a correct model. The retained nominal experiment reaches all 100 fixed targets, while controlled dynamics mismatch causes a large performance drop. This establishes a concrete correction problem for RL rather than an artificially weak baseline.

## MPC

`NonlinearMPC` uses direct shooting through the analytical RK4 plant. SLSQP handles torque bounds and nonlinear state constraints. This implementation is a reference optimiser for correctness studies, not a real-time claim.

## Residual RL

In residual mode the policy acts inside a bounded correction envelope rather than commanding all torque. This preserves the physics controller as a useful prior and makes the learned action interpretable as compensation for modelling error.

Direct-torque mode is retained as an ablation.

## Causal context adaptation

The context encoder is not given masses, friction, motor gains or payload at runtime. It receives only a fixed history of:

```text
observation_t, action_t, observation_(t+1) - observation_t
```

Ground-truth dynamics parameters are auxiliary supervision during encoder training. The latent representation is therefore inferable from signals that would exist on a real robot.

## Fault model

Faults are explicit changes to the plant, not changes secretly communicated to the policy. Current controlled faults include motor-gain degradation and payload changes at a configured step.

## Residual dynamics and uncertainty

The learned dynamics model predicts the difference between observed acceleration and the acceleration predicted by the nominal model from the **commanded** torque. This distinction is important: using the post-actuator applied torque would hide motor-gain error from the residual model.

Independent bootstrap minibatches create an ensemble. Prediction disagreement is used as epistemic uncertainty. The associated gate reduces learned residual authority but is intentionally not described as a certificate.

## Safety semantics

The safety layer contains no feasibility slack. It builds hard affine torque constraints and projects the candidate command exactly in two dimensions by enumerating active sets of size zero, one and two.

If the polytope is empty, the result is an explicit failure. When `require_safety=True`, the runtime stack does not silently execute an uncertified fallback.

The guarantee remains model-relative: a mathematically exact HOCBF based on an incorrect dynamics model is not a hardware safety proof.

## Off-policy reproducibility

A useful SAC checkpoint is more than neural weights. Exact continuation also depends on replay contents, replay RNG, environment state, delay queue and all RNG states that affect future samples.

Training checkpoint v2 therefore stores:

- agent weights, target networks and optimisers;
- entropy state and PyTorch RNG;
- replay contents, indices and RNG;
- full environment constructor configuration and current state;
- domain randomisation/fault configuration;
- delayed-command queue;
- NumPy and Python RNG states;
- loop counters and trainer update cadence.

`load_training_session()` reconstructs the session instead of requiring the caller to reproduce hidden CLI defaults.

## Model selection

`best.pt` is selected only on a dedicated deterministic validation seed set. Selection is lexicographic:

1. higher validation success rate;
2. higher mean return as the tie-breaker.

The held-out seed range is separate and is only used for final reporting.

## Scientific rules

1. Held-out evaluation never selects checkpoints.
2. Final learned comparisons use at least five independent training seeds.
3. Variation across trained models is reported separately from episode-level binomial uncertainty.
4. A failed training run is first treated as a debugging signal, not evidence that an algorithm cannot solve the task.
5. No README metric is added without a retained raw artifact.
6. Softened or heuristic mechanisms are never described as hard safety guarantees.
7. Unvalidated simulator/hardware integrations remain roadmap items rather than implemented claims.
