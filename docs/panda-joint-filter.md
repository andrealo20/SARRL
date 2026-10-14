# Panda joint-filter comparison

The cumulative delay bank remains unchanged. The candidate projects its sent
torque against joint-position, velocity and actuator bounds using causal
measured state, the selected physical estimate and its own sent history.
The filter receives no plant object, true parameter, applied age, delivered
command or task-phase input. The original five RLS estimators observe the
actual filtered sent command and retain their cumulative updates.

## Frozen formulation

From the selected model, compute `B = inverse(M)` and `d = -B * bias`, so the
estimated acceleration is `B * torque + d`. Freeze these coefficients at the
measured state for the prediction. For every possible constant age from zero
through four control periods, integrate the known preceding sent commands
until the new command would arrive. Missing startup commands are known zeros.
Prefix checks include exact quadratic position extrema and velocity endpoints
under this frozen affine model. This covers a bounded constant FIFO delay;
varying packet ages and off-grid delays are outside the comparison.

At every predicted arrival state impose lower/upper relative-degree-two joint
barrier inequalities with both gains five, and next-period velocity bounds.
The initial position barriers and their first-order conditions must be
nonnegative. Position bounds have a 0.03 rad reserve and velocity bounds a
0.10 rad/s reserve. The control period is 10 ms and the maximum delay is 40 ms.
The QP minimizes half the squared change in torque divided componentwise by
the actuator limits. Torque bounds are explicit constraints.

The numerical projection uses [SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)
with analytic derivatives, at most 100 iterations and objective tolerance
`1e-12`. An independent check requires scaled primal residual at least `-1e-8`,
nonnegative dual stationarity residual at most `1e-6`, and complementarity at
most `1e-6`. A solver success flag alone cannot accept a torque. An unsuccessful
projection is classified with a separate linear feasibility solve. The
[MuJoCo dynamics documentation](https://mujoco.readthedocs.io/en/stable/computation/)
describes the mass matrix and bias terms used by the existing model API.

An invalid prefix, outside-domain prediction or unsuccessful QP latches an
abort. The system then sends clipped estimated-model braking torque
`bias - 20 * M * measured_velocity` through the unchanged FIFO for the rest of
the fixed horizon. It does not clear the transport queue or stop simulation.
Braking is never labelled feasible or safe. An aborted episode cannot count
as a controller success, even if the physical target happens to be reached.

These checks concern an estimated model with frozen coefficients and noisy
measurements. They are not calibrated uncertainty bounds or a physical safety
certificate. Physical position, velocity and contacts are measured at every
2 ms plant substep. General collision avoidance, hardware and sim-to-real
remain outside scope.

## Paired grid and provenance

Twenty fresh seeds, `9805800` through `9805819`, cross twelve conditions:
ordinary 2 s moves or fast 0.25 s moves, clean/noisy measurements and constant
delays 0/20/40 ms. Every episode lasts 3 s including settling. Both arms use
the unchanged wide task sampler, identical payload/friction draws and paired
sensor streams. The arms are cumulative alignment and filtered alignment.
The grid retains 480 episodes and 24 first-seed traces including all physical
substep states. Source, model, protocol, reproduction and completion hashes
are retained. Original physical integration is unchanged; its new recorder
is checked for exact agreement with the original environment.

The fast duration increases the prescribed quintic peak speed eightfold on
the same displacement. Physical violations are measured rather than inferred
from that reference; the informativeness criterion below can still fail.

Freeze source, this protocol and the reserved `9805800..9805899` block before
execution. Reproduce every one of the 4,000 v2.5 reference-validation records
before opening the new seeds. Use Python 3.12, NumPy 2.5.2, MuJoCo 3.13.0 and
SciPy 1.18.1, four workers and one numerical thread per worker. Only the
canonical output is permitted. Interrupted runs require the same source and
runtime; completed output is never replaced. Reproduction uses 6,000,000
physical substeps and the fresh grid uses 720,000. Affine prediction performs
no virtual MuJoCo stepping.

```bash
python -m tools.run_panda_joint_filter --workers 4 --reproduce-only
python -m tools.run_panda_joint_filter --workers 4 --resume
python -m tools.audit_panda_joint_filter
```

## Exploratory decision

All criteria must pass to advance to independent confirmation:

1. At least 10% of the 120 fast baseline episodes are physically unsafe, so
   the grid can demonstrate a reduction. Otherwise the grid is uninformative.
2. The paired fast unsafe fraction decreases by at least ten percentage points.
3. The candidate has zero measured unsafe episodes across all 240 episodes.
4. No cell loses task successes against cumulative alignment. Every slow cell
   has zero candidate aborts and at most 5 mrad paired mean RMS increase for
   each joint.

Report every condition, abort, physical violation and failed criterion. These
point-estimate engineering criteria do not establish population safety or
statistical noninferiority. Evaluate this decision before freezing and running
the separate planar residual-learning study in the combined v2.6 release.
No filter gain, reserve, task speed or abort rule is tuned on this block.
