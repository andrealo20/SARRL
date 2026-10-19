"""Validated mean-value maps of the normalized Panda ODE and structured feedback."""

import numpy as np

from sarrl.safety.panda_affine_state import concatenate, mean_value
from sarrl.safety.panda_center_arithmetic import _compiled_signature, _evaluate, _export
from sarrl.safety.panda_local_coefficients import PandaLocalCoefficients, PandaSnapshotDynamics
from sarrl.safety.panda_regional_arithmetic import _global_bounds
from sarrl.safety.panda_state_region import EnclosureUnavailable, VectorBox
from sarrl.safety.panda_structured_backup import PandaParameterBox
from sarrl.safety.panda_validated_dynamics import arithmetic, balls, exact, interval, point


def solve(matrix, rhs, backend):
    try:
        answer = backend.arb_mat(matrix.tolist()).solve(
            backend.arb_mat(rhs.tolist()), algorithm="precond"
        )
    except (ValueError, ZeroDivisionError) as failure:
        raise EnclosureUnavailable("validated_differential_solve_unavailable") from failure
    return np.array(
        [[answer[i, j] for j in range(answer.ncols())] for i in range(answer.nrows())], dtype=object
    )


class PandaMeanValueDynamics(PandaSnapshotDynamics):
    """Ideal-model derivatives; nonzero unstructured physical remainders are unsupported."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if np.any(self.mass_remainder) or np.any(self.bias_remainder):
            raise ValueError("mean-value derivatives require the declared normalized ideal model")
        with arithmetic(self.precision_bits) as backend:
            bounds = _global_bounds(self._snapshot, backend.arb, lambda x: exact(x, backend))
            self._mass_qq = tuple(_export(group[1]).upper for group in bounds)

    def jets(self, q, v):
        if not self.q_domain.contains(q) or not self.v_domain.contains(v):
            raise EnclosureUnavailable("state_region_not_covered")
        key = (self.precision_bits, *(x.tobytes() for x in (q.lower, q.upper, v.lower, v.upper)))
        with self._cache_lock, arithmetic(self.precision_bits) as backend:
            if _compiled_signature(self._snapshot) != self._signature:
                raise ValueError("compiled snapshot changed after capture")
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            (nominal, g0), (payload, gp) = _evaluate(
                self._snapshot,
                None,
                None,
                backend.arb,
                backend.fmpq,
                state_arrays=(balls(q, backend), balls(v, backend)),
                include_gravity_q=True,
            )
            result = (
                PandaLocalCoefficients(
                    nominal,
                    payload,
                    q,
                    v,
                    self.precision_bits,
                    backend.__version__,
                    self._source_hash,
                    self._signature,
                ),
                g0,
                gp,
            )
            if self._cache_size:
                self._cache[key] = result
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            return result

    def coefficients(self, q, v):
        return self.jets(q, v)[0]

    def derivatives(self, q, v, mass, backend):
        coefficients, g0, gp = self.jets(q, v)
        nominal, payload = coefficients.nominal, coefficients.payload
        mq = balls(nominal.mass_q, backend) + mass * balls(payload.mass_q, backend)
        matrix = balls(nominal.mass, backend) + mass * balls(payload.mass, backend)
        hq = balls(g0, backend) + mass * balls(gp, backend)
        gamma = (mq + mq.swapaxes(1, 2) - mq.transpose(2, 0, 1)) / 2
        hessian = point(self._mass_qq[0], backend) + abs(mass) * point(self._mass_qq[1], backend)
        gamma_q = (hessian + hessian.swapaxes(1, 2) + hessian.transpose(2, 0, 1, 3)) / 2
        velocity = balls(v, backend)
        hv = point(np.diag(self._snapshot.dof_damping), backend)
        for i, axis in np.ndindex((7, 7)):
            error = sum(
                (
                    gamma_q[i, j, k, axis] * abs(velocity[j]) * abs(velocity[k])
                    for j, k in np.ndindex((7, 7))
                ),
                backend.arb(0),
            )
            hq[i, axis] += error * interval(-1, 1, backend)
            hv[i, axis] += sum(
                ((gamma[i, axis, k] + gamma[i, k, axis]) * velocity[k] for k in range(7)),
                backend.arb(0),
            )
        return coefficients, matrix, mq, hq, hv

    def acceleration_jacobian(self, parameters, q, v, torque, backend):
        mass = interval(parameters.lower[0], parameters.upper[0], backend)
        coefficients, matrix, mq, hq, hv = self.derivatives(q, v, mass, backend)
        acceleration = balls(self.acceleration(parameters, q, v, torque), backend)
        aq = -hq.copy()
        for k in range(7):
            aq[:, k] -= mq[:, :, k] @ acceleration
        b = np.array(
            [
                interval(lo, hi, backend)
                for lo, hi in zip(parameters.lower[1:], parameters.upper[1:], strict=True)
            ],
            dtype=object,
        )
        am = -(
            balls(coefficients.payload.mass, backend) @ acceleration
            + balls(coefficients.payload.bias, backend)
        )
        rhs = np.column_stack(
            (aq, -hv - np.diag(b), am, -np.diag(balls(v, backend)), point(np.eye(7), backend))
        )
        return solve(matrix, rhs, backend)

    def acceleration_affine(self, parameters, q, v, torque, backend):
        combined = concatenate((q, v, parameters, torque))
        domain = combined.hull(backend)
        positions = VectorBox(domain.lower[:7], domain.upper[:7])
        velocities = VectorBox(domain.lower[7:14], domain.upper[7:14])
        physical = PandaParameterBox(domain.lower[14:22], domain.upper[14:22])
        held = VectorBox(domain.lower[22:], domain.upper[22:])
        center = self.acceleration(
            PandaParameterBox(parameters.center, parameters.center),
            VectorBox.point(q.center),
            VectorBox.point(v.center),
            VectorBox.point(torque.center),
        )
        jacobian = self.acceleration_jacobian(physical, positions, velocities, held, backend)
        return mean_value(combined, balls(center, backend), jacobian, backend)

    def feedback_affine(self, context, q, v, backend):
        combined = concatenate((q, v))
        domain = combined.hull(backend)
        positions = VectorBox(domain.lower[:7], domain.upper[:7])
        velocities = VectorBox(domain.lower[7:], domain.upper[7:])
        _, matrix, mq, hq, hv = self.derivatives(
            positions, velocities, exact(context.parameters[0], backend), backend
        )
        desired = -exact(context.kp, backend) * (
            balls(positions, backend) - point(context.anchor, backend)
        )
        desired -= exact(context.kd, backend) * balls(velocities, backend)
        jq = hq - exact(context.kp, backend) * matrix
        for k in range(7):
            jq[:, k] += mq[:, :, k] @ desired
        jv = (
            hv
            + point(np.diag(context.parameters[1:]), backend)
            - exact(context.kd, backend) * matrix
        )
        value = self.torque(
            context, VectorBox.point(q.center), VectorBox.point(v.center), np.zeros(7)
        )
        return mean_value(combined, balls(value, backend), np.column_stack((jq, jv)), backend)

    def jerk(self, parameters, q, v, torque, acceleration, backend):
        jacobian = self.acceleration_jacobian(parameters, q, v, torque, backend)
        return jacobian[:, :7] @ balls(v, backend) + jacobian[:, 7:14] @ balls(
            acceleration, backend
        )
