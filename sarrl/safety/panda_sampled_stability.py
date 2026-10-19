"""Offline equilibrium linearization of the normalized model and a held feedback command."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sarrl.robots.panda import model_provenance
from sarrl.safety.panda_analytic_region import _checked_packaged_model
from sarrl.safety.panda_backup import readonly
from sarrl.safety.panda_center_arithmetic import (
    ArrayEnclosure,
    _compiled_signature,
    _evaluate,
    _export,
)
from sarrl.safety.panda_structured_backup import PandaStructuredContext, parameters8
from sarrl.safety.panda_validated_dynamics import arithmetic, balls, exact, point


@dataclass(frozen=True)
class SampledStabilityEnclosure:
    mass: ArrayEnclosure
    gravity_q: ArrayEnclosure
    equilibrium_torque: ArrayEnclosure
    plant_generator: ArrayEnclosure
    feedback_jacobian: ArrayEnclosure
    held_flow: ArrayEnclosure
    sampled_jacobian: ArrayEnclosure
    eigen_real: ArrayEnclosure
    eigen_imag: ArrayEnclosure
    eigen_modulus: ArrayEnclosure
    lag: int
    precision_bits: int
    model_xml_sha256: str
    compiled_constants_sha256: str

    @property
    def instability_verified(self):
        return bool(np.any(self.eigen_modulus.lower > 1))

    @property
    def local_schur_stability_verified(self):
        return bool(np.all(self.eigen_modulus.upper < 1))


def _matrix(values, backend):
    return backend.arb_mat(np.asarray(values, dtype=object).tolist())


def _entries(matrix):
    return np.array(
        [[matrix[i, j] for j in range(matrix.ncols())] for i in range(matrix.nrows())], dtype=object
    )


def held_feedback_map(
    mass, gravity_q, damping, actual_friction, estimated_friction, *, kp, kd, control_dt, backend
):
    """Exact held linear flow at a matched-mass rest equilibrium, with ball operations.

    Plant state is [delta_q, delta_v] and held torque is an additional seven
    constant coordinates. This is not the continuous-feedback or Euler map.
    """
    m, g = balls(mass, backend), balls(gravity_q, backend)
    d0, b, bhat = [point(x, backend) for x in (damping, actual_friction, estimated_friction)]
    inverse = _matrix(m, backend).inv()
    inverse_array = _entries(inverse)
    generator = np.full((21, 21), backend.arb(0), dtype=object)
    generator[:7, 7:14] = point(np.eye(7), backend)
    generator[7:14, :7] = -inverse_array @ g
    generator[7:14, 7:14] = -inverse_array @ np.diag(d0 + b)
    generator[7:14, 14:] = inverse_array
    feedback = np.column_stack(
        (g - exact(kp, backend) * m, np.diag(d0 + bhat) - exact(kd, backend) * m)
    )
    flow = (_matrix(generator, backend) * exact(control_dt, backend)).exp()
    flow_array = _entries(flow)
    sampled = flow_array[:14, :14] + flow_array[:14, 14:] @ feedback
    return generator, feedback, flow_array, sampled


def delayed_feedback_map(flow, feedback, lag, backend):
    """Keep the last lag held torques; unused older queue coordinates only add zero modes."""
    if type(lag) is not int or not 0 <= lag <= 4:
        raise ValueError("lag must be an integer from zero through four")
    if lag == 0:
        return flow[:14, :14] + flow[:14, 14:] @ feedback
    result = np.full((14 + 7 * lag, 14 + 7 * lag), backend.arb(0), dtype=object)
    result[:14, :14], result[:14, 14:21] = flow[:14, :14], flow[:14, 14:]
    if lag > 1:
        result[14:-7, 21:] = point(np.eye(7 * (lag - 1)), backend)
    result[-7:, :14] = feedback
    return result


def enclose_sampled_stability(
    context, actual_parameters, *, control_dt=0.01, lag=0, precision_bits=128
):
    """Isolate sampled Jacobian eigenvalues at a matched-mass rest point.

    Mass equality establishes the exact equilibrium for the declared law; viscous
    parameters may differ because their force vanishes at zero velocity. An
    unstable multiplier refutes local asymptotic stability here, not every
    possible larger invariant set or any measured physical trajectory.
    """
    if not isinstance(context, PandaStructuredContext):
        raise ValueError("a structured context is required")
    parameters = readonly(parameters8(actual_parameters, "actual_parameters"))
    if parameters[0] != context.parameters[0]:
        raise ValueError("the equilibrium screen requires matched payload mass")
    if not np.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be finite and positive")
    if type(lag) is not int or not 0 <= lag <= 4:
        raise ValueError("lag must be an integer from zero through four")
    source_hash = model_provenance()["files"]["panda.xml"]
    model = _checked_packaged_model()
    with arithmetic(precision_bits) as backend:
        (nominal, g0), (payload, gp) = _evaluate(
            model,
            context.anchor,
            np.zeros(7),
            backend.arb,
            backend.fmpq,
            include_gravity_q=True,
        )
        mass = _export(
            balls(nominal.mass, backend)
            + exact(parameters[0], backend) * balls(payload.mass, backend)
        )
        gravity = _export(balls(g0, backend) + exact(parameters[0], backend) * balls(gp, backend))
        equilibrium_torque = _export(
            balls(nominal.gravity, backend)
            + exact(parameters[0], backend) * balls(payload.gravity, backend)
        )
        generator, feedback, flow, sampled = held_feedback_map(
            mass,
            gravity,
            model.dof_damping,
            parameters[1:],
            context.parameters[1:],
            kp=context.kp,
            kd=context.kd,
            control_dt=control_dt,
            backend=backend,
        )
        sampled = delayed_feedback_map(flow, feedback, lag, backend)
        eigenvalues = backend.acb_mat(_matrix(sampled, backend)).eig(algorithm="rump")
        if len(eigenvalues) != 14 + 7 * lag or any(not x.is_finite() for x in eigenvalues):
            raise ValueError("sampled eigenvalue isolation is incomplete")
        # Sorting is only for report order; it never supplies a spectral decision.
        eigenvalues.sort(key=lambda x: (float(x.real.mid()), float(x.imag.mid())))
        real, imag, modulus = [
            _export(np.array([f(x) for x in eigenvalues], dtype=object))
            for f in (lambda x: x.real, lambda x: x.imag, abs)
        ]
    if model_provenance()["files"]["panda.xml"] != source_hash:
        raise ValueError("model source changed during sampled stability evaluation")
    return SampledStabilityEnclosure(
        mass,
        gravity,
        equilibrium_torque,
        _export_at(generator, precision_bits),
        _export_at(feedback, precision_bits),
        _export_at(flow, precision_bits),
        _export_at(sampled, precision_bits),
        real,
        imag,
        modulus,
        lag,
        precision_bits,
        source_hash,
        _compiled_signature(model),
    )


def _export_at(values, precision_bits):
    # Export directed Arb endpoints under the same working precision as the calculation.
    with arithmetic(precision_bits):
        return _export(values)
