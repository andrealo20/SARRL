"""Optional ball-arithmetic enclosures of normalized Panda center coefficients."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction
from threading import RLock

import numpy as np

from sarrl.robots.panda import PAYLOAD_RADIUS, model_provenance, vector7
from sarrl.safety.panda_analytic_region import _checked_packaged_model
from sarrl.safety.panda_backup import readonly

_ARITHMETIC_LOCK = RLock()


def _outward(value, *, upper):
    """Convert an exact rational endpoint and verify the rounding direction."""
    try:
        result = float(value)
    except OverflowError as failure:
        raise ValueError("coefficient endpoint exceeds binary64 range") from failure
    if not math.isfinite(result):
        raise ValueError("coefficient endpoint must be finite")
    direction = math.inf if upper else -math.inf
    if (Fraction(result) < value) if upper else (Fraction(result) > value):
        result = math.nextafter(result, direction)
    if not math.isfinite(result) or (
        (Fraction(result) < value) if upper else (Fraction(result) > value)
    ):
        raise ValueError("outward endpoint conversion failed")
    return result


def _endpoint(value):
    mantissa, exponent = map(int, value.man_exp())
    return Fraction(mantissa * 2**exponent) if exponent >= 0 else Fraction(mantissa, 2**-exponent)


@dataclass(frozen=True)
class ArrayEnclosure:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        lower, upper = np.asarray(self.lower, dtype=float), np.asarray(self.upper, dtype=float)
        if (
            lower.shape != upper.shape
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower > upper)
        ):
            raise ValueError("enclosure endpoints must be finite, ordered and have matching shapes")
        object.__setattr__(self, "lower", readonly(lower))
        object.__setattr__(self, "upper", readonly(upper))

    def absolute_error(self, reference):
        """Outward absolute error relative to the supplied binary64 array."""
        reference = np.asarray(reference, dtype=float)
        if reference.shape != self.lower.shape or not np.all(np.isfinite(reference)):
            raise ValueError("reference must be finite and match the enclosure shape")
        error = np.empty(reference.shape)
        for index in np.ndindex(reference.shape):
            center = Fraction(float(reference[index]))
            distance = max(
                abs(center - Fraction(float(self.lower[index]))),
                abs(center - Fraction(float(self.upper[index]))),
            )
            error[index] = _outward(distance, upper=True)
        return readonly(error)


@dataclass(frozen=True)
class CenterCoefficientEnclosure:
    mass: ArrayEnclosure
    mass_q: ArrayEnclosure
    gravity: ArrayEnclosure
    bias: ArrayEnclosure


@dataclass(frozen=True)
class PandaCenterEnclosure:
    nominal: CenterCoefficientEnclosure
    payload: CenterCoefficientEnclosure
    precision_bits: int
    backend_version: str
    model_xml_sha256: str
    compiled_constants_sha256: str
    q: np.ndarray
    velocity: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "q", readonly(vector7(self.q, "q")))
        object.__setattr__(self, "velocity", readonly(vector7(self.velocity, "velocity")))


def _export(values):
    lower, upper = np.empty(values.shape), np.empty(values.shape)
    for index in np.ndindex(values.shape):
        value = values[index]
        if not value.is_finite():
            raise ValueError("nonfinite ball coefficient")
        lower[index] = _outward(_endpoint(value.lower()), upper=False)
        upper[index] = _outward(_endpoint(value.upper()), upper=True)
    return ArrayEnclosure(lower, upper)


def _compiled_signature(model):
    digest = hashlib.sha256()
    names = (
        "body_parentid",
        "body_jntnum",
        "body_jntadr",
        "jnt_bodyid",
        "jnt_qposadr",
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_pos",
        "body_quat",
        "jnt_axis",
        "qpos0",
        "dof_armature",
        "dof_damping",
    )
    entries = [(name, np.asarray(getattr(model, name))) for name in names]
    entries += [
        ("gravity", np.asarray(model.opt.gravity)),
        ("payload_radius", np.array([PAYLOAD_RADIUS])),
    ]
    for name, values in entries:
        dtype = "<i8" if values.dtype.kind in "iu" else "<f8"
        canonical = np.ascontiguousarray(values, dtype=dtype)
        digest.update(f"{name}:{dtype}:{canonical.shape}:".encode("ascii"))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _evaluate(model, q, velocity, arb, fmpq):
    """Evaluate exact-input normalized geometry using only ball scalar operations."""

    def exact(value):
        numerator, denominator = float(value).as_integer_ratio()
        return arb(fmpq(numerator, denominator))

    def array(values):
        values = np.asarray(values)
        return np.array([exact(x) for x in values.flat], dtype=object).reshape(values.shape)

    def zeros(shape):
        return np.full(shape, arb(0), dtype=object)

    def eye(size):
        result = zeros((size, size))
        for i in range(size):
            result[i, i] = arb(1)
        return result

    def unit(values):
        result = array(values)
        norm = sum(x * x for x in result).sqrt()
        if not norm > 0:
            raise ValueError("normalized frame or axis is unverified")
        return result / norm

    def rotation(quaternion):
        w, x, y, z = unit(quaternion)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=object,
        )

    q, velocity = array(q), array(velocity)
    transforms = np.repeat(eye(4)[None, :, :], model.nbody, axis=0)
    first, second = zeros((model.nbody, 7, 4, 4)), zeros((model.nbody, 7, 7, 4, 4))
    ancestors = np.zeros((model.nbody, 7), dtype=bool)
    local_axes = [unit(axis) for axis in model.jnt_axis]
    for body in range(1, model.nbody):
        parent = model.body_parentid[body]
        joint = model.body_jntadr[body] if model.body_jntnum[body] else -1
        ancestors[body] = ancestors[parent]
        local, dlocal, ddlocal = eye(4), zeros((7, 4, 4)), zeros((7, 7, 4, 4))
        fixed = rotation(model.body_quat[body])
        local[:3, :3], local[:3, 3] = fixed, array(model.body_pos[body])
        if joint >= 0:
            ancestors[body, joint] = True
            x, y, z = local_axes[joint]
            cross = np.array([[arb(0), -z, y], [z, arb(0), -x], [-y, x, arb(0)]], dtype=object)
            square = cross @ cross
            angle = q[joint] - exact(model.qpos0[model.jnt_qposadr[joint]])
            sine, cosine = angle.sin_cos()
            local[:3, :3] = fixed @ (eye(3) + sine * cross + (1 - cosine) * square)
            dlocal[joint, :3, :3] = fixed @ (cosine * cross + sine * square)
            ddlocal[joint, joint, :3, :3] = fixed @ (-sine * cross + cosine * square)
        transforms[body] = transforms[parent] @ local
        first[body] = first[parent] @ local + transforms[parent] @ dlocal
        second[body] = (
            second[parent] @ local
            + first[parent, :, None] @ dlocal[None, :]
            + first[parent, None, :] @ dlocal[:, None]
            + transforms[parent] @ ddlocal
        )
    axes = np.column_stack(
        [transforms[b, :3, :3] @ local_axes[j] for j, b in enumerate(model.jnt_bodyid)]
    )
    axes_q = np.stack(
        [first[b, :, :3, :3] @ local_axes[j] for j, b in enumerate(model.jnt_bodyid)], axis=2
    )
    gravity_acceleration = array(model.opt.gravity)

    def accumulate(bodies, masses, inertias, armature, damping):
        mass, derivative, gravity = np.diag(armature), zeros((7, 7, 7)), zeros((7,))
        for body, body_mass, inertia in zip(bodies, masses, inertias, strict=True):
            com = np.array([*array(model.body_ipos[body]), arb(1)], dtype=object)
            linear = (first[body] @ com)[:, :3].T
            linear_q = (second[body] @ com)[:, :, :3].transpose(2, 0, 1)
            rotate = transforms[body, :3, :3]
            angular = rotate.T @ axes
            angular_q = np.swapaxes(first[body, :, :3, :3], 1, 2) @ axes + rotate.T @ axes_q
            angular[:, ~ancestors[body]], angular_q[:, :, ~ancestors[body]] = arb(0), arb(0)
            mass += body_mass * (linear.T @ linear) + angular.T @ inertia @ angular
            gravity -= body_mass * (gravity_acceleration @ linear)
            for k in range(7):
                term = (
                    body_mass * (linear_q[:, :, k].T @ linear) + angular_q[k].T @ inertia @ angular
                )
                derivative[:, :, k] += term + term.T
        bias = gravity + damping * velocity
        for i, j, k in np.ndindex((7, 7, 7)):
            gamma = (derivative[i, j, k] + derivative[i, k, j] - derivative[j, k, i]) / 2
            bias[i] += gamma * velocity[j] * velocity[k]
        return CenterCoefficientEnclosure(*map(_export, (mass, derivative, gravity, bias)))

    bodies, inertias = model.jnt_bodyid, []
    for body in bodies:
        rotate = rotation(model.body_iquat[body])
        inertias.append(rotate @ np.diag(array(model.body_inertia[body])) @ rotate.T)
    nominal = accumulate(
        bodies,
        array(model.body_mass[bodies]),
        inertias,
        array(model.dof_armature),
        array(model.dof_damping),
    )
    sphere = (arb(2) / 5) * exact(PAYLOAD_RADIUS) ** 2
    payload = accumulate(
        [model.body("attachment").id], [arb(1)], [eye(3) * sphere], zeros((7,)), zeros((7,))
    )
    return nominal, payload


def enclose_panda_center(q, velocity, *, precision_bits=128):
    """Enclose one exact-input normalized model center, outside the control loop.

    This does not validate regional Hessians, inverse bounds or prefix arithmetic.
    The process must not mutate FLINT's global precision concurrently. A lock
    serializes calls here; it cannot protect unrelated changes to that context.
    """
    q, velocity = vector7(q, "q").copy(), vector7(velocity, "velocity").copy()
    if (
        isinstance(precision_bits, bool)
        or not isinstance(precision_bits, int)
        or not 80 <= precision_bits <= 1024
    ):
        raise ValueError("precision_bits must be an integer from 80 through 1024")
    try:
        import flint
    except ImportError as failure:
        raise ImportError("Center arithmetic needs the numerics extra") from failure
    if flint.__version__ != "0.9.0":
        raise RuntimeError("Center arithmetic requires python-flint 0.9.0")
    source_hash = model_provenance()["files"]["panda.xml"]
    model = _checked_packaged_model()
    with _ARITHMETIC_LOCK, flint.ctx.workprec(precision_bits):
        nominal, payload = _evaluate(model, q, velocity, flint.arb, flint.fmpq)
    if model_provenance()["files"]["panda.xml"] != source_hash:
        raise RuntimeError("model source changed during the center audit")
    return PandaCenterEnclosure(
        nominal,
        payload,
        precision_bits,
        flint.__version__,
        source_hash,
        _compiled_signature(model),
        q,
        velocity,
    )
