"""Direct state-box kinematics for the normalized Panda model, evaluated offline."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

from sarrl.robots.panda import model_provenance
from sarrl.safety.panda_analytic_region import _checked_packaged_model
from sarrl.safety.panda_center_arithmetic import (
    CenterCoefficientEnclosure,
    _compiled_signature,
    _evaluate,
)
from sarrl.safety.panda_state_region import EnclosureUnavailable, VectorBox
from sarrl.safety.panda_validated_dynamics import PandaValidatedDynamics, arithmetic, balls


@dataclass(frozen=True)
class PandaLocalCoefficients:
    nominal: CenterCoefficientEnclosure
    payload: CenterCoefficientEnclosure
    q_domain: VectorBox
    v_domain: VectorBox
    precision_bits: int
    backend_version: str
    model_xml_sha256: str
    compiled_constants_sha256: str


def enclose_local_coefficients(q_domain, v_domain, *, precision_bits=128):
    """Evaluate normalized geometry and signed Christoffel sums on continuous boxes.

    No global gravity or mass-Hessian majorant enters the bias. Ball dependency
    loss remains possible. This is a coefficient enclosure, not a physical or
    terminal certificate.
    """
    if not isinstance(q_domain, VectorBox) or not isinstance(v_domain, VectorBox):
        raise ValueError("local domains must be VectorBox instances")
    source_hash = model_provenance()["files"]["panda.xml"]
    model = _checked_packaged_model()
    signature = _compiled_signature(model)
    with arithmetic(precision_bits) as backend:
        nominal, payload = _evaluate(
            model,
            None,
            None,
            backend.arb,
            backend.fmpq,
            state_arrays=(balls(q_domain, backend), balls(v_domain, backend)),
        )
    if model_provenance()["files"]["panda.xml"] != source_hash:
        raise ValueError("model source changed during local coefficient evaluation")
    return PandaLocalCoefficients(
        nominal,
        payload,
        q_domain,
        v_domain,
        precision_bits,
        backend.__version__,
        source_hash,
        signature,
    )


class PandaLocalDynamics(PandaValidatedDynamics):
    """Use direct local coefficient boxes with the existing validated solve and FIFO."""

    def coefficients(self, q, v):
        if not self.q_domain.contains(q) or not self.v_domain.contains(v):
            raise EnclosureUnavailable("state_region_not_covered")
        return enclose_local_coefficients(q, v, precision_bits=self.precision_bits)


class PandaSnapshotDynamics(PandaValidatedDynamics):
    """Offline compiled snapshot with bounded memoization of identical coefficient boxes.

    The snapshot stays bound to its captured model/source identity. It does not
    reload assets on every query or follow later disk changes. Campaign callers
    must check source/asset provenance before and after their run. Used compiled
    constants are checked on every query, including cache hits. No real-time
    execution claim is made.
    """

    def __init__(self, *args, cache_size=128, **kwargs):
        super().__init__(*args, **kwargs)
        if type(cache_size) is not int or not 0 <= cache_size <= 512:
            raise ValueError("cache_size must be an integer from zero through 512")
        self._source_hash = model_provenance()["files"]["panda.xml"]
        self._snapshot = _checked_packaged_model()
        self._signature = _compiled_signature(self._snapshot)
        self._cache_size, self._cache = cache_size, OrderedDict()
        self._cache_lock = RLock()

    def coefficients(self, q, v):
        if not self.q_domain.contains(q) or not self.v_domain.contains(v):
            raise EnclosureUnavailable("state_region_not_covered")
        key = (
            self.precision_bits,
            *(array.tobytes() for array in (q.lower, q.upper, v.lower, v.upper)),
        )
        with self._cache_lock, arithmetic(self.precision_bits) as backend:
            if _compiled_signature(self._snapshot) != self._signature:
                raise ValueError("compiled snapshot changed after capture")
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            nominal, payload = _evaluate(
                self._snapshot,
                None,
                None,
                backend.arb,
                backend.fmpq,
                state_arrays=(balls(q, backend), balls(v, backend)),
            )
            result = PandaLocalCoefficients(
                nominal,
                payload,
                q,
                v,
                self.precision_bits,
                backend.__version__,
                self._source_hash,
                self._signature,
            )
            if self._cache_size:
                self._cache[key] = result
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            return result
