"""Direct state-box kinematics for the normalized Panda model, evaluated offline."""

from __future__ import annotations

from dataclasses import dataclass

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
