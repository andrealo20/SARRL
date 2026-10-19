"""A separate backup context whose viscous compensation uses a physical lower bound."""

from __future__ import annotations

import numpy as np

from sarrl.safety.panda_structured_backup import PandaParameterBox, PandaStructuredContext


def bounded_friction_context(context, parameters):
    """Cap backup viscous feedforward without modifying the retained estimator.

    Each compensated coefficient is at most the declared physical lower bound.
    This prevents compensation from exceeding known viscous dissipation. It
    does not establish sampled/delayed nonlinear stability or a physical bound.
    """
    if not isinstance(context, PandaStructuredContext) or not isinstance(
        parameters, PandaParameterBox
    ):
        raise ValueError("a structured context and physical parameter box are required")
    effective = context.parameters.copy()
    effective[1:] = np.minimum(effective[1:], parameters.lower[1:])
    return PandaStructuredContext(context.anchor, effective, kp=context.kp, kd=context.kd)
