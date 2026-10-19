"""Retain one continuous payload cell across every interval and causal delay branch."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from sarrl.safety.panda_center_arithmetic import _outward
from sarrl.safety.panda_prefix_audit import audit_prefix
from sarrl.safety.panda_structured_backup import PandaParameterBox
from sarrl.safety.panda_structured_prefix import PandaPrefixResult
from sarrl.safety.panda_validated_dynamics import mass_cells
from sarrl.safety.panda_validated_prefix import PandaValidatedPrefix


@dataclass(frozen=True)
class PandaParameterPrefixCase:
    parameters: PandaParameterBox
    prefix: PandaPrefixResult


@dataclass(frozen=True)
class PandaParameterPrefixResult:
    cases: tuple[PandaParameterPrefixCase, ...]

    @property
    def prefix_contained(self):
        return bool(self.cases) and all(case.prefix.prefix_contained for case in self.cases)

    @property
    def terminal_verified(self):
        return False


def parameter_cells(parameters, count):
    """Cover exact rational mass cells by outward binary64 subboxes, preserving all friction."""
    result = []
    for low, high in mass_cells(parameters, count):
        lower, upper = parameters.lower.copy(), parameters.upper.copy()
        lower[0], upper[0] = _outward(low, upper=False), _outward(high, upper=True)
        result.append(PandaParameterBox(lower, upper))
    return tuple(result)


def _checker(template, parameters):
    if not isinstance(template, PandaValidatedPrefix):
        raise ValueError("parameter prefixes require a validated prefix checker")
    return PandaValidatedPrefix(
        template.region,
        parameters,
        template.positions.lower,
        template.positions.upper,
        template.velocities.upper,
        template.torques.upper,
        template.errors,
        template.config,
        precision_bits=template.precision_bits,
    )


def verify_parameter_prefix(
    checker, context, measured_q, measured_v, candidate, history, *, intervals, cells=16
):
    """Verify every cell without merging its physical state into another parameter branch.

    The law and estimate stay identical across cells. This is a universal check,
    not a controller supplied with the true payload or delay. A single failed
    cell/lag keeps the complete result unverified.
    """
    cases = []
    for parameters in parameter_cells(checker.parameters, cells):
        result = _checker(checker, parameters).verify_prefix(
            context,
            measured_q,
            measured_v,
            candidate,
            history,
            intervals=intervals,
        )
        cases.append(PandaParameterPrefixCase(parameters, result))
    return PandaParameterPrefixResult(tuple(cases))


def audit_parameter_prefix(
    checker, context, measured_q, measured_v, candidate, history, result, *, intervals, cells=16
):
    """Verify exact rational coverage and replay each cell's queue/propagation records."""
    targets = mass_cells(checker.parameters, cells)
    if len(result.cases) != len(targets):
        raise ValueError("parameter prefix audit: missing payload cell")
    checked = 0
    for case, (low, high) in zip(result.cases, targets, strict=True):
        parameters = case.parameters
        if (
            Fraction(float(parameters.lower[0])) > low
            or Fraction(float(parameters.upper[0])) < high
            or np.any(parameters.lower < checker.parameters.lower)
            or np.any(parameters.upper > checker.parameters.upper)
            or not np.array_equal(parameters.lower[1:], checker.parameters.lower[1:])
            or not np.array_equal(parameters.upper[1:], checker.parameters.upper[1:])
        ):
            raise ValueError("parameter prefix audit: incomplete parameter coverage")
        replay = audit_prefix(
            _checker(checker, parameters),
            context,
            measured_q,
            measured_v,
            candidate,
            history,
            case.prefix,
            intervals=intervals,
        )
        checked += replay["checked_intervals"]
    if result.terminal_verified:
        raise ValueError("parameter prefix audit: no terminal witness is established")
    return {
        "checked_parameter_cells": len(result.cases),
        "checked_intervals": checked,
        "prefix_contained": result.prefix_contained,
        "terminal_verified": False,
        "physical_binding_verified": False,
    }
