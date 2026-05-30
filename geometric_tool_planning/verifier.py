"""Bounded semantic verification and repair for intent contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Tuple

from .contracts import IntentContractV0, validate_intent_contract_v0
from .errors import IntentSchemaError, IntentSemanticError

_MIN_CLEARANCE_M = 0.0
_MAX_CLEARANCE_M = 0.12


# Report the outcome of schema validation plus bounded semantic repairs.
@dataclass(frozen=True)
class VerificationReportV0:
    """Verification status for one intent-contract payload."""

    schema_valid: bool
    semantic_valid: bool
    geometric_valid: bool
    repaired: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Normalize one non-zero 2D direction vector or reject degenerate inputs.
def _normalize_direction_xy(direction_xy: Tuple[float, float]) -> Tuple[float, float]:
    """Return a unit-length direction vector."""
    norm = math.hypot(direction_xy[0], direction_xy[1])
    if norm <= 1e-8:
        raise IntentSemanticError("direction_xy must be non-zero.")
    return (direction_xy[0] / norm, direction_xy[1] / norm)


# Clamp one scalar value into a closed interval.
def _clamp(value: float, *, lower: float, upper: float) -> float:
    """Return a scalar clamped to the provided inclusive bounds."""
    return max(lower, min(upper, value))


# Validate, repair, and report one MVP intent payload before geometric refinement.
def verify_and_repair_intent_contract(
    payload: dict[str, Any],
) -> tuple[IntentContractV0, VerificationReportV0]:
    """Return a repaired typed contract plus a structured verification report."""
    try:
        contract = validate_intent_contract_v0(payload)
    except IntentSchemaError:
        raise

    warnings: list[str] = []
    repaired = False

    direction_xy = _normalize_direction_xy(contract.direction_xy)
    if direction_xy != contract.direction_xy:
        repaired = True
        warnings.append("direction_xy normalized to unit length.")

    target_hint_uv = (
        _clamp(contract.target_hint_uv[0], lower=0.0, upper=1.0),
        _clamp(contract.target_hint_uv[1], lower=0.0, upper=1.0),
    )
    if target_hint_uv != contract.target_hint_uv:
        repaired = True
        warnings.append("target_hint_uv clamped to [0, 1].")

    clearance_m = _clamp(contract.clearance_m, lower=_MIN_CLEARANCE_M, upper=_MAX_CLEARANCE_M)
    if clearance_m != contract.clearance_m:
        repaired = True
        warnings.append(f"clearance_m clamped to [{_MIN_CLEARANCE_M:.2f}, {_MAX_CLEARANCE_M:.2f}].")

    repaired_contract = IntentContractV0(
        schema_version=contract.schema_version,
        skill=contract.skill,
        direction_xy=direction_xy,
        target_hint_uv=target_hint_uv,
        clearance_m=clearance_m,
        amplitude=contract.amplitude,
    )
    report = VerificationReportV0(
        schema_valid=True,
        semantic_valid=True,
        geometric_valid=True,
        repaired=repaired,
        warnings=warnings,
    )
    return repaired_contract, report
