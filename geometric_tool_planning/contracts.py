"""Typed intent-contract schema for deterministic geometric refinement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from .errors import IntentSchemaError

_SUPPORTED_AMPLITUDES = ("small", "medium", "large")


# Store the normalized, typed MVP intent payload used by the verifier and refiner.
@dataclass(frozen=True)
class IntentContractV0:
    """Typed MVP intent contract for geometric swing refinement."""

    schema_version: str
    skill: str
    direction_xy: Tuple[float, float]
    target_hint_uv: Tuple[float, float]
    clearance_m: float
    amplitude: str


# Parse and validate one fixed-length numeric vector from an untyped payload field.
def _parse_vector2(values: Any, *, field_name: str) -> Tuple[float, float]:
    """Return one finite 2D vector from a list/tuple payload field."""
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise IntentSchemaError(f"{field_name} must be a 2-element list.")
    parsed = (float(values[0]), float(values[1]))
    if any(not math.isfinite(value) for value in parsed):
        raise IntentSchemaError(f"{field_name} must contain only finite numbers.")
    return parsed


# Reject unexpected or missing top-level keys before field-by-field parsing.
def _require_keys(payload: dict[str, Any], keys: Iterable[str]) -> None:
    """Ensure the provided payload exposes all required top-level keys."""
    missing = [key for key in keys if key not in payload]
    if missing:
        raise IntentSchemaError(f"Missing required keys: {missing}.")


# Validate and coerce one untyped JSON payload into a typed MVP intent contract.
def validate_intent_contract_v0(payload: dict[str, Any]) -> IntentContractV0:
    """Return a typed `IntentContractV0` after schema validation."""
    if not isinstance(payload, dict):
        raise IntentSchemaError("Intent payload must be a JSON object.")

    _require_keys(
        payload,
        (
            "schema_version",
            "skill",
            "direction_xy",
            "target_hint_uv",
            "clearance_m",
            "amplitude",
        ),
    )

    schema_version = str(payload["schema_version"])
    if schema_version != "intent_v0":
        raise IntentSchemaError("schema_version must equal 'intent_v0'.")

    skill = str(payload["skill"])
    if skill != "swing_down":
        raise IntentSchemaError("skill must equal 'swing_down'.")

    amplitude = str(payload["amplitude"])
    if amplitude not in _SUPPORTED_AMPLITUDES:
        raise IntentSchemaError(
            f"amplitude must be one of {_SUPPORTED_AMPLITUDES}, got '{amplitude}'."
        )

    clearance_m = float(payload["clearance_m"])
    if not math.isfinite(clearance_m):
        raise IntentSchemaError("clearance_m must be finite.")

    return IntentContractV0(
        schema_version=schema_version,
        skill=skill,
        direction_xy=_parse_vector2(payload["direction_xy"], field_name="direction_xy"),
        target_hint_uv=_parse_vector2(payload["target_hint_uv"], field_name="target_hint_uv"),
        clearance_m=clearance_m,
        amplitude=amplitude,
    )
