"""Typed schema and validation for geometric goal parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ..errors import SchemaValidationError

SE3PoseSequence = List[Tuple[float, float, float, float, float, float, float]]


@dataclass(frozen=True)
class GeometricGoalV1:
    """Represent validated geometric parameters for deterministic goal conversion."""

    schema_version: str
    task_label: str
    object_frame: str
    contact_point_object: Tuple[float, float, float]
    approach_direction_object: Tuple[float, float, float]
    tool_axis_object: Tuple[float, float, float]
    pregrasp_offset_m: float
    grasp_depth_m: float
    lift_height_m: float
    timing_s: Dict[str, float]


# Validate a payload dict and return a typed GeometricGoalV1 object.
def validate_geometric_goal_v1(payload: Dict[str, Any]) -> GeometricGoalV1:
    """Validate the canonical v1 schema for geometric goal parameters."""
    if not isinstance(payload, dict):
        raise SchemaValidationError("Geometric goal payload must be a dictionary.")

    # Require versioned schema to avoid silent contract drift.
    schema_version = payload.get("schema_version")
    if schema_version != "v1":
        raise SchemaValidationError("Expected schema_version='v1'.")

    # Validate mandatory string identifiers.
    task_label = _validate_nonempty_string("task_label", payload.get("task_label"))
    object_frame = _validate_nonempty_string("object_frame", payload.get("object_frame"))

    # Validate geometric vectors in object frame.
    contact_point_object = _validate_vector3(
        "contact_point_object", payload.get("contact_point_object")
    )
    approach_direction_object = _validate_vector3(
        "approach_direction_object", payload.get("approach_direction_object")
    )
    tool_axis_object = _validate_vector3("tool_axis_object", payload.get("tool_axis_object"))

    # Validate scalar distances in meters.
    pregrasp_offset_m = _validate_nonnegative_float(
        "pregrasp_offset_m", payload.get("pregrasp_offset_m")
    )
    grasp_depth_m = _validate_nonnegative_float("grasp_depth_m", payload.get("grasp_depth_m"))
    lift_height_m = _validate_nonnegative_float("lift_height_m", payload.get("lift_height_m"))

    # Validate timing dictionary with required positive durations.
    timing_s = _validate_timing_s(payload.get("timing_s"))

    return GeometricGoalV1(
        schema_version=schema_version,
        task_label=task_label,
        object_frame=object_frame,
        contact_point_object=contact_point_object,
        approach_direction_object=approach_direction_object,
        tool_axis_object=tool_axis_object,
        pregrasp_offset_m=pregrasp_offset_m,
        grasp_depth_m=grasp_depth_m,
        lift_height_m=lift_height_m,
        timing_s=timing_s,
    )


# Validate that a field is a non-empty string.
def _validate_nonempty_string(name: str, value: Any) -> str:
    """Return a stripped non-empty string or raise a schema error."""
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"Field '{name}' must be a non-empty string.")
    return value.strip()


# Validate a 3D numeric vector and coerce it to a float tuple.
def _validate_vector3(name: str, value: Any) -> Tuple[float, float, float]:
    """Return a validated 3D vector as a float tuple."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise SchemaValidationError(f"Field '{name}' must be a list/tuple of length 3.")

    # Coerce values to floats and reject invalid entries.
    values: List[float] = []
    for idx, item in enumerate(value):
        try:
            values.append(float(item))
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(
                f"Field '{name}' has a non-numeric value at index {idx}."
            ) from exc

    return (values[0], values[1], values[2])


# Validate a scalar field that must be a non-negative float.
def _validate_nonnegative_float(name: str, value: Any) -> float:
    """Return a non-negative float or raise a schema error."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"Field '{name}' must be numeric.") from exc

    if parsed < 0.0:
        raise SchemaValidationError(f"Field '{name}' must be non-negative.")
    return parsed


# Validate timing fields and enforce required positive durations.
def _validate_timing_s(value: Any) -> Dict[str, float]:
    """Return validated approach/close/lift timing values in seconds."""
    if not isinstance(value, dict):
        raise SchemaValidationError("Field 'timing_s' must be a dictionary.")

    required_keys = ("approach", "close", "lift")
    parsed: Dict[str, float] = {}
    for key in required_keys:
        if key not in value:
            raise SchemaValidationError(f"Field 'timing_s' is missing key '{key}'.")
        try:
            parsed_value = float(value[key])
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(f"Field 'timing_s.{key}' must be numeric.") from exc
        if parsed_value <= 0.0:
            raise SchemaValidationError(f"Field 'timing_s.{key}' must be > 0.")
        parsed[key] = parsed_value

    return parsed
