"""Unit tests for geometric goal schema validation."""

import pytest

from llm_runtime.errors import SchemaValidationError
from llm_runtime.goals.schema import validate_geometric_goal_v1


# Build a valid canonical schema payload for reuse across tests.
def _valid_payload() -> dict:
    """Return a minimal valid v1 payload for schema tests."""
    return {
        "schema_version": "v1",
        "task_label": "swing_down",
        "object_frame": "claw_hammer_head",
        "contact_point_object": [0.01, 0.02, 0.03],
        "approach_direction_object": [0.0, 0.0, -1.0],
        "tool_axis_object": [1.0, 0.0, 0.0],
        "pregrasp_offset_m": 0.08,
        "grasp_depth_m": 0.02,
        "lift_height_m": 0.05,
        "timing_s": {"approach": 1.2, "close": 0.4, "lift": 0.8},
    }


# Verify a valid payload parses into the typed schema object.
def test_validate_geometric_goal_v1_success() -> None:
    """Ensure valid payloads are accepted and normalized."""
    parsed = validate_geometric_goal_v1(_valid_payload())
    assert parsed.schema_version == "v1"
    assert parsed.task_label == "swing_down"
    assert parsed.contact_point_object == (0.01, 0.02, 0.03)


# Verify version mismatch fails validation.
def test_validate_geometric_goal_v1_rejects_wrong_version() -> None:
    """Ensure schema version mismatch is rejected."""
    payload = _valid_payload()
    payload["schema_version"] = "v2"
    with pytest.raises(SchemaValidationError, match="schema_version"):
        validate_geometric_goal_v1(payload)


# Verify non-3D vectors fail validation.
def test_validate_geometric_goal_v1_rejects_bad_vector_shape() -> None:
    """Ensure vector fields require exactly three numeric values."""
    payload = _valid_payload()
    payload["approach_direction_object"] = [1.0, 2.0]
    with pytest.raises(SchemaValidationError, match="approach_direction_object"):
        validate_geometric_goal_v1(payload)


# Verify timing dictionary requires all mandatory keys.
def test_validate_geometric_goal_v1_rejects_missing_timing_key() -> None:
    """Ensure timing_s includes approach/close/lift keys."""
    payload = _valid_payload()
    payload["timing_s"] = {"approach": 1.0, "close": 0.5}
    with pytest.raises(SchemaValidationError, match="timing_s"):
        validate_geometric_goal_v1(payload)


# Verify negative scalar distances fail validation.
def test_validate_geometric_goal_v1_rejects_negative_distance() -> None:
    """Ensure distance-like scalar fields are non-negative."""
    payload = _valid_payload()
    payload["lift_height_m"] = -0.01
    with pytest.raises(SchemaValidationError, match="lift_height_m"):
        validate_geometric_goal_v1(payload)
