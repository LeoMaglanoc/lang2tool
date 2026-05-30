"""Unit tests for high-level LLM parametric goal generation orchestration."""

from __future__ import annotations

import pytest

from llm_runtime.errors import ConverterError
from llm_runtime.llm.generator import LLMParametricGoalGenerator


# Fake parameter client returning a pre-defined payload.
class _FakeParamClient:
    """Provide canned raw LLM parameters for generator tests."""

    # Store canned payload and record invocation args.
    def __init__(self, payload: dict) -> None:
        """Initialize with a fixed raw payload."""
        self._payload = payload
        self.calls = []

    # Return fixed payload regardless of instruction/context.
    def generate_raw_params(self, user_instruction: str, scene_context=None) -> dict:
        """Record call and return canned raw parameter payload."""
        self.calls.append({"user_instruction": user_instruction, "scene_context": scene_context})
        return self._payload


# Fake deterministic converter returning a canned pose sequence.
class _FakeConverter:
    """Capture validated params and return deterministic sequence output."""

    # Initialize with canned output and capture container.
    def __init__(self, sequence):
        """Initialize fake converter with expected return sequence."""
        self._sequence = sequence
        self.received = None

    # Capture params and return deterministic sequence.
    def to_pose_sequence(self, params):
        """Return canned sequence after storing received parameters."""
        self.received = params
        return self._sequence


# Build a valid raw payload for shared generator test setup.
def _valid_payload() -> dict:
    """Return a valid geometric goal payload for orchestration tests."""
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


# Verify full orchestration from raw params to converted sequence.
def test_generate_pose_sequence_success() -> None:
    """Ensure generator validates params and forwards them to converter."""
    expected_sequence = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)]
    generator = LLMParametricGoalGenerator(
        param_client=_FakeParamClient(_valid_payload()),
        converter=_FakeConverter(expected_sequence),
    )

    sequence = generator.generate_pose_sequence("do a top grasp", {"object": "hammer"})

    assert sequence == expected_sequence


# Verify converter failures are wrapped in ConverterError.
def test_generate_pose_sequence_wraps_converter_failure() -> None:
    """Ensure converter exceptions are surfaced through ConverterError."""

    # Fake converter that always fails for this test.
    class _FailingConverter:
        """Raise an exception to simulate converter runtime failure."""

        # Raise to simulate converter failure path.
        def to_pose_sequence(self, params):
            """Raise a runtime error for error-path testing."""
            raise RuntimeError("converter failed")

    generator = LLMParametricGoalGenerator(
        param_client=_FakeParamClient(_valid_payload()),
        converter=_FailingConverter(),
    )

    with pytest.raises(ConverterError, match="Failed converting"):
        generator.generate_pose_sequence("do a top grasp")
