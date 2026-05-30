"""Mock LLM parameter client returning canned payloads for all DexToolBench tasks."""

from __future__ import annotations

from typing import Any, Dict, Optional

# Canned geometric goal payloads for each DexToolBench task label.
# Geometry is reasoned from the physical task: approach/tool_axis in object frame,
# lift heights in metres matching typical manipulation distances.
_TASK_PAYLOADS: Dict[str, Dict[str, Any]] = {
    "swing_down": {
        "schema_version": "v1",
        "task_label": "swing_down",
        "object_frame": "claw_hammer_head",
        "contact_point_object": [0.0, 0.0, 0.0],
        "approach_direction_object": [0.0, 0.0, -1.0],
        "tool_axis_object": [1.0, 0.0, 0.0],
        "pregrasp_offset_m": 0.08,
        "grasp_depth_m": 0.02,
        "lift_height_m": 0.10,
        "timing_s": {"approach": 1.2, "close": 0.4, "lift": 0.8},
    },
    "swing_side": {
        "schema_version": "v1",
        "task_label": "swing_side",
        "object_frame": "claw_hammer_head",
        "contact_point_object": [0.0, 0.0, 0.0],
        "approach_direction_object": [0.0, -1.0, 0.0],
        "tool_axis_object": [1.0, 0.0, 0.0],
        "pregrasp_offset_m": 0.08,
        "grasp_depth_m": 0.02,
        "lift_height_m": 0.08,
        "timing_s": {"approach": 1.2, "close": 0.4, "lift": 0.8},
    },
    "draw_smile": {
        "schema_version": "v1",
        "task_label": "draw_smile",
        "object_frame": "marker_tip",
        "contact_point_object": [0.0, 0.0, 0.0],
        "approach_direction_object": [0.0, 0.0, -1.0],
        "tool_axis_object": [0.0, 1.0, 0.0],
        "pregrasp_offset_m": 0.06,
        "grasp_depth_m": 0.02,
        "lift_height_m": 0.06,
        "timing_s": {"approach": 1.0, "close": 0.3, "lift": 0.6},
    },
    "write_c": {
        "schema_version": "v1",
        "task_label": "write_c",
        "object_frame": "marker_tip",
        "contact_point_object": [0.0, 0.0, 0.0],
        "approach_direction_object": [0.0, 0.0, -1.0],
        "tool_axis_object": [0.0, 1.0, 0.0],
        "pregrasp_offset_m": 0.06,
        "grasp_depth_m": 0.02,
        "lift_height_m": 0.06,
        "timing_s": {"approach": 1.0, "close": 0.3, "lift": 0.6},
    },
    "wipe_smile": {
        "schema_version": "v1",
        "task_label": "wipe_smile",
        "object_frame": "eraser_face",
        "contact_point_object": [0.0, 0.0, 0.0],
        "approach_direction_object": [0.0, 0.0, -1.0],
        "tool_axis_object": [1.0, 0.0, 0.0],
        "pregrasp_offset_m": 0.05,
        "grasp_depth_m": 0.02,
        "lift_height_m": 0.05,
        "timing_s": {"approach": 1.0, "close": 0.3, "lift": 0.5},
    },
    "wipe_c": {
        "schema_version": "v1",
        "task_label": "wipe_c",
        "object_frame": "eraser_face",
        "contact_point_object": [0.0, 0.0, 0.0],
        "approach_direction_object": [0.0, 0.0, -1.0],
        "tool_axis_object": [1.0, 0.0, 0.0],
        "pregrasp_offset_m": 0.05,
        "grasp_depth_m": 0.02,
        "lift_height_m": 0.05,
        "timing_s": {"approach": 1.0, "close": 0.3, "lift": 0.5},
    },
    "sweep_forward": {
        "schema_version": "v1",
        "task_label": "sweep_forward",
        "object_frame": "brush_head",
        "contact_point_object": [0.0, 0.0, 0.0],
        "approach_direction_object": [0.0, -1.0, 0.0],
        "tool_axis_object": [0.0, 0.0, 1.0],
        "pregrasp_offset_m": 0.07,
        "grasp_depth_m": 0.02,
        "lift_height_m": 0.08,
        "timing_s": {"approach": 1.1, "close": 0.4, "lift": 0.7},
    },
    "sweep_right": {
        "schema_version": "v1",
        "task_label": "sweep_right",
        "object_frame": "brush_head",
        "contact_point_object": [0.0, 0.0, 0.0],
        "approach_direction_object": [0.0, -1.0, 0.0],
        "tool_axis_object": [1.0, 0.0, 0.0],
        "pregrasp_offset_m": 0.07,
        "grasp_depth_m": 0.02,
        "lift_height_m": 0.08,
        "timing_s": {"approach": 1.1, "close": 0.4, "lift": 0.7},
    },
    "serve_plate": {
        "schema_version": "v1",
        "task_label": "serve_plate",
        "object_frame": "spatula_blade",
        "contact_point_object": [0.0, 0.0, 0.0],
        "approach_direction_object": [0.0, 0.0, -1.0],
        "tool_axis_object": [1.0, 0.0, 0.0],
        "pregrasp_offset_m": 0.06,
        "grasp_depth_m": 0.02,
        "lift_height_m": 0.07,
        "timing_s": {"approach": 1.0, "close": 0.3, "lift": 0.6},
    },
    "flip_over": {
        "schema_version": "v1",
        "task_label": "flip_over",
        "object_frame": "spatula_blade",
        "contact_point_object": [0.0, 0.0, 0.0],
        "approach_direction_object": [0.0, 0.0, -1.0],
        "tool_axis_object": [0.0, 1.0, 0.0],
        "pregrasp_offset_m": 0.06,
        "grasp_depth_m": 0.02,
        "lift_height_m": 0.09,
        "timing_s": {"approach": 1.0, "close": 0.3, "lift": 0.8},
    },
    "spin_vertical": {
        "schema_version": "v1",
        "task_label": "spin_vertical",
        "object_frame": "screwdriver_tip",
        "contact_point_object": [0.0, 0.0, 0.0],
        "approach_direction_object": [0.0, 0.0, -1.0],
        "tool_axis_object": [0.0, 1.0, 0.0],
        "pregrasp_offset_m": 0.07,
        "grasp_depth_m": 0.02,
        "lift_height_m": 0.08,
        "timing_s": {"approach": 1.0, "close": 0.3, "lift": 0.7},
    },
    "spin_horizontal": {
        "schema_version": "v1",
        "task_label": "spin_horizontal",
        "object_frame": "screwdriver_tip",
        "contact_point_object": [0.0, 0.0, 0.0],
        "approach_direction_object": [0.0, -1.0, 0.0],
        "tool_axis_object": [0.0, 0.0, 1.0],
        "pregrasp_offset_m": 0.07,
        "grasp_depth_m": 0.02,
        "lift_height_m": 0.08,
        "timing_s": {"approach": 1.0, "close": 0.3, "lift": 0.7},
    },
}

# Fall-through default task when no keyword matches.
_DEFAULT_TASK = "swing_down"


# Select the payload whose task label appears as a substring of the instruction.
def _match_task(instruction: str) -> str:
    """Return the task label whose key is a substring of instruction (lowercased)."""
    lowered = instruction.lower()
    for label in _TASK_PAYLOADS:
        if label in lowered:
            return label
    return _DEFAULT_TASK


class MockLLMParamClient:
    """Return canned geometric goal payloads without any network calls.

    Matches task labels by substring and falls back to 'swing_down'.
    Signature-compatible with OpenAIParamClient.generate_raw_params().
    """

    # Perform keyword matching and return the corresponding canned payload.
    def generate_raw_params(
        self, user_instruction: str, scene_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Return a deep-copy of the canned payload for the matched task label."""
        label = _match_task(user_instruction)
        # Return a shallow copy so callers cannot mutate the registry.
        return dict(_TASK_PAYLOADS[label])
