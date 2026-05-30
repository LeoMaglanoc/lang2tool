"""User-facing chat presentation helpers shared by the LLM eval entrypoints."""

from __future__ import annotations

from typing import Optional, Sequence

from llm_runtime import ToolCommand, ToolCommandIntent

_CM_PER_M = 100.0
_AXIS_EPSILON_M = 1e-6

DEFAULT_ASSISTANT_GREETING = "🤖 Hey, I'm your friendly robot assistant. How can I help you?"


# Format one non-negative translation magnitude in meters as rounded centimeters for chat.
def _format_cm(distance_m: float) -> str:
    """Return one user-facing centimeter string."""
    centimeters = round(abs(float(distance_m)) * _CM_PER_M)
    return f"{centimeters} cm"


# Choose one emoji prefix for a user-facing movement direction phrase.
def _movement_emoji(direction_text: str) -> str:
    """Return one direction-aware emoji prefix for movement summaries."""
    if "left" in direction_text:
        return "⬅️"
    if "right" in direction_text:
        return "➡️"
    if "forward" in direction_text:
        return "⬆️"
    if "backward" in direction_text:
        return "⬇️"
    if "up" in direction_text:
        return "⬆️"
    if "down" in direction_text:
        return "⬇️"
    return "🛠️"


# Convert one translation delta into a short human-facing direction phrase.
def _describe_translation_delta(
    delta_translation_m: Optional[Sequence[float]],
    frame: Optional[str],
) -> Optional[str]:
    """Return a human-facing movement phrase for one translation delta."""
    if delta_translation_m is None:
        return None
    dx, dy, dz = [float(value) for value in list(delta_translation_m)[:3]]
    axis_candidates = [
        (abs(dx), dx, "to the left", "to the right"),
        (abs(dy), dy, "forward", "backward"),
        (abs(dz), dz, "up", "down"),
    ]
    axis_magnitude, signed_value, negative_text, positive_text = max(
        axis_candidates, key=lambda item: item[0]
    )
    if axis_magnitude <= _AXIS_EPSILON_M:
        return None
    if frame == "camera_spawn":
        direction = negative_text if signed_value < 0.0 else positive_text
    else:
        direction = "along the requested frame direction"
    return f"{_movement_emoji(direction)} I moved the tool {_format_cm(axis_magnitude)} {direction} for you."


# Choose one emoji prefix for a natural-language strike target description.
def _target_emoji(target_description: str) -> str:
    """Return one target-aware emoji prefix for strike-target summaries."""
    lowered = target_description.lower()
    if "left" in lowered:
        return "⬅️"
    if "right" in lowered:
        return "➡️"
    if "up" in lowered:
        return "⬆️"
    if "down" in lowered:
        return "⬇️"
    return "🎯"


# Return one compact human-facing tool name for chat summaries.
def _tool_label(object_name: Optional[str]) -> str:
    """Return a user-facing tool label for one internal object identifier."""
    if object_name in {"claw_hammer", "mallet_hammer", "cuboid_hammer_v014"}:
        return "hammer"
    if object_name in {
        "long_screwdriver",
        "short_screwdriver",
        "cylinder_screwdriver_v3009",
    }:
        return "screwdriver"
    if object_name:
        return object_name
    return "tool"


# Return the user-facing Lie motion name implied by one selected object.
def _lie_motion_label(object_name: Optional[str]) -> str:
    """Return twist for screwdriver Lie commands and swing otherwise."""
    if object_name in {
        "long_screwdriver",
        "short_screwdriver",
        "cylinder_screwdriver_v3009",
    }:
        return "twist"
    return "swing"


# Build one user-facing chat summary for a queued tool command.
def format_chat_command_summary(command: ToolCommand) -> str:
    """Return a friendly human-facing summary of one tool command."""
    if command.intent == ToolCommandIntent.GRASP_TOOL:
        return "🤖 I'm re-grasping the tool for you."
    if command.intent == ToolCommandIntent.RELEASE_TOOL:
        return "🖐️ I'm releasing the tool now."
    if command.intent == ToolCommandIntent.SET_GOALS:
        goal_count = len(command.goals or [])
        if goal_count <= 1:
            return "🎯 I updated the goal pose for you."
        return f"🛤️ I updated the goal trajectory with {goal_count} waypoints."
    if command.intent == ToolCommandIntent.EXECUTE_LIE_TRAJECTORY:
        if command.target_description:
            if command.object_name is not None:
                return (
                    f"{_target_emoji(command.target_description)} I chose the "
                    f"{_tool_label(command.object_name)} and set up a "
                    f"{_lie_motion_label(command.object_name)} trajectory toward "
                    f"{command.target_description}."
                )
            return (
                f"{_target_emoji(command.target_description)} I set up a swing trajectory "
                f"toward {command.target_description}."
            )
        if command.strike_target_xy is not None and len(command.strike_target_xy) == 2:
            x_value = float(command.strike_target_xy[0])
            y_value = float(command.strike_target_xy[1])
            if command.object_name is not None:
                return (
                    f"🎯 I chose the {_tool_label(command.object_name)} and set up a "
                    f"{_lie_motion_label(command.object_name)} trajectory toward the requested "
                    f"strike point at x={x_value:.2f}, y={y_value:.2f}."
                )
            return (
                "🎯 I set up a swing trajectory toward the requested strike point "
                f"at x={x_value:.2f}, y={y_value:.2f}."
            )
        return f"🎯 I set up the {_lie_motion_label(command.object_name)} trajectory for you."
    if command.intent == ToolCommandIntent.EXECUTE_PREDEFINED_SWING:
        if command.object_name is not None:
            return f"🔁 I chose the {_tool_label(command.object_name)} and started the predefined motion."
        return "🔁 I started the predefined motion for you."
    if command.intent == ToolCommandIntent.SWITCH_ACTIVE_OBJECT:
        if command.object_name:
            return f"🔄 I switched to {command.object_name}."
        return "🔄 I switched the active tool for you."
    if command.intent == ToolCommandIntent.MOVE_TOOL:
        translation_summary = _describe_translation_delta(
            command.delta_translation_m,
            command.delta_frame,
        )
        if translation_summary is not None:
            return translation_summary
        if command.semantic_target is not None:
            return f"🧭 I reoriented the tool toward the {command.semantic_target} pose for you."
        return "🛠️ I adjusted the tool pose for you."
    if command.intent == ToolCommandIntent.NOOP:
        return "🤖 I'm ready for the next command."
    return "🤖 I applied the requested command."
