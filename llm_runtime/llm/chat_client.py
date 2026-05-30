"""Chat-style LLM clients for the viser GUI chat panel.

Provides ``MockChatClient`` (offline, keyword-based) and ``OpenAIChatClient``
(live, tool-call-based) behind a common ``send()`` interface.  The
``build_chat_client`` factory selects the backend from a string key.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dextoolbench.llm_supported_objects import (
    SUPPORTED_LLM_OBJECT_FAMILIES,
    supported_llm_object_family,
)

from ..goals.converter import GeometricPoseConverter
from ..goals.schema import validate_geometric_goal_v1
from ..types import ToolCommand, ToolCommandIntent
from .mock_client import _DEFAULT_TASK, _TASK_PAYLOADS
from .tool_schema import (
    APPLY_GOAL_DELTA_TOOL,
    EXECUTE_LIE_TRAJECTORY_TOOL,
    EXECUTE_PREDEFINED_SWING_TOOL,
    GENERATE_GOALS_TOOL,
    GET_SIM_STATE_TOOL,
    GRASP_TOOL_COMMAND,
    RELEASE_TOOL_COMMAND,
    SWITCH_ACTIVE_OBJECT_TOOL,
)

if TYPE_CHECKING:
    from openai import OpenAI

    from .config import GoalGeneratorConfig


@dataclass
class ChatMessage:
    """A single turn in a chat conversation."""

    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class ChatResponse:
    """Response from a chat client.

    ``goals`` is a list of waypoints [[x,y,z,qx,qy,qz,qw], ...] when the
    LLM generated task goals, or ``None`` when the message was informational.
    """

    text: str
    goals: Optional[List[List[float]]] = field(default=None)
    command: Optional[ToolCommand] = field(default=None)
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)


# Return the task label if a keyword is found in instruction, else None.
def _explicit_match_task(instruction: str) -> Optional[str]:
    """Return task label if present as underscore or space-separated phrase."""
    lowered = instruction.lower()
    for label in _TASK_PAYLOADS:
        if label in lowered or label.replace("_", " ") in lowered:
            return label
    return None


# Validate canned task payload and convert to world-frame waypoint list.
def _convert_payload_to_goals(task_label: str, start_pose: List[float]) -> List[List[float]]:
    """Convert one mock task payload to a list of 7D goals."""
    raw = dict(_TASK_PAYLOADS[task_label])
    params = validate_geometric_goal_v1(raw)
    converter = GeometricPoseConverter(object_ref_pose=tuple(start_pose))
    pose_seq = converter.to_pose_sequence(params)
    return [list(wp) for wp in pose_seq]


# Coerce chat content into plain strings to avoid SDK sentinel serialization errors.
def _normalize_message_content(value: Any) -> str:
    """Return a JSON-safe text value for chat messages."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return ""


# Match the OpenAI SDK serializer failure shape seen in Isaac-Sim environments.
def _is_omit_serialization_error(exc: Exception) -> bool:
    """Return True when an exception matches the Omit JSON-serialization SDK bug."""
    text = str(exc)
    return "Omit" in text and "JSON serializable" in text


# Format one static strike-geometry payload into concise prompt text.
def _format_static_strike_context(static_context: Dict[str, Any]) -> str:
    """Return one compact instruction block describing immutable tabletop geometry."""
    if not isinstance(static_context, dict) or not static_context.get("available"):
        return ""
    table_region = static_context.get("table_target_region", {})
    named_points = static_context.get("named_strike_points", {})
    point_payload = {}
    if isinstance(named_points, dict):
        points = named_points.get("points", {})
        if isinstance(points, dict):
            point_payload = points.get("target_a", {})
    if not isinstance(point_payload, dict):
        point_payload = {}
    frame_description = str(static_context.get("frame_description", "")).strip()
    region_text = (
        f"x in [{table_region.get('x_min')}, {table_region.get('x_max')}], "
        f"y in [{table_region.get('y_min')}, {table_region.get('y_max')}]"
        if isinstance(table_region, dict)
        else ""
    )
    target_a_text = ""
    if point_payload:
        target_a_text = f" Fixed strike point a is at {point_payload.get('world_xy')}."
    return (
        f"Static tabletop targeting geometry: {frame_description} "
        f"Valid strike_target_xy bounds are {region_text}.{target_a_text} "
        "Choose explicit strike_target_xy coordinates inside those bounds."
    ).strip()


# Format a concise natural-language summary of current sim state payload.
def _format_sim_state_summary(sim_state: Dict[str, Any]) -> str:
    """Return compact state summary for chat replies."""
    object_name = str(sim_state.get("object_name", "unknown"))
    object_pose = sim_state.get("object_pose_xyzw", [])
    goal_pose = sim_state.get("goal_pose_xyzw", [])
    runtime_mode = str(sim_state.get("runtime_mode", "unknown"))
    pose_semantics = sim_state.get("pose_semantics", {})
    active_strike_target = sim_state.get("active_strike_target", {})
    semantic_targets = (
        pose_semantics.get("semantic_targets", []) if isinstance(pose_semantics, dict) else []
    )
    axes_local = pose_semantics.get("axes_local", {}) if isinstance(pose_semantics, dict) else {}
    primary_axis = axes_local.get("primary_axis")
    head_axis = axes_local.get("head_axis")
    tip_axis = axes_local.get("tip_axis")
    face_normal = axes_local.get("face_normal")
    target_text = ", ".join(str(t).replace("_", " ") for t in semantic_targets) or "none"

    summary = (
        f"The current object is {object_name} in {runtime_mode} mode. "
        f"Its current pose is {object_pose} (xyzw), and the goal pose is {goal_pose} (xyzw). "
        f"Supported semantic pose targets are {target_text}."
    )
    if all(v is not None for v in (primary_axis, head_axis, tip_axis, face_normal)):
        summary += (
            " In the object-local frame, the primary axis is "
            f"{primary_axis}, the head axis is {head_axis}, the tip axis is {tip_axis}, "
            f"and the face normal is {face_normal}."
        )
    if isinstance(active_strike_target, dict) and active_strike_target.get("available"):
        summary += (
            " The currently active Lie strike target is at "
            f"{active_strike_target.get('world_xy')}."
        )
    return summary


# Return the latest user message text from outbound chat history.
def _get_latest_user_text(messages: List[ChatMessage]) -> str:
    """Return newest user message content or empty text."""
    for msg in reversed(messages):
        if msg.role == "user":
            return str(msg.content or "")
    return ""


# Return one explicitly named supported object when user text disambiguates the instance.
def _explicit_supported_object_from_text(text: str) -> Optional[str]:
    """Return a supported object name when the user text names one specific instance."""
    lowered = str(text).lower().replace("-", " ").replace("_", " ")
    alias_map = {
        "claw_hammer": ("claw hammer",),
        "mallet_hammer": ("mallet hammer",),
        "cuboid_hammer_v014": ("cuboid hammer v014", "cuboid hammer", "primitive hammer"),
        "long_screwdriver": ("long screwdriver",),
        "short_screwdriver": ("short screwdriver",),
        "cylinder_screwdriver_v3009": (
            "cylinder screwdriver v3009",
            "cylinder screwdriver",
            "primitive screwdriver",
        ),
    }
    for object_name, aliases in alias_map.items():
        if any(alias in lowered for alias in aliases):
            return object_name
    return None


# Return one ambiguous supported family name when the text names only a family label.
def _ambiguous_supported_family_from_text(text: str) -> Optional[str]:
    """Return hammer or screwdriver when the user named only the family, not the instance."""
    lowered = str(text).lower().replace("-", " ").replace("_", " ")
    if _explicit_supported_object_from_text(lowered) is not None:
        return None
    family_aliases = {
        "hammer": (r"\bhammer\b",),
        "screwdriver": (r"\bscrewdriver\b",),
    }
    for family_name, patterns in family_aliases.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            return family_name
    return None


# Return the canonical object implied by executable motion verbs in user text.
def _motion_default_object_from_text(text: str) -> Optional[str]:
    """Return the default supported object for explicit swing or twist motion verbs."""
    lowered = str(text).lower()
    screwdriver_patterns = (
        r"\btwist(?:ing)?\b",
        r"\bscrew(?:ing)?\b",
        r"\bscrew\s+in\b",
        r"\bdrive\b",
        r"\bdriving\b",
    )
    hammer_patterns = (
        r"\bswing(?:ing)?\b",
        r"\bstrike\b(?!\s+point)",
        r"\bstriking\b(?!\s+point)",
        r"\bhit(?:ting)?\b",
    )
    if any(re.search(pattern, lowered) for pattern in screwdriver_patterns):
        return "long_screwdriver"
    if any(re.search(pattern, lowered) for pattern in hammer_patterns):
        return "claw_hammer"
    return None


# Return one clarification question when the latest user request leaves the instance ambiguous.
def _clarify_supported_family_request(text: str) -> Optional[str]:
    """Return a clarification question for ambiguous hammer/screwdriver family requests."""
    if _motion_default_object_from_text(text) is not None:
        return None
    family_name = _ambiguous_supported_family_from_text(text)
    if family_name is None:
        return None
    family_objects = SUPPORTED_LLM_OBJECT_FAMILIES[family_name]
    readable_names = " or ".join(object_name.replace("_", " ") for object_name in family_objects)
    return f"Which {family_name} do you want: {readable_names}?"


# Map user natural language to one core semantic pose ontology label.
def _semantic_target_from_text(text: str) -> Optional[str]:
    """Return ontology target name when text clearly requests semantic pose."""
    lowered = str(text).lower()
    if "upright" in lowered:
        return "upright"
    if "flat" in lowered or "lay flat" in lowered:
        return "flat"
    if "head down" in lowered:
        return "head_down"
    if "tip forward" in lowered:
        return "tip_forward"
    if "face table" in lowered or "toward the table" in lowered:
        return "face_table"
    return None


# Return whether user text appears to request a swing strike target.
def _is_swing_target_request(text: str) -> bool:
    """Return whether the text looks like a tabletop strike-placement request."""
    lowered = str(text).lower()
    swing_tokens = ("swing", "strike", "hit")
    location_tokens = ("strike point", "striking point", "side of the table", "table")
    return any(token in lowered for token in swing_tokens) and any(
        token in lowered for token in location_tokens
    )


# Return whether user text appears to request the recorded predefined swing directly.
def _is_predefined_swing_request(text: str) -> bool:
    """Return whether the text asks for the recorded predefined swing motion."""
    lowered = str(text).lower()
    return (
        "predefined swing" in lowered
        or "predefined motion" in lowered
        or "predefined screwdriver" in lowered
    )


# Return whether user text asks to retarget the active tool rather than select a tool family.
def _is_object_preserving_target_request(text: str) -> bool:
    """Return whether the request should stay on the current exact object."""
    lowered = str(text).lower()
    preserving_tokens = ("target", "aim", "place", "move", "shift", "rotate")
    return any(token in lowered for token in preserving_tokens)


# Return whether the latest predefined-motion request explicitly names one supported object instance.
def _predefined_request_explicit_object(text: str) -> Optional[str]:
    """Return one explicit supported object when the predefined request names the instance."""
    if not _is_predefined_swing_request(text):
        return None
    return _explicit_supported_object_from_text(text)


# Return whether a Lie-placement request explicitly named one supported object instance.
def _lie_request_explicit_object(text: str) -> Optional[str]:
    """Return one explicit supported object when a strike/twist request names the instance."""
    lowered = str(text).lower()
    if not (_is_swing_target_request(text) or "twist" in lowered):
        return None
    return _explicit_supported_object_from_text(text)


# Return one implicit default object selected from the request verb family.
def _default_object_for_motion_request(text: str) -> Optional[str]:
    """Return the default object implied by generic hammer or screwdriver verbs."""
    motion_object = _motion_default_object_from_text(text)
    if motion_object is not None:
        return motion_object
    if _is_object_preserving_target_request(text):
        return None
    return None


# Enforce deterministic Lie object routing after model tool-call parsing.
def _repair_lie_command_object_from_text(command: ToolCommand, text: str) -> ToolCommand:
    """Patch Lie command object_name so motion-family verbs cannot execute the wrong tool."""
    explicit_object = _explicit_supported_object_from_text(text)
    if explicit_object is not None:
        command.object_name = explicit_object
        return command
    default_object = _default_object_for_motion_request(text)
    if default_object is not None:
        command.object_name = default_object
    elif _is_object_preserving_target_request(text):
        command.object_name = None
    return command


# Resolve one mock strike target from static geometry for deterministic dev/test behavior.
def _mock_strike_target_xy_from_text(
    text: str, static_context: Dict[str, Any]
) -> Optional[List[float]]:
    """Return one deterministic mock strike target inferred from user text."""
    if not isinstance(static_context, dict) or not static_context.get("available"):
        return None
    named_points = static_context.get("named_strike_points", {})
    point_payload = {}
    if isinstance(named_points, dict):
        points = named_points.get("points", {})
        if isinstance(points, dict):
            point_payload = points.get("target_a", {})
    table_region = static_context.get("table_target_region", {})
    lowered = str(text).lower()
    if "strike point a" in lowered or "striking point a" in lowered:
        if isinstance(point_payload, dict):
            world_xy = point_payload.get("world_xy")
            if isinstance(world_xy, list) and len(world_xy) == 2:
                return [float(world_xy[0]), float(world_xy[1])]
    if "right side of the table" in lowered and isinstance(table_region, dict):
        y_mid = 0.0
        if isinstance(point_payload, dict):
            world_xy = point_payload.get("world_xy")
            if isinstance(world_xy, list) and len(world_xy) == 2:
                y_mid = float(world_xy[1])
        return [float(table_region.get("x_max", 0.0)), y_mid]
    if "left side of the table" in lowered and isinstance(table_region, dict):
        y_mid = 0.0
        if isinstance(point_payload, dict):
            world_xy = point_payload.get("world_xy")
            if isinstance(world_xy, list) and len(world_xy) == 2:
                y_mid = float(world_xy[1])
        return [float(table_region.get("x_min", 0.0)), y_mid]
    if "front of the table" in lowered and isinstance(table_region, dict):
        return [0.0, float(table_region.get("y_max", 0.0))]
    if "back of the table" in lowered and isinstance(table_region, dict):
        return [0.0, float(table_region.get("y_min", 0.0))]
    return None


class MockChatClient:
    """Offline keyword-based chat client for development / CI use."""

    # No constructor arguments needed — everything comes from pre-baked payloads.
    def __init__(self) -> None:
        pass

    # Process a chat turn and return text plus optional goal waypoints.
    def send(self, messages: List[ChatMessage], sim_context: Dict[str, Any]) -> ChatResponse:
        """Return canned goals when task labels are detected in latest user message."""
        latest_user = ""
        for msg in reversed(messages):
            if msg.role == "user":
                latest_user = msg.content
                break
        clarification_text = _clarify_supported_family_request(latest_user)
        if clarification_text is not None:
            return ChatResponse(text=clarification_text)

        current_obj = sim_context.get("current_object", "unknown")
        current_task = sim_context.get("current_task")
        start_pose = sim_context.get("start_pose", [0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 1.0])
        static_context = sim_context.get("static_strike_context", {})
        strike_target_xy = _mock_strike_target_xy_from_text(latest_user, static_context)
        if _is_predefined_swing_request(latest_user):
            tool_result = {
                "ok": True,
                "command": "execute_predefined_swing",
                "replace_active_goals": True,
                "object_name": None,
            }
            return ChatResponse(
                text="Executing the predefined motion trajectory.",
                command=ToolCommand(
                    intent=ToolCommandIntent.EXECUTE_PREDEFINED_SWING,
                    replace_active_goals=True,
                ),
                tool_trace=[
                    {
                        "tool_name": "execute_predefined_swing",
                        "arguments": {},
                        "result": tool_result,
                    }
                ],
            )
        if strike_target_xy is not None:
            tool_arguments = {
                "strike_target_xy": strike_target_xy,
                "target_description": latest_user,
                "replace_active_goals": True,
            }
            tool_result = {
                "ok": True,
                "command": "execute_lie_trajectory",
                **tool_arguments,
            }
            return ChatResponse(
                text=f"Executing Lie swing trajectory toward {strike_target_xy}.",
                command=ToolCommand(
                    intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY,
                    strike_target_xy=strike_target_xy,
                    target_description=latest_user,
                    replace_active_goals=True,
                ),
                tool_trace=[
                    {
                        "tool_name": "execute_lie_trajectory",
                        "arguments": tool_arguments,
                        "result": tool_result,
                    }
                ],
            )

        label = _explicit_match_task(latest_user)
        if label is not None:
            goals = _convert_payload_to_goals(label, start_pose)
            text = f"Setting up goals for **{label}**. Click **Run** to start the episode."
            return ChatResponse(
                text=text,
                goals=goals,
                tool_trace=[
                    {
                        "tool_name": "generate_goals",
                        "arguments": dict(_TASK_PAYLOADS[label]),
                        "result": {"ok": True, "num_goals": len(goals)},
                    }
                ],
            )

        if current_task:
            text = (
                f"Currently: **{current_obj}** / **{current_task}**. "
                f"Say a task name (e.g. '{_DEFAULT_TASK}') to generate goals."
            )
        else:
            text = (
                f"Currently: **{current_obj}**. "
                f"Say a task name (e.g. '{_DEFAULT_TASK}') to generate goals."
            )
        return ChatResponse(text=text, goals=None)


class OpenAIChatClient:
    """Live chat client backed by OpenAI's Chat Completions API with tool use."""

    # Store client and config for use in send().
    def __init__(self, openai_client: "OpenAI", config: "GoalGeneratorConfig") -> None:
        """Initialise with an authenticated OpenAI client and generation config."""
        self._client = openai_client
        self._config = config

    # Call Chat Completions over raw HTTP to bypass SDK JSON encoding sentinels.
    def _create_chat_completion_via_http(self, request_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """POST /v1/chat/completions directly and return decoded JSON payload."""
        api_key = getattr(self._client, "api_key", None) or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for HTTP fallback.")
        base_url_raw = str(getattr(self._client, "base_url", "https://api.openai.com/v1"))
        base_url = base_url_raw.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        url = f"{base_url}/chat/completions"
        body = json.dumps(request_kwargs).encode("utf-8")
        req = Request(
            url=url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI HTTP fallback failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenAI HTTP fallback connection error: {exc}") from exc

    # Create one Chat Completions request payload from current conversation state.
    def _build_request_kwargs(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return OpenAI chat-completions payload with tools and configured decoding knobs."""
        request_kwargs: Dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "tools": [
                GENERATE_GOALS_TOOL,
                APPLY_GOAL_DELTA_TOOL,
                SWITCH_ACTIVE_OBJECT_TOOL,
                EXECUTE_LIE_TRAJECTORY_TOOL,
                EXECUTE_PREDEFINED_SWING_TOOL,
                GET_SIM_STATE_TOOL,
                GRASP_TOOL_COMMAND,
                RELEASE_TOOL_COMMAND,
            ],
        }
        if self._config.temperature is not None:
            request_kwargs["temperature"] = self._config.temperature
        if self._config.max_output_tokens is not None:
            request_kwargs["max_tokens"] = self._config.max_output_tokens
        return request_kwargs

    # Execute one chat completion request via SDK and fallback to HTTP on Omit bug.
    def _create_chat_completion(self, request_kwargs: Dict[str, Any]) -> Any:
        """Return model message payload for one chat-completions request."""
        try:
            response_obj: Any = self._client.chat.completions.create(**request_kwargs)
            return response_obj.choices[0].message
        except TypeError as exc:
            if not _is_omit_serialization_error(exc):
                raise
            response_payload = self._create_chat_completion_via_http(request_kwargs)
            return response_payload["choices"][0]["message"]

    # Parse tool_call shape from dict or SDK object into id/name/arguments.
    def _parse_tool_call(self, tool_call: Any, index: int) -> tuple[str, str, Any]:
        """Return normalized (tool_call_id, function_name, function_arguments)."""
        if isinstance(tool_call, dict):
            call_id = str(tool_call.get("id", f"call_{index}"))
            function_name = str(tool_call.get("function", {}).get("name", ""))
            function_args = tool_call.get("function", {}).get("arguments")
            return call_id, function_name, function_args
        call_id = str(getattr(tool_call, "id", f"call_{index}"))
        function = getattr(tool_call, "function", None)
        function_name = str(getattr(function, "name", ""))
        function_args = getattr(function, "arguments", None)
        return call_id, function_name, function_args

    # Decode function arguments from string/dict into plain dict.
    def _parse_function_args(self, function_args: Any) -> Dict[str, Any]:
        """Return tool arguments as a dictionary."""
        if isinstance(function_args, str):
            parsed = json.loads(function_args) if function_args.strip() else {}
            if not isinstance(parsed, dict):
                raise TypeError("Tool arguments JSON must decode to an object.")
            return parsed
        if isinstance(function_args, dict):
            return function_args
        raise TypeError(f"Unexpected function arguments type: {type(function_args).__name__}")

    # Return tool arguments for audit logging without failing the chat loop.
    def _safe_parse_function_args(self, function_args: Any) -> Dict[str, Any]:
        """Return parsed tool arguments, or an error payload when parsing fails."""
        try:
            return self._parse_function_args(function_args)
        except Exception as exc:
            return {"_parse_error": f"{type(exc).__name__}: {exc}"}

    # Execute one normalized tool call and return (tool_result, optional command, optional goals).
    def _execute_tool_call(
        self,
        function_name: str,
        function_args: Any,
        *,
        start_pose: List[float],
        sim_context: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Optional[ToolCommand], Optional[List[List[float]]]]:
        """Execute one tool call locally and return structured outputs for loop continuation."""
        parsed_args = self._parse_function_args(function_args)
        if function_name == "get_sim_state":
            sim_state = sim_context.get("sim_state", {})
            if not isinstance(sim_state, dict):
                sim_state = {}
            return (
                {
                    "ok": True,
                    "sim_state": sim_state,
                    "summary": _format_sim_state_summary(sim_state),
                },
                None,
                None,
            )

        if function_name == "generate_goals":
            params = validate_geometric_goal_v1(parsed_args)
            converter = GeometricPoseConverter(object_ref_pose=tuple(start_pose))
            pose_seq = converter.to_pose_sequence(params)
            goals = [list(wp) for wp in pose_seq]
            return {"ok": True, "num_goals": len(goals)}, None, goals

        if function_name == "apply_goal_delta":
            delta_translation = parsed_args.get("delta_translation_m", [0.0, 0.0, 0.0])
            delta_euler = parsed_args.get("delta_euler_rad", [0.0, 0.0, 0.0])
            frame = str(parsed_args.get("frame", "camera_spawn"))
            semantic_target_raw = parsed_args.get("semantic_target")
            semantic_target = str(semantic_target_raw) if semantic_target_raw is not None else None
            preserve_position_raw = parsed_args.get("semantic_preserve_position")
            preserve_position = (
                bool(preserve_position_raw) if preserve_position_raw is not None else None
            )
            command = ToolCommand(
                intent=ToolCommandIntent.MOVE_TOOL,
                delta_translation_m=[float(v) for v in delta_translation],
                delta_euler_rad=[float(v) for v in delta_euler],
                delta_frame=frame,
                semantic_target=semantic_target,
                semantic_preserve_position=preserve_position,
            )
            return (
                {
                    "ok": True,
                    "command": "apply_goal_delta",
                    "delta_translation_m": command.delta_translation_m,
                    "delta_euler_rad": command.delta_euler_rad,
                    "frame": command.delta_frame,
                    "semantic_target": command.semantic_target,
                    "semantic_preserve_position": command.semantic_preserve_position,
                },
                command,
                None,
            )

        if function_name == "execute_lie_trajectory":
            strike_target_xy_raw = parsed_args.get("strike_target_xy", [])
            target_description_raw = parsed_args.get("target_description")
            replace_active_goals_raw = parsed_args.get("replace_active_goals")
            object_name_raw = parsed_args.get("object_name")
            replace_active_goals = (
                True if replace_active_goals_raw is None else bool(replace_active_goals_raw)
            )
            command = ToolCommand(
                intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY,
                strike_target_xy=[float(v) for v in strike_target_xy_raw],
                target_description=(
                    str(target_description_raw) if target_description_raw is not None else None
                ),
                replace_active_goals=replace_active_goals,
                object_name=str(object_name_raw) if object_name_raw is not None else None,
            )
            return (
                {
                    "ok": True,
                    "command": "execute_lie_trajectory",
                    "strike_target_xy": command.strike_target_xy,
                    "target_description": command.target_description,
                    "replace_active_goals": command.replace_active_goals,
                    "object_name": command.object_name,
                },
                command,
                None,
            )

        if function_name == "execute_predefined_swing":
            object_name_raw = parsed_args.get("object_name")
            command = ToolCommand(
                intent=ToolCommandIntent.EXECUTE_PREDEFINED_SWING,
                replace_active_goals=True,
                object_name=str(object_name_raw) if object_name_raw is not None else None,
            )
            return (
                {
                    "ok": True,
                    "command": "execute_predefined_swing",
                    "replace_active_goals": command.replace_active_goals,
                    "object_name": command.object_name,
                },
                command,
                None,
            )

        if function_name == "switch_active_object":
            object_name_raw = parsed_args.get("object_name")
            command = ToolCommand(
                intent=ToolCommandIntent.SWITCH_ACTIVE_OBJECT,
                object_name=str(object_name_raw) if object_name_raw is not None else None,
            )
            return (
                {
                    "ok": True,
                    "command": "switch_active_object",
                    "object_name": command.object_name,
                },
                command,
                None,
            )

        if function_name == "release_tool":
            command = ToolCommand(intent=ToolCommandIntent.RELEASE_TOOL)
            return {"ok": True, "command": "release_tool"}, command, None

        if function_name == "grasp_tool":
            command = ToolCommand(intent=ToolCommandIntent.GRASP_TOOL)
            return {"ok": True, "command": "grasp_tool"}, command, None

        return {"ok": False, "error": f"Unsupported tool: {function_name}"}, None, None

    # Send the conversation to OpenAI and return a ChatResponse.
    def send(self, messages: List[ChatMessage], sim_context: Dict[str, Any]) -> ChatResponse:
        """Generate assistant text and optional goals from OpenAI chat/tool outputs."""
        current_obj = sim_context.get("current_object", "unknown")
        current_task = sim_context.get("current_task")
        start_pose = sim_context.get("start_pose", [0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 1.0])
        latest_user_text = _get_latest_user_text(messages)
        clarification_text = _clarify_supported_family_request(latest_user_text)
        if clarification_text is not None:
            return ChatResponse(text=clarification_text)
        explicit_predefined_object = _predefined_request_explicit_object(latest_user_text)
        static_context = sim_context.get("static_strike_context", {})
        static_context_text = _format_static_strike_context(static_context)
        active_family = None
        try:
            active_family = supported_llm_object_family(str(current_obj))
        except ValueError:
            active_family = None

        system_content = (
            "You are a robot manipulation assistant. "
            f"The current object is '{current_obj}'. "
            "When semantic pose meaning is needed (e.g., upright/flat/head-down/tip-forward/face-table), "
            "call get_sim_state first to read pose and semantics metadata. "
            "For any swing-placement or twist-placement request, call get_sim_state first to read "
            "the current object pose and runtime state, then call execute_lie_trajectory with explicit "
            "strike_target_xy coordinates and the chosen object_name. "
            "Verb routing policy: target/aim/place/move/shift/rotate keeps the current exact object. "
            "Swing/strike/hit/hammer defaults to claw_hammer unless the user explicitly names a different supported object instance. "
            "Twist/screw/drive defaults to long_screwdriver unless the user explicitly names a different supported object instance, and generic twist requests should not ask for screwdriver clarification. "
            "Hammer objects are claw_hammer, mallet_hammer, and cuboid_hammer_v014. "
            "Screwdriver objects are long_screwdriver, short_screwdriver, and cylinder_screwdriver_v3009. "
            "For screwdrivers, strike_target_xy is the tabletop hover target above which the tool should twist. "
            "When the user explicitly asks for predefined motion, call execute_predefined_swing. "
            "Keep the current object for predefined motion unless the user explicitly named a different supported object instance in that same request. "
            "When the user explicitly asks to switch to one supported object, call switch_active_object. "
            "When the user asks to move/shift/rotate relative to the current goal, "
            "call apply_goal_delta. "
            "When the user asks to grasp/regrasp/pick up the current tool, call grasp_tool. "
            "When the user asks to release/open hand/drop tool, call release_tool. "
            "When the user asks to perform a new manipulation task, call generate_goals. "
            "If the user names only a family such as hammer or screwdriver and does not specify the instance, "
            "ask a short clarification question instead of calling any tool. "
            "In your assistant wording, explicitly mention which tool you selected whenever you choose one. "
            "When replying, use short natural-language sentences and avoid schema-style dumps "
            "of raw field names and arrays. Otherwise respond conversationally."
        )
        if current_task:
            system_content += f" Current task context: '{current_task}'."
        if static_context_text:
            system_content += f" {static_context_text}"
        if active_family is not None:
            system_content += (
                f" The active object family is '{active_family}', but do not assume the user wants the "
                "currently active instance when they mention only the family name."
            )

        openai_msgs: List[Dict[str, Any]] = [{"role": "system", "content": system_content}]
        for msg in messages:
            openai_msgs.append(
                {"role": msg.role, "content": _normalize_message_content(msg.content)}
            )

        goals: Optional[List[List[float]]] = None
        command: Optional[ToolCommand] = None
        assistant_text = ""
        tool_trace: List[Dict[str, Any]] = []

        max_rounds = max(1, int(getattr(self._config, "max_tool_round_trips", 5)))
        for _ in range(max_rounds):
            request_kwargs = self._build_request_kwargs(openai_msgs)
            msg_obj: Any = self._create_chat_completion(request_kwargs)
            assistant_content = _normalize_message_content(
                msg_obj.get("content")
                if isinstance(msg_obj, dict)
                else getattr(msg_obj, "content", None)
            )
            if assistant_content:
                assistant_text = assistant_content

            tool_calls = (
                msg_obj.get("tool_calls")
                if isinstance(msg_obj, dict)
                else getattr(msg_obj, "tool_calls", None)
            )
            if not tool_calls:
                break

            assistant_with_tools: Dict[str, Any] = {
                "role": "assistant",
                "content": assistant_content or "",
            }
            if isinstance(msg_obj, dict):
                assistant_with_tools["tool_calls"] = msg_obj.get("tool_calls", [])
            else:
                tool_calls_payload = []
                for idx, tc in enumerate(tool_calls):
                    call_id, fn_name, fn_args = self._parse_tool_call(tc, idx)
                    tool_calls_payload.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": (
                                    fn_args
                                    if isinstance(fn_args, str)
                                    else json.dumps(fn_args or {})
                                ),
                            },
                        }
                    )
                assistant_with_tools["tool_calls"] = tool_calls_payload
            openai_msgs.append(assistant_with_tools)

            for idx, tool_call in enumerate(tool_calls):
                call_id, function_name, function_args = self._parse_tool_call(tool_call, idx)
                trace_entry: Dict[str, Any] = {
                    "tool_name": function_name,
                    "arguments": self._safe_parse_function_args(function_args),
                }
                try:
                    tool_result, parsed_command, parsed_goals = self._execute_tool_call(
                        function_name=function_name,
                        function_args=function_args,
                        start_pose=start_pose,
                        sim_context=sim_context,
                    )
                except Exception as exc:
                    tool_result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    parsed_command = None
                    parsed_goals = None
                trace_entry["result"] = tool_result
                tool_trace.append(trace_entry)
                if parsed_goals is not None:
                    goals = parsed_goals
                if parsed_command is not None:
                    if parsed_command.intent == ToolCommandIntent.EXECUTE_LIE_TRAJECTORY:
                        parsed_command = _repair_lie_command_object_from_text(
                            parsed_command,
                            latest_user_text,
                        )
                    if (
                        parsed_command.intent == ToolCommandIntent.EXECUTE_PREDEFINED_SWING
                        and explicit_predefined_object is None
                    ):
                        parsed_command.object_name = None
                    command = parsed_command
                if (
                    not assistant_text
                    and function_name == "get_sim_state"
                    and isinstance(tool_result, dict)
                    and isinstance(tool_result.get("summary"), str)
                ):
                    assistant_text = str(tool_result["summary"])
                openai_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(tool_result),
                    }
                )

        if not assistant_text:
            assistant_text = "(No response from model.)"

        semantic_requested = _semantic_target_from_text(latest_user_text) is not None
        swing_requested = _is_swing_target_request(latest_user_text)
        predefined_swing_requested = _is_predefined_swing_request(latest_user_text)
        if (
            (semantic_requested or swing_requested or predefined_swing_requested)
            and command is None
            and goals is None
        ):
            clarification = (
                "I fetched the current sim state but did not execute a motion command yet. "
                "Please confirm a concrete semantic target, tabletop strike location, or "
                "predefined swing request."
            )
            assistant_text = (
                f"{assistant_text}\n{clarification}" if assistant_text else clarification
            )
        elif (
            command is None
            and goals is None
            and any(
                token in assistant_text
                for token in (
                    "primary_axis",
                    "head_axis",
                    "tip_axis",
                    "face_normal",
                    "axes_local",
                    "semantic_targets",
                )
            )
        ):
            assistant_text = _format_sim_state_summary(sim_context.get("sim_state", {}))

        return ChatResponse(
            text=assistant_text,
            goals=goals,
            command=command,
            tool_trace=tool_trace,
        )


# Return the appropriate chat client for the given backend string.
def build_chat_client(
    backend: str, model: Optional[str] = None
) -> Union[MockChatClient, OpenAIChatClient]:
    """Build mock or OpenAI chat client from backend selector."""
    if backend == "mock":
        return MockChatClient()

    if backend == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for backend='openai'. "
                "Install it with: pip install openai"
            ) from exc
        from pathlib import Path

        from dotenv import load_dotenv

        from .config import GoalGeneratorConfig

        _env_file = Path(__file__).resolve().parents[2] / ".env"
        load_dotenv(_env_file, override=False)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                f"OPENAI_API_KEY is not set. Add it to {_env_file} or export it in your shell."
            )
        default_config = GoalGeneratorConfig()
        resolved_model = model or os.environ.get("OPENAI_MODEL", default_config.model)
        config = GoalGeneratorConfig(model=resolved_model)
        client = OpenAI(api_key=api_key)
        return OpenAIChatClient(openai_client=client, config=config)

    raise ValueError(f"Unknown chat backend: {backend!r}. Choose 'mock' or 'openai'.")
