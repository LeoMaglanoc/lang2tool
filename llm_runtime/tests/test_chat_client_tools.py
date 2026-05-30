"""Unit tests for OpenAI and mock chat tool-call parsing paths."""

from __future__ import annotations

import json

from geometric_tool_planning import get_llm_static_strike_context
from llm_runtime.llm.chat_client import ChatMessage, OpenAIChatClient
from llm_runtime.llm.config import GoalGeneratorConfig
from llm_runtime.types import ToolCommandIntent


# Fake completion response matching minimal SDK shape used by OpenAIChatClient.
class _FakeCompletionResponse:
    """Expose a choices[0].message payload for chat-client tests."""

    # Store one fake message object on the first choice.
    def __init__(self, message) -> None:
        """Initialize fake response with one message."""
        self.choices = [type("Choice", (), {"message": message})()]


# Fake chat completions endpoint returning one pre-baked message.
class _FakeCompletions:
    """Return deterministic fake completion payloads."""

    # Store payload that create() should return.
    def __init__(self, responses) -> None:
        """Initialize fake completions endpoint with one or more responses."""
        self._responses = list(responses)
        self.calls = []

    # Return the pre-baked response object.
    def create(self, **kwargs):
        """Return deterministic fake completion output."""
        self.calls.append(kwargs)
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)


# Fake OpenAI client exposing chat.completions.create.
class _FakeOpenAIClient:
    """Provide minimal chat.completions.create shape for tests."""

    # Build chat namespace with fake completions handler.
    def __init__(self, responses) -> None:
        """Initialize fake client with deterministic response."""
        self.chat = type("Chat", (), {"completions": _FakeCompletions(responses)})()
        self.api_key = "test-key"
        self.base_url = "https://api.openai.com/v1"


# Verify the shared static strike-context explains front/back using camera-relative wording.
def test_get_llm_static_strike_context_describes_front_back_relative_to_camera() -> None:
    """Ensure front/back tabletop language matches the camera-facing convention."""
    static_context = get_llm_static_strike_context(
        object_name="claw_hammer",
        task_name="swing_down",
    )

    frame_description = static_context["frame_description"]

    assert "front of the table" in frame_description
    assert "farther away from the camera" in frame_description
    assert "back of the table" in frame_description
    assert "closer to the camera" in frame_description


# Verify semantic fields from apply_goal_delta tool call map into ToolCommand.
def test_openai_chat_client_parses_semantic_apply_goal_delta_command() -> None:
    """Ensure semantic apply_goal_delta tool args are parsed into ToolCommand."""
    tool_call = {
        "function": {
            "name": "apply_goal_delta",
            "arguments": json.dumps(
                {
                    "delta_translation_m": [0.0, 0.0, 0.0],
                    "delta_euler_rad": [0.0, 0.0, 0.0],
                    "frame": "camera_spawn",
                    "target": "all_active_goals",
                    "semantic_target": "upright",
                    "semantic_preserve_position": True,
                }
            ),
        }
    }
    first = {"content": "", "tool_calls": [tool_call]}
    second = {"content": "Done.", "tool_calls": None}
    fake_client = _FakeOpenAIClient(
        [_FakeCompletionResponse(first), _FakeCompletionResponse(second)]
    )
    chat_client = OpenAIChatClient(fake_client, GoalGeneratorConfig(model="gpt-4o"))

    response = chat_client.send(
        messages=[ChatMessage(role="user", content="make the tool upright")],
        sim_context={"current_object": "claw_hammer", "sim_state": {}},
    )

    assert response.command is not None
    assert response.command.intent == ToolCommandIntent.MOVE_TOOL
    assert response.command.semantic_target == "upright"
    assert response.command.semantic_preserve_position is True


# Verify get_sim_state tool call returns a natural-language state summary.
def test_openai_chat_client_handles_get_sim_state_tool_call() -> None:
    """Ensure get_sim_state tool call yields state summary text and no command."""
    tool_call = {
        "function": {
            "name": "get_sim_state",
            "arguments": "{}",
        }
    }
    first = {"content": "", "tool_calls": [tool_call]}
    second = {"content": "State inspected.", "tool_calls": None}
    fake_client = _FakeOpenAIClient(
        [_FakeCompletionResponse(first), _FakeCompletionResponse(second)]
    )
    chat_client = OpenAIChatClient(fake_client, GoalGeneratorConfig(model="gpt-4o"))

    response = chat_client.send(
        messages=[ChatMessage(role="user", content="what is the current pose?")],
        sim_context={
            "current_object": "claw_hammer",
            "sim_state": {
                "object_name": "claw_hammer",
                "runtime_mode": "policy",
                "object_pose_xyzw": [0.0, 0.8, 0.75, 0.0, 0.0, 0.0, 1.0],
                "goal_pose_xyzw": [0.0, 0.8, 0.80, 0.0, 0.0, 0.0, 1.0],
                "pose_semantics": {"semantic_targets": ["upright", "flat"]},
            },
        },
    )

    assert response.command is None
    assert "State inspected." in response.text


# Verify semantic intent without write tool call yields clarification text.
def test_openai_chat_client_semantic_intent_without_write_call_requests_clarification() -> None:
    """Ensure semantic requests do not auto-execute when model omits a write command."""
    get_state_call = {"function": {"name": "get_sim_state", "arguments": "{}"}}
    first = {"content": "", "tool_calls": [get_state_call]}
    second = {"content": "I checked the state.", "tool_calls": None}
    fake_client = _FakeOpenAIClient(
        [_FakeCompletionResponse(first), _FakeCompletionResponse(second)]
    )
    chat_client = OpenAIChatClient(fake_client, GoalGeneratorConfig(model="gpt-4o"))

    response = chat_client.send(
        messages=[ChatMessage(role="user", content="move tool upright")],
        sim_context={
            "current_object": "claw_hammer",
            "sim_state": {
                "object_name": "claw_hammer",
                "runtime_mode": "policy",
                "object_pose_xyzw": [0.0, 0.8, 0.75, 0.0, 0.0, 0.0, 1.0],
                "goal_pose_xyzw": [0.0, 0.8, 0.80, 0.0, 0.0, 0.0, 1.0],
                "pose_semantics": {"semantic_targets": ["upright", "flat"]},
            },
        },
    )

    assert response.command is None
    assert "did not execute a motion command" in response.text


# Verify ambiguous supported-family requests ask for clarification before any tool call runs.
def test_openai_chat_client_requests_clarification_for_ambiguous_screwdriver() -> None:
    """Ambiguous screwdriver-family requests should ask the user which instance they want."""
    fake_client = _FakeOpenAIClient([])
    chat_client = OpenAIChatClient(fake_client, GoalGeneratorConfig(model="gpt-4o"))

    response = chat_client.send(
        messages=[ChatMessage(role="user", content="use screwdriver")],
        sim_context={"current_object": "claw_hammer", "sim_state": {}},
    )

    assert response.command is None
    assert "which screwdriver" in response.text.lower()
    assert fake_client.chat.completions.calls == []


# Verify ambiguous hammer-family requests ask for clarification before any tool call runs.
def test_openai_chat_client_requests_clarification_for_ambiguous_hammer() -> None:
    """Ambiguous hammer-family requests should ask the user which instance they want."""
    fake_client = _FakeOpenAIClient([])
    chat_client = OpenAIChatClient(fake_client, GoalGeneratorConfig(model="gpt-4o"))

    response = chat_client.send(
        messages=[ChatMessage(role="user", content="switch to the hammer")],
        sim_context={"current_object": "long_screwdriver", "sim_state": {}},
    )

    assert response.command is None
    assert "which hammer" in response.text.lower()
    assert fake_client.chat.completions.calls == []


# Rewrite schema-like pose-semantics text into natural-language summary.
def test_openai_chat_client_rewrites_schema_like_semantics_text() -> None:
    """Ensure schema-style semantic dumps are replaced by a natural-language summary."""
    get_state_call = {"function": {"name": "get_sim_state", "arguments": "{}"}}
    first = {"content": "", "tool_calls": [get_state_call]}
    second = {
        "content": (
            "Pose semantics: semantic_targets=[upright, flat], "
            "axes_local={primary_axis:[1,0,0],head_axis:[1,0,0],"
            "tip_axis:[1,0,0],face_normal:[0,0,1]}"
        ),
        "tool_calls": None,
    }
    fake_client = _FakeOpenAIClient(
        [_FakeCompletionResponse(first), _FakeCompletionResponse(second)]
    )
    chat_client = OpenAIChatClient(fake_client, GoalGeneratorConfig(model="gpt-4o"))

    response = chat_client.send(
        messages=[ChatMessage(role="user", content="show me pose semantics for this object")],
        sim_context={
            "current_object": "claw_hammer",
            "sim_state": {
                "object_name": "claw_hammer",
                "runtime_mode": "policy",
                "object_pose_xyzw": [0.0, 0.8, 0.75, 0.0, 0.0, 0.0, 1.0],
                "goal_pose_xyzw": [0.0, 0.8, 0.80, 0.0, 0.0, 0.0, 1.0],
                "pose_semantics": {
                    "semantic_targets": ["upright", "flat", "head_down"],
                    "axes_local": {
                        "primary_axis": [1.0, 0.0, 0.0],
                        "head_axis": [1.0, 0.0, 0.0],
                        "tip_axis": [1.0, 0.0, 0.0],
                        "face_normal": [0.0, 0.0, 1.0],
                    },
                },
            },
        },
    )

    assert response.command is None
    assert "The current object is claw_hammer" in response.text
    assert "semantic pose targets are upright, flat, head down" in response.text
    assert "primary_axis" not in response.text


# Verify generic swing requests default to the canonical hammer object.
def test_openai_chat_client_parses_execute_lie_trajectory_command() -> None:
    """Ensure coordinate-driven swing requests route to the default hammer."""
    tool_call = {
        "function": {
            "name": "execute_lie_trajectory",
            "arguments": json.dumps(
                {
                    "strike_target_xy": [0.18, 0.04],
                    "target_description": "right side of the table",
                    "replace_active_goals": True,
                }
            ),
        }
    }
    first = {"content": "", "tool_calls": [tool_call]}
    second = {"content": "Swinging to the requested tabletop strike point.", "tool_calls": None}
    fake_client = _FakeOpenAIClient(
        [_FakeCompletionResponse(first), _FakeCompletionResponse(second)]
    )
    chat_client = OpenAIChatClient(fake_client, GoalGeneratorConfig(model="gpt-4o"))

    response = chat_client.send(
        messages=[ChatMessage(role="user", content="swing on the right side of the table")],
        sim_context={
            "current_object": "claw_hammer",
            "static_strike_context": {
                "available": True,
                "table_target_region": {
                    "x_min": -0.1975,
                    "x_max": 0.1975,
                    "y_min": -0.16,
                    "y_max": 0.16,
                },
                "frame_description": "Use world-table XY coordinates.",
                "named_strike_points": {
                    "available": True,
                    "points": {
                        "target_a": {"aliases": ["a"], "world_xy": [0.12, 0.04]},
                    },
                },
            },
            "sim_state": {
                "object_name": "claw_hammer",
                "runtime_mode": "policy",
                "object_pose_xyzw": [0.0, 0.8, 0.75, 0.0, 0.0, 0.0, 1.0],
                "goal_pose_xyzw": [0.0, 0.8, 0.80, 0.0, 0.0, 0.0, 1.0],
                "pose_semantics": {"semantic_targets": ["upright"]},
                "active_strike_target": {"available": False, "world_xy": None, "world_xyz": None},
            },
        },
    )

    assert response.command is not None
    assert response.command.intent == ToolCommandIntent.EXECUTE_LIE_TRAJECTORY
    assert response.command.strike_target_xy == [0.18, 0.04]
    assert response.command.target_description == "right side of the table"
    assert response.command.replace_active_goals is True
    assert response.command.object_name == "claw_hammer"


# Verify explicit object-switch text is routed through an OpenAI tool call.
def test_openai_chat_client_parses_switch_active_object_command() -> None:
    """OpenAI chat should parse switch_active_object tool calls into ToolCommand."""
    tool_call = {
        "function": {
            "name": "switch_active_object",
            "arguments": json.dumps({"object_name": "long_screwdriver"}),
        }
    }
    first = {"content": "", "tool_calls": [tool_call]}
    second = {"content": "I chose the screwdriver and switched to it.", "tool_calls": None}
    fake_client = _FakeOpenAIClient(
        [_FakeCompletionResponse(first), _FakeCompletionResponse(second)]
    )
    chat_client = OpenAIChatClient(fake_client, GoalGeneratorConfig(model="gpt-4o"))

    response = chat_client.send(
        messages=[ChatMessage(role="user", content="switch to the long screwdriver")],
        sim_context={"current_object": "claw_hammer", "sim_state": {}},
    )

    assert response.command is not None
    assert response.command.intent == ToolCommandIntent.SWITCH_ACTIVE_OBJECT
    assert response.command.object_name == "long_screwdriver"
    assert "screwdriver" in response.text.lower()


# Verify explicit switching also accepts the primitive training-object additions.
def test_openai_chat_client_parses_switch_active_object_command_for_primitive_hammer() -> None:
    """OpenAI chat should parse primitive hammer switch tool calls into ToolCommand."""
    tool_call = {
        "function": {
            "name": "switch_active_object",
            "arguments": json.dumps({"object_name": "cuboid_hammer_v014"}),
        }
    }
    first = {"content": "", "tool_calls": [tool_call]}
    second = {"content": "I chose the primitive hammer and switched to it.", "tool_calls": None}
    fake_client = _FakeOpenAIClient(
        [_FakeCompletionResponse(first), _FakeCompletionResponse(second)]
    )
    chat_client = OpenAIChatClient(fake_client, GoalGeneratorConfig(model="gpt-4o"))

    response = chat_client.send(
        messages=[ChatMessage(role="user", content="switch to the cuboid hammer")],
        sim_context={"current_object": "claw_hammer", "sim_state": {}},
    )

    assert response.command is not None
    assert response.command.intent == ToolCommandIntent.SWITCH_ACTIVE_OBJECT
    assert response.command.object_name == "cuboid_hammer_v014"
    assert "hammer" in response.text.lower()


# Verify twist placement defaults to the canonical screwdriver when no instance is named.
def test_openai_chat_client_parses_twist_request_with_tool_choice() -> None:
    """OpenAI chat should route generic twist requests to the default screwdriver."""
    tool_call = {
        "function": {
            "name": "execute_lie_trajectory",
            "arguments": json.dumps(
                {
                    "object_name": "long_screwdriver",
                    "strike_target_xy": [0.0, -0.12],
                    "target_description": "back of the table",
                    "replace_active_goals": True,
                }
            ),
        }
    }
    first = {"content": "", "tool_calls": [tool_call]}
    second = {
        "content": "I chose the screwdriver and set up a twist on the back of the table.",
        "tool_calls": None,
    }
    fake_client = _FakeOpenAIClient(
        [_FakeCompletionResponse(first), _FakeCompletionResponse(second)]
    )
    chat_client = OpenAIChatClient(fake_client, GoalGeneratorConfig(model="gpt-4o"))

    response = chat_client.send(
        messages=[ChatMessage(role="user", content="twist on the back of the table")],
        sim_context={"current_object": "claw_hammer", "sim_state": {}},
    )

    assert response.command is not None
    assert response.command.intent == ToolCommandIntent.EXECUTE_LIE_TRAJECTORY
    assert response.command.object_name == "long_screwdriver"
    assert response.command.target_description == "back of the table"
    assert "screwdriver" in response.text.lower()


# Verify same-target twist requests override the active hammer object.
def test_openai_chat_client_routes_same_target_twist_to_default_screwdriver() -> None:
    """Same-target twist requests should default to the canonical screwdriver."""
    tool_call = {
        "function": {
            "name": "execute_lie_trajectory",
            "arguments": json.dumps(
                {
                    "object_name": "claw_hammer",
                    "strike_target_xy": [0.12, 0.04],
                    "target_description": "same target",
                    "replace_active_goals": True,
                }
            ),
        }
    }
    first = {"content": "", "tool_calls": [tool_call]}
    second = {"content": "Twisting on the same target.", "tool_calls": None}
    fake_client = _FakeOpenAIClient(
        [_FakeCompletionResponse(first), _FakeCompletionResponse(second)]
    )
    chat_client = OpenAIChatClient(fake_client, GoalGeneratorConfig(model="gpt-4o"))

    response = chat_client.send(
        messages=[ChatMessage(role="user", content="twist on the same target")],
        sim_context={"current_object": "claw_hammer", "sim_state": {}},
    )

    assert response.command is not None
    assert response.command.intent == ToolCommandIntent.EXECUTE_LIE_TRAJECTORY
    assert response.command.object_name == "long_screwdriver"
    assert response.command.strike_target_xy == [0.12, 0.04]
    assert response.command.target_description == "same target"


# Verify generic family wording in executable twist requests does not ask clarification.
def test_openai_chat_client_twist_with_screwdriver_defaults_without_clarification() -> None:
    """Executable generic screwdriver twist requests should default to the long screwdriver."""
    tool_call = {
        "function": {
            "name": "execute_lie_trajectory",
            "arguments": json.dumps(
                {
                    "strike_target_xy": [0.12, 0.04],
                    "target_description": "same target",
                    "replace_active_goals": True,
                }
            ),
        }
    }
    first = {"content": "", "tool_calls": [tool_call]}
    second = {"content": "I chose the screwdriver and set up the twist.", "tool_calls": None}
    fake_client = _FakeOpenAIClient(
        [_FakeCompletionResponse(first), _FakeCompletionResponse(second)]
    )
    chat_client = OpenAIChatClient(fake_client, GoalGeneratorConfig(model="gpt-4o"))

    response = chat_client.send(
        messages=[ChatMessage(role="user", content="twist with screwdriver on the same target")],
        sim_context={"current_object": "claw_hammer", "sim_state": {}},
    )

    assert response.command is not None
    assert response.command.intent == ToolCommandIntent.EXECUTE_LIE_TRAJECTORY
    assert response.command.object_name == "long_screwdriver"
    assert fake_client.chat.completions.calls != []


# Verify explicit short-screwdriver twist requests survive model object mistakes.
def test_openai_chat_client_repairs_twist_request_to_explicit_short_screwdriver() -> None:
    """Explicit screwdriver instances should override incorrect model-selected objects."""
    tool_call = {
        "function": {
            "name": "execute_lie_trajectory",
            "arguments": json.dumps(
                {
                    "object_name": "claw_hammer",
                    "strike_target_xy": [0.12, 0.04],
                    "target_description": "same target",
                    "replace_active_goals": True,
                }
            ),
        }
    }
    first = {"content": "", "tool_calls": [tool_call]}
    second = {"content": "I chose the short screwdriver.", "tool_calls": None}
    fake_client = _FakeOpenAIClient(
        [_FakeCompletionResponse(first), _FakeCompletionResponse(second)]
    )
    chat_client = OpenAIChatClient(fake_client, GoalGeneratorConfig(model="gpt-4o"))

    response = chat_client.send(
        messages=[
            ChatMessage(role="user", content="twist with short screwdriver on the same target")
        ],
        sim_context={"current_object": "claw_hammer", "sim_state": {}},
    )

    assert response.command is not None
    assert response.command.intent == ToolCommandIntent.EXECUTE_LIE_TRAJECTORY
    assert response.command.object_name == "short_screwdriver"


# Verify plain strike requests default to the canonical hammer when no instance is named.
def test_openai_chat_client_routes_plain_strike_request_to_default_hammer() -> None:
    """Plain strike requests should route to the default hammer object."""
    tool_call = {
        "function": {
            "name": "execute_lie_trajectory",
            "arguments": json.dumps(
                {
                    "object_name": "long_screwdriver",
                    "strike_target_xy": [0.18, 0.04],
                    "target_description": "right side of the table",
                    "replace_active_goals": True,
                }
            ),
        }
    }
    first = {"content": "", "tool_calls": [tool_call]}
    second = {"content": "Swinging to the requested tabletop strike point.", "tool_calls": None}
    fake_client = _FakeOpenAIClient(
        [_FakeCompletionResponse(first), _FakeCompletionResponse(second)]
    )
    chat_client = OpenAIChatClient(fake_client, GoalGeneratorConfig(model="gpt-4o"))

    response = chat_client.send(
        messages=[ChatMessage(role="user", content="strike on the right side of the table")],
        sim_context={"current_object": "short_screwdriver", "sim_state": {}},
    )

    assert response.command is not None
    assert response.command.intent == ToolCommandIntent.EXECUTE_LIE_TRAJECTORY
    assert response.command.object_name == "claw_hammer"
    assert response.command.strike_target_xy == [0.18, 0.04]


# Verify explicit targeting verbs keep the current object even if the model names another one.
def test_openai_chat_client_keeps_current_object_for_target_request() -> None:
    """Target requests should preserve the active object unless the user names another instance."""
    tool_call = {
        "function": {
            "name": "execute_lie_trajectory",
            "arguments": json.dumps(
                {
                    "object_name": "claw_hammer",
                    "strike_target_xy": [0.12, 0.04],
                    "target_description": "strike point a",
                    "replace_active_goals": True,
                }
            ),
        }
    }
    first = {"content": "", "tool_calls": [tool_call]}
    second = {"content": "Targeting strike point a with the current tool.", "tool_calls": None}
    fake_client = _FakeOpenAIClient(
        [_FakeCompletionResponse(first), _FakeCompletionResponse(second)]
    )
    chat_client = OpenAIChatClient(fake_client, GoalGeneratorConfig(model="gpt-4o"))

    response = chat_client.send(
        messages=[ChatMessage(role="user", content="target strike point a")],
        sim_context={"current_object": "mallet_hammer", "sim_state": {}},
    )

    assert response.command is not None
    assert response.command.intent == ToolCommandIntent.EXECUTE_LIE_TRAJECTORY
    assert response.command.object_name is None
    assert response.command.strike_target_xy == [0.12, 0.04]


# Verify predefined twist playback uses an OpenAI tool call that names the screwdriver explicitly.
def test_openai_chat_client_parses_predefined_twist_with_tool_choice() -> None:
    """OpenAI chat should parse screwdriver predefined tool calls returned by the model."""
    tool_call = {
        "function": {
            "name": "execute_predefined_swing",
            "arguments": json.dumps({"object_name": "long_screwdriver"}),
        }
    }
    first = {"content": "", "tool_calls": [tool_call]}
    second = {
        "content": "I chose the screwdriver and started the predefined twist motion.",
        "tool_calls": None,
    }
    fake_client = _FakeOpenAIClient(
        [_FakeCompletionResponse(first), _FakeCompletionResponse(second)]
    )
    chat_client = OpenAIChatClient(fake_client, GoalGeneratorConfig(model="gpt-4o"))

    response = chat_client.send(
        messages=[ChatMessage(role="user", content="do predefined twist motion")],
        sim_context={"current_object": "claw_hammer", "sim_state": {}},
    )

    assert response.command is not None
    assert response.command.intent == ToolCommandIntent.EXECUTE_PREDEFINED_SWING


# Verify plain predefined requests stay on the current object even if the model names another tool.
def test_openai_chat_client_keeps_current_object_for_plain_predefined_request() -> None:
    """Plain predefined motion requests should not switch objects unless the user named one explicitly."""
    tool_call = {
        "function": {
            "name": "execute_predefined_swing",
            "arguments": json.dumps({"object_name": "long_screwdriver"}),
        }
    }
    first = {"content": "", "tool_calls": [tool_call]}
    second = {
        "content": "Starting the predefined motion on the current tool.",
        "tool_calls": None,
    }
    fake_client = _FakeOpenAIClient(
        [_FakeCompletionResponse(first), _FakeCompletionResponse(second)]
    )
    chat_client = OpenAIChatClient(fake_client, GoalGeneratorConfig(model="gpt-4o"))

    response = chat_client.send(
        messages=[ChatMessage(role="user", content="please do predefined motion")],
        sim_context={"current_object": "claw_hammer", "sim_state": {}},
    )

    assert response.command is not None
    assert response.command.intent == ToolCommandIntent.EXECUTE_PREDEFINED_SWING
    assert response.command.object_name is None


# Verify explicit predefined requests may still target a specific supported object instance.
def test_openai_chat_client_allows_explicit_object_for_predefined_request() -> None:
    """Predefined requests may switch when the user explicitly names the object instance."""
    tool_call = {
        "function": {
            "name": "execute_predefined_swing",
            "arguments": json.dumps({"object_name": "long_screwdriver"}),
        }
    }
    first = {"content": "", "tool_calls": [tool_call]}
    second = {
        "content": "Starting the predefined screwdriver motion.",
        "tool_calls": None,
    }
    fake_client = _FakeOpenAIClient(
        [_FakeCompletionResponse(first), _FakeCompletionResponse(second)]
    )
    chat_client = OpenAIChatClient(fake_client, GoalGeneratorConfig(model="gpt-4o"))

    response = chat_client.send(
        messages=[
            ChatMessage(
                role="user",
                content="please do predefined motion with the long screwdriver",
            )
        ],
        sim_context={"current_object": "claw_hammer", "sim_state": {}},
    )

    assert response.command is not None
    assert response.command.intent == ToolCommandIntent.EXECUTE_PREDEFINED_SWING
    assert response.command.object_name == "long_screwdriver"
    assert response.command.object_name == "long_screwdriver"
    assert "screwdriver" in response.text.lower()


# Verify predefined swing tool calls map into the dedicated runtime command.
def test_openai_chat_client_parses_execute_predefined_swing_command() -> None:
    """Ensure predefined swing tool calls are parsed into ToolCommand."""
    tool_call = {
        "function": {
            "name": "execute_predefined_swing",
            "arguments": json.dumps({}),
        }
    }
    first = {"content": "", "tool_calls": [tool_call]}
    second = {"content": "Executing the predefined swing.", "tool_calls": None}
    fake_client = _FakeOpenAIClient(
        [_FakeCompletionResponse(first), _FakeCompletionResponse(second)]
    )
    chat_client = OpenAIChatClient(fake_client, GoalGeneratorConfig(model="gpt-4o"))

    response = chat_client.send(
        messages=[ChatMessage(role="user", content="please do predefined swing")],
        sim_context={
            "current_object": "claw_hammer",
            "static_strike_context": {
                "available": True,
                "table_target_region": {
                    "x_min": -0.1975,
                    "x_max": 0.1975,
                    "y_min": -0.16,
                    "y_max": 0.16,
                },
                "frame_description": "Use world-table XY coordinates.",
                "named_strike_points": {
                    "available": True,
                    "points": {
                        "target_a": {"aliases": ["a"], "world_xy": [0.12, 0.04]},
                    },
                },
            },
            "sim_state": {
                "object_name": "claw_hammer",
                "runtime_mode": "policy",
                "object_pose_xyzw": [0.0, 0.8, 0.75, 0.0, 0.0, 0.0, 1.0],
                "goal_pose_xyzw": [0.0, 0.8, 0.80, 0.0, 0.0, 0.0, 1.0],
                "pose_semantics": {"semantic_targets": ["upright"]},
                "active_strike_target": {"available": False, "world_xy": None, "world_xyz": None},
            },
        },
    )

    assert response.command is not None
    assert response.command.intent == ToolCommandIntent.EXECUTE_PREDEFINED_SWING
