"""Types for tool-centered command parsing and execution runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ToolCommandIntent(str, Enum):
    """Supported tool-centered chat intents."""

    NOOP = "noop"
    GRASP_TOOL = "grasp_tool"
    MOVE_TOOL = "move_tool"
    RELEASE_TOOL = "release_tool"
    SET_GOALS = "set_goals"
    EXECUTE_LIE_TRAJECTORY = "execute_lie_trajectory"
    EXECUTE_PREDEFINED_SWING = "execute_predefined_swing"
    SWITCH_ACTIVE_OBJECT = "switch_active_object"


class RuntimeMode(str, Enum):
    """Execution states for release/freeze handling during live eval."""

    POLICY = "policy"
    WAIT_PRE_RELEASE = "wait_pre_release"
    OPEN_HAND = "open_hand"
    FROZEN_AFTER_RELEASE = "frozen_after_release"


@dataclass
class ToolCommand:
    """Parsed command payload consumed by the eval runtime."""

    intent: ToolCommandIntent
    delta_translation_m: Optional[List[float]] = None
    delta_euler_rad: Optional[List[float]] = None
    delta_frame: Optional[str] = None
    semantic_target: Optional[str] = None
    semantic_preserve_position: Optional[bool] = None
    goals: Optional[List[List[float]]] = None
    strike_target_xy: Optional[List[float]] = None
    target_description: Optional[str] = None
    replace_active_goals: Optional[bool] = None
    object_name: Optional[str] = None
    raw_text: str = ""


@dataclass
class ToolRuntimeState:
    """Mutable runtime state for release and freeze control."""

    mode: RuntimeMode = RuntimeMode.POLICY
    pre_release_steps_left: int = 0
    open_steps_left: int = 0
    meta: dict = field(default_factory=dict)
