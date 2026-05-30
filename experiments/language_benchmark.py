"""Run the thesis language-grounding benchmark over curated prompt/intent pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import tyro

from experiments.common import (
    default_experiment_name,
    ensure_experiment_dirs,
    ensure_experiment_metadata,
    save_trial_summaries,
    write_json,
)
from experiments.result_schema import LanguageTrialResult, to_dict
from llm_runtime.llm.chat_client import ChatMessage, build_chat_client
from llm_runtime.types import ToolCommand, ToolCommandIntent

_STATIC_STRIKE_CONTEXT = {
    "available": True,
    "frame_description": (
        "Table x increases from left to right, and y increases from back to front."
    ),
    "table_target_region": {
        "x_min": -0.1975,
        "x_max": 0.1975,
        "y_min": -0.16,
        "y_max": 0.16,
    },
    "named_strike_points": {
        "points": {
            "target_a": {
                "world_xy": [0.0, 0.0],
            }
        }
    },
}

_DEFAULT_START_POSE = [0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0]
_TARGET_EPS = 1e-6


# Store one curated language benchmark prompt with its gold structured intent.
@dataclass(frozen=True)
class LanguageBenchmarkPrompt:
    """Ground-truth annotation for one curated language-grounding prompt."""

    prompt_id: str
    prompt_family: str
    prompt_variant: str
    prompt_text: str
    active_object_context: str
    expected_outcome_type: str
    expected_intent: Optional[str]
    expected_object_name: Optional[str]
    expected_target_label: Optional[str]
    expected_clarification: bool


_DEFAULT_PROMPTS: Sequence[LanguageBenchmarkPrompt] = (
    LanguageBenchmarkPrompt(
        prompt_id="clarify_hammer_family",
        prompt_family="clarification_or_ambiguity",
        prompt_variant="hammer_family_only",
        prompt_text="switch to the hammer",
        active_object_context="claw_hammer",
        expected_outcome_type="clarification",
        expected_intent=None,
        expected_object_name=None,
        expected_target_label=None,
        expected_clarification=True,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="clarify_screwdriver_family",
        prompt_family="clarification_or_ambiguity",
        prompt_variant="screwdriver_family_only",
        prompt_text="use the screwdriver",
        active_object_context="long_screwdriver",
        expected_outcome_type="clarification",
        expected_intent=None,
        expected_object_name=None,
        expected_target_label=None,
        expected_clarification=True,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="clarify_generic_hammer_swap",
        prompt_family="clarification_or_ambiguity",
        prompt_variant="generic_hammer_swap",
        prompt_text="make it a hammer instead",
        active_object_context="long_screwdriver",
        expected_outcome_type="clarification",
        expected_intent=None,
        expected_object_name=None,
        expected_target_label=None,
        expected_clarification=True,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="clarify_generic_screwdriver_swap",
        prompt_family="clarification_or_ambiguity",
        prompt_variant="generic_screwdriver_swap",
        prompt_text="can you use a screwdriver instead?",
        active_object_context="claw_hammer",
        expected_outcome_type="clarification",
        expected_intent=None,
        expected_object_name=None,
        expected_target_label=None,
        expected_clarification=True,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="clarify_other_hammer",
        prompt_family="clarification_or_ambiguity",
        prompt_variant="other_hammer_instance",
        prompt_text="switch from this to the other hammer",
        active_object_context="mallet_hammer",
        expected_outcome_type="clarification",
        expected_intent=None,
        expected_object_name=None,
        expected_target_label=None,
        expected_clarification=True,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="clarify_conflicting_tool_family",
        prompt_family="clarification_or_ambiguity",
        prompt_variant="conflicting_tool_family",
        prompt_text="use a hammer or screwdriver for this",
        active_object_context="claw_hammer",
        expected_outcome_type="clarification",
        expected_intent=None,
        expected_object_name=None,
        expected_target_label=None,
        expected_clarification=True,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="switch_specific_object",
        prompt_family="explicit_object_switching",
        prompt_variant="switch_short_screwdriver",
        prompt_text="switch to the short screwdriver",
        active_object_context="claw_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.SWITCH_ACTIVE_OBJECT.value,
        expected_object_name="short_screwdriver",
        expected_target_label=None,
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="switch_long_screwdriver",
        prompt_family="explicit_object_switching",
        prompt_variant="switch_long_screwdriver",
        prompt_text="switch to the long screwdriver",
        active_object_context="claw_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.SWITCH_ACTIVE_OBJECT.value,
        expected_object_name="long_screwdriver",
        expected_target_label=None,
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="switch_mallet_hammer",
        prompt_family="explicit_object_switching",
        prompt_variant="switch_mallet_hammer",
        prompt_text="make the mallet hammer active",
        active_object_context="short_screwdriver",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.SWITCH_ACTIVE_OBJECT.value,
        expected_object_name="mallet_hammer",
        expected_target_label=None,
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="switch_cuboid_hammer",
        prompt_family="explicit_object_switching",
        prompt_variant="switch_cuboid_hammer",
        prompt_text="select the cuboid hammer",
        active_object_context="claw_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.SWITCH_ACTIVE_OBJECT.value,
        expected_object_name="cuboid_hammer_v014",
        expected_target_label=None,
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="switch_cylinder_screwdriver",
        prompt_family="explicit_object_switching",
        prompt_variant="switch_cylinder_screwdriver",
        prompt_text="change to the cylinder screwdriver",
        active_object_context="long_screwdriver",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.SWITCH_ACTIVE_OBJECT.value,
        expected_object_name="cylinder_screwdriver_v3009",
        expected_target_label=None,
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="switch_specific_object_paraphrase",
        prompt_family="paraphrase_robustness",
        prompt_variant="switch_short_screwdriver_paraphrase",
        prompt_text="change the active tool to the short screwdriver",
        active_object_context="claw_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.SWITCH_ACTIVE_OBJECT.value,
        expected_object_name="short_screwdriver",
        expected_target_label=None,
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="switch_mallet_paraphrase",
        prompt_family="paraphrase_robustness",
        prompt_variant="switch_mallet_paraphrase",
        prompt_text="set the current object to the mallet hammer",
        active_object_context="claw_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.SWITCH_ACTIVE_OBJECT.value,
        expected_object_name="mallet_hammer",
        expected_target_label=None,
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="strike_target_a_paraphrase",
        prompt_family="paraphrase_robustness",
        prompt_variant="strike_target_a_paraphrase",
        prompt_text="please aim the claw hammer at strike point a",
        active_object_context="mallet_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY.value,
        expected_object_name="claw_hammer",
        expected_target_label="strike_point_a",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="twist_target_a_paraphrase",
        prompt_family="paraphrase_robustness",
        prompt_variant="twist_target_a_paraphrase",
        prompt_text="spin above strike point a with the short screwdriver",
        active_object_context="long_screwdriver",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY.value,
        expected_object_name="short_screwdriver",
        expected_target_label="strike_point_a",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="predefined_paraphrase",
        prompt_family="paraphrase_robustness",
        prompt_variant="predefined_motion_paraphrase",
        prompt_text="run the canned motion for the current tool",
        active_object_context="short_screwdriver",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_PREDEFINED_SWING.value,
        expected_object_name="short_screwdriver",
        expected_target_label="predefined_motion",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="grasp_paraphrase",
        prompt_family="paraphrase_robustness",
        prompt_variant="grasp_tool_paraphrase",
        prompt_text="pick up the current tool again",
        active_object_context="claw_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.GRASP_TOOL.value,
        expected_object_name="claw_hammer",
        expected_target_label=None,
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="generic_hammer_default",
        prompt_family="default_family_selection",
        prompt_variant="generic_hammer_right_side",
        prompt_text="swing on the right side of the table",
        active_object_context="mallet_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY.value,
        expected_object_name="claw_hammer",
        expected_target_label="right_side_of_table",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="generic_hammer_target_a_default",
        prompt_family="default_family_selection",
        prompt_variant="generic_hammer_target_a",
        prompt_text="hit strike point a",
        active_object_context="mallet_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY.value,
        expected_object_name="claw_hammer",
        expected_target_label="strike_point_a",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="generic_hammer_left_default",
        prompt_family="default_family_selection",
        prompt_variant="generic_hammer_left_side",
        prompt_text="hammer the left side of the table",
        active_object_context="cuboid_hammer_v014",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY.value,
        expected_object_name="claw_hammer",
        expected_target_label="left_side_of_table",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="specific_hammer_target_a",
        prompt_family="target_grounding",
        prompt_variant="named_hammer_target_a",
        prompt_text="use the mallet hammer and strike point a",
        active_object_context="claw_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY.value,
        expected_object_name="mallet_hammer",
        expected_target_label="strike_point_a",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="specific_twist_default_target",
        prompt_family="default_family_selection",
        prompt_variant="specific_screwdriver_default_target",
        prompt_text="twist with the long screwdriver",
        active_object_context="claw_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY.value,
        expected_object_name="long_screwdriver",
        expected_target_label="unspecified_target",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="generic_twist_target_a_default",
        prompt_family="default_family_selection",
        prompt_variant="generic_twist_target_a",
        prompt_text="twist at strike point a",
        active_object_context="short_screwdriver",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY.value,
        expected_object_name="long_screwdriver",
        expected_target_label="strike_point_a",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="generic_twist_left_default",
        prompt_family="default_family_selection",
        prompt_variant="generic_twist_left_side",
        prompt_text="drive the screw on the left side of the table",
        active_object_context="cylinder_screwdriver_v3009",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY.value,
        expected_object_name="long_screwdriver",
        expected_target_label="left_side_of_table",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="current_tool_target_default",
        prompt_family="default_family_selection",
        prompt_variant="current_tool_target_a",
        prompt_text="place the current tool over strike point a",
        active_object_context="short_screwdriver",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY.value,
        expected_object_name="short_screwdriver",
        expected_target_label="strike_point_a",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="generic_screwdriver_default_target",
        prompt_family="default_family_selection",
        prompt_variant="generic_screwdriver_default_target",
        prompt_text="drive with a screwdriver",
        active_object_context="short_screwdriver",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY.value,
        expected_object_name="long_screwdriver",
        expected_target_label="unspecified_target",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="target_grounding_right_side",
        prompt_family="target_grounding",
        prompt_variant="right_side_table",
        prompt_text="swing on the right side of the table",
        active_object_context="claw_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY.value,
        expected_object_name="claw_hammer",
        expected_target_label="right_side_of_table",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="target_grounding_custom_left",
        prompt_family="target_grounding",
        prompt_variant="left_side_table",
        prompt_text="swing on the left side of the table",
        active_object_context="claw_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY.value,
        expected_object_name="claw_hammer",
        expected_target_label="left_side_of_table",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="target_grounding_short_screwdriver_front",
        prompt_family="target_grounding",
        prompt_variant="short_screwdriver_front_side",
        prompt_text="twist near the front of the table with the short screwdriver",
        active_object_context="long_screwdriver",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY.value,
        expected_object_name="short_screwdriver",
        expected_target_label="front_side_of_table",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="target_grounding_cylinder_right",
        prompt_family="target_grounding",
        prompt_variant="cylinder_screwdriver_right_side",
        prompt_text="drive on the right side of the table with the cylinder screwdriver",
        active_object_context="long_screwdriver",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY.value,
        expected_object_name="cylinder_screwdriver_v3009",
        expected_target_label="right_side_of_table",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="target_grounding_current_tool_back",
        prompt_family="target_grounding",
        prompt_variant="current_tool_back_side",
        prompt_text="aim the current tool at the back of the table",
        active_object_context="claw_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY.value,
        expected_object_name="claw_hammer",
        expected_target_label="back_side_of_table",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="target_grounding_mallet_left",
        prompt_family="target_grounding",
        prompt_variant="mallet_left_side",
        prompt_text="aim the mallet hammer at the left side of the table",
        active_object_context="claw_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY.value,
        expected_object_name="mallet_hammer",
        expected_target_label="left_side_of_table",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="target_grounding_cuboid_back",
        prompt_family="target_grounding",
        prompt_variant="cuboid_hammer_back_side",
        prompt_text="strike the back of the table with the cuboid hammer",
        active_object_context="claw_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY.value,
        expected_object_name="cuboid_hammer_v014",
        expected_target_label="back_side_of_table",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="predefined_keep_current",
        prompt_family="lifecycle_and_predefined_commands",
        prompt_variant="predefined_keep_current",
        prompt_text="please do predefined motion",
        active_object_context="short_screwdriver",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_PREDEFINED_SWING.value,
        expected_object_name="short_screwdriver",
        expected_target_label="predefined_motion",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="predefined_explicit_object",
        prompt_family="lifecycle_and_predefined_commands",
        prompt_variant="predefined_explicit_object",
        prompt_text="run predefined motion with the cuboid hammer",
        active_object_context="claw_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_PREDEFINED_SWING.value,
        expected_object_name="cuboid_hammer_v014",
        expected_target_label="predefined_motion",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="predefined_explicit_screwdriver",
        prompt_family="lifecycle_and_predefined_commands",
        prompt_variant="predefined_explicit_screwdriver",
        prompt_text="run predefined motion with the long screwdriver",
        active_object_context="claw_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_PREDEFINED_SWING.value,
        expected_object_name="long_screwdriver",
        expected_target_label="predefined_motion",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="semantic_pose_request",
        prompt_family="semantic_pose_edits",
        prompt_variant="upright_pose_edit",
        prompt_text="make it upright",
        active_object_context="long_screwdriver",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.MOVE_TOOL.value,
        expected_object_name="long_screwdriver",
        expected_target_label="upright",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="grasp_tool",
        prompt_family="lifecycle_and_predefined_commands",
        prompt_variant="grasp_active_tool",
        prompt_text="grasp the tool",
        active_object_context="claw_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.GRASP_TOOL.value,
        expected_object_name="claw_hammer",
        expected_target_label=None,
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="release_tool",
        prompt_family="lifecycle_and_predefined_commands",
        prompt_variant="release_active_tool",
        prompt_text="release the tool",
        active_object_context="claw_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.RELEASE_TOOL.value,
        expected_object_name="claw_hammer",
        expected_target_label=None,
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="generate_task_goals",
        prompt_family="lifecycle_and_predefined_commands",
        prompt_variant="set_task_goals",
        prompt_text="set up swing down",
        active_object_context="claw_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.EXECUTE_LIE_TRAJECTORY.value,
        expected_object_name="claw_hammer",
        expected_target_label="unspecified_target",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="generate_task_goals_paraphrase",
        prompt_family="lifecycle_and_predefined_commands",
        prompt_variant="generate_swing_goals",
        prompt_text="generate goals for swing down",
        active_object_context="claw_hammer",
        expected_outcome_type="goals",
        expected_intent=ToolCommandIntent.SET_GOALS.value,
        expected_object_name="claw_hammer",
        expected_target_label="swing_down",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="semantic_pose_request_flat",
        prompt_family="semantic_pose_edits",
        prompt_variant="flat_pose_edit",
        prompt_text="lay it flat",
        active_object_context="long_screwdriver",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.MOVE_TOOL.value,
        expected_object_name="long_screwdriver",
        expected_target_label="flat",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="semantic_pose_request_head_down",
        prompt_family="semantic_pose_edits",
        prompt_variant="head_down_pose_edit",
        prompt_text="turn the tool head down",
        active_object_context="claw_hammer",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.MOVE_TOOL.value,
        expected_object_name="claw_hammer",
        expected_target_label="head_down",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="semantic_pose_request_tip_forward",
        prompt_family="semantic_pose_edits",
        prompt_variant="tip_forward_pose_edit",
        prompt_text="point the tip forward",
        active_object_context="long_screwdriver",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.MOVE_TOOL.value,
        expected_object_name="long_screwdriver",
        expected_target_label="tip_forward",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="semantic_pose_request_face_table",
        prompt_family="semantic_pose_edits",
        prompt_variant="face_table_pose_edit",
        prompt_text="rotate it so the face points toward the table",
        active_object_context="cuboid_hammer_v014",
        expected_outcome_type="command",
        expected_intent=ToolCommandIntent.MOVE_TOOL.value,
        expected_object_name="cuboid_hammer_v014",
        expected_target_label="face_table",
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="unsupported_out_of_scope_real_robot",
        prompt_family="unsupported_or_out_of_scope",
        prompt_variant="real_robot_request",
        prompt_text="drive the real robot to my desk and hand me the tool",
        active_object_context="claw_hammer",
        expected_outcome_type="text",
        expected_intent=None,
        expected_object_name=None,
        expected_target_label=None,
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="unsupported_out_of_scope_explanation",
        prompt_family="unsupported_or_out_of_scope",
        prompt_variant="explain_training_history",
        prompt_text="explain how this policy was trained in detail",
        active_object_context="claw_hammer",
        expected_outcome_type="text",
        expected_intent=None,
        expected_object_name=None,
        expected_target_label=None,
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="unsupported_out_of_scope_web_request",
        prompt_family="unsupported_or_out_of_scope",
        prompt_variant="open_external_application",
        prompt_text="open a video website and show me screwdriver tutorials",
        active_object_context="long_screwdriver",
        expected_outcome_type="text",
        expected_intent=None,
        expected_object_name=None,
        expected_target_label=None,
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="unsupported_out_of_scope_training_request",
        prompt_family="unsupported_or_out_of_scope",
        prompt_variant="train_new_policy",
        prompt_text="train a brand new policy for this tool right now",
        active_object_context="claw_hammer",
        expected_outcome_type="text",
        expected_intent=None,
        expected_object_name=None,
        expected_target_label=None,
        expected_clarification=False,
    ),
    LanguageBenchmarkPrompt(
        prompt_id="unsupported_out_of_scope_delete_results",
        prompt_family="unsupported_or_out_of_scope",
        prompt_variant="delete_experiment_results",
        prompt_text="delete the experiment results folder before continuing",
        active_object_context="claw_hammer",
        expected_outcome_type="text",
        expected_intent=None,
        expected_object_name=None,
        expected_target_label=None,
        expected_clarification=False,
    ),
)


# Configure one language benchmark run from the CLI.
@dataclass
class LanguageBenchmarkArgs:
    """CLI arguments for the thesis language-grounding benchmark."""

    results_dir: Path = Path("experiments/results")
    """Root directory under which experiment folders are created."""

    experiment_name: Optional[str] = None
    """Optional stable experiment name. If omitted, one timestamped name is generated."""

    backend: str = "openai"
    """Chat backend selector used by the language benchmark."""

    model: Optional[str] = None
    """Optional OpenAI model override for backend='openai'."""

    max_trials: Optional[int] = None
    """Optional hard cap on the number of prompts to evaluate."""


# Build the sim-context payload used to evaluate one language prompt.
def _build_language_sim_context(active_object_context: str) -> Dict[str, Any]:
    """Return one synthetic chat runtime context for language benchmark prompts."""
    sim_state = {
        "object_name": active_object_context,
        "object_pose_xyzw": list(_DEFAULT_START_POSE),
        "goal_pose_xyzw": list(_DEFAULT_START_POSE),
        "runtime_mode": "policy",
        "pose_semantics": {
            "semantic_targets": ["upright", "flat", "head_down", "tip_forward", "face_table"],
        },
        "active_strike_target": {"available": True, "world_xy": [0.0, 0.0]},
    }
    return {
        "current_object": active_object_context,
        "current_task": None,
        "start_pose": list(_DEFAULT_START_POSE),
        "sim_state": sim_state,
        "static_strike_context": dict(_STATIC_STRIKE_CONTEXT),
    }


# Resolve the effective object targeted by one predicted command or generated-goals response.
def _resolve_predicted_object_name(
    command: Optional[ToolCommand], active_object_context: str, predicted_outcome_type: str
) -> Optional[str]:
    """Return the effective object chosen by one chat response."""
    if predicted_outcome_type == "goals":
        return active_object_context
    if command is None:
        return None
    if command.object_name is not None:
        return command.object_name
    if command.intent in (
        ToolCommandIntent.MOVE_TOOL,
        ToolCommandIntent.EXECUTE_LIE_TRAJECTORY,
        ToolCommandIntent.EXECUTE_PREDEFINED_SWING,
        ToolCommandIntent.GRASP_TOOL,
        ToolCommandIntent.RELEASE_TOOL,
    ):
        return active_object_context
    return None


# Return one JSON-safe dictionary for a parsed tool command.
def _serialize_tool_command(command: Optional[ToolCommand]) -> Optional[Dict[str, Any]]:
    """Return a compact serializable command payload for language audit logs."""
    if command is None:
        return None
    payload = asdict(command)
    payload["intent"] = command.intent.value
    return {key: value for key, value in payload.items() if value is not None and value != ""}


# Return the expected command shape or target constraint for one benchmark prompt.
def _expected_tool_call(prompt: LanguageBenchmarkPrompt) -> Optional[Dict[str, Any]]:
    """Return the gold tool-call contract used for language audit display."""
    if prompt.expected_intent is None:
        return None
    payload: Dict[str, Any] = {"intent": prompt.expected_intent}
    if prompt.expected_object_name is not None:
        payload["object_name"] = prompt.expected_object_name
    if prompt.expected_target_label is not None:
        payload["target_label"] = prompt.expected_target_label
        payload["target_constraint"] = _target_constraint_description(prompt.expected_target_label)
    return payload


# Return a human-readable target constraint for expected tool-call display.
def _target_constraint_description(target_label: str) -> Dict[str, Any]:
    """Return the coordinate constraint implied by one target label."""
    region = _STATIC_STRIKE_CONTEXT["table_target_region"]
    if target_label == "left_side_of_table":
        return {"x": f"< {-_TARGET_EPS}", "bounds": region}
    if target_label == "right_side_of_table":
        return {"x": f"> {_TARGET_EPS}", "bounds": region}
    if target_label == "front_side_of_table":
        return {"y": f"> {_TARGET_EPS}", "bounds": region}
    if target_label == "back_side_of_table":
        return {"y": f"< {-_TARGET_EPS}", "bounds": region}
    if target_label == "strike_point_a":
        return {"strike_target_xy": [0.0, 0.0], "tolerance": _TARGET_EPS}
    return {"label": target_label}


# Return whether a coordinate lies inside the benchmark tabletop target region.
def _is_inside_table_bounds(strike_target_xy: Sequence[float]) -> bool:
    """Return True when one XY coordinate is inside the configured narrow table."""
    region = _STATIC_STRIKE_CONTEXT["table_target_region"]
    x_value = float(strike_target_xy[0])
    y_value = float(strike_target_xy[1])
    return float(region["x_min"]) <= x_value <= float(region["x_max"]) and float(
        region["y_min"]
    ) <= y_value <= float(region["y_max"])


# Evaluate one predicted target against the expected semantic target label.
def _evaluate_target_match(
    expected_target_label: Optional[str],
    command: Optional[ToolCommand],
    goals: Optional[List[List[float]]],
    predicted_target_label: Optional[str],
) -> Dict[str, Any]:
    """Return a detailed target-match decision for one language trial."""
    if expected_target_label is None:
        return {
            "match": predicted_target_label is None,
            "reason": "no_target_expected",
        }
    if goals is not None:
        return {
            "match": predicted_target_label == expected_target_label,
            "reason": "goals_target_label_match",
        }
    if command is None:
        return {"match": False, "reason": "missing_command"}
    if command.intent != ToolCommandIntent.EXECUTE_LIE_TRAJECTORY:
        return {
            "match": predicted_target_label == expected_target_label,
            "reason": "non_lie_target_label_match",
        }
    strike_target_xy = command.strike_target_xy or []
    if len(strike_target_xy) != 2:
        return {"match": expected_target_label == "unspecified_target", "reason": "missing_xy"}
    x_value = float(strike_target_xy[0])
    y_value = float(strike_target_xy[1])
    inside_bounds = _is_inside_table_bounds(strike_target_xy)
    if expected_target_label == "unspecified_target":
        return {
            "match": len(strike_target_xy) == 2,
            "reason": "xy_present_for_unspecified_target",
            "strike_target_xy": [x_value, y_value],
            "inside_table_bounds": inside_bounds,
        }
    if expected_target_label == "strike_point_a":
        match = abs(x_value) < _TARGET_EPS and abs(y_value) < _TARGET_EPS
        return {
            "match": match,
            "reason": "strike_point_a_origin_check",
            "strike_target_xy": [x_value, y_value],
            "inside_table_bounds": inside_bounds,
        }
    if expected_target_label == "left_side_of_table":
        match = inside_bounds and x_value < -_TARGET_EPS
        reason = "left_half_x_negative" if match else "not_left_half_or_out_of_bounds"
    elif expected_target_label == "right_side_of_table":
        match = inside_bounds and x_value > _TARGET_EPS
        reason = "right_half_x_positive" if match else "not_right_half_or_out_of_bounds"
    elif expected_target_label == "front_side_of_table":
        match = inside_bounds and y_value > _TARGET_EPS
        reason = "front_half_y_positive" if match else "not_front_half_or_out_of_bounds"
    elif expected_target_label == "back_side_of_table":
        match = inside_bounds and y_value < -_TARGET_EPS
        reason = "back_half_y_negative" if match else "not_back_half_or_out_of_bounds"
    else:
        match = predicted_target_label == expected_target_label
        reason = "target_label_match"
    return {
        "match": bool(match),
        "reason": reason,
        "strike_target_xy": [x_value, y_value],
        "inside_table_bounds": inside_bounds,
    }


# Map one predicted command/goals response into a concise target label for scoring.
def _resolve_predicted_target_label(
    command: Optional[ToolCommand],
    goals: Optional[List[List[float]]],
) -> Optional[str]:
    """Return one normalized target label for language-stage scoring."""
    if goals is not None:
        return "swing_down"
    if command is None:
        return None
    if command.intent == ToolCommandIntent.EXECUTE_PREDEFINED_SWING:
        return "predefined_motion"
    if command.intent == ToolCommandIntent.MOVE_TOOL:
        return command.semantic_target
    if command.intent == ToolCommandIntent.EXECUTE_LIE_TRAJECTORY:
        strike_target_xy = command.strike_target_xy or []
        if len(strike_target_xy) != 2:
            return "unspecified_target"
        if (
            abs(float(strike_target_xy[0])) < _TARGET_EPS
            and abs(float(strike_target_xy[1])) < _TARGET_EPS
        ):
            return "strike_point_a"
        return "custom_xy"
    return None


# Return one normalized outcome type label for a chat response.
def _resolve_predicted_outcome_type(
    command: Optional[ToolCommand], goals: Optional[List[List[float]]], assistant_text: str
) -> str:
    """Return one compact outcome category used by the language benchmark."""
    if goals is not None:
        return "goals"
    if command is not None:
        return "command"
    if assistant_text.strip().endswith("?"):
        return "clarification"
    return "text"


# Flatten one language benchmark result so it can be stored in the summary CSV.
def _flatten_language_result(result: LanguageTrialResult) -> Dict[str, Any]:
    """Return one summary row for the language benchmark trials table."""
    return {
        "trial_id": result.trial_id,
        "prompt_id": result.prompt_id,
        "prompt_family": result.prompt_family,
        "prompt_variant": result.prompt_variant,
        "backend": result.backend,
        "active_object_context": result.active_object_context,
        "expected_outcome_type": result.expected_outcome_type,
        "expected_intent": result.expected_intent,
        "expected_object_name": result.expected_object_name,
        "expected_target_label": result.expected_target_label,
        "expected_clarification": bool(result.expected_clarification),
        "predicted_outcome_type": result.predicted_outcome_type,
        "predicted_intent": result.predicted_intent,
        "predicted_object_name": result.predicted_object_name,
        "predicted_target_label": result.predicted_target_label,
        "predicted_clarification": bool(result.predicted_clarification),
        "exact_match": bool(result.exact_match),
        "object_match": bool(result.object_match),
        "intent_match": bool(result.intent_match),
        "target_match": bool(result.target_match),
        "clarification_match": bool(result.clarification_match),
        "error": result.error,
    }


# Compute compact aggregate statistics for the language benchmark section.
def _build_language_aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Return aggregate metrics for the language benchmark summary."""
    num_trials = len(rows)
    if num_trials == 0:
        return {
            "num_trials": 0,
            "exact_match_accuracy": 0.0,
            "intent_accuracy": 0.0,
            "object_accuracy": 0.0,
            "target_accuracy": 0.0,
            "clarification_accuracy": 0.0,
            "by_prompt_family": [],
        }
    family_rows: List[Dict[str, Any]] = []
    family_names = sorted({str(row["prompt_family"]) for row in rows})
    for family_name in family_names:
        family_subset = [row for row in rows if str(row["prompt_family"]) == family_name]
        family_count = len(family_subset)
        family_rows.append(
            {
                "prompt_family": family_name,
                "num_trials": int(family_count),
                "exact_match_accuracy": float(
                    sum(bool(row["exact_match"]) for row in family_subset) / family_count
                ),
                "intent_accuracy": float(
                    sum(bool(row["intent_match"]) for row in family_subset) / family_count
                ),
                "object_accuracy": float(
                    sum(bool(row["object_match"]) for row in family_subset) / family_count
                ),
                "target_accuracy": float(
                    sum(bool(row["target_match"]) for row in family_subset) / family_count
                ),
                "clarification_accuracy": float(
                    sum(bool(row["clarification_match"]) for row in family_subset) / family_count
                ),
            }
        )
    return {
        "num_trials": int(num_trials),
        "exact_match_accuracy": float(sum(bool(row["exact_match"]) for row in rows) / num_trials),
        "intent_accuracy": float(sum(bool(row["intent_match"]) for row in rows) / num_trials),
        "object_accuracy": float(sum(bool(row["object_match"]) for row in rows) / num_trials),
        "target_accuracy": float(sum(bool(row["target_match"]) for row in rows) / num_trials),
        "clarification_accuracy": float(
            sum(bool(row["clarification_match"]) for row in rows) / num_trials
        ),
        "by_prompt_family": family_rows,
    }


# Run the full language benchmark and save raw prompt results plus summary tables.
def run_language_benchmark(args: LanguageBenchmarkArgs) -> Path:
    """Execute the language-grounding benchmark and return the saved experiment root."""
    experiment_name = args.experiment_name or default_experiment_name("language_benchmark")
    experiment_dir = ensure_experiment_dirs(args.results_dir, experiment_name)
    ensure_experiment_metadata(experiment_dir, experiment_name)
    write_json(experiment_dir / "language" / "config.json", asdict(args))
    raw_language_dir = experiment_dir / "language" / "raw"
    for stale_raw_path in raw_language_dir.glob("*.json"):
        stale_raw_path.unlink()

    chat_client = build_chat_client(args.backend, model=args.model)
    rows: List[Dict[str, Any]] = []

    for prompt_index, prompt in enumerate(_DEFAULT_PROMPTS):
        if args.max_trials is not None and prompt_index >= int(args.max_trials):
            break
        trial_id = f"{prompt_index:04d}_{prompt.prompt_id}"
        try:
            response = chat_client.send(
                messages=[ChatMessage(role="user", content=prompt.prompt_text)],
                sim_context=_build_language_sim_context(prompt.active_object_context),
            )
            predicted_outcome_type = _resolve_predicted_outcome_type(
                response.command,
                response.goals,
                response.text,
            )
            predicted_intent = (
                ToolCommandIntent.SET_GOALS.value
                if response.goals is not None
                else (response.command.intent.value if response.command is not None else None)
            )
            predicted_object_name = _resolve_predicted_object_name(
                response.command,
                prompt.active_object_context,
                predicted_outcome_type,
            )
            predicted_target_label = _resolve_predicted_target_label(
                response.command, response.goals
            )
            target_evaluation = _evaluate_target_match(
                prompt.expected_target_label,
                response.command,
                response.goals,
                predicted_target_label,
            )
            predicted_clarification = predicted_outcome_type == "clarification"
            result = LanguageTrialResult(
                trial_id=trial_id,
                prompt_id=prompt.prompt_id,
                prompt_family=prompt.prompt_family,
                prompt_variant=prompt.prompt_variant,
                prompt_text=prompt.prompt_text,
                backend=args.backend,
                active_object_context=prompt.active_object_context,
                expected_outcome_type=prompt.expected_outcome_type,
                expected_intent=prompt.expected_intent,
                expected_object_name=prompt.expected_object_name,
                expected_target_label=prompt.expected_target_label,
                expected_clarification=bool(prompt.expected_clarification),
                predicted_outcome_type=predicted_outcome_type,
                predicted_intent=predicted_intent,
                predicted_object_name=predicted_object_name,
                predicted_target_label=predicted_target_label,
                predicted_clarification=predicted_clarification,
                assistant_text=response.text,
                exact_match=(
                    predicted_outcome_type == prompt.expected_outcome_type
                    and predicted_intent == prompt.expected_intent
                    and predicted_object_name == prompt.expected_object_name
                    and bool(target_evaluation["match"])
                    and predicted_clarification == bool(prompt.expected_clarification)
                ),
                object_match=predicted_object_name == prompt.expected_object_name,
                intent_match=predicted_intent == prompt.expected_intent,
                target_match=bool(target_evaluation["match"]),
                clarification_match=(
                    predicted_clarification == bool(prompt.expected_clarification)
                ),
                expected_tool_call=_expected_tool_call(prompt),
                predicted_tool_call=_serialize_tool_command(response.command),
                target_evaluation=target_evaluation,
                predicted_tool_trace=list(response.tool_trace),
                metrics={"num_goals": len(response.goals or [])},
            )
        except Exception as exc:
            result = LanguageTrialResult(
                trial_id=trial_id,
                prompt_id=prompt.prompt_id,
                prompt_family=prompt.prompt_family,
                prompt_variant=prompt.prompt_variant,
                prompt_text=prompt.prompt_text,
                backend=args.backend,
                active_object_context=prompt.active_object_context,
                expected_outcome_type=prompt.expected_outcome_type,
                expected_intent=prompt.expected_intent,
                expected_object_name=prompt.expected_object_name,
                expected_target_label=prompt.expected_target_label,
                expected_clarification=bool(prompt.expected_clarification),
                predicted_outcome_type="error",
                predicted_intent=None,
                predicted_object_name=None,
                predicted_target_label=None,
                predicted_clarification=False,
                assistant_text="",
                exact_match=False,
                object_match=False,
                intent_match=False,
                target_match=False,
                clarification_match=False,
                expected_tool_call=_expected_tool_call(prompt),
                predicted_tool_call=None,
                target_evaluation={"match": False, "reason": "exception"},
                predicted_tool_trace=[],
                error=str(exc),
            )
        write_json(experiment_dir / "language" / "raw" / f"{trial_id}.json", to_dict(result))
        rows.append(_flatten_language_result(result))

    aggregate_payload = {
        "backend": args.backend,
        "model": args.model,
        **_build_language_aggregate(rows),
    }
    save_trial_summaries(experiment_dir, "language", rows, aggregate_payload)
    return experiment_dir


# Parse CLI arguments and execute the language benchmark.
def main() -> None:
    """Entry point for the language benchmark CLI."""
    args = tyro.cli(LanguageBenchmarkArgs)
    experiment_dir = run_language_benchmark(args)
    print(f"Saved language benchmark to {experiment_dir}")


if __name__ == "__main__":
    main()
