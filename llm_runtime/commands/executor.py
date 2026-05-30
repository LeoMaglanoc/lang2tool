"""Command execution/dispatch for tool-centered runtime control."""

from __future__ import annotations

from ..constants import DEFAULT_RELEASE_OPEN_STEPS
from ..fsm import enter_frozen, enter_policy
from ..goals.ops import (
    camera_spawn_delta_to_world,
    quat_from_euler_xyz,
    shift_goals_pose_delta,
    shift_goals_pose_quat_delta,
)
from ..semantic_pose import compute_semantic_quat_delta, quat_mul_xyzw
from ..types import ToolCommandIntent


class ToolCommandExecutor:
    """Execute queued ToolCommand intents against a runner-like runtime object."""

    # Execute and dispatch queued commands from runtime._tool_command_queue.
    def process_queued_commands(self, runtime) -> None:
        """Consume queued commands and apply policy-compatible runtime actions."""
        if not runtime._llm_tool_control_enabled or not runtime._llm_live_command_queue_enabled:
            return
        commands = runtime._tool_command_queue.pop_all()
        if not commands:
            return
        for command in commands:
            if command.intent == ToolCommandIntent.NOOP:
                continue
            if command.intent == ToolCommandIntent.GRASP_TOOL:
                self._handle_grasp_tool(runtime, command)
                continue
            if command.intent == ToolCommandIntent.SET_GOALS and command.goals is not None:
                enter_policy(runtime._tool_runtime_state)
                runtime._apply_goals_live(command.goals)
                runtime._write_llm_debug_event(
                    {
                        "event": "tool_cmd_applied",
                        "intent": command.intent.value,
                        "num_goals": len(command.goals),
                    }
                )
                continue
            if command.intent == ToolCommandIntent.MOVE_TOOL:
                self._handle_move_tool(runtime, command)
                continue
            if command.intent == ToolCommandIntent.EXECUTE_LIE_TRAJECTORY:
                self._handle_execute_lie_trajectory(runtime, command)
                continue
            if command.intent == ToolCommandIntent.EXECUTE_PREDEFINED_SWING:
                self._handle_execute_predefined_swing(runtime, command)
                continue
            if command.intent == ToolCommandIntent.SWITCH_ACTIVE_OBJECT:
                self._handle_switch_active_object(runtime, command)
                continue
            if command.intent == ToolCommandIntent.RELEASE_TOOL:
                self._handle_release_tool(runtime, command)

    # Route GRASP_TOOL command: return to policy mode and apply hover grasp goal.
    def _handle_grasp_tool(self, runtime, command) -> None:
        """Apply grasp behavior and unpause interactive controls when needed."""
        enter_policy(runtime._tool_runtime_state)
        grasp_goals = runtime._build_grasp_hover_goals()
        runtime._apply_goals_live(grasp_goals)
        if runtime.viser is not None and runtime.viser.is_paused:
            runtime.viser.is_paused = False
            if hasattr(runtime.viser, "pause_button"):
                runtime.viser.pause_button.name = "Pause"
            if hasattr(runtime.viser, "_chat_pause_button"):
                runtime.viser._chat_pause_button.name = "Pause"
        runtime._write_llm_debug_event(
            {
                "event": "tool_cmd_applied",
                "intent": command.intent.value,
                "num_goals": len(grasp_goals),
                "hover_offset_m": float(
                    getattr(runtime._eval_args, "llm_grasp_hover_offset_m", 0.03)
                ),
            }
        )

    # Route MOVE_TOOL command: shift current goals and apply shifted sequence.
    def _handle_move_tool(self, runtime, command) -> None:
        """Apply translation+rotation delta over active goals and re-apply sequence."""
        if command.delta_translation_m is None and command.semantic_target is None:
            return
        delta_frame = command.delta_frame or "camera_spawn"
        delta_translation = command.delta_translation_m or [0.0, 0.0, 0.0]
        delta_euler = command.delta_euler_rad or [0.0, 0.0, 0.0]
        current_goals = runtime._get_current_goal_sequence()
        min_pose_z = runtime.TABLE_Z + float(getattr(runtime._eval_args, "z_offset", 0.03))
        shifted_goals = []
        if command.semantic_target is None:
            shifted_goals = shift_goals_pose_delta(
                goals=current_goals,
                delta_translation_m=delta_translation,
                delta_euler_rad=delta_euler,
                min_pose_z=min_pose_z,
                delta_frame=delta_frame,
            )
        else:
            if delta_frame != "camera_spawn":
                raise ValueError(f"Unsupported delta frame: {delta_frame}")
            preserve_position = (
                True
                if command.semantic_preserve_position is None
                else command.semantic_preserve_position
            )
            _, object_pose, _, _, _, _, _ = runtime._get_state()
            dq_semantic = compute_semantic_quat_delta(
                runtime.object_name, object_pose[3:7], command.semantic_target
            )
            dq_euler = quat_from_euler_xyz(
                float(delta_euler[0]),
                float(delta_euler[1]),
                float(delta_euler[2]),
            )
            dq_total = quat_mul_xyzw(dq_semantic, dq_euler)
            world_delta = (
                [0.0, 0.0, 0.0]
                if preserve_position
                else camera_spawn_delta_to_world(delta_translation)
            )
            shifted_goals = shift_goals_pose_quat_delta(
                goals=current_goals,
                world_delta_translation_m=world_delta,
                delta_quat_xyzw=dq_total,
                min_pose_z=min_pose_z,
            )
        if shifted_goals:
            enter_policy(runtime._tool_runtime_state)
            runtime._apply_goals_live(shifted_goals)
            runtime._write_llm_debug_event(
                {
                    "event": "tool_cmd_applied",
                    "intent": command.intent.value,
                    "delta_translation_m": delta_translation,
                    "delta_euler_rad": delta_euler,
                    "delta_frame": delta_frame,
                    "semantic_target": command.semantic_target,
                    "semantic_preserve_position": command.semantic_preserve_position,
                    "num_goals": len(shifted_goals),
                }
            )

    # Route RELEASE_TOOL command: apply pre-release goals and enter pre-release mode.
    def _handle_release_tool(self, runtime, command) -> None:
        """Freeze arm + open hand behavior until a new LLM goal command resumes policy."""
        enter_frozen(runtime._tool_runtime_state)
        runtime._write_llm_debug_event(
            {
                "event": "tool_cmd_applied",
                "intent": command.intent.value,
                "mode": runtime._tool_runtime_state.mode.value,
                "open_steps": int(
                    getattr(runtime, "_llm_release_open_steps", DEFAULT_RELEASE_OPEN_STEPS)
                ),
            }
        )

    # Route EXECUTE_LIE_TRAJECTORY command: compile coordinate-driven swing goals and apply them.
    def _handle_execute_lie_trajectory(self, runtime, command) -> None:
        """Compile one coordinate-driven Lie swing trajectory and apply it as active goals."""
        if command.strike_target_xy is None:
            raise ValueError("EXECUTE_LIE_TRAJECTORY requires strike_target_xy.")
        if command.object_name is not None and command.object_name != runtime.object_name:
            runtime._switch_active_object(command.object_name)
        compiled_goals = runtime._compile_lie_trajectory_goals(
            command.strike_target_xy,
            target_description=command.target_description,
        )
        enter_policy(runtime._tool_runtime_state)
        runtime._apply_goals_live(compiled_goals)
        runtime._write_llm_debug_event(
            {
                "event": "tool_cmd_applied",
                "intent": command.intent.value,
                "object_name": getattr(runtime, "object_name", None),
                "task_name": getattr(runtime, "task_name", None),
                "strike_target_xy": command.strike_target_xy,
                "target_description": command.target_description,
                "replace_active_goals": command.replace_active_goals,
                "num_goals": len(compiled_goals),
            }
        )

    # Route EXECUTE_PREDEFINED_SWING command: load recorded swing goals and apply them live.
    def _handle_execute_predefined_swing(self, runtime, command) -> None:
        """Load one recorded predefined swing trajectory and apply it as active goals."""
        if command.object_name is not None and command.object_name != runtime.object_name:
            runtime._switch_active_object(command.object_name)
        predefined_goals = runtime._load_predefined_swing_goals()
        enter_policy(runtime._tool_runtime_state)
        runtime._apply_goals_live(predefined_goals)
        runtime._write_llm_debug_event(
            {
                "event": "tool_cmd_applied",
                "intent": command.intent.value,
                "object_name": getattr(runtime, "object_name", None),
                "task_name": getattr(runtime, "task_name", None),
                "num_goals": len(predefined_goals),
            }
        )

    # Route SWITCH_ACTIVE_OBJECT command: swap the active preloaded object/runtime bundle.
    def _handle_switch_active_object(self, runtime, command) -> None:
        """Switch the active preloaded object/runtime bundle in-place."""
        if command.object_name is None:
            raise ValueError("SWITCH_ACTIVE_OBJECT requires object_name.")
        runtime._switch_active_object(command.object_name)
        runtime._write_llm_debug_event(
            {
                "event": "tool_cmd_applied",
                "intent": command.intent.value,
                "object_name": command.object_name,
            }
        )
