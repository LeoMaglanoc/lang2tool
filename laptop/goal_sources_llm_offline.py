"""CPU-only offline LLM trajectory viewer for predefined and Lie tool motions."""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import tyro

# Support script-style execution by ensuring the repo root is importable.
if (
    importlib.util.find_spec("geometric_tool_planning") is None
    or importlib.util.find_spec("laptop") is None
):
    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from dextoolbench.eval_config import (
    DEFAULT_CONTROL_HZ,
    DEFAULT_LLM_BACKEND,
    DEFAULT_LLM_LIE_HORIZONTAL_STRIKE_CLEARANCE_M,
    DEFAULT_LLM_LIE_SCREWDRIVER_TWIST_EXTRA_HOVER_M,
    DEFAULT_LLM_LIE_TRAINING_RESAMPLING_ENABLED,
    DEFAULT_LLM_LIE_TRAINING_RESAMPLING_MIN_WAYPOINTS,
    DEFAULT_LLM_LIE_TRAINING_RESAMPLING_POS_SCALE_M,
    DEFAULT_LLM_LIE_TRAINING_RESAMPLING_ROT_SCALE_DEG,
    DEFAULT_LLM_LIE_TRAINING_RESAMPLING_TARGET_COST,
    DEFAULT_LLM_LIE_TRAINING_TARGET_VOLUME_MAXS,
    DEFAULT_LLM_LIE_TRAINING_TARGET_VOLUME_MINS,
    DEFAULT_LLM_LIE_TRAINING_VOLUME_CLAMP_ENABLED,
    DEFAULT_LLM_LIE_WAYPOINT_TABLE_CLEARANCE_M,
    DEFAULT_OBJECT_NAME,
    DEFAULT_TASK_NAME,
    DEFAULT_VISER_PORT,
    DEFAULT_Z_OFFSET_M,
)
from dextoolbench.llm_lie_trajectory import compile_llm_lie_trajectory
from dextoolbench.object_start_poses import get_default_start_pose
from geometric_tool_planning import (
    GoalSourceArtifact,
    get_llm_static_strike_context,
    get_named_strike_point_payload,
    get_predefined_goal_sequence,
    get_predefined_strike_target_xy,
    is_target_a_strike_target_xy,
    sample_interval_sec,
)
from geometric_tool_planning.viewer import TABLE_TOP_Z, ToolTrajectoryViewer
from laptop.utils import log_info, render_chat_html, to_json_compatible
from llm_runtime import ToolCommand, ToolCommandIntent
from llm_runtime.chat.presentation import (
    DEFAULT_ASSISTANT_GREETING as _DEFAULT_ASSISTANT_GREETING,
)
from llm_runtime.chat.presentation import format_chat_command_summary
from llm_runtime.chat.service import ChatService
from llm_runtime.llm.chat_client import build_chat_client
from llm_runtime.semantic_pose import get_object_pose_semantics_payload

_ACTIVE_TARGET_OVERLAY_EPSILON_M = 1e-4
_SUPPORTED_OBJECT_TASKS = {
    "claw_hammer": "swing_down",
    "long_screwdriver": "spin_vertical",
}


@dataclass
class OfflineLLMGoalSourcesViewerArgs:
    """CLI args for the laptop-friendly offline LLM trajectory viewer."""

    object_name: str = DEFAULT_OBJECT_NAME
    """Supported object for offline viewer playback."""

    task_name: str = DEFAULT_TASK_NAME
    """Supported task for offline viewer playback."""

    llm_backend: str = DEFAULT_LLM_BACKEND
    """LLM backend: 'mock' or 'openai'."""

    llm_debug_log_path: Optional[Path] = None
    """Optional JSONL path for best-effort LLM chat debug logging."""

    llm_startup_chat_message: Optional[str] = None
    """Optional startup chat message injected once before the viewer loop begins."""

    control_hz: float = DEFAULT_CONTROL_HZ
    """Nominal playback control rate used to derive artifact durations."""

    z_offset: float = DEFAULT_Z_OFFSET_M
    """Vertical offset applied to the default object start pose for Lie motion compilation."""

    llm_lie_horizontal_strike_clearance_m: float = DEFAULT_LLM_LIE_HORIZONTAL_STRIKE_CLEARANCE_M
    """Execution-only clearance above the table during the horizontal Lie strike phase."""

    llm_lie_waypoint_table_clearance_m: float = DEFAULT_LLM_LIE_WAYPOINT_TABLE_CLEARANCE_M
    """Minimum clearance the tool support points must maintain above the table."""

    llm_lie_screwdriver_twist_extra_hover_m: float = DEFAULT_LLM_LIE_SCREWDRIVER_TWIST_EXTRA_HOVER_M
    """Extra hover margin added on top of screwdriver twist body-clearance solves."""

    llm_lie_training_resampling_enabled: bool = DEFAULT_LLM_LIE_TRAINING_RESAMPLING_ENABLED
    """Resample dense Lie waypoints toward RL-like step scales before workspace clamping."""

    llm_lie_training_resampling_pos_scale_m: float = DEFAULT_LLM_LIE_TRAINING_RESAMPLING_POS_SCALE_M
    """Translation scale used by the shared Lie training-distribution resampler."""

    llm_lie_training_resampling_rot_scale_deg: float = (
        DEFAULT_LLM_LIE_TRAINING_RESAMPLING_ROT_SCALE_DEG
    )
    """Rotation scale used by the shared Lie training-distribution resampler."""

    llm_lie_training_resampling_target_cost: float = DEFAULT_LLM_LIE_TRAINING_RESAMPLING_TARGET_COST
    """Combined SE(3) target progress between consecutive resampled Lie waypoints."""

    llm_lie_training_resampling_min_waypoints: int = (
        DEFAULT_LLM_LIE_TRAINING_RESAMPLING_MIN_WAYPOINTS
    )
    """Minimum waypoint count retained by the shared Lie training-distribution resampler."""

    llm_lie_training_volume_clamp_enabled: bool = DEFAULT_LLM_LIE_TRAINING_VOLUME_CLAMP_ENABLED
    """Clamp Lie waypoint positions into the RL training target volume after compilation."""

    llm_lie_training_target_volume_mins: List[float] = field(
        default_factory=lambda: list(DEFAULT_LLM_LIE_TRAINING_TARGET_VOLUME_MINS)
    )
    """Inclusive XYZ lower bounds for the Lie training-distribution clamp."""

    llm_lie_training_target_volume_maxs: List[float] = field(
        default_factory=lambda: list(DEFAULT_LLM_LIE_TRAINING_TARGET_VOLUME_MAXS)
    )
    """Inclusive XYZ upper bounds for the Lie training-distribution clamp."""

    port: int = DEFAULT_VISER_PORT
    """Viser server port for the trajectory viewer."""

    startup_only: bool = False
    """Initialize the offline viewer stack once and exit without entering the serve loop."""


# Mirror one one-shot goal list into a down-then-up cycle without duplicating turnaround endpoints.
def _build_mirrored_cycle(goals: Sequence[Sequence[float]]) -> List[List[float]]:
    """Return one mirrored cycle for a forward-only goal list."""
    forward_goals = [list(goal) for goal in goals]
    if len(forward_goals) <= 1:
        return forward_goals
    return forward_goals + [list(goal) for goal in reversed(forward_goals[1:-1])]


# Build one cyclic Lie playback chunk using either mirrored or forward-repeat semantics.
def _build_lie_cycle(
    goals: Sequence[Sequence[float]],
    *,
    llm_raw: Optional[Dict[str, object]],
) -> List[List[float]]:
    """Return one Lie playback chunk with object-specific cycle semantics."""
    if isinstance(llm_raw, dict) and llm_raw.get("cycle_style") == "forward_repeat":
        return [list(goal) for goal in goals]
    return _build_mirrored_cycle(goals)


class OfflineLLMGoalSourcesViewerRunner:
    """Own the offline viewer-only LLM chat flow and active trajectory replacement."""

    # Build the viewer, initial artifacts, and chat state for one offline LLM session.
    def __init__(self, args: OfflineLLMGoalSourcesViewerArgs) -> None:
        self._args = args
        self.object_name, self.task_name = self._resolve_supported_object_task(
            args.object_name, args.task_name
        )
        self._llm_debug_log_path = args.llm_debug_log_path
        self._chat_history: List[Tuple[str, str]] = [("assistant", _DEFAULT_ASSISTANT_GREETING)]
        self._chat_client = build_chat_client(args.llm_backend)
        self._chat_service = ChatService()
        self._viewer_state_by_object = {
            object_name: self._build_object_viewer_state(object_name)
            for object_name in _SUPPORTED_OBJECT_TASKS
        }
        current_state = self._viewer_state_by_object[self.object_name]
        self._current_start_pose = list(current_state["start_pose"])
        self.viewer = ToolTrajectoryViewer(
            object_name=self.object_name,
            artifacts=current_state["artifacts"],
            port=args.port,
            use_tabs=True,
            preloaded_object_names=tuple(_SUPPORTED_OBJECT_TASKS),
        )
        self._target_a_world_xyz = current_state["target_a_world_xyz"]
        self._named_strike_point_marker = None
        self._named_strike_point_label = None
        self._active_strike_target_marker = None
        self._active_strike_target_label = None
        self._chat_html = None
        self._chat_input = None
        self._chat_input_update_suppressed = False
        self._install_strike_target_overlays()
        self._install_chat_panel()
        self._refresh_strike_target_overlays()
        if self.viewer.object_dropdown is not None:
            self.viewer.object_dropdown.on_update(lambda _: self._on_object_dropdown_change())

    # Resolve one supported object/task pair for the offline multi-object viewer.
    def _resolve_supported_object_task(self, object_name: str, task_name: str) -> Tuple[str, str]:
        """Return one validated object/task pair supported by the offline viewer."""
        resolved_object_name = str(object_name)
        if resolved_object_name not in _SUPPORTED_OBJECT_TASKS:
            raise ValueError(
                "goal_sources_llm_offline.py currently supports only "
                "claw_hammer / swing_down and long_screwdriver / spin_vertical."
            )
        resolved_task_name = _SUPPORTED_OBJECT_TASKS[resolved_object_name]
        if task_name not in (resolved_task_name, DEFAULT_TASK_NAME):
            raise ValueError(
                f"Unsupported task `{task_name}` for `{resolved_object_name}`; "
                f"use `{resolved_task_name}`."
            )
        return resolved_object_name, resolved_task_name

    # Build one preloaded viewer-state bundle for a supported object/task pair.
    def _build_object_viewer_state(self, object_name: str) -> Dict[str, object]:
        """Return prebuilt artifacts and metadata for one supported object."""
        task_name = _SUPPORTED_OBJECT_TASKS[object_name]
        previous_object_name = getattr(self, "object_name", None)
        previous_task_name = getattr(self, "task_name", None)
        previous_start_pose = getattr(self, "_current_start_pose", None)
        self.object_name = object_name
        self.task_name = task_name
        self._current_start_pose = self._build_default_start_pose()
        predefined_artifact = self._build_predefined_swing_artifact()
        predefined_target_xy = get_predefined_strike_target_xy(
            object_name=self.object_name,
            task_name=self.task_name,
        )
        lie_artifact = self._build_lie_artifact(
            strike_target_xy=(
                predefined_target_xy if predefined_target_xy is not None else [0.0, 0.0]
            ),
            target_description=(
                "strike point a" if self.object_name == "claw_hammer" else "predefined target"
            ),
        )
        state = {
            "task_name": task_name,
            "start_pose": list(self._current_start_pose),
            "artifacts_by_mode": {
                predefined_artifact.mode: predefined_artifact,
                lie_artifact.mode: lie_artifact,
            },
            "artifacts": [predefined_artifact, lie_artifact],
            "target_a_world_xyz": self._resolve_target_a_world_xyz(),
        }
        if previous_object_name is not None and previous_task_name is not None:
            self.object_name = previous_object_name
            self.task_name = previous_task_name
            if previous_start_pose is not None:
                self._current_start_pose = previous_start_pose
        return state

    # Return the default world-frame object start pose used as Lie compilation anchor.
    def _build_default_start_pose(self) -> List[float]:
        """Return the default start pose with the same z-offset used by live LLM mode."""
        start_pose = list(get_default_start_pose(self.object_name))
        start_pose[2] = max(start_pose[2] + float(self._args.z_offset), TABLE_TOP_Z)
        return start_pose

    # Build one GoalSourceArtifact from a pose sequence and semantic strike target metadata.
    def _build_artifact(
        self,
        *,
        mode: str,
        goals: List[List[float]],
        strike_target_xy: Sequence[float],
        llm_raw: Optional[Dict[str, object]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> GoalSourceArtifact:
        """Return one viewer artifact with consistent timing and strike-target metadata."""
        duration_sec = float(len(goals) / max(self._args.control_hz, 1e-6))
        artifact_metadata: Dict[str, object] = dict(metadata or {})
        artifact_spec = dict(artifact_metadata.get("spec", {}))
        artifact_spec["strike_target_xy"] = [float(value) for value in strike_target_xy]
        artifact_metadata["spec"] = artifact_spec
        return GoalSourceArtifact(
            mode=mode,
            goals=[list(goal) for goal in goals],
            duration_sec=duration_sec,
            sample_interval_sec=sample_interval_sec(duration_sec, len(goals)),
            metrics={},
            metadata=artifact_metadata,
            llm_raw=llm_raw,
            execution_metrics={},
        )

    # Build the mirrored predefined motion artifact shown in offline LLM mode.
    def _build_predefined_swing_artifact(self) -> GoalSourceArtifact:
        """Return the recorded predefined motion as a mirrored cyclic playback artifact."""
        goals = _build_mirrored_cycle(
            get_predefined_goal_sequence(
                object_name=self.object_name,
                task_name=self.task_name,
                include_start_pose=False,
            )
        )
        strike_target_xy = get_predefined_strike_target_xy(
            object_name=self.object_name,
            task_name=self.task_name,
        )
        return self._build_artifact(
            mode="predefined",
            goals=goals,
            strike_target_xy=strike_target_xy,
            metadata={"source": "predefined_motion"},
        )

    # Build one cyclic Lie motion artifact from explicit tabletop strike-target coordinates.
    def _build_lie_artifact(
        self,
        *,
        strike_target_xy: Sequence[float],
        target_description: Optional[str],
    ) -> GoalSourceArtifact:
        """Return one cyclic Lie motion artifact for the requested tabletop strike target."""
        clamped_target_xy, spec, compiled_goals, clamp_summary = compile_llm_lie_trajectory(
            object_name=self.object_name,
            task_name=self.task_name,
            pivot_point=self._current_start_pose[:3],
            strike_target_xy=strike_target_xy,
            horizontal_strike_clearance_m=float(self._args.llm_lie_horizontal_strike_clearance_m),
            waypoint_table_clearance_m=float(self._args.llm_lie_waypoint_table_clearance_m),
            screwdriver_twist_extra_hover_m=float(
                self._args.llm_lie_screwdriver_twist_extra_hover_m
            ),
            training_resampling_enabled=bool(
                getattr(self._args, "llm_lie_training_resampling_enabled", True)
            ),
            training_resampling_pos_scale_m=float(
                getattr(
                    self._args,
                    "llm_lie_training_resampling_pos_scale_m",
                    DEFAULT_LLM_LIE_TRAINING_RESAMPLING_POS_SCALE_M,
                )
            ),
            training_resampling_rot_scale_deg=float(
                getattr(
                    self._args,
                    "llm_lie_training_resampling_rot_scale_deg",
                    DEFAULT_LLM_LIE_TRAINING_RESAMPLING_ROT_SCALE_DEG,
                )
            ),
            training_resampling_target_cost=float(
                getattr(
                    self._args,
                    "llm_lie_training_resampling_target_cost",
                    DEFAULT_LLM_LIE_TRAINING_RESAMPLING_TARGET_COST,
                )
            ),
            training_resampling_min_waypoints=int(
                getattr(
                    self._args,
                    "llm_lie_training_resampling_min_waypoints",
                    DEFAULT_LLM_LIE_TRAINING_RESAMPLING_MIN_WAYPOINTS,
                )
            ),
            training_volume_clamp_enabled=bool(
                getattr(self._args, "llm_lie_training_volume_clamp_enabled", True)
            ),
            training_target_volume_mins=list(
                getattr(
                    self._args,
                    "llm_lie_training_target_volume_mins",
                    DEFAULT_LLM_LIE_TRAINING_TARGET_VOLUME_MINS,
                )
            ),
            training_target_volume_maxs=list(
                getattr(
                    self._args,
                    "llm_lie_training_target_volume_maxs",
                    DEFAULT_LLM_LIE_TRAINING_TARGET_VOLUME_MAXS,
                )
            ),
        )
        goals = _build_lie_cycle(compiled_goals, llm_raw=spec)
        return self._build_artifact(
            mode="llm_lie",
            goals=goals,
            strike_target_xy=clamped_target_xy,
            llm_raw=spec,
            metadata={
                "source": "llm_lie",
                "spec": dict(spec),
                "target_description": target_description,
                "training_distribution_resampling": spec.get(
                    "training_distribution_resampling", {}
                ),
                "training_distribution_clamp": clamp_summary,
            },
        )

    # Replace one existing viewer mode artifact in place and switch playback to it immediately.
    def _install_active_artifact(self, artifact: GoalSourceArtifact) -> None:
        """Replace one stored viewer artifact and make it the active playback source."""
        object_state = self._viewer_state_by_object[self.object_name]
        object_state["artifacts_by_mode"][artifact.mode] = artifact
        object_state["artifacts"] = [
            object_state["artifacts_by_mode"][mode] for mode in ("predefined", "llm_lie")
        ]
        self.viewer.artifacts_by_mode[artifact.mode] = artifact
        if artifact.mode not in self.viewer.mode_order:
            self.viewer.mode_order.append(artifact.mode)
        self.viewer._activate_source(artifact.mode)
        self._refresh_strike_target_overlays()
        self._write_llm_debug_event(
            {
                "event": "offline_viewer_artifact_installed",
                "object_name": self.object_name,
                "task_name": self.task_name,
                "mode": artifact.mode,
                "goals": artifact.goals,
                "strike_target_xy": artifact.metadata.get("spec", {}).get("strike_target_xy"),
                "llm_raw": artifact.llm_raw,
                "metadata": artifact.metadata,
            }
        )

    # Resolve the persistent named strike point a marker from the recorded predefined swing target.
    def _resolve_target_a_world_xyz(self) -> Optional[Tuple[float, float, float]]:
        """Return the world XYZ location of strike point a when available."""
        if self.object_name != "claw_hammer":
            return None
        strike_target_xy = get_predefined_strike_target_xy(
            object_name=self.object_name,
            task_name=self.task_name,
        )
        if strike_target_xy is None:
            return None
        return (float(strike_target_xy[0]), float(strike_target_xy[1]), float(TABLE_TOP_Z))

    # Create separate fixed and active strike-target overlays for offline LLM mode.
    def _install_strike_target_overlays(self) -> None:
        """Install one fixed strike point marker and one active strike target marker."""
        self._named_strike_point_marker = self.viewer.server.scene.add_icosphere(
            "/offline_debug/strike_point_a_marker",
            radius=0.012,
            color=(255, 140, 0),
        )
        self._named_strike_point_label = self.viewer.server.scene.add_label(
            "/offline_debug/strike_point_a_label",
            text="strike point a",
        )
        self._active_strike_target_marker = self.viewer.server.scene.add_icosphere(
            "/offline_debug/active_strike_target_marker",
            radius=0.01,
            color=(0, 255, 170),
        )
        self._active_strike_target_label = self.viewer.server.scene.add_label(
            "/offline_debug/active_strike_target_label",
            text="active target",
        )

    # Update the fixed and active strike-target overlays to match the current artifact.
    def _refresh_strike_target_overlays(self) -> None:
        """Update the fixed and active strike-target overlays for the active playback artifact."""
        if self._target_a_world_xyz is None:
            self._named_strike_point_marker.visible = False
            self._named_strike_point_label.visible = False
        else:
            target_a_position = self._target_a_world_xyz
            self._named_strike_point_marker.visible = True
            self._named_strike_point_marker.position = target_a_position
            self._named_strike_point_label.visible = True
            self._named_strike_point_label.position = (
                target_a_position[0],
                target_a_position[1],
                target_a_position[2] + 0.03,
            )

        active_target_xy = list(
            self.viewer._active_artifact().metadata.get("spec", {}).get("strike_target_xy", [])
        )
        if len(active_target_xy) != 2:
            self._active_strike_target_marker.visible = False
            self._active_strike_target_label.visible = False
            return
        active_position = (
            float(active_target_xy[0]),
            float(active_target_xy[1]),
            float(TABLE_TOP_Z),
        )
        matches_target_a = self._target_a_world_xyz is not None and is_target_a_strike_target_xy(
            active_target_xy,
            object_name=self.object_name,
            task_name=self.task_name,
            tolerance_m=_ACTIVE_TARGET_OVERLAY_EPSILON_M,
        )
        self._active_strike_target_marker.visible = not matches_target_a
        self._active_strike_target_label.visible = not matches_target_a
        self._active_strike_target_marker.position = active_position
        self._active_strike_target_label.position = (
            active_position[0],
            active_position[1],
            active_position[2] + 0.025,
        )

    # Return offline viewer-state payload used by the chat client instead of live sim state.
    def _build_offline_sim_context(self) -> Dict[str, object]:
        """Return one chat grounding payload from the active offline viewer artifact."""
        active_artifact = self.viewer._active_artifact()
        current_pose = [
            float(value) for value in active_artifact.goals[self.viewer.playback.frame_index]
        ]
        strike_target_xy = list(
            active_artifact.metadata.get("spec", {}).get("strike_target_xy", [])
        )
        return {
            "object_name": self.object_name,
            "task_name": self.task_name,
            "current_object": self.object_name,
            "current_task": self.task_name,
            "runtime_mode": "offline_viewer",
            "object_pose_xyzw": current_pose,
            "goal_pose_xyzw": current_pose,
            "pose_semantics": get_object_pose_semantics_payload(self.object_name),
            "active_strike_target": {
                "available": len(strike_target_xy) == 2,
                "world_xy": strike_target_xy if len(strike_target_xy) == 2 else None,
                "world_xyz": (
                    [float(strike_target_xy[0]), float(strike_target_xy[1]), float(TABLE_TOP_Z)]
                    if len(strike_target_xy) == 2
                    else None
                ),
                "description": active_artifact.metadata.get("target_description"),
            },
            "named_strike_points": get_named_strike_point_payload(
                object_name=self.object_name,
                task_name=self.task_name,
            ),
            "static_strike_context": get_llm_static_strike_context(
                object_name=self.object_name,
                task_name=self.task_name,
            ),
        }

    # Append one best-effort JSONL debug event when llm_debug_log_path is configured.
    def _write_llm_debug_event(self, payload: Dict[str, object]) -> None:
        """Write one JSONL debug event when llm_debug_log_path is configured."""
        if self._llm_debug_log_path is None:
            return
        self._llm_debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        event_payload = {"timestamp_utc": datetime.utcnow().isoformat() + "Z"}
        event_payload.update(to_json_compatible(payload))
        with self._llm_debug_log_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(event_payload) + "\n")

    # Render one short user-facing summary of the applied offline command.
    def _format_executed_command_summary(self, command: ToolCommand) -> str:
        """Return one compact summary of the applied offline command."""
        return format_chat_command_summary(command)

    # Apply one supported LLM command by rebuilding the active viewer artifact in place.
    def _apply_supported_command(self, command: ToolCommand) -> str:
        """Apply one supported offline command and return a user-facing summary."""
        if command.intent == ToolCommandIntent.SWITCH_ACTIVE_OBJECT:
            if command.object_name is None:
                raise ValueError("switch_active_object requires object_name.")
            self._switch_active_object(command.object_name)
            return self._format_executed_command_summary(command)
        if command.intent == ToolCommandIntent.EXECUTE_LIE_TRAJECTORY:
            if command.strike_target_xy is None:
                raise ValueError("execute_lie_trajectory requires strike_target_xy.")
            if command.object_name is not None and command.object_name != self.object_name:
                self._switch_active_object(command.object_name)
            artifact = self._build_lie_artifact(
                strike_target_xy=command.strike_target_xy,
                target_description=command.target_description,
            )
            self._install_active_artifact(artifact)
            return self._format_executed_command_summary(command)
        if command.intent == ToolCommandIntent.EXECUTE_PREDEFINED_SWING:
            if command.object_name is not None and command.object_name != self.object_name:
                self._switch_active_object(command.object_name)
            artifact = self._build_predefined_swing_artifact()
            self._install_active_artifact(artifact)
            return self._format_executed_command_summary(command)
        raise ValueError(f"Unsupported command `{command.intent.value}` in offline viewer mode.")

    # Switch the active object bundle in response to UI or chat commands.
    def _switch_active_object(self, object_name: str) -> None:
        """Swap the active preloaded object, artifacts, and overlay metadata."""
        resolved_object_name, resolved_task_name = self._resolve_supported_object_task(
            object_name, _SUPPORTED_OBJECT_TASKS[str(object_name)]
        )
        self.object_name = resolved_object_name
        self.task_name = resolved_task_name
        object_state = self._viewer_state_by_object[self.object_name]
        self._current_start_pose = list(object_state["start_pose"])
        self._target_a_world_xyz = object_state["target_a_world_xyz"]
        self.viewer.switch_object(self.object_name, object_state["artifacts"])
        self._refresh_strike_target_overlays()

    # Apply object switching from the controls dropdown without going through the LLM.
    def _on_object_dropdown_change(self) -> None:
        """Switch the active object when the viewer object dropdown changes."""
        if self.viewer._suppress_callbacks or self.viewer.object_dropdown is None:
            return
        self._switch_active_object(str(self.viewer.object_dropdown.value))

    # Refresh the rendered chat HTML after appending or replacing chat history entries.
    def _update_chat_history(self) -> None:
        """Mirror the current chat history into the viewer HTML panel."""
        if self._chat_html is None:
            return
        self._chat_html.content = render_chat_html(self._chat_history)

    # Consume one pending chat input value, clear the text box, and avoid recursive update callbacks.
    def _consume_chat_input_value(self) -> None:
        """Read one pending chat input value and submit it once without recursive callbacks."""
        if self._chat_input is None or self._chat_input_update_suppressed:
            return
        text_value = str(self._chat_input.value)
        if "\n" not in text_value:
            return
        message_text = text_value.replace("\n", "").strip()
        self._chat_input_update_suppressed = True
        try:
            self._chat_input.value = ""
        finally:
            self._chat_input_update_suppressed = False
        if message_text:
            self._handle_chat_send(message_text)

    # Process one user chat message and apply only the supported offline commands.
    def _handle_chat_send(self, message_text: str) -> None:
        """Process one chat turn against the offline viewer state."""
        stripped_text = message_text.strip()
        if not stripped_text:
            return
        self._chat_history.append(("user", stripped_text))
        self._chat_history.append(("assistant", "typing..."))
        self._update_chat_history()

        try:
            response = self._chat_service.send_to_llm(
                self._chat_client,
                self._chat_history[:-1],
                sim_context=self._build_offline_sim_context(),
            )
            if response.command is not None:
                assistant_text = self._apply_supported_command(response.command)
            else:
                assistant_text = response.text
        except Exception as exc:
            assistant_text = f"Error while handling chat: {exc}"
            self._write_llm_debug_event(
                {
                    "event": "offline_llm_chat_error",
                    "message_text": stripped_text,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        self._chat_history[-1] = ("assistant", assistant_text)
        self._update_chat_history()

    # Install the dedicated chat HTML panel and text input in the viewer chat tab.
    def _install_chat_panel(self) -> None:
        """Populate the viewer chat tab with history and a text input."""
        chat_container = self.viewer.chat_gui_container or self.viewer.server.gui
        with chat_container:
            self.viewer.server.gui.add_markdown("## LLM Chat")
            self.viewer.server.gui.add_markdown(
                "Switch between hammer and screwdriver, request Lie placement, or ask for "
                "the predefined motion."
            )
            self._chat_html = self.viewer.server.gui.add_html(render_chat_html(self._chat_history))
            self._chat_input = self.viewer.server.gui.add_text(
                "Message (Enter to send)",
                initial_value="",
                multiline=True,
            )

            @self._chat_input.on_update
            def _(_) -> None:
                self._consume_chat_input_value()

    # Run the offline viewer loop forever and optionally send one startup chat command.
    def run_forever(self) -> None:
        """Serve the offline viewer and animate the active tool indefinitely."""
        if self._args.llm_startup_chat_message:
            self._handle_chat_send(self._args.llm_startup_chat_message)
        log_info(f"Offline laptop viewer running at {getattr(self.viewer.server, 'url', '')}")
        self.viewer.run_forever()


# Parse CLI args and run the offline laptop LLM trajectory playback loop.
def main() -> None:
    """Entry point for the laptop-friendly offline LLM trajectory viewer."""
    args = tyro.cli(OfflineLLMGoalSourcesViewerArgs)
    runner = OfflineLLMGoalSourcesViewerRunner(args)
    if args.startup_only:
        if args.llm_startup_chat_message:
            runner._handle_chat_send(args.llm_startup_chat_message)
        log_info("Offline laptop viewer startup completed.")
        return
    runner.run_forever()


if __name__ == "__main__":
    main()
