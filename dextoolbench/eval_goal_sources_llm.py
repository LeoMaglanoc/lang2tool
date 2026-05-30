"""LLM-driven kinematics-only trajectory viewer for Lie motions and predefined replay."""

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

# Support script-style execution (`python3 dextoolbench/eval_goal_sources_llm.py`) by ensuring
# repo root is importable before local package imports.
if importlib.util.find_spec("geometric_tool_planning") is None:
    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from dextoolbench.eval import _render_chat_html, _to_json_compatible, log_warn
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
    DEFAULT_TASK_NAME,
    DEFAULT_VISER_PORT,
    DEFAULT_Z_OFFSET_M,
)
from dextoolbench.llm_lie_trajectory import compile_llm_lie_trajectory
from dextoolbench.llm_supported_objects import (
    SUPPORTED_LLM_OBJECT_TASKS,
    supported_llm_data_structure,
    supported_llm_object_family,
    supported_llm_object_names,
    supported_llm_task_name,
)
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
from llm_runtime import ToolCommand, ToolCommandIntent
from llm_runtime.chat.presentation import (
    DEFAULT_ASSISTANT_GREETING as _DEFAULT_ASSISTANT_GREETING,
)
from llm_runtime.chat.presentation import (
    format_chat_command_summary,
)
from llm_runtime.chat.service import ChatService
from llm_runtime.llm.chat_client import ChatResponse, build_chat_client
from llm_runtime.semantic_pose import get_object_pose_semantics_payload

_ACTIVE_TARGET_OVERLAY_EPSILON_M = 1e-4


# Build a flat ordered list of selectable objects grouped by category.
def _build_object_selection_choices(
    data_structure: Dict[str, Dict[str, List[str]]], preferred_category: Optional[str] = None
) -> List[Tuple[str, str]]:
    """Return ordered `(category, object_name)` pairs for terminal selection."""
    categories = list(data_structure.keys())
    if preferred_category and preferred_category in data_structure:
        categories = [preferred_category] + [c for c in categories if c != preferred_category]

    choices: List[Tuple[str, str]] = []
    for category in categories:
        for object_name in data_structure[category].keys():
            choices.append((category, object_name))
    return choices


# Prompt the user to choose one object name from all available tools.
def _prompt_for_object_name(
    data_structure: Dict[str, Dict[str, List[str]]], preferred_category: Optional[str] = None
) -> str:
    """Interactively select an object by index or exact name; re-prompts on invalid input."""
    if not sys.stdin.isatty():
        raise ValueError("No TTY available. Pass `--object_name` for non-interactive startup.")

    choices = _build_object_selection_choices(
        data_structure=data_structure, preferred_category=preferred_category
    )
    if not choices:
        raise ValueError("No objects available in metadata for selection.")

    print("\nAvailable tools:")
    current_category = None
    for idx, (category, object_name) in enumerate(choices, start=1):
        if category != current_category:
            print(f"\n[{category}]")
            current_category = category
        print(f"  {idx:2d}) {object_name}")

    while True:
        raw = input("\nSelect tool by number or name (q to quit): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            raise SystemExit("Tool selection aborted by user.")
        if not raw:
            print("Please enter a number or object name.")
            continue

        if raw.isdigit():
            selected_idx = int(raw)
            if 1 <= selected_idx <= len(choices):
                return choices[selected_idx - 1][1]
            print(f"Invalid index: {selected_idx}. Choose between 1 and {len(choices)}.")
            continue

        exact_matches = [obj for _, obj in choices if obj == raw]
        if len(exact_matches) == 1:
            return exact_matches[0]
        insensitive_matches = [obj for _, obj in choices if obj.lower() == raw.lower()]
        if len(insensitive_matches) == 1:
            return insensitive_matches[0]
        print(f"Unknown object '{raw}'. Enter a listed index or exact object name.")


# Return one validated startup object restricted to the supported LLM viewer objects.
def _validate_supported_startup_object_name(object_name: str) -> str:
    """Return one validated supported startup object name."""
    resolved_object_name = str(object_name)
    if resolved_object_name not in SUPPORTED_LLM_OBJECT_TASKS:
        raise ValueError(
            f"Unsupported --object_name `{resolved_object_name}`. "
            f"Supported objects: {', '.join(supported_llm_object_names())}."
        )
    return resolved_object_name


# Resolve startup object name, prompting in terminal when not provided by CLI.
def _resolve_startup_object_name(args: "LLMGoalSourcesViewerArgs") -> str:
    """Return startup object name from args or interactive terminal selection."""
    if args.object_name:
        return _validate_supported_startup_object_name(args.object_name)
    return _prompt_for_object_name(data_structure=supported_llm_data_structure())


@dataclass
class LLMGoalSourcesViewerArgs:
    """CLI args for the viewer-only LLM trajectory playback entrypoint."""

    object_name: Optional[str] = None
    """Optional supported object. If omitted, startup prompts for tool selection in terminal."""

    task_name: tyro.conf.Suppress[str] = DEFAULT_TASK_NAME
    """Suppressed because the viewer derives the supported task from the selected object."""

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


class LLMGoalSourcesViewerRunner:
    """Own the viewer-only LLM chat flow and active trajectory replacement."""

    # Build the viewer, initial artifacts, and chat state for one viewer-only LLM session.
    def __init__(self, args: LLMGoalSourcesViewerArgs) -> None:
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
            for object_name in SUPPORTED_LLM_OBJECT_TASKS
        }
        current_state = self._viewer_state_by_object[self.object_name]
        self._current_start_pose = list(current_state["start_pose"])
        self.viewer = ToolTrajectoryViewer(
            object_name=self.object_name,
            artifacts=current_state["artifacts"],
            port=args.port,
            use_tabs=True,
            preloaded_object_names=tuple(SUPPORTED_LLM_OBJECT_TASKS),
        )
        self._target_a_world_xyz = current_state["target_a_world_xyz"]
        self._named_strike_point_marker = None
        self._named_strike_point_label = None
        self._active_strike_target_marker = None
        self._active_strike_target_label = None
        self._chat_html = None
        self._chat_input = None
        self._install_strike_target_overlays()
        self._install_chat_panel()
        self._refresh_strike_target_overlays()
        if self.viewer.object_dropdown is not None:
            self.viewer.object_dropdown.on_update(lambda _: self._on_object_dropdown_change())

    # Resolve one supported object/task pair for the multi-object viewer.
    def _resolve_supported_object_task(self, object_name: str, task_name: str) -> Tuple[str, str]:
        """Return one validated object/task pair supported by the viewer."""
        resolved_object_name = str(object_name)
        if resolved_object_name not in SUPPORTED_LLM_OBJECT_TASKS:
            raise ValueError(
                "eval_goal_sources_llm.py currently supports only "
                "claw_hammer, mallet_hammer, cuboid_hammer_v014, long_screwdriver, "
                "short_screwdriver, and cylinder_screwdriver_v3009."
            )
        resolved_task_name = supported_llm_task_name(resolved_object_name)
        if task_name not in (resolved_task_name, DEFAULT_TASK_NAME):
            raise ValueError(
                f"Unsupported task `{task_name}` for `{resolved_object_name}`; "
                f"use `{resolved_task_name}`."
            )
        return resolved_object_name, resolved_task_name

    # Build one preloaded viewer-state bundle for a supported object/task pair.
    def _build_object_viewer_state(self, object_name: str) -> Dict[str, object]:
        """Return prebuilt artifacts and metadata for one supported object."""
        task_name = supported_llm_task_name(object_name)
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
                "strike point a"
                if supported_llm_object_family(self.object_name) == "hammer"
                else "predefined target"
            ),
        )
        target_a_world_xyz = self._resolve_target_a_world_xyz()
        state = {
            "task_name": task_name,
            "start_pose": list(self._current_start_pose),
            "artifacts_by_mode": {
                predefined_artifact.mode: predefined_artifact,
                lie_artifact.mode: lie_artifact,
            },
            "artifacts": [predefined_artifact, lie_artifact],
            "target_a_world_xyz": target_a_world_xyz,
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
        artifact_spec["strike_target_xy"] = [float(v) for v in strike_target_xy]
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

    # Build the mirrored predefined motion artifact shown in viewer-only LLM mode.
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
                    "training_distribution_resampling",
                    {},
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

    # Resolve the fixed named strike-point-a world position shared with eval_llm overlays.
    def _resolve_target_a_world_xyz(self) -> Optional[Tuple[float, float, float]]:
        """Return the fixed strike-point-a world position when available."""
        if supported_llm_object_family(self.object_name) != "hammer":
            return None
        payload = get_named_strike_point_payload(
            object_name=self.object_name,
            task_name=self.task_name,
        )
        target_a_world_xyz = payload.get("points", {}).get("target_a", {}).get("world_xyz")
        if not isinstance(target_a_world_xyz, list) or len(target_a_world_xyz) != 3:
            return None
        return tuple(float(value) for value in target_a_world_xyz)

    # Add separate fixed and active strike-target overlay handles on top of the trajectory viewer.
    def _install_strike_target_overlays(self) -> None:
        """Create separate fixed and active strike-target overlays for viewer-only LLM mode."""
        self.viewer.strike_target_marker.visible = False
        self._named_strike_point_marker = self.viewer.server.scene.add_icosphere(
            "/llm_goal_sources/named_strike_point_a",
            radius=0.017,
            color=(255, 215, 0),
            visible=False,
        )
        self._named_strike_point_label = self.viewer.server.scene.add_label(
            "/llm_goal_sources/named_strike_point_a_label",
            text="strike point a",
            position=(0.0, 0.0, 0.0),
        )
        self._named_strike_point_label.visible = False
        self._active_strike_target_marker = self.viewer.server.scene.add_icosphere(
            "/llm_goal_sources/active_strike_target",
            radius=0.018,
            color=(255, 99, 71),
            visible=False,
        )
        self._active_strike_target_label = self.viewer.server.scene.add_label(
            "/llm_goal_sources/active_strike_target_label",
            text="active swing target",
            position=(0.0, 0.0, 0.0),
        )
        self._active_strike_target_label.visible = False

    # Update the viewer-only fixed and active strike-target overlays to match eval_llm semantics.
    def _refresh_strike_target_overlays(self) -> None:
        """Show strike point a persistently and suppress the active marker when targets match."""
        if self._target_a_world_xyz is not None:
            self._named_strike_point_marker.position = self._target_a_world_xyz
            self._named_strike_point_marker.visible = True
            self._named_strike_point_label.position = (
                self._target_a_world_xyz[0],
                self._target_a_world_xyz[1],
                self._target_a_world_xyz[2] + 0.035,
            )
            self._named_strike_point_label.visible = True
        else:
            self._named_strike_point_marker.visible = False
            self._named_strike_point_label.visible = False
        active_target_xy = (
            self.viewer._active_artifact().metadata.get("spec", {}).get("strike_target_xy", [])
        )
        if not isinstance(active_target_xy, list) or len(active_target_xy) != 2:
            self._active_strike_target_marker.visible = False
            self._active_strike_target_label.visible = False
            return
        if self._target_a_world_xyz is not None and is_target_a_strike_target_xy(
            active_target_xy,
            object_name=self.object_name,
            task_name=self.task_name,
            tolerance_m=_ACTIVE_TARGET_OVERLAY_EPSILON_M,
        ):
            self._active_strike_target_marker.visible = False
            self._active_strike_target_label.visible = False
            return
        active_world_xyz = (
            float(active_target_xy[0]),
            float(active_target_xy[1]),
            float(TABLE_TOP_Z),
        )
        self._active_strike_target_marker.position = active_world_xyz
        self._active_strike_target_marker.visible = True
        self._active_strike_target_label.position = (
            active_world_xyz[0],
            active_world_xyz[1],
            active_world_xyz[2] + 0.035,
        )
        self._active_strike_target_label.visible = True

    # Render a compact execution summary for one supported viewer-only command.
    def _format_executed_command_summary(self, command: ToolCommand) -> str:
        """Return a short user-facing summary of the applied viewer-only command."""
        return format_chat_command_summary(command)

    # Apply one supported LLM command by rebuilding the active viewer artifact in place.
    def _apply_supported_command(self, command: ToolCommand) -> str:
        """Apply one Lie or predefined-motion command and return its summary text."""
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
        raise ValueError(f"Unsupported command `{command.intent.value}` in viewer-only mode.")

    # Switch the active object bundle in response to UI or chat commands.
    def _switch_active_object(self, object_name: str) -> None:
        """Swap the active preloaded object, artifacts, and overlay metadata."""
        resolved_object_name, resolved_task_name = self._resolve_supported_object_task(
            object_name, supported_llm_task_name(str(object_name))
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

    # Return the current viewer pose plus static strike context consumed by the LLM backends.
    def _build_llm_sim_context(self) -> Dict[str, object]:
        """Return static strike geometry and current playback-frame pose for LLM calls."""
        active_artifact = self.viewer._active_artifact()
        current_pose = [float(v) for v in active_artifact.goals[self.viewer.playback.frame_index]]
        strike_target_xy = list(
            active_artifact.metadata.get("spec", {}).get("strike_target_xy", [])
        )
        active_strike_target = {
            "available": len(strike_target_xy) == 2,
            "world_xy": strike_target_xy if len(strike_target_xy) == 2 else None,
            "world_xyz": (
                [float(strike_target_xy[0]), float(strike_target_xy[1]), float(TABLE_TOP_Z)]
                if len(strike_target_xy) == 2
                else None
            ),
            "description": active_artifact.metadata.get("target_description"),
        }
        return {
            "current_object": self.object_name,
            "current_task": self.task_name,
            "start_pose": list(self._current_start_pose),
            "static_strike_context": get_llm_static_strike_context(
                object_name=self.object_name,
                task_name=self.task_name,
            ),
            "sim_state": {
                "object_name": self.object_name,
                "runtime_mode": "viewer_only",
                "object_pose_xyzw": current_pose,
                "goal_pose_xyzw": current_pose,
                "pose_semantics": get_object_pose_semantics_payload(self.object_name),
                "active_strike_target": active_strike_target,
            },
        }

    # Append one best-effort JSONL debug event when llm_debug_log_path is configured.
    def _write_llm_debug_event(self, event: Dict[str, object]) -> None:
        """Write one JSONL debug event when llm_debug_log_path is configured."""
        if self._llm_debug_log_path is None:
            return
        self._llm_debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ts": datetime.utcnow().isoformat() + "Z", **event}
        with self._llm_debug_log_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(_to_json_compatible(payload)) + "\n")

    # Attach one lightweight chat panel to the existing trajectory viewer sidebar.
    def _install_chat_panel(self) -> None:
        """Attach a simple chat history + input area to the viewer sidebar."""
        gui = self.viewer.server.gui
        chat_container = self.viewer.chat_gui_container or gui
        with chat_container:
            gui.add_markdown("## LLM Chat")
            gui.add_markdown(
                "Supported commands: switch between hammer and screwdriver, Lie placement "
                "requests, and `please do predefined motion`."
            )
            self._chat_html = gui.add_html(_render_chat_html(self._chat_history))
            self._chat_input = gui.add_text(
                "Message (Enter to send)",
                initial_value="",
                multiline=True,
            )
        self._chat_input.on_update(lambda _: self._on_chat_input_update())

    # Submit the chat input when the user presses Enter in the multiline text box.
    def _on_chat_input_update(self) -> None:
        """Submit one chat message when the text widget receives an Enter key newline."""
        if self._chat_input is None:
            return
        value = self._chat_input.value
        if "\n" not in value:
            return
        message = value.replace("\n", "").strip()
        self._chat_input.value = ""
        if message:
            self._handle_chat_send(message)

    # Re-render the chat history after each user or assistant turn.
    def _update_chat_history(self) -> None:
        """Render the current chat history into the viewer sidebar HTML widget."""
        if self._chat_html is not None:
            self._chat_html.content = _render_chat_html(self._chat_history)

    # Process one user chat message and apply only the supported viewer-only commands.
    def _handle_chat_send(self, message: str) -> None:
        """Call the LLM backend and rebuild the active playback trajectory when supported."""
        self._chat_history.append(("user", message))
        if self._chat_html is not None:
            self._chat_html.content = _render_chat_html(
                self._chat_history + [("assistant", "typing...")]
            )

        sim_context = self._build_llm_sim_context()
        try:
            response = self._chat_service.send_to_llm(
                self._chat_client,
                self._chat_history,
                sim_context,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            log_warn(f"LLM chat exception: {exc}")
            log_warn(tb)
            self._write_llm_debug_event(
                {
                    "event": "llm_chat_exception",
                    "message": message,
                    "sim_context": sim_context,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": tb,
                }
            )
            response = ChatResponse(text=f"[Error: {exc}]")

        assistant_lines: List[str] = []
        executed_command_summary: str | None = None
        if response.goals is not None:
            assistant_lines.append(
                "Viewer-only mode does not support generic goal lists; use Lie swing placement "
                "or `please do predefined swing`."
            )
        if response.command is not None:
            try:
                executed_command_summary = self._apply_supported_command(response.command)
                self._write_llm_debug_event(
                    {
                        "event": "viewer_command_applied",
                        "intent": response.command.intent.value,
                        "strike_target_xy": response.command.strike_target_xy,
                        "target_description": response.command.target_description,
                    }
                )
            except Exception as exc:
                assistant_lines.append(str(exc))
        if executed_command_summary is not None:
            assistant_lines.insert(0, executed_command_summary)
        elif response.text:
            assistant_lines.insert(0, response.text)

        assistant_text = "\n".join(line for line in assistant_lines if line)
        self._chat_history.append(("assistant", assistant_text))
        self._update_chat_history()

    # Run one optional startup message, then hand over to the trajectory viewer loop.
    def run_forever(self) -> None:
        """Serve the viewer and optionally inject one startup chat message."""
        if self._args.llm_startup_chat_message:
            self._handle_chat_send(self._args.llm_startup_chat_message)
        self.viewer.run_forever()


# Parse CLI args and run the viewer-only LLM trajectory playback loop.
def main() -> None:
    """Entry point for the viewer-only LLM trajectory playback script."""
    args = tyro.cli(LLMGoalSourcesViewerArgs)
    args.object_name = _resolve_startup_object_name(args)
    runner = LLMGoalSourcesViewerRunner(args)
    runner.run_forever()


if __name__ == "__main__":
    main()
