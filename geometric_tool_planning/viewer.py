"""Live Viser viewer for kinematics-only tool trajectory inspection."""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from dextoolbench.objects import NAME_TO_OBJECT
from dextoolbench.viser_camera import zoom_connected_viser_clients
from isaacgymenvs.utils.utils import get_repo_root_dir
from llm_runtime.semantic_pose import get_object_pose_semantics

from .orchestrator_types import GoalSourceArtifact

MAX_EFFECTIVE_SPEED_MULTIPLIER = 1.0
DEFAULT_EFFECTIVE_SPEED_MULTIPLIER = 1.0
TABLE_Z = 0.38
TABLE_TOP_Z = TABLE_Z + 0.15
SEMANTIC_AXIS_LENGTH = 0.12


# Convert an XYZW quaternion into the WXYZ convention expected by Viser.
def quaternion_xyzw_to_wxyz(quaternion_xyzw: Sequence[float]) -> tuple[float, float, float, float]:
    """Return one quaternion reordered from XYZW into WXYZ."""
    return (
        float(quaternion_xyzw[3]),
        float(quaternion_xyzw[0]),
        float(quaternion_xyzw[1]),
        float(quaternion_xyzw[2]),
    )


# Return the generic table URDF path used for kinematics-only visualization.
def generic_table_urdf_path() -> Path:
    """Return the absolute path to the default narrow table URDF."""
    return get_repo_root_dir() / "assets" / "urdf" / "table_narrow.urdf"


# Return the generic robot URDF path used for cached robot playback visualization.
def generic_robot_urdf_path() -> Path:
    """Return the absolute path to the default Kuka-ShARPA robot URDF."""
    return (
        get_repo_root_dir()
        / "assets"
        / "urdf"
        / "kuka_sharpa_description"
        / "iiwa14_left_sharpa_adjusted_restricted.urdf"
    )


# Normalize a 3D vector for viewer-side semantic overlays.
def normalize3(vector: Sequence[float]) -> np.ndarray:
    """Return a normalized 3D numpy vector."""
    values = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(values)
    if norm <= 1e-8:
        raise ValueError("Semantic overlay vector must be non-zero.")
    return values / norm


# Convert an XYZW quaternion into a 3x3 rotation matrix.
def quaternion_xyzw_to_rotation_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    """Return a 3x3 rotation matrix for an XYZW quaternion."""
    x_value, y_value, z_value, w_value = [float(value) for value in quaternion_xyzw]
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y_value * y_value + z_value * z_value),
                2.0 * (x_value * y_value - z_value * w_value),
                2.0 * (x_value * z_value + y_value * w_value),
            ],
            [
                2.0 * (x_value * y_value + z_value * w_value),
                1.0 - 2.0 * (x_value * x_value + z_value * z_value),
                2.0 * (y_value * z_value - x_value * w_value),
            ],
            [
                2.0 * (x_value * z_value - y_value * w_value),
                2.0 * (y_value * z_value + x_value * w_value),
                1.0 - 2.0 * (x_value * x_value + y_value * y_value),
            ],
        ],
        dtype=float,
    )


# Return a valid XY coordinate from a metadata payload when one is present.
def _target_xy_from_metadata(metadata: dict[str, Any]) -> Optional[np.ndarray]:
    """Return one explicit target XY from artifact metadata when available."""
    for key in ("target_xy", "strike_target_xy"):
        target_xy = metadata.get(key)
        if isinstance(target_xy, (list, tuple)) and len(target_xy) >= 2:
            return np.asarray([float(target_xy[0]), float(target_xy[1])], dtype=float)
    nested_metadata = metadata.get("metadata")
    if isinstance(nested_metadata, dict):
        nested_target_xy = _target_xy_from_metadata(nested_metadata)
        if nested_target_xy is not None:
            return nested_target_xy
    spec = metadata.get("spec")
    if isinstance(spec, dict):
        spec_target_xy = spec.get("strike_target_xy")
        if isinstance(spec_target_xy, (list, tuple)) and len(spec_target_xy) >= 2:
            return np.asarray([float(spec_target_xy[0]), float(spec_target_xy[1])], dtype=float)
    return None


# Infer the implied strike target on the table for the active artifact.
def infer_strike_target_xy(artifact: GoalSourceArtifact, object_name: str) -> np.ndarray:
    """Return the table-plane strike target for one artifact."""
    metadata_target_xy = _target_xy_from_metadata(artifact.metadata)
    if metadata_target_xy is not None:
        return metadata_target_xy
    strike_point_local = np.asarray(
        get_object_pose_semantics(object_name).strike_point_local, dtype=float
    )
    final_pose = artifact.goals[-1]
    final_rotation = quaternion_xyzw_to_rotation_matrix(final_pose[3:])
    final_position = np.asarray(final_pose[:3], dtype=float)
    final_strike_point = final_position + final_rotation @ strike_point_local
    return np.asarray(final_strike_point[:2], dtype=float)


# Normalize optional experiment target grid coordinates for table visualization.
def normalize_target_grid_xy(
    target_grid_xy: Optional[Sequence[Sequence[float]]],
) -> tuple[tuple[float, float], ...]:
    """Return valid XY grid coordinates as immutable float pairs."""
    if target_grid_xy is None:
        return ()
    normalized_points = []
    for target_xy in target_grid_xy:
        if len(target_xy) < 2:
            raise ValueError("target_grid_xy entries must contain at least x and y.")
        normalized_points.append((float(target_xy[0]), float(target_xy[1])))
    return tuple(normalized_points)


# Score one pose by how well its object semantic axis matches the replay jump target.
def _semantic_target_error_frame_score(
    pose: Sequence[float],
    object_name: str,
) -> tuple[float, bool]:
    """Return one semantic score and threshold-valid flag for target-error replay jumps."""
    semantics = get_object_pose_semantics(object_name)
    tool_rotation = quaternion_xyzw_to_rotation_matrix(pose[3:])
    if "hammer" in object_name:
        axis_world = tool_rotation @ normalize3(semantics.strike_face_normal_local)
        score = 1.0 + float(axis_world[2])
        return score, float(axis_world[2]) <= -math.cos(math.radians(5.0))
    if "screwdriver" in object_name:
        axis_world = tool_rotation @ normalize3(semantics.primary_axis_local)
        score = 1.0 - abs(float(axis_world[2]))
        return score, score <= 1.0 - math.cos(math.radians(5.0))
    return 0.0, False


# Return the current semantic target-error frame for one commanded trajectory.
def target_error_frame_index(
    artifact: GoalSourceArtifact,
    object_name: Optional[str] = None,
) -> int:
    """Return the frame whose tool pose is used by the semantic target-error metric."""
    if len(artifact.goals) == 0:
        return 0
    if object_name is None or ("hammer" not in object_name and "screwdriver" not in object_name):
        return len(artifact.goals) - 1
    scored_frames = [
        (index, *_semantic_target_error_frame_score(goal, object_name))
        for index, goal in enumerate(artifact.goals)
    ]
    valid_indices = [index for index, _, is_valid in scored_frames if is_valid]
    if valid_indices:
        return valid_indices[len(valid_indices) // 2]
    return min(scored_frames, key=lambda item: item[1])[0]


# Return one transformed semantic point in world coordinates for the current tool pose.
def semantic_point_world(
    pose: Sequence[float],
    point_local: Sequence[float],
) -> np.ndarray:
    """Return one semantic point transformed from tool-local to world coordinates."""
    tool_rotation = quaternion_xyzw_to_rotation_matrix(pose[3:])
    tool_position = np.asarray(pose[:3], dtype=float)
    return tool_position + tool_rotation @ np.asarray(point_local, dtype=float)


# Intersect the semantic blue strike-face axis with a horizontal table plane.
def semantic_strike_axis_table_intersection(
    pose: Sequence[float],
    object_name: str,
    *,
    table_top_z: float = TABLE_TOP_Z,
) -> Optional[np.ndarray]:
    """Return the blue-axis table-plane intersection, or None for parallel axes."""
    semantics = get_object_pose_semantics(object_name)
    tool_rotation = quaternion_xyzw_to_rotation_matrix(pose[3:])
    axis_origin = semantic_point_world(pose, semantics.strike_point_local)
    axis_direction = tool_rotation @ normalize3(semantics.strike_face_normal_local)
    if abs(float(axis_direction[2])) <= 1e-8:
        return None
    line_scale = (float(table_top_z) - float(axis_origin[2])) / float(axis_direction[2])
    return axis_origin + line_scale * axis_direction


# Build the semantic overlay geometries used by both kinematics and policy viewers.
def build_semantic_overlay_geometry(
    pose: Sequence[float],
    object_name: str,
    strike_target_xy: Sequence[float],
) -> dict[str, np.ndarray]:
    """Return world-space semantic overlay geometry for one tool pose."""
    semantics = get_object_pose_semantics(object_name)
    tool_rotation = quaternion_xyzw_to_rotation_matrix(pose[3:])
    head_axis_origin = semantic_point_world(pose, semantics.head_center_local)
    strike_face_axis_origin = semantic_point_world(pose, semantics.strike_point_local)
    head_axis_endpoint = head_axis_origin + (
        tool_rotation @ normalize3(semantics.head_axis_local) * SEMANTIC_AXIS_LENGTH
    )
    strike_face_axis_endpoint = strike_face_axis_origin + (
        tool_rotation @ normalize3(semantics.strike_face_normal_local) * SEMANTIC_AXIS_LENGTH
    )
    strike_target_position = np.asarray(
        [float(strike_target_xy[0]), float(strike_target_xy[1]), TABLE_TOP_Z],
        dtype=float,
    )
    implied_contact_position = semantic_strike_axis_table_intersection(
        pose,
        object_name,
    )
    if implied_contact_position is None:
        implied_contact_position = np.asarray([np.nan, np.nan, np.nan], dtype=float)
    return {
        "head_axis_points": np.asarray([[head_axis_origin, head_axis_endpoint]], dtype=float),
        "strike_face_axis_points": np.asarray(
            [[strike_face_axis_origin, strike_face_axis_endpoint]],
            dtype=float,
        ),
        "strike_target_position": strike_target_position,
        "implied_contact_position": implied_contact_position,
    }


@dataclass
class ToolPlaybackState:
    """Deterministic playback state for one active trajectory source."""

    mode: str
    num_frames: int
    duration_sec: float
    frame_progress: float = 0.0
    paused: bool = False
    speed_multiplier: float = 1.0
    _last_tick_sec: Optional[float] = None

    # Build playback state directly from one stored trajectory artifact.
    @classmethod
    def from_artifact(cls, artifact: GoalSourceArtifact) -> "ToolPlaybackState":
        """Return initialized playback state for one artifact."""
        return cls(
            mode=artifact.mode,
            num_frames=len(artifact.goals),
            duration_sec=float(artifact.duration_sec),
        )

    @property
    def frame_index(self) -> int:
        """Return the active integer frame index."""
        return min(int(self.frame_progress), self.max_frame_index)

    @property
    def max_frame_index(self) -> int:
        """Return the last valid frame index for the current artifact."""
        return max(self.num_frames - 1, 0)

    # Replace the active artifact and reset playback to the first frame.
    def set_artifact(self, artifact: GoalSourceArtifact) -> None:
        """Adopt one artifact and reset playback to its first frame."""
        self.mode = artifact.mode
        self.num_frames = len(artifact.goals)
        self.duration_sec = float(artifact.duration_sec)
        self.frame_progress = 0.0
        self.paused = False
        self._last_tick_sec = None

    # Force playback to one concrete frame index without changing pause state.
    def set_frame_index(self, frame_index: int) -> None:
        """Clamp and store one explicit frame index."""
        self.frame_progress = float(int(np.clip(frame_index, 0, self.max_frame_index)))

    # Restart playback from the beginning of the active trajectory.
    def restart(self) -> None:
        """Reset playback to the first frame."""
        self.frame_progress = 0.0
        self._last_tick_sec = None

    # Toggle paused playback and reset the wall-clock anchor when needed.
    def set_paused(self, paused: bool, now_sec: Optional[float] = None) -> None:
        """Pause or resume playback from the current frame."""
        self.paused = paused
        self._last_tick_sec = None if paused else now_sec

    # Update the playback speed multiplier used during wall-clock advancement.
    def set_speed_multiplier(self, speed_multiplier: float) -> None:
        """Store one positive playback speed multiplier."""
        if speed_multiplier < 0.0:
            raise ValueError("speed_multiplier must be non-negative.")
        self.speed_multiplier = float(speed_multiplier)

    # Advance the playback state using wall-clock time and return the active frame.
    def advance(self, now_sec: float) -> int:
        """Advance one looping playback state to the current wall-clock time."""
        if self.num_frames <= 1:
            return 0
        if self.paused:
            return self.frame_index
        if self._last_tick_sec is None:
            self._last_tick_sec = now_sec
            return self.frame_index

        elapsed_sec = max(0.0, now_sec - self._last_tick_sec)
        self._last_tick_sec = now_sec
        interval_count = float(self.max_frame_index)
        progress_delta = (elapsed_sec * self.speed_multiplier / self.duration_sec) * interval_count
        self.frame_progress += progress_delta
        while self.frame_progress > interval_count:
            self.frame_progress -= interval_count
        return self.frame_index


class ToolTrajectoryViewer:
    """Live Viser viewer that animates one tool mesh through compared trajectories."""

    # Build one live viewer from compared artifacts and the selected tool name.
    def __init__(
        self,
        object_name: str,
        artifacts: Sequence[GoalSourceArtifact],
        port: int = 8080,
        use_tabs: bool = False,
        preloaded_object_names: Optional[Sequence[str]] = None,
        show_robot: bool = False,
        show_goal: bool = False,
        show_target_error_jump: bool = False,
        initial_speed_multiplier: float = DEFAULT_EFFECTIVE_SPEED_MULTIPLIER,
        max_speed_multiplier: float = MAX_EFFECTIVE_SPEED_MULTIPLIER,
        target_grid_xy: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        """Create one mesh-only Viser viewer for the compared trajectories."""
        try:
            import viser  # type: ignore
            from viser.extras import ViserUrdf  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional package/runtime
            raise RuntimeError(
                "Viser visualization requested but the viser package is unavailable."
            ) from exc

        self._viser = viser
        self._viser_urdf = ViserUrdf
        self.object_name = object_name
        self._preloaded_object_names = tuple(
            dict.fromkeys(preloaded_object_names or [object_name]).keys()
        )
        self.artifacts_by_mode: Dict[str, GoalSourceArtifact] = {
            artifact.mode: artifact for artifact in artifacts
        }
        self.mode_order = [artifact.mode for artifact in artifacts]
        self.playback = ToolPlaybackState.from_artifact(self.artifacts_by_mode[self.mode_order[0]])
        self._suppress_callbacks = False
        self._use_tabs = bool(use_tabs)
        self.controls_gui_container = None
        self.chat_gui_container = None
        self.object_dropdown = None
        self._object_markdown = None
        self._show_robot = bool(show_robot)
        self._show_goal = bool(show_goal)
        self._show_target_error_jump = bool(show_target_error_jump)
        self._target_grid_xy = normalize_target_grid_xy(target_grid_xy)
        self._target_grid_markers: list[object] = []
        self.target_grid_checkbox = None
        self.target_error_button = None
        self._initial_speed_multiplier = float(initial_speed_multiplier)
        self._max_speed_multiplier = float(max_speed_multiplier)
        if self._initial_speed_multiplier < 0.0:
            raise ValueError("initial_speed_multiplier must be non-negative.")
        if self._max_speed_multiplier <= 0.0:
            raise ValueError("max_speed_multiplier must be positive.")
        if self._initial_speed_multiplier > self._max_speed_multiplier:
            raise ValueError("initial_speed_multiplier must not exceed max_speed_multiplier.")
        self.robot = None
        self._tool_frames: Dict[str, object] = {}
        self._goal_frames: Dict[str, object] = {}
        self.goal_frame = None
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self._setup_scene()

    # Build the minimal scene graph and GUI controls for the live tool viewer.
    def _setup_scene(self) -> None:
        """Initialize one world frame, tool mesh, and playback GUI."""

        @self.server.on_client_connect
        def _(client) -> None:
            client.camera.position = (0.0, -0.6, 0.55)
            client.camera.look_at = (0.0, 0.0, 0.25)

        self.server.scene.add_frame("/world", show_axes=True, axes_length=0.12, axes_radius=0.003)
        self.server.scene.add_grid("/grid", width=0.8, height=0.8, cell_size=0.05)
        self.server.scene.add_frame(
            "/table",
            position=(0.0, 0.0, TABLE_Z),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            show_axes=False,
        )
        self._add_urdf_safe(
            generic_table_urdf_path(),
            root_node_name="/table",
        )
        if self._show_robot:
            self.server.scene.add_frame(
                "/robot",
                position=(0.0, 0.8, 0.0),
                wxyz=(1.0, 0.0, 0.0, 0.0),
                show_axes=False,
            )
            self.robot = self._viser_urdf(
                self.server,
                generic_robot_urdf_path(),
                root_node_name="/robot",
                load_meshes=True,
                load_collision_meshes=False,
            )
            self.robot.update_cfg(np.zeros(29, dtype=float))
        if self._show_goal:
            for preloaded_object_name in self._preloaded_object_names:
                goal_frame = self.server.scene.add_frame(
                    f"/goal/{preloaded_object_name}",
                    show_axes=False,
                    axes_length=0.08,
                    axes_radius=0.002,
                    visible=(preloaded_object_name == self.object_name),
                )
                self._add_urdf_safe(
                    NAME_TO_OBJECT[preloaded_object_name].urdf_path,
                    root_node_name=f"/goal/{preloaded_object_name}",
                    mesh_color_override=(0, 255, 0, 0.5),
                )
                self._goal_frames[preloaded_object_name] = goal_frame
            self.goal_frame = self._goal_frames.get(self.object_name)
        for preloaded_object_name in self._preloaded_object_names:
            tool_frame = self.server.scene.add_frame(
                f"/tool/{preloaded_object_name}",
                show_axes=False,
                axes_length=0.08,
                axes_radius=0.002,
                visible=(preloaded_object_name == self.object_name),
            )
            self._add_urdf_safe(
                NAME_TO_OBJECT[preloaded_object_name].urdf_path,
                root_node_name=f"/tool/{preloaded_object_name}",
            )
            self._tool_frames[preloaded_object_name] = tool_frame
        self.tool_frame = self._tool_frames[self.object_name]
        self.head_axis_line = self.server.scene.add_line_segments(
            "/tool_debug/head_axis",
            points=np.zeros((1, 2, 3), dtype=float),
            colors=np.asarray([[(255, 0, 0), (255, 0, 0)]], dtype=np.uint8),
            line_width=4.0,
        )
        self.strike_face_axis_line = self.server.scene.add_line_segments(
            "/tool_debug/strike_face_axis",
            points=np.zeros((1, 2, 3), dtype=float),
            colors=np.asarray([[(0, 102, 255), (0, 102, 255)]], dtype=np.uint8),
            line_width=4.0,
        )
        self.strike_target_marker = self.server.scene.add_icosphere(
            "/tool_debug/strike_target",
            radius=0.01,
            color=(255, 215, 0),
        )
        self.implied_contact_marker = self.server.scene.add_icosphere(
            "/tool_debug/implied_contact",
            radius=0.01,
            color=(0, 200, 0),
        )
        for index, target_xy in enumerate(self._target_grid_xy):
            marker = self.server.scene.add_icosphere(
                f"/experiment_target_grid/target_{index:02d}",
                radius=0.006,
                color=(255, 128, 0),
                position=(float(target_xy[0]), float(target_xy[1]), TABLE_TOP_Z + 0.003),
                visible=False,
            )
            self._target_grid_markers.append(marker)
        if self._use_tabs:
            tab_group = self.server.gui.add_tab_group()
            self.chat_gui_container = tab_group.add_tab("LLM Chat")
            self.controls_gui_container = tab_group.add_tab("Controls")
        else:
            self.controls_gui_container = self.server.gui
            self.chat_gui_container = self.server.gui

        controls_ctx = self.controls_gui_container if self._use_tabs else nullcontext()
        with controls_ctx:
            self.server.gui.add_markdown("# Goal Source Tool Viewer")
            self._object_markdown = self.server.gui.add_markdown(
                f"**Object:** `{self.object_name}`  \n"
                "Select one source and inspect the live tool motion plus semantic overlays."
            )
            self.server.gui.add_markdown(
                "**Semantic Overlay Legend**  \n"
                "- Red line: semantic head/handle axis  \n"
                "- Blue line: semantic strike/twist axis  \n"
                "- Requested strike target: gold marker on the table  \n"
                "- Implied contact point: green marker on the table"
            )
            self.server.gui.add_markdown("---")
            if len(self._preloaded_object_names) > 1:
                self.object_dropdown = self.server.gui.add_dropdown(
                    "Object",
                    tuple(self._preloaded_object_names),
                    initial_value=self.object_name,
                )
            self.source_dropdown = self.server.gui.add_dropdown(
                "Source",
                tuple(self.mode_order),
                initial_value=self.playback.mode,
            )
            self.pause_button = self.server.gui.add_button("Pause")
            self.restart_button = self.server.gui.add_button("Restart")
            if self._show_target_error_jump:
                self.target_error_button = self.server.gui.add_button("Jump to Target Error Pose")
            self._add_camera_zoom_controls()
            self.speed_slider = self.server.gui.add_slider(
                "Speed",
                min=0.0,
                max=self._max_speed_multiplier,
                step=0.05,
                initial_value=self._initial_speed_multiplier,
            )
            if self._target_grid_markers:
                self.target_grid_checkbox = self.server.gui.add_checkbox(
                    "Show Target Grid",
                    initial_value=False,
                )
            self.frame_slider = self.server.gui.add_slider(
                "Frame",
                min=0,
                max=self.playback.max_frame_index,
                step=1,
                initial_value=0,
            )
            self.metrics_markdown = self.server.gui.add_markdown("")

        self.source_dropdown.on_update(lambda _: self._on_source_change())
        self.pause_button.on_click(lambda _: self._on_pause_toggle())
        self.restart_button.on_click(lambda _: self._on_restart())
        if self.target_error_button is not None:
            self.target_error_button.on_click(lambda _: self._on_target_error_jump())
        self.speed_slider.on_update(lambda _: self._on_speed_change())
        if self.target_grid_checkbox is not None:
            self.target_grid_checkbox.on_update(lambda _: self._on_target_grid_toggle())
        self.frame_slider.on_update(lambda _: self._on_frame_scrub())

    # Load one URDF into Viser while tolerating mesh-conversion failures.
    def _add_urdf_safe(self, path: Path, *, root_node_name: str, mesh_color_override=None):
        """Return one best-effort ViserUrdf handle for the requested path."""
        try:
            if mesh_color_override is None:
                return self._viser_urdf(self.server, path, root_node_name=root_node_name)
            return self._viser_urdf(
                self.server,
                path,
                root_node_name=root_node_name,
                mesh_color_override=mesh_color_override,
            )
        except Exception:
            return None

    # Update the current object label shown in the controls panel.
    def _refresh_object_markdown(self) -> None:
        """Render one compact object summary above the viewer controls."""
        if self._object_markdown is None:
            return
        self._object_markdown.content = (
            f"**Object:** `{self.object_name}`  \n"
            "Select one source and inspect the live tool motion plus semantic overlays."
        )

    # Add keyboard-free zoom controls for laptop-friendly camera adjustment.
    def _add_camera_zoom_controls(self) -> None:
        """Add zoom buttons that move connected client cameras along their current view rays."""
        self.zoom_in_button = self.server.gui.add_button("Zoom In")
        self.zoom_out_button = self.server.gui.add_button("Zoom Out")
        self.zoom_in_button.on_click(lambda _: self._zoom_connected_clients(zoom_in=True))
        self.zoom_out_button.on_click(lambda _: self._zoom_connected_clients(zoom_in=False))

    # Apply one zoom step to every connected client camera.
    def _zoom_connected_clients(self, *, zoom_in: bool) -> None:
        """Move all connected viewer cameras closer to or farther from the current look-at point."""
        zoom_connected_viser_clients(self.server, zoom_in=zoom_in)

    # Return the currently selected artifact from the viewer state.
    def _active_artifact(self) -> GoalSourceArtifact:
        """Return the artifact selected in the source dropdown."""
        artifact = self.artifacts_by_mode.get(self.playback.mode)
        if artifact is not None:
            return artifact
        fallback_mode = self.mode_order[0]
        fallback_artifact = self.artifacts_by_mode[fallback_mode]
        self.playback.set_artifact(fallback_artifact)
        return fallback_artifact

    # Recompute the frame slider label to show time and sample progress.
    def _refresh_frame_label(self) -> None:
        """Update the frame slider label for the active source."""
        artifact = self._active_artifact()
        current_time_sec = self.playback.frame_index * artifact.sample_interval_sec
        total_time_sec = max(artifact.duration_sec - artifact.sample_interval_sec, 0.0)
        self.frame_slider.label = (
            f"Frame ({self.playback.mode}) "
            f"{current_time_sec:.3f}s/{total_time_sec:.3f}s "
            f"[{self.playback.frame_index:03d}/{self.playback.max_frame_index:03d}]"
        )

    # Update the per-source summary panel shown under the controls.
    def _refresh_metrics(self) -> None:
        """Render one compact markdown summary for the active source."""
        artifact = self._active_artifact()
        effective_speed = self._effective_speed_multiplier()
        lines = [
            f"**Active Source:** `{artifact.mode}`",
            f"- duration_sec: `{artifact.duration_sec:.3f}`",
            f"- num_samples: `{len(artifact.goals)}`",
            f"- effective_speed_multiplier: `{effective_speed:.3f}`",
        ]
        if artifact.metrics:
            for metric_key, metric_value in artifact.metrics.items():
                if isinstance(metric_value, float):
                    lines.append(f"- {metric_key}: `{metric_value:.4f}`")
                elif isinstance(metric_value, int):
                    lines.append(f"- {metric_key}: `{metric_value}`")
                elif isinstance(metric_value, list):
                    lines.append(f"- {metric_key}: `{len(metric_value)} values`")
                else:
                    lines.append(f"- {metric_key}: `{metric_value}`")
        else:
            lines.append("- reference path")
        self.metrics_markdown.content = "\n".join(lines)

    # Return the effective playback speed multiplier stored in the GUI slider.
    def _effective_speed_multiplier(self) -> float:
        """Return the playback multiplier selected directly in the speed slider."""
        return float(self.speed_slider.value)

    # Apply the target-grid checkbox state to every experiment grid marker.
    def _set_target_grid_visible(self, visible: bool) -> None:
        """Show or hide all experiment target-grid markers."""
        for marker in self._target_grid_markers:
            marker.visible = bool(visible)

    # Return the cached robot joints for the active frame when the artifact provides them.
    def _active_robot_joint_positions(self) -> Optional[np.ndarray]:
        """Return the current frame's cached robot joints or None when unavailable."""
        robot_joint_positions_by_frame = self._active_artifact().metadata.get(
            "robot_joint_positions_by_frame"
        )
        if not isinstance(robot_joint_positions_by_frame, list):
            return None
        if not robot_joint_positions_by_frame:
            return None
        frame_index = min(self.playback.frame_index, len(robot_joint_positions_by_frame) - 1)
        joint_positions = robot_joint_positions_by_frame[frame_index]
        if joint_positions is None:
            return None
        return np.asarray(joint_positions, dtype=float)

    # Return the cached goal pose for the active frame when the artifact provides it.
    def _active_goal_pose(self) -> Optional[np.ndarray]:
        """Return the current frame's cached goal pose or None when unavailable."""
        goal_poses_by_frame = self._active_artifact().metadata.get("goal_poses_by_frame")
        if not isinstance(goal_poses_by_frame, list):
            return None
        if not goal_poses_by_frame:
            return None
        frame_index = min(self.playback.frame_index, len(goal_poses_by_frame) - 1)
        goal_pose = goal_poses_by_frame[frame_index]
        if goal_pose is None:
            return None
        return np.asarray(goal_pose, dtype=float)

    # Clamp playback state before indexing the active artifact's frame list.
    def _sync_playback_bounds(self, artifact: GoalSourceArtifact) -> None:
        """Keep playback progress valid for the active artifact."""
        if not artifact.goals:
            raise ValueError(f"Artifact `{artifact.mode}` does not contain any replay frames.")
        if hasattr(self.playback, "num_frames"):
            self.playback.num_frames = len(artifact.goals)
        max_frame_index = max(len(artifact.goals) - 1, 0)
        if hasattr(self.playback, "frame_progress"):
            self.playback.frame_progress = float(
                np.clip(self.playback.frame_progress, 0.0, float(max_frame_index))
            )
        elif hasattr(self.playback, "set_frame_index"):
            clamped_index = int(np.clip(self.playback.frame_index, 0, max_frame_index))
            self.playback.set_frame_index(clamped_index)
        else:
            self.playback.frame_index = int(np.clip(self.playback.frame_index, 0, max_frame_index))

    # Activate one specific source and sync the shared playback controls.
    def _activate_source(self, mode: str) -> None:
        """Switch playback to one concrete source mode."""
        artifact = self.artifacts_by_mode[mode]
        self.playback.set_artifact(artifact)
        self.playback.set_speed_multiplier(self._effective_speed_multiplier())
        self.pause_button.name = "Pause"
        self._suppress_callbacks = True
        self.source_dropdown.value = mode
        self.frame_slider.max = self.playback.max_frame_index
        self.frame_slider.value = 0
        self._suppress_callbacks = False
        self._refresh_metrics()
        self._apply_current_pose()

    # Switch the active object and artifact set while reusing the preloaded scene nodes.
    def switch_object(self, object_name: str, artifacts: Sequence[GoalSourceArtifact]) -> None:
        """Adopt one preloaded object and replace the active source set for it."""
        if object_name not in self._tool_frames:
            raise ValueError(f"Object `{object_name}` was not preloaded into the viewer.")
        self.object_name = object_name
        self.artifacts_by_mode = {artifact.mode: artifact for artifact in artifacts}
        self.mode_order = [artifact.mode for artifact in artifacts]
        for candidate_object_name, tool_frame in self._tool_frames.items():
            tool_frame.visible = candidate_object_name == object_name
        self.tool_frame = self._tool_frames[object_name]
        for candidate_object_name, goal_frame in self._goal_frames.items():
            goal_frame.visible = candidate_object_name == object_name
        self.goal_frame = self._goal_frames.get(object_name)
        self._suppress_callbacks = True
        if self.object_dropdown is not None:
            self.object_dropdown.value = object_name
        self.source_dropdown.options = tuple(self.mode_order)
        self._suppress_callbacks = False
        self._refresh_object_markdown()
        self._activate_source(self.mode_order[0])

    # Apply the active frame pose to the tool mesh root frame.
    def _apply_current_pose(self) -> None:
        """Update the tool frame from the current playback frame."""
        artifact = self._active_artifact()
        ToolTrajectoryViewer._sync_playback_bounds(self, artifact)
        pose = artifact.goals[self.playback.frame_index]
        strike_target_xy = infer_strike_target_xy(artifact, self.object_name)
        overlay_geometry = build_semantic_overlay_geometry(
            pose,
            self.object_name,
            strike_target_xy,
        )
        self.tool_frame.position = np.asarray(pose[:3], dtype=float)
        self.tool_frame.wxyz = quaternion_xyzw_to_wxyz(pose[3:])
        goal_pose = self._active_goal_pose()
        if self.goal_frame is not None and goal_pose is not None:
            self.goal_frame.position = np.asarray(goal_pose[:3], dtype=float)
            self.goal_frame.wxyz = quaternion_xyzw_to_wxyz(goal_pose[3:])
        robot_joint_positions = self._active_robot_joint_positions()
        if self.robot is not None and robot_joint_positions is not None:
            self.robot.update_cfg(robot_joint_positions)
        self.head_axis_line.points = overlay_geometry["head_axis_points"]
        self.strike_face_axis_line.points = overlay_geometry["strike_face_axis_points"]
        self.strike_target_marker.position = overlay_geometry["strike_target_position"]
        self.implied_contact_marker.position = overlay_geometry["implied_contact_position"]
        self._refresh_frame_label()

    # Handle one source selection change from the viewer dropdown.
    def _on_source_change(self) -> None:
        """Switch the active trajectory source and reset playback."""
        if self._suppress_callbacks:
            return
        self._activate_source(str(self.source_dropdown.value))

    # Toggle paused playback from the viewer control button.
    def _on_pause_toggle(self) -> None:
        """Pause or resume live playback."""
        next_paused = not self.playback.paused
        self.playback.set_paused(next_paused, now_sec=time.monotonic())
        self.pause_button.name = "Play" if next_paused else "Pause"

    # Restart playback from the first frame and resume autoplay.
    def _on_restart(self) -> None:
        """Reset playback to frame zero and resume autoplay."""
        self.playback.restart()
        self.playback.set_paused(False, now_sec=None)
        self.pause_button.name = "Pause"
        self._suppress_callbacks = True
        self.frame_slider.value = 0
        self._suppress_callbacks = False
        self._apply_current_pose()

    # Jump to the commanded pose used by the semantic target-error metric.
    def _on_target_error_jump(self) -> None:
        """Pause playback on the current trajectory's semantic target-error pose."""
        frame_index = target_error_frame_index(self._active_artifact(), self.object_name)
        self.playback.set_frame_index(frame_index)
        self.playback.set_paused(True, now_sec=None)
        self.pause_button.name = "Play"
        self._suppress_callbacks = True
        self.frame_slider.value = frame_index
        self._suppress_callbacks = False
        self._apply_current_pose()

    # Store one new playback speed multiplier from the GUI slider.
    def _on_speed_change(self) -> None:
        """Apply one updated playback speed multiplier."""
        self.playback.set_speed_multiplier(self._effective_speed_multiplier())
        self._refresh_frame_label()
        self._refresh_metrics()
        self._apply_current_pose()

    # Handle one target-grid visibility update from the GUI checkbox.
    def _on_target_grid_toggle(self) -> None:
        """Apply the selected target-grid visibility state."""
        if self.target_grid_checkbox is None:
            return
        self._set_target_grid_visible(bool(self.target_grid_checkbox.value))

    # Seek to one explicit frame index from the GUI scrub slider.
    def _on_frame_scrub(self) -> None:
        """Seek playback to the selected frame and pause there."""
        if self._suppress_callbacks:
            return
        self.playback.set_frame_index(int(self.frame_slider.value))
        self.playback.set_paused(True, now_sec=None)
        self.pause_button.name = "Play"
        self._apply_current_pose()

    # Advance playback one step and mirror the resulting state into the GUI.
    def tick(self, now_sec: float) -> int:
        """Advance playback one iteration and return the active frame index."""
        frame_index = self.playback.advance(now_sec)
        self._suppress_callbacks = True
        self.frame_slider.value = frame_index
        self._suppress_callbacks = False
        self._apply_current_pose()
        return frame_index

    # Run the live viewer loop until the process is terminated by the user or tests.
    def run_forever(self) -> None:
        """Serve the Viser viewer and animate the active tool indefinitely."""
        print(f"Viser running at {getattr(self.server, 'url', 'http://localhost:8080')}")
        while True:  # pragma: no cover - interactive runtime
            self.tick(time.monotonic())
            time.sleep(1.0 / 60.0)
