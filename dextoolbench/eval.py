"""Evaluation script for dexterous manipulation with viser visualization."""

# NOTE: torch must be imported AFTER isaacgym imports
# isort: off
import torch

# isort: on

import html
import importlib.util
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio
import numpy as np
import tyro
from termcolor import colored

try:
    import viser
    from viser.extras import ViserUrdf
except Exception:  # pragma: no cover - optional visualization dependency
    viser = None
    ViserUrdf = None

# Support script-style execution (`python3 dextoolbench/eval.py`) by ensuring repo root is importable.
if importlib.util.find_spec("compat") is None:
    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
from dextoolbench.eval_config import (
    DEFAULT_EVAL_SUCCESS_TOLERANCE_M,
    DEFAULT_MAX_REALTIME_FACTOR,
    DEFAULT_OBJECT_CATEGORY,
    DEFAULT_OBJECT_NAME,
    DEFAULT_TASK_NAME,
    DEFAULT_VISER_PORT,
    DEFAULT_Z_OFFSET_M,
    OBJECT_CATEGORY_TO_TABLE_URDF,
    TABLE_URDF,
    TABLE_Z,
)
from dextoolbench.metadata import DEXTOOLBENCH_DATA_STRUCTURE, OBJECT_NAME_TO_CATEGORY
from dextoolbench.objects import NAME_TO_OBJECT
from dextoolbench.predefined_baselines import resolve_predefined_trajectory_path
from dextoolbench.shutdown_utils import close_simulation_app_with_timeout
from dextoolbench.viser_camera import zoom_connected_viser_clients
from experiments.trial_stopping import (
    TrialStopConfig,
    TrialStopResult,
    TrialStopSignals,
    evaluate_trial_stop,
)
from geometric_tool_planning.viewer import TABLE_TOP_Z, build_semantic_overlay_geometry
from isaacgymenvs.utils.utils import get_repo_root_dir

_SIDEBAR_IMG_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "urdf"
    / "dextoolbench"
    / "dextoolbench_objects_sidebar.png"
)
_SIDEBAR_IMG_CACHE: Optional[np.ndarray] = None

_CATEGORY_DESCRIPTIONS = {
    "hammer": "Swing a hammer to hit a nail.",
    "spatula": "Flip or serve food with a spatula.",
    "eraser": "Wipe a whiteboard with an eraser.",
    "screwdriver": "Drive a screw from the top or side.",
    "marker": "Write shapes on a whiteboard.",
    "brush": "Sweep debris forward across the table.",
}

_INTERACTIVE_EVAL_EPISODE_LENGTH_STEPS = 1_000_000_000


# Lazily load the object-overview image once for the interactive sidebar.
def _load_sidebar_image() -> Optional[np.ndarray]:
    """Return RGB image array for the interactive sidebar preview."""
    global _SIDEBAR_IMG_CACHE
    if _SIDEBAR_IMG_CACHE is not None:
        return _SIDEBAR_IMG_CACHE
    if not _SIDEBAR_IMG_PATH.exists():
        return None
    try:
        from PIL import Image as PILImage

        _SIDEBAR_IMG_CACHE = np.asarray(PILImage.open(_SIDEBAR_IMG_PATH).convert("RGB"))
    except Exception as exc:
        log_warn(f"Failed to load sidebar image {_SIDEBAR_IMG_PATH}: {exc}")
        _SIDEBAR_IMG_CACHE = None
    return _SIDEBAR_IMG_CACHE


# Patch PIL save behavior used by trimesh->glTF export in viser.
# Some in-memory PNG images can miss `.filename`, which raises in PIL.Image.save.
def _patch_pillow_image_save_for_viser():
    """Patch PIL.Image.Image.save to tolerate images without filename attribute."""
    try:
        from PIL import Image as PILImage
    except ImportError:
        return

    if getattr(PILImage.Image.save, "_simtoolreal_filename_patch", False):
        return

    original_save = PILImage.Image.save

    # Ensure in-memory images from texture pipelines always expose `.filename`.
    def _safe_save(self, *args, **kwargs):
        if not hasattr(self, "filename"):
            self.filename = ""
        return original_save(self, *args, **kwargs)

    _safe_save._simtoolreal_filename_patch = True
    PILImage.Image.save = _safe_save


def quat_xyzw_to_wxyz(q):
    """Convert quaternion from xyzw to wxyz format."""
    return (q[3], q[0], q[1], q[2])


def log_info(text):
    print(colored(text, "cyan"))


def log_success(text):
    print(colored(text, "green"))


def log_warn(text):
    print(colored(text, "yellow"))


# ---------------------------------------------------------------------------
# Chat UI helpers — WhatsApp-style HTML bubbles rendered in viser sidebar
# ---------------------------------------------------------------------------

_CHAT_EMPTY_HTML = (
    "<div style='font-size:12px;color:#888;padding:4px;'><em>No messages yet.</em></div>"
)


def _render_chat_html(history: list) -> str:
    """Render full chat history as WhatsApp-style HTML bubbles."""
    if not history:
        return (
            "<div id='chat-history' style='max-height:320px;overflow-y:auto;padding-right:4px;'>"
            f"{_CHAT_EMPTY_HTML}</div>"
            "<script>const c=document.getElementById('chat-history');"
            "if(c){c.scrollTop=c.scrollHeight;}</script>"
        )
    parts = []
    for role, text in history:
        safe_text = html.escape(str(text), quote=True)
        if role == "user":
            bubble = (
                f"<div style='text-align:right;margin:4px 0;'>"
                f"<span style='display:inline-block;background:#DCF8C6;"
                f"border-radius:8px;padding:5px 9px;max-width:85%;"
                f"word-wrap:break-word;font-size:12px;'>{safe_text}</span></div>"
            )
        else:
            bubble = (
                f"<div style='text-align:left;margin:4px 0;'>"
                f"<span style='display:inline-block;background:#F0F0F0;"
                f"border-radius:8px;padding:5px 9px;max-width:85%;"
                f"word-wrap:break-word;font-size:12px;'>{safe_text}</span></div>"
            )
        parts.append(bubble)
    body = "".join(parts)
    return (
        "<div id='chat-history' style='max-height:320px;overflow-y:auto;padding-right:4px;'>"
        f"{body}</div>"
        "<script>const c=document.getElementById('chat-history');"
        "if(c){c.scrollTop=c.scrollHeight;}</script>"
    )


# Recursively convert tensors/numpy/scalars to JSON-serializable Python types.
def _to_json_compatible(value):
    """Convert nested values to JSON-compatible primitives."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(v) for v in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# Build a runtime snapshot payload that can be persisted as JSON for parity debugging.
def build_runtime_snapshot_payload(env, *, control_hz: float, control_dt: float) -> dict:
    """Collect an Isaac Lab runtime snapshot from an eval environment."""
    if hasattr(env, "get_runtime_snapshot"):
        snapshot = _to_json_compatible(env.get_runtime_snapshot())
    else:
        # Fallback when env does not expose get_runtime_snapshot.
        snapshot = {
            "backend": "unknown",
            "policy_interface": {
                "num_observations": 140,
                "num_actions": 29,
                "dof_lower_limits": _to_json_compatible(
                    getattr(env, "arm_hand_dof_lower_limits", None)
                ),
                "dof_upper_limits": _to_json_compatible(
                    getattr(env, "arm_hand_dof_upper_limits", None)
                ),
            },
            "materials": {
                "bind_counts": _to_json_compatible(getattr(env, "_debug_material_bind_counts", {})),
            },
        }
    snapshot["control"] = {
        "is_lab_env": True,
        "control_hz": float(control_hz),
        "control_dt": float(control_dt),
    }
    return snapshot


class ViserServer:
    """Viser-based visualization server for robot manipulation."""

    def __init__(
        self,
        object_name: str,
        task_name: str,
        num_keypoints: int,
        table_urdf: str,
        policy_name: str,
        data_structure: dict,
        port: int = DEFAULT_VISER_PORT,
        use_llm: bool = False,
        llm_backend: str = "mock",
        enable_task_selection: bool = True,
        preloaded_object_names: Optional[Sequence[str]] = None,
        preloaded_table_urdfs: Optional[Sequence[str]] = None,
        target_grid_xy: Optional[Sequence[Sequence[float]]] = None,
    ):
        # Guard visualization setup when optional viser deps are unavailable.
        if viser is None or ViserUrdf is None:
            raise RuntimeError(
                "Viser visualization is unavailable in this environment. "
                "Install compatible `viser`/`websockets` to use --enable-viser."
            )
        self.port = port
        self.num_keypoints = num_keypoints
        self.is_paused = False
        self.stop_requested = False
        self.reset_requested = False
        self.show_keypoints = True
        # Selection state: set by GUI dropdowns, consumed by EvalRunner on Run/Reset.
        self.pending_object_name = object_name
        self.pending_task_name = task_name
        self.enable_task_selection = enable_task_selection
        self.selection_changed = False
        self.pending_goal_source: Optional[str] = None
        self.goal_source_changed = False
        self._suppress_callbacks = False
        self._data_structure = data_structure
        # LLM chat state: pending goals from the LLM, cleared after _apply_pending_selection.
        self.pending_llm_goals: Optional[List[List[float]]] = None
        # Tab handle for the Controls tab; None when tab layout is not used (use_llm=False).
        self._controls_tab = None
        self._run_callback = lambda: None
        self._reset_callback = lambda: None
        self._stop_callback = lambda: None
        self._semantic_overlay_object_name: Optional[str] = None
        self._semantic_overlay_strike_target_xy: Optional[np.ndarray] = None
        self._named_strike_points_context: Dict[str, Tuple[float, float, float]] = {}
        self._active_lie_strike_target_context: Optional[Tuple[float, float, float]] = None
        self._target_grid_xy = self._normalize_target_grid_xy(target_grid_xy)
        self._target_grid_markers: List[object] = []
        self.target_grid_checkbox = None
        self._preloaded_object_names = tuple(
            dict.fromkeys(preloaded_object_names or [object_name]).keys()
        )
        self._preloaded_table_urdfs = tuple(
            dict.fromkeys(preloaded_table_urdfs or [table_urdf]).keys()
        )
        _patch_pillow_image_save_for_viser()
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.table_urdf = table_urdf
        self._setup_scene(
            object_name=object_name,
            task_name=task_name,
            policy_name=policy_name,
            use_llm=use_llm,
            llm_backend=llm_backend,
            enable_task_selection=enable_task_selection,
        )

    # Build the interactive scene graph and control sidebar layout.
    def _setup_scene(
        self,
        object_name: str,
        task_name: str,
        policy_name: str,
        use_llm: bool = False,
        llm_backend: str = "mock",
        enable_task_selection: bool = True,
    ):
        """Initialize the 3D scene with robot, table, object, and GUI elements."""

        # Some URDF meshes/textures can fail conversion inside viser/trimesh.
        # Returns the ViserUrdf handle so callers can remove meshes later.
        def _add_urdf_safe(path, root_node_name: str, mesh_color_override=None):
            try:
                if mesh_color_override is None:
                    return ViserUrdf(self.server, path, root_node_name=root_node_name)
                else:
                    return ViserUrdf(
                        self.server,
                        path,
                        root_node_name=root_node_name,
                        mesh_color_override=mesh_color_override,
                    )
            except Exception as exc:
                log_warn(f"Failed to load URDF in Viser ({root_node_name}): {exc}")
                return None

        @self.server.on_client_connect
        def _(client):
            client.camera.position = (0.0, -1.0, 1.0)
            client.camera.look_at = (0.0, 0.0, 0.5)

        # Ground grid
        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        # Robot
        robot_urdf = (
            get_repo_root_dir()
            / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
        )
        self.server.scene.add_frame(
            "/robot", position=(0, 0.8, 0), wxyz=(1, 0, 0, 0), show_axes=False
        )
        self.robot = None
        try:
            self.robot = ViserUrdf(self.server, robot_urdf, root_node_name="/robot")
            self.robot.update_cfg(np.zeros(29))
        except Exception as exc:
            log_warn(f"Failed to load URDF in Viser (/robot): {exc}")

        # Table / object / goal meshes — keep handles in dictionaries so supported LLM
        # entrypoints can preload all variants and switch visibility in place.
        self._add_urdf_safe = _add_urdf_safe
        self._table_frames: Dict[str, object] = {}
        self._table_urdf_handles: Dict[str, object] = {}
        self._object_frames: Dict[str, object] = {}
        self._goal_frames: Dict[str, object] = {}
        self._object_urdf_handles: Dict[str, object] = {}
        self._goal_urdf_handles: Dict[str, object] = {}
        for preloaded_table_urdf in self._preloaded_table_urdfs:
            self._ensure_preloaded_table(preloaded_table_urdf)
        for preloaded_object_name in self._preloaded_object_names:
            self._ensure_preloaded_object_meshes(preloaded_object_name)
        self._switch_preloaded_scene_nodes(object_name, self.table_urdf)
        self.head_axis_line = self.server.scene.add_line_segments(
            "/semantic_overlays/head_axis",
            points=np.zeros((1, 2, 3), dtype=float),
            colors=np.asarray([[[255, 215, 0], [255, 215, 0]]], dtype=np.uint8),
            line_width=4.0,
            visible=False,
        )
        self.strike_face_axis_line = self.server.scene.add_line_segments(
            "/semantic_overlays/strike_face_axis",
            points=np.zeros((1, 2, 3), dtype=float),
            colors=np.asarray([[[255, 140, 0], [255, 140, 0]]], dtype=np.uint8),
            line_width=4.0,
            visible=False,
        )
        self.strike_target_marker = self.server.scene.add_icosphere(
            "/semantic_overlays/strike_target",
            radius=0.02,
            color=(255, 215, 0),
            visible=False,
        )
        self.implied_contact_marker = self.server.scene.add_icosphere(
            "/semantic_overlays/implied_contact",
            radius=0.02,
            color=(0, 200, 0),
            visible=False,
        )
        self.named_strike_point_markers = {
            "target_a": self.server.scene.add_icosphere(
                "/named_strike_points/target_a",
                radius=0.017,
                color=(255, 215, 0),
                visible=False,
            )
        }
        self.named_strike_point_labels = {
            "target_a": self.server.scene.add_label(
                "/named_strike_points/target_a_label",
                text="strike point a",
                position=(0.0, 0.0, 0.0),
            )
        }
        self.named_strike_point_labels["target_a"].visible = False
        self.active_lie_strike_target_marker = self.server.scene.add_icosphere(
            "/named_strike_points/active_lie_target",
            radius=0.018,
            color=(255, 99, 71),
            visible=False,
        )
        self.active_lie_strike_target_label = self.server.scene.add_label(
            "/named_strike_points/active_lie_target_label",
            text="active swing target",
            position=(0.0, 0.0, 0.0),
        )
        self.active_lie_strike_target_label.visible = False
        self._add_target_grid_markers()

        # Keypoint spheres (red for object, green for goal)
        self.obj_keypoint_spheres = []
        self.goal_keypoint_spheres = []
        self.obj_keypoint_spheres_fixed_size = []
        self.goal_keypoint_spheres_fixed_size = []
        for i in range(self.num_keypoints):
            self.obj_keypoint_spheres.append(
                self.server.scene.add_icosphere(
                    f"/obj_keypoint_{i}", radius=0.01, color=(255, 0, 0)
                )
            )
            self.goal_keypoint_spheres.append(
                self.server.scene.add_icosphere(
                    f"/goal_keypoint_{i}", radius=0.01, color=(0, 255, 0)
                )
            )
            self.obj_keypoint_spheres_fixed_size.append(
                self.server.scene.add_icosphere(
                    f"/obj_keypoint_{i}_fixed_size",
                    radius=0.01,
                    color=(255, 0, 0),
                    opacity=0.6,
                )
            )
            self.goal_keypoint_spheres_fixed_size.append(
                self.server.scene.add_icosphere(
                    f"/goal_keypoint_{i}_fixed_size",
                    radius=0.01,
                    color=(0, 255, 0),
                    opacity=0.6,
                )
            )

        # GUI elements — when use_llm is active, split into Controls / Chat tabs.
        # Otherwise use the flat sidebar layout (no tabs).
        if use_llm:
            self._tab_group = self.server.gui.add_tab_group()
            self._chat_tab = self._tab_group.add_tab("Chat")
            # Create Chat first so it becomes the default opened tab on startup.
            self._controls_tab = self._tab_group.add_tab("Controls")
        _ctrl_ctx = self._controls_tab if self._controls_tab is not None else nullcontext()

        with _ctrl_ctx:
            sidebar_image = _load_sidebar_image()
            if sidebar_image is not None:
                self.server.gui.add_image(sidebar_image, label="DexToolBench", format="jpeg")
                self.server.gui.add_markdown("---")
            self.server.gui.add_markdown(f"**Policy:** {policy_name}")
            self.server.gui.add_markdown("---")

            # Cascading selection dropdowns: Category → Object → Task.
            # Changes are buffered in pending_object_name/pending_task_name and applied on Run/Reset.
            initial_category = OBJECT_NAME_TO_CATEGORY[object_name]
            initial_objects = list(self._data_structure[initial_category].keys())
            initial_tasks = self._data_structure[initial_category][object_name]

            self.category_dropdown = self.server.gui.add_dropdown(
                "Category", list(self._data_structure.keys()), initial_value=initial_category
            )
            self.object_dropdown = self.server.gui.add_dropdown(
                "Object", initial_objects, initial_value=object_name
            )
            if enable_task_selection:
                self.task_dropdown = self.server.gui.add_dropdown(
                    "Task", initial_tasks, initial_value=task_name
                )
            self.category_dropdown.on_update(lambda _: self._on_category_change())
            self.object_dropdown.on_update(lambda _: self._on_object_change())
            if enable_task_selection:
                self.task_dropdown.on_update(lambda _: self._on_task_change())

            self.server.gui.add_markdown("---")
            self.category_description_text = self.server.gui.add_markdown("")
            self._update_category_description(initial_category)
            self.server.gui.add_markdown("---")
            self.active_label = self.server.gui.add_markdown(
                self._format_active_label(object_name, task_name)
            )
            self.server.gui.add_markdown("---")
            self.progress_text = self.server.gui.add_markdown("**Progress:** --")
            self.stats_text = self.server.gui.add_markdown("**Stats:** No episodes completed")
            self.object_state_text = self.server.gui.add_markdown("**Object State:** --")
            self.server.gui.add_markdown("---")

            self.keypoint_toggle = self.server.gui.add_checkbox(
                "Show Keypoints", initial_value=True
            )
            self.keypoint_toggle.on_update(lambda _: self._toggle_keypoints())
            self.keypoint_toggle_fixed_size = self.server.gui.add_checkbox(
                "Show Keypoints Fixed Size", initial_value=True
            )
            self.keypoint_toggle_fixed_size.on_update(lambda _: self._toggle_keypoints_fixed_size())
            self.keypoint_toggle_fixed_size.value = (
                False  # start as True, then set to False to hide them
            )
            if self._target_grid_markers:
                self.target_grid_checkbox = self.server.gui.add_checkbox(
                    "Show Target Grid",
                    initial_value=False,
                )
                self.target_grid_checkbox.on_update(lambda _: self._on_target_grid_toggle())

        # Chat tab is only created when use_llm=True.
        if use_llm:
            with self._chat_tab:
                self._setup_chat_panel(llm_backend)

    # Normalize optional experiment target-grid coordinates for table visualization.
    @staticmethod
    def _normalize_target_grid_xy(
        target_grid_xy: Optional[Sequence[Sequence[float]]],
    ) -> Tuple[Tuple[float, float], ...]:
        """Return validated XY target-grid points as immutable float tuples."""
        if target_grid_xy is None:
            return ()
        normalized_grid: List[Tuple[float, float]] = []
        for target_xy in target_grid_xy:
            if len(target_xy) < 2:
                raise ValueError("target_grid_xy entries must contain at least x and y.")
            normalized_grid.append((float(target_xy[0]), float(target_xy[1])))
        return tuple(normalized_grid)

    # Add hidden target-grid marker handles for optional benchmark grid visualization.
    def _add_target_grid_markers(self) -> None:
        """Create hidden target-grid markers on the table plane."""
        for index, target_xy in enumerate(self._target_grid_xy):
            marker = self.server.scene.add_icosphere(
                f"/experiment_target_grid/target_{index:02d}",
                radius=0.006,
                color=(255, 128, 0),
                position=self._target_grid_marker_position(target_xy),
                visible=False,
            )
            self._target_grid_markers.append(marker)

    # Return one grid marker position just above the rendered tabletop.
    @staticmethod
    def _target_grid_marker_position(target_xy: Sequence[float]) -> Tuple[float, float, float]:
        """Return world XYZ for one visible target-grid marker."""
        return (float(target_xy[0]), float(target_xy[1]), float(TABLE_TOP_Z) + 0.003)

    # Return one stable scene-node suffix for a preloaded table URDF path.
    @staticmethod
    def _table_node_suffix(table_urdf: str) -> str:
        """Return one safe scene-node suffix for a table URDF asset path."""
        return str(table_urdf).replace("/", "_").replace(".", "_").replace("-", "_")

    # Ensure one table URDF is present in the scene graph and cache its frame + handle.
    def _ensure_preloaded_table(self, table_urdf: str) -> None:
        """Create one cached table frame/mesh pair when it has not been loaded yet."""
        if table_urdf in self._table_frames:
            return
        suffix = self._table_node_suffix(table_urdf)
        root_node_name = f"/table/{suffix}"
        table_frame = self.server.scene.add_frame(
            root_node_name,
            position=(0, 0, TABLE_Z),
            wxyz=(1, 0, 0, 0),
            show_axes=False,
            visible=False,
        )
        table_urdf_path = get_repo_root_dir() / "assets" / table_urdf
        self._table_frames[table_urdf] = table_frame
        self._table_urdf_handles[table_urdf] = self._add_urdf_safe(
            table_urdf_path,
            root_node_name=root_node_name,
            mesh_color_override=(255, 255, 255, 1.0),
        )

    # Ensure one object and goal mesh pair is present in the scene graph and cache their handles.
    def _ensure_preloaded_object_meshes(self, object_name: str) -> None:
        """Create one cached object/goal frame pair when it has not been loaded yet."""
        if object_name in self._object_frames:
            return
        object_urdf = NAME_TO_OBJECT[object_name].urdf_path
        object_root = f"/object/{object_name}"
        goal_root = f"/goal/{object_name}"
        object_frame = self.server.scene.add_frame(
            object_root,
            show_axes=True,
            axes_length=0.1,
            axes_radius=0.001,
            visible=False,
        )
        goal_frame = self.server.scene.add_frame(
            goal_root,
            show_axes=True,
            axes_length=0.1,
            axes_radius=0.001,
            visible=False,
        )
        self._object_frames[object_name] = object_frame
        self._goal_frames[object_name] = goal_frame
        self._object_urdf_handles[object_name] = self._add_urdf_safe(
            object_urdf,
            root_node_name=object_root,
        )
        self._goal_urdf_handles[object_name] = self._add_urdf_safe(
            object_urdf,
            root_node_name=goal_root,
            mesh_color_override=(0, 255, 0, 0.5),
        )

    # Switch visible table/object/goal scene nodes without recreating already-cached meshes.
    def _switch_preloaded_scene_nodes(self, object_name: str, table_urdf: str) -> None:
        """Activate one cached table/object/goal triple and hide the rest."""
        self._ensure_preloaded_table(table_urdf)
        self._ensure_preloaded_object_meshes(object_name)
        for candidate_table_urdf, table_frame in self._table_frames.items():
            table_frame.visible = candidate_table_urdf == table_urdf
        for candidate_object_name, object_frame in self._object_frames.items():
            object_frame.visible = candidate_object_name == object_name
        for candidate_object_name, goal_frame in self._goal_frames.items():
            goal_frame.visible = candidate_object_name == object_name
        self.table_urdf = table_urdf
        self.table_frame = self._table_frames[table_urdf]
        self._table_urdf_handle = self._table_urdf_handles.get(table_urdf)
        self.object_frame = self._object_frames[object_name]
        self.goal_frame = self._goal_frames[object_name]
        self._object_urdf_handle = self._object_urdf_handles.get(object_name)
        self._goal_urdf_handle = self._goal_urdf_handles.get(object_name)

    def _toggle_keypoints(self):
        """Toggle visibility of keypoint spheres."""
        self.show_keypoints = self.keypoint_toggle.value
        for sphere in self.obj_keypoint_spheres + self.goal_keypoint_spheres:
            sphere.visible = self.show_keypoints

    def _toggle_keypoints_fixed_size(self):
        """Toggle visibility of keypoint spheres fixed size."""
        self.show_keypoints_fixed_size = self.keypoint_toggle_fixed_size.value
        for sphere in self.obj_keypoint_spheres_fixed_size + self.goal_keypoint_spheres_fixed_size:
            sphere.visible = self.show_keypoints_fixed_size

    # Add the LLM chat panel and mirrored episode-control buttons.
    def _setup_chat_panel(self, llm_backend: str) -> None:
        """Populate the Chat tab with HTML bubbles and Enter-to-send input."""
        # Small info note; the tab label already identifies this as the chat panel.
        self.server.gui.add_markdown(f"*backend: `{llm_backend}`*")
        # HTML area for styled message bubbles (full history, scrollable).
        self._chat_html = self.server.gui.add_html(_CHAT_EMPTY_HTML)
        # multiline=True → Textarea; Enter inserts \n, detected in on_update to submit.
        self._chat_input = self.server.gui.add_text(
            "Message (Enter to send)", initial_value="", multiline=True
        )
        self._chat_input.on_update(lambda _: self._on_chat_input_update())
        # Callback registered by EvalRunner; None until wired up.
        self._chat_send_callback = None
        self.server.gui.add_markdown("---")
        # Mirror core controls in Chat tab for fast command/episode iteration.
        self._chat_run_button = self.server.gui.add_button("Run Episode")
        self._chat_pause_button = self.server.gui.add_button("Pause")
        self._chat_stop_button = self.server.gui.add_button("Stop Episode")
        self._chat_reset_button = self.server.gui.add_button("Reset")
        self._chat_run_button.on_click(lambda _: self._run_callback())
        self._chat_pause_button.on_click(lambda _: self._toggle_pause())
        self._chat_stop_button.on_click(lambda _: self._stop_callback())
        self._chat_reset_button.on_click(lambda _: self._reset_callback())

    def register_chat_callback(self, callback) -> None:
        """Register the callback that EvalRunner uses to process outgoing messages."""
        self._chat_send_callback = callback

    def _on_chat_input_update(self) -> None:
        """Fire send when user presses Enter (multiline=True inserts \\n on Enter)."""
        if not hasattr(self, "_chat_input"):
            return
        val = self._chat_input.value
        if "\n" in val:
            msg = val.replace("\n", "").strip()
            self._chat_input.value = ""
            if msg and self._chat_send_callback is not None:
                self._chat_send_callback(msg)

    def update_chat_history(self, history: list) -> None:
        """Re-render the full conversation as WhatsApp-style HTML bubbles."""
        if hasattr(self, "_chat_html"):
            self._chat_html.content = _render_chat_html(history)

    # Set pause state and keep all pause-button labels synchronized.
    def _set_pause_state(self, paused: bool) -> None:
        """Set pause state and update pause button labels in all tabs."""
        self.is_paused = paused
        if hasattr(self, "pause_button"):
            self.pause_button.name = "Resume" if self.is_paused else "Pause"
        if hasattr(self, "_chat_pause_button"):
            self._chat_pause_button.name = "Resume" if self.is_paused else "Pause"

    # Toggle pause state from GUI input.
    def _toggle_pause(self):
        """Toggle pause state."""
        self._set_pause_state(not self.is_paused)
        log_info(f"Paused: {self.is_paused}")

    # --- Cascading dropdown handlers ---

    # Rebuild object/task options when the selected category changes.
    def _on_category_change(self):
        """Repopulate object and task dropdowns when category changes."""
        if self._suppress_callbacks:
            return
        cat = self.category_dropdown.value
        objects = list(self._data_structure[cat].keys())
        obj = objects[0]
        tasks = self._data_structure[cat][obj]
        self._suppress_callbacks = True
        try:
            self.object_dropdown.options = objects
            self.object_dropdown.value = obj
            if self.enable_task_selection:
                self.task_dropdown.options = tasks
                self.task_dropdown.value = tasks[0]
        finally:
            self._suppress_callbacks = False
        self._update_category_description(cat)
        self._mark_selection_changed()

    # Refresh category helper text shown in the sidebar.
    def _update_category_description(self, category: str) -> None:
        """Update the sidebar description for the selected tool category."""
        if not hasattr(self, "category_description_text"):
            return
        description = _CATEGORY_DESCRIPTIONS.get(category, "No description available.")
        self.category_description_text.content = f"**Category Intent:** {description}"

    def _on_object_change(self):
        """Repopulate task dropdown when object changes."""
        if self._suppress_callbacks:
            return
        cat = self.category_dropdown.value
        obj = self.object_dropdown.value
        tasks = self._data_structure.get(cat, {}).get(obj, [])
        if self.enable_task_selection and not tasks:
            return
        self._suppress_callbacks = True
        try:
            if self.enable_task_selection:
                self.task_dropdown.options = tasks
                self.task_dropdown.value = tasks[0]
        finally:
            self._suppress_callbacks = False
        self._mark_selection_changed()

    def _on_task_change(self):
        """Record selection change when task dropdown changes."""
        if self._suppress_callbacks:
            return
        self._mark_selection_changed()

    def _mark_selection_changed(self):
        """Buffer the current dropdown values as a pending selection."""
        self.pending_object_name = self.object_dropdown.value
        if self.enable_task_selection:
            self.pending_task_name = self.task_dropdown.value
        self.selection_changed = True
        if self.enable_task_selection:
            log_info(
                f"Selection pending: {self.pending_object_name} / {self.pending_task_name}"
                " (takes effect on Run or Reset)"
            )
        else:
            log_info(
                f"Selection pending: {self.pending_object_name} " "(takes effect on Run or Reset)"
            )

    # Add a dropdown for goal-source comparison mode selection.
    def add_goal_source_selector(self, options, initial_value: str) -> None:
        """Add a buffered goal-source selector to the controls panel."""
        _ctrl_ctx = self._controls_tab if self._controls_tab is not None else nullcontext()
        with _ctrl_ctx:
            self.server.gui.add_markdown("---")
            self.goal_source_dropdown = self.server.gui.add_dropdown(
                "Goal Source",
                tuple(options),
                initial_value=initial_value,
            )
            self.goal_source_summary = self.server.gui.add_markdown("")
            self.goal_source_dropdown.on_update(lambda _: self._on_goal_source_change())
        self.pending_goal_source = initial_value
        self.goal_source_changed = False

    # Buffer the selected goal-source mode until run/reset applies it.
    def _on_goal_source_change(self) -> None:
        """Record pending goal-source changes from the controls panel."""
        if self._suppress_callbacks or not hasattr(self, "goal_source_dropdown"):
            return
        self.pending_goal_source = str(self.goal_source_dropdown.value)
        self.goal_source_changed = True
        log_info(
            f"Goal source pending: {self.pending_goal_source} " "(takes effect on Run or Reset)"
        )

    # Update the goal-source summary block shown under the selector.
    def update_goal_source_summary(self, markdown: str) -> None:
        """Render markdown summary text for the active goal source."""
        if hasattr(self, "goal_source_summary"):
            self.goal_source_summary.content = markdown

    # Synchronize the selector UI after the active goal source changes.
    def sync_goal_source(self, goal_source: str) -> None:
        """Update the goal-source dropdown without retriggering callbacks."""
        if not hasattr(self, "goal_source_dropdown"):
            return
        self._suppress_callbacks = True
        try:
            self.goal_source_dropdown.value = goal_source
        finally:
            self._suppress_callbacks = False
        self.pending_goal_source = goal_source
        self.goal_source_changed = False

    # Toggle the semantic overlay primitives together so policy-mode debug state stays consistent.
    def _set_semantic_overlay_visibility(self, visible: bool) -> None:
        """Set visibility for all semantic goal-overlay handles."""
        self.head_axis_line.visible = visible
        self.strike_face_axis_line.visible = visible
        self.strike_target_marker.visible = visible
        self.implied_contact_marker.visible = visible

    # Toggle the always-on named strike-point marker set together.
    def _set_named_strike_point_visibility(self, visible: bool) -> None:
        """Set visibility for named strike-point markers and labels."""
        for marker in self.named_strike_point_markers.values():
            marker.visible = visible
        for label in self.named_strike_point_labels.values():
            label.visible = visible

    # Toggle the active Lie strike-target overlay primitives together.
    def _set_active_lie_strike_target_visibility(self, visible: bool) -> None:
        """Set visibility for the active Lie strike-target marker and label."""
        self.active_lie_strike_target_marker.visible = visible
        self.active_lie_strike_target_label.visible = visible

    # Toggle the experiment target-grid markers together.
    def _set_target_grid_visible(self, visible: bool) -> None:
        """Show or hide all experiment target-grid markers."""
        for marker in self._target_grid_markers:
            marker.visible = bool(visible)

    # Handle one target-grid checkbox update from the controls panel.
    def _on_target_grid_toggle(self) -> None:
        """Apply the selected target-grid visibility state."""
        if self.target_grid_checkbox is None:
            return
        self._set_target_grid_visible(bool(self.target_grid_checkbox.value))

    # Clear the semantic overlay context when the active goal source has no semantic swing target.
    def clear_semantic_goal_overlay_context(self) -> None:
        """Drop semantic goal-overlay context and hide the overlay handles."""
        self._semantic_overlay_object_name = None
        self._semantic_overlay_strike_target_xy = None
        self._set_semantic_overlay_visibility(False)

    # Store the semantic overlay context used to render the active goal-source swing overlays.
    def set_semantic_goal_overlay_context(
        self,
        object_name: Optional[str],
        strike_target_xy: Optional[Tuple[float, float] | np.ndarray],
    ) -> None:
        """Remember semantic overlay inputs for the active goal source."""
        if object_name is None or strike_target_xy is None:
            self.clear_semantic_goal_overlay_context()
            return
        self._semantic_overlay_object_name = object_name
        self._semantic_overlay_strike_target_xy = np.asarray(strike_target_xy, dtype=float)

    # Clear the fixed named strike-point overlay context when unavailable.
    def clear_named_strike_points_context(self) -> None:
        """Drop fixed named strike-point overlay context and hide the marker set."""
        self._named_strike_points_context = {}
        self._set_named_strike_point_visibility(False)

    # Store the fixed named strike-point positions used by LLM-mode overlays.
    def set_named_strike_points_context(
        self,
        named_points: Dict[str, Tuple[float, float, float]],
    ) -> None:
        """Remember fixed named strike-point positions and update the marker set."""
        if not named_points:
            self.clear_named_strike_points_context()
            return
        self._named_strike_points_context = {
            name: (float(value[0]), float(value[1]), float(value[2]))
            for name, value in named_points.items()
        }
        for point_name, position in self._named_strike_points_context.items():
            if point_name not in self.named_strike_point_markers:
                continue
            self.named_strike_point_markers[point_name].position = position
            label_position = (position[0], position[1], position[2] + 0.035)
            self.named_strike_point_labels[point_name].position = label_position
        self._set_named_strike_point_visibility(True)

    # Clear the active Lie strike-target overlay context when no swing target is active.
    def clear_active_lie_strike_target_context(self) -> None:
        """Drop active Lie strike-target overlay context and hide the marker."""
        self._active_lie_strike_target_context = None
        self._set_active_lie_strike_target_visibility(False)

    # Store the active Lie strike-target position used by LLM-mode overlays.
    def set_active_lie_strike_target_context(
        self,
        position: Optional[Tuple[float, float, float]],
    ) -> None:
        """Remember one active Lie strike target and update the marker position."""
        if position is None:
            self.clear_active_lie_strike_target_context()
            return
        self._active_lie_strike_target_context = (
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )
        self.active_lie_strike_target_marker.position = self._active_lie_strike_target_context
        self.active_lie_strike_target_label.position = (
            self._active_lie_strike_target_context[0],
            self._active_lie_strike_target_context[1],
            self._active_lie_strike_target_context[2] + 0.035,
        )
        self._set_active_lie_strike_target_visibility(True)

    # Recompute semantic overlay geometry from the current goal pose on each viewer refresh.
    def _update_semantic_goal_overlays(self, goal_pose) -> None:
        """Update semantic goal-overlay handles from the current goal pose."""
        if (
            self._semantic_overlay_object_name is None
            or self._semantic_overlay_strike_target_xy is None
        ):
            self._set_semantic_overlay_visibility(False)
            return
        try:
            overlay_geometry = build_semantic_overlay_geometry(
                goal_pose,
                self._semantic_overlay_object_name,
                self._semantic_overlay_strike_target_xy,
            )
        except Exception as exc:
            log_warn(f"Failed to update semantic goal overlays: {exc}")
            self._set_semantic_overlay_visibility(False)
            return
        self.head_axis_line.points = overlay_geometry["head_axis_points"]
        self.strike_face_axis_line.points = overlay_geometry["strike_face_axis_points"]
        self.strike_target_marker.position = tuple(overlay_geometry["strike_target_position"])
        self.implied_contact_marker.position = tuple(overlay_geometry["implied_contact_position"])
        self._set_semantic_overlay_visibility(True)

    def _remove_urdf(self, handle):
        """Remove all mesh nodes from a ViserUrdf handle (best-effort)."""
        if handle is None:
            return
        try:
            # ViserUrdf stores meshes in ._meshes (dict[str, MeshHandle]).
            for mesh in handle._meshes.values():
                mesh.remove()
        except Exception:
            pass

    def rebuild_scene(self, object_name: str, task_name: Optional[str], table_urdf: str):
        """Swap object and table meshes after a selection change."""
        self._switch_preloaded_scene_nodes(object_name, table_urdf)
        self.active_label.content = self._format_active_label(object_name, task_name)

    # Build active-selection label text for the controls panel.
    def _format_active_label(self, object_name: str, task_name: Optional[str] = None) -> str:
        """Return active-selection label text for controls panel."""
        if self.enable_task_selection:
            active_task = task_name if task_name is not None else self.pending_task_name
            return f"**Active:** {object_name} / {active_task}"
        return f"**Active:** {object_name}"

    # Wire control callbacks and render control buttons in the controls tab.
    def add_controls(
        self,
        run_callback,
        reset_callback,
        stop_callback,
        *,
        include_camera_zoom_controls: bool = False,
    ):
        """Add run/pause/stop/reset buttons inside the controls panel."""
        self._run_callback = run_callback
        self._reset_callback = reset_callback
        self._stop_callback = stop_callback
        _ctrl_ctx = self._controls_tab if self._controls_tab is not None else nullcontext()
        with _ctrl_ctx:
            self.server.gui.add_button("Run Episode").on_click(lambda _: run_callback())
            self.pause_button = self.server.gui.add_button("Pause")
            self.pause_button.on_click(lambda _: self._toggle_pause())
            self.server.gui.add_button("Stop Episode").on_click(lambda _: stop_callback())
            self.server.gui.add_button("Reset").on_click(lambda _: reset_callback())
            if include_camera_zoom_controls:
                self._add_camera_zoom_controls()

    # Add keyboard-free zoom controls for viewers that need button-based camera adjustment.
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

    def update(
        self,
        joint_pos,
        object_pose,
        goal_pose,
        obj_keypoints=None,
        goal_keypoints=None,
        obj_keypoints_fixed_size=None,
        goal_keypoints_fixed_size=None,
    ):
        """Update visualization with current state."""
        if self.robot is not None:
            self.robot.update_cfg(joint_pos)
        self.object_frame.position = object_pose[:3]
        self.object_frame.wxyz = quat_xyzw_to_wxyz(object_pose[3:7])
        self.goal_frame.position = goal_pose[:3]
        self.goal_frame.wxyz = quat_xyzw_to_wxyz(goal_pose[3:7])
        self._update_semantic_goal_overlays(goal_pose)

        if obj_keypoints is not None:
            for i, sphere in enumerate(self.obj_keypoint_spheres):
                sphere.position = tuple(obj_keypoints[i])
        if goal_keypoints is not None:
            for i, sphere in enumerate(self.goal_keypoint_spheres):
                sphere.position = tuple(goal_keypoints[i])
        if obj_keypoints_fixed_size is not None:
            for i, sphere in enumerate(self.obj_keypoint_spheres_fixed_size):
                sphere.position = tuple(obj_keypoints_fixed_size[i])
        if goal_keypoints_fixed_size is not None:
            for i, sphere in enumerate(self.goal_keypoint_spheres_fixed_size):
                sphere.position = tuple(goal_keypoints_fixed_size[i])

    def update_progress(self, current: int, total: int, timestep: int, control_hz: float = 60.0):
        """Update progress display."""
        pct = 100 * current / total if total > 0 else 0
        self.progress_text.content = (
            f"**Time:** {timestep / control_hz:.1f}s | **Goal:** {current}/{total} ({pct:.0f}%)"
        )

    def update_object_state(self, object_state: np.ndarray):
        object_pos = object_state[:3]
        self.object_state_text.content = (
            f"**Object State:** {object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f}"
        )

    def update_stats(self, num_episodes: int, avg_goal_pct: float, avg_time_sec: float):
        """Update statistics display."""
        self.stats_text.content = f"**Episodes:** {num_episodes} | **Avg Goal:** {avg_goal_pct:.1f}% | **Avg Time:** {avg_time_sec:.1f}s"

    def get_frame(self) -> np.ndarray:
        """Capture current view as image."""
        clients = list(self.server.get_clients().values())
        if clients:
            return clients[0].camera.get_render(height=480, width=640)
        return np.zeros((480, 640, 3), dtype=np.uint8)


class EvalRunner:
    """Runs policy evaluation with viser visualization."""

    def __init__(
        self,
        env,
        config_path: Path,
        checkpoint_path: Path,
        object_name: str,
        task_name: str,
        table_urdf: str,
        output_dir: Optional[Path] = None,
        record_video: bool = False,
        policy_name: str = None,
        enable_viser: bool = False,
        interactive_autorun: bool = False,
        exit_after_episodes: int = 0,
        telemetry_json_path: Optional[Path] = None,
        max_realtime_factor: float = 1.10,
        enable_trial_stopping: bool = False,
        trial_timeout_sec: Optional[float] = None,
        eval_args=None,
        app_launcher=None,
        data_structure: Optional[dict] = None,
    ):
        self.env = env
        from compat.legacy_env_wrapper import LegacyEnvWrapper
        from deployment.rl_player import RlPlayer

        # Always use Isaac Lab (DirectRLEnv) path via the compatibility wrapper.
        self._legacy_env_wrapper = LegacyEnvWrapper(env)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_act = 29
        # Isaac Lab control rate is derived from sim dt * decimation.
        if hasattr(env, "cfg"):
            dt = float(getattr(env.cfg.sim, "dt", 1.0 / 60.0))
            decimation = int(getattr(env.cfg, "decimation", 2))
            self.control_hz = 1.0 / (dt * decimation)
        else:
            self.control_hz = 60.0
        self.control_dt = 1.0 / self.control_hz
        self.record_fps = 10
        self.record_interval = int(self.control_hz / self.record_fps)
        self.interactive_autorun = interactive_autorun
        self.exit_after_episodes = exit_after_episodes
        self.telemetry_json_path = telemetry_json_path
        self.max_realtime_factor = max_realtime_factor
        self.enable_trial_stopping = bool(enable_trial_stopping)
        self.trial_stop_config = TrialStopConfig(
            timeout_sec=float(trial_timeout_sec) if trial_timeout_sec is not None else None
        )
        self._last_trial_stop: Optional[TrialStopResult] = None
        self.object_name = object_name
        self.task_name = task_name
        self._current_table_urdf = table_urdf
        # Stored for env recreation when the GUI selection changes.
        self._eval_args = eval_args
        self._app_launcher = app_launcher
        self._data_structure = data_structure or DEXTOOLBENCH_DATA_STRUCTURE

        # Joint limits for denormalization (via LegacyEnvWrapper which exposes Lab tensors).
        self.joint_lower = (
            self._legacy_env_wrapper.arm_hand_dof_lower_limits[: self.n_act].cpu().numpy()
        )
        self.joint_upper = (
            self._legacy_env_wrapper.arm_hand_dof_upper_limits[: self.n_act].cpu().numpy()
        )

        # Load policy checkpoint.
        self.policy = RlPlayer(
            140, self.n_act, config_path, checkpoint_path, self.device, env.num_envs
        )

        self.output_dir = output_dir

        # Recording setup
        self.record_video = record_video
        self.episode_count = 0
        self.episode_goal_pcts = []
        self.episode_lengths = []
        self.episode_traces = []
        self.episode_trial_stops = []
        self.episode_reset_signal_summaries = []
        self._current_reset_signal_summary = self._new_reset_signal_summary()
        self._current_reset_signal_details: Dict[str, Any] = {}
        if self.record_video:
            assert (
                self.output_dir is not None
            ), "Output directory must be provided if recording video"
            self.session_dir = self.output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.session_dir.mkdir(parents=True, exist_ok=True)
            log_info(f"Recording to: {self.session_dir}")

        # Visualization
        if policy_name is None:
            policy_name = checkpoint_path.name

        # Optionally create the Viser visualization server.
        # Read use_llm/llm_backend from eval_args so subclasses (e.g. LLMEvalRunner) can
        # request the chat tab without overriding this constructor.
        self.viser = None
        if enable_viser:
            preloaded_object_names = getattr(eval_args, "preloaded_object_names", None)
            preloaded_table_urdfs = getattr(eval_args, "preloaded_table_urdfs", None)
            target_grid_xy = getattr(eval_args, "target_grid_xy", None)
            self.viser = ViserServer(
                object_name=object_name,
                task_name=task_name,
                num_keypoints=env.num_keypoints,
                table_urdf=table_urdf,
                policy_name=policy_name,
                data_structure=self._data_structure,
                use_llm=getattr(eval_args, "use_llm", False),
                llm_backend=getattr(eval_args, "llm_backend", "mock"),
                enable_task_selection=not bool(getattr(eval_args, "llm_taskless_mode", False)),
                preloaded_object_names=preloaded_object_names,
                preloaded_table_urdfs=preloaded_table_urdfs,
                target_grid_xy=target_grid_xy,
            )
        self.obs = self._reset()
        self._write_runtime_snapshot()
        # Show the real initial pose immediately (before clicking "Run Episode").
        if self.viser is not None:
            self.viser.update(*self._get_state())
            obj_state = torch.cat([self.env.object_pos[0], self.env.object_rot[0]], dim=-1)
            self.viser.update_object_state(obj_state.cpu().numpy())
        self._write_telemetry(
            status="idle",
            step=0,
            sim_time_sec=0.0,
            wall_time_sec=0.0,
            realtime_factor=0.0,
        )

    # Persist a one-shot runtime snapshot used for parity debugging and e2e artifacts.
    def _write_runtime_snapshot(self) -> None:
        """Write `runtime_snapshot.json` under output_dir when available."""
        if self.output_dir is None:
            return
        snapshot = build_runtime_snapshot_payload(
            self.env,
            control_hz=self.control_hz,
            control_dt=self.control_dt,
        )
        snapshot["meta"] = {
            "object_name": self.object_name,
            "task_name": self.task_name,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        }
        output_path = self.output_dir / "runtime_snapshot.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(_to_json_compatible(snapshot), f, indent=2)

    # Build policy observation block slices for per-component telemetry stats.
    @staticmethod
    def _obs_block_slices():
        return [
            ("joint_pos", 0, 29),
            ("joint_vel", 29, 58),
            ("prev_action_targets", 58, 87),
            ("palm_pos", 87, 90),
            ("palm_rot", 90, 94),
            ("object_rot", 94, 98),
            ("fingertip_pos_rel_palm", 98, 113),
            ("keypoints_rel_palm", 113, 125),
            ("keypoints_rel_goal", 125, 137),
            ("object_scales", 137, 140),
        ]

    # Compute compact scalar stats for a 1D vector to keep telemetry lightweight.
    @staticmethod
    def _vector_stats(values: torch.Tensor) -> dict:
        if values.numel() == 0:
            return {"min": 0.0, "max": 0.0, "mean_abs": 0.0, "l2": 0.0, "clip_abs_ge_10": 0.0}
        abs_v = values.abs()
        return {
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "mean_abs": float(abs_v.mean().item()),
            "l2": float(torch.linalg.norm(values).item()),
            "clip_abs_ge_10": float((abs_v >= 9.999).float().mean().item()),
        }

    # Compute per-block observation stats used for parity diagnostics.
    def _obs_block_stats(self, policy_obs: Optional[torch.Tensor]) -> dict:
        if policy_obs is None or policy_obs.ndim != 2:
            return {}
        obs0 = policy_obs[0].detach().cpu()
        out = {}
        for name, start, end in self._obs_block_slices():
            out[name] = self._vector_stats(obs0[start:end])
        return out

    # Keep eval reset handshake aligned with legacy: one zero-action step produces initial policy obs.
    def _reset(self):
        """Reset environment and return policy observation via Isaac Lab wrapper."""
        return self._legacy_env_wrapper.reset(device=self.device)

    def _hard_reset(self) -> torch.Tensor:
        """Force all envs back to spawn state and return the first policy observation.

        Unlike _reset() (which only takes a zero-action step and works only after done=True),
        this triggers the env's actual reset logic regardless of episode state.
        DirectRLEnv.reset() calls _reset_idx for all envs and returns (obs_dict, info).
        """
        obs_dict, _ = self.env.reset()
        return obs_dict["policy"].to(self.device)

    # Reset aggregate episode counters so new selections start with clean stats.
    def _reset_episode_tracking(self) -> None:
        """Clear per-run episode counters and aggregates."""
        self.episode_count = 0
        self.episode_goal_pcts = []
        self.episode_lengths = []
        self.episode_traces = []
        self.episode_trial_stops = []
        self.episode_reset_signal_summaries = []
        self._current_reset_signal_summary = self._new_reset_signal_summary()
        self._current_reset_signal_details = {}

    # Temporarily override env no-reset mode after the reset handshake has restored spawn state.
    def _set_env_force_no_reset(self, enabled: bool) -> Optional[bool]:
        """Set env force-no-reset mode and return the previous value when available."""
        if not hasattr(self.env, "cfg") or not hasattr(self.env.cfg, "force_no_reset"):
            return None
        previous = bool(self.env.cfg.force_no_reset)
        self.env.cfg.force_no_reset = bool(enabled)
        return previous

    # Restore env no-reset mode when an episode exits or aborts.
    def _restore_env_force_no_reset(self, previous: Optional[bool]) -> None:
        """Restore a previously captured force-no-reset value when available."""
        if previous is None:
            return
        if hasattr(self.env, "cfg") and hasattr(self.env.cfg, "force_no_reset"):
            self.env.cfg.force_no_reset = bool(previous)

    # Build a fresh per-episode reset-condition summary payload.
    def _new_reset_signal_summary(self) -> Dict[str, Any]:
        """Return counters and first-step markers for passive reset-condition signals."""
        return {
            "dropped_count": 0,
            "dropped_first_step": None,
            "object_z_low_count": 0,
            "object_z_low_first_step": None,
        }

    # Read live env state and derive passive reset-condition diagnostics.
    def _collect_reset_signal_details(self, step: int) -> Dict[str, Any]:
        """Return reset-condition signal details derived from the current live state."""
        if hasattr(self.env, "_populate_sim_buffers"):
            self.env._populate_sim_buffers()
        object_z = (
            float(self.env.object_pos[0, 2].item()) if hasattr(self.env, "object_pos") else 0.0
        )
        object_init_z = (
            float(self.env.object_init_state[0, 2].item())
            if hasattr(self.env, "object_init_state")
            else object_z
        )
        lifted_object = (
            bool(self.env.lifted_object[0].item()) if hasattr(self.env, "lifted_object") else False
        )
        env_debug_dropped = (
            bool(self.env._debug_done_dropped[0].item())
            if hasattr(self.env, "_debug_done_dropped")
            else False
        )
        env_debug_object_z_low = (
            bool(self.env._debug_done_object_z_low[0].item())
            if hasattr(self.env, "_debug_done_object_z_low")
            else False
        )
        return {
            "dropped": bool(object_z < object_init_z and lifted_object),
            "object_z_low": bool(object_z < 0.1),
            "object_z": object_z,
            "object_init_z": object_init_z,
            "lifted_object": lifted_object,
            "env_debug_dropped": env_debug_dropped,
            "env_debug_object_z_low": env_debug_object_z_low,
            "step": int(step),
        }

    # Accumulate one step of passive reset-condition diagnostics.
    def _record_reset_signal_details(self, details: Dict[str, Any]) -> None:
        """Update per-episode reset-condition counters from one signal payload."""
        step = int(details.get("step", 0))
        if bool(details.get("dropped", False)):
            self._current_reset_signal_summary["dropped_count"] += 1
            if self._current_reset_signal_summary["dropped_first_step"] is None:
                self._current_reset_signal_summary["dropped_first_step"] = step
        if bool(details.get("object_z_low", False)):
            self._current_reset_signal_summary["object_z_low_count"] += 1
            if self._current_reset_signal_summary["object_z_low_first_step"] is None:
                self._current_reset_signal_summary["object_z_low_first_step"] = step

    # Capture the commanded and executed object pose for one control step.
    def _capture_execution_trace_step(self, step: int) -> Dict[str, Any]:
        """Return one compact execution-trace sample for the current env state."""
        if hasattr(self.env, "_populate_sim_buffers"):
            self.env._populate_sim_buffers()
        obs_np = self.obs[0].cpu().numpy()
        joint_pos = (
            0.5 * (obs_np[: self.n_act] + 1.0) * (self.joint_upper - self.joint_lower)
            + self.joint_lower
        )
        object_pose = torch.cat([self.env.object_pos[0], self.env.object_rot[0]], dim=-1).cpu()
        goal_pose = self.env.goal_pose[0].cpu()
        return {
            "step": int(step),
            "sim_time_sec": float(step / max(self.control_hz, 1e-6)),
            "object_pose": object_pose.tolist(),
            "goal_pose": goal_pose.tolist(),
            "success_count": self._success_count(),
            "success_target": self._success_target(),
            "reset_signals": dict(self._current_reset_signal_details),
            "trial_stop": self._trial_stop_payload(),
            "robot_joint_positions": [float(value) for value in joint_pos.tolist()],
        }

    # Wait for a running interactive episode loop to exit before mutating env state.
    def _wait_for_episode_stop(self) -> None:
        """Block until no episode loop is running."""
        while getattr(self, "_run_episode_in_progress", False):
            time.sleep(0.05)

    # Reset policy state and force the env back to its initial pose.
    def _reset_policy_and_env_state(self) -> None:
        """Reset policy hidden state and refresh the first observation after a hard env reset."""
        self.policy.reset()
        # Isaac Lab reset restores spawn tensors, but the interactive eval path also needs the
        # legacy zero-action handshake to flush the reset robot DOF state and rebuild policy obs.
        self._hard_reset()
        self.obs = self._legacy_env_wrapper.reset(device=self.device)

    # Refresh the viewer after a reset, goal swap, or env recreation.
    def _refresh_viewer_state(
        self,
        *,
        rebuild_scene: bool = False,
        reset_stats: bool = False,
        reset_progress: bool = True,
    ) -> None:
        """Update viewer scene, state, progress, and optional aggregate stats."""
        if self.viser is None:
            return
        if rebuild_scene:
            self.viser.rebuild_scene(self.object_name, self.task_name, self._current_table_urdf)
            self.viser._update_category_description(OBJECT_NAME_TO_CATEGORY[self.object_name])
        self.viser.update(*self._get_state())
        if reset_progress:
            self.viser.update_progress(0, self._success_target(), 0, self.control_hz)
        if reset_stats:
            self.viser.update_stats(0, 0.0, 0.0)

    # Recreate the eval env for a new object/task pair and refresh runner-side references.
    def _recreate_environment(
        self,
        object_name: str,
        task_name: str,
        *,
        custom_goals: Optional[List[List[float]]] = None,
        build_env_fn=None,
    ):
        """Close the current env, build a replacement env, and install it on the runner."""
        from compat.legacy_env_wrapper import LegacyEnvWrapper

        if self._eval_args is None:
            raise RuntimeError("Cannot recreate environment: eval_args not stored.")
        if build_env_fn is None:
            build_env_fn = _build_eval_env
        try:
            self.env.close()
        except Exception as exc:
            log_warn(f"env.close() raised: {exc}")
        if build_env_fn is _build_eval_env:
            new_env, new_table_urdf, new_start_pose = build_env_fn(
                self._eval_args,
                object_name,
                task_name,
                self._app_launcher,
                custom_goals=custom_goals,
            )
        else:
            new_env, new_table_urdf, new_start_pose = build_env_fn(
                self._eval_args,
                object_name,
                self._app_launcher,
            )
        self.env = new_env
        self._legacy_env_wrapper = LegacyEnvWrapper(new_env)
        self.object_name = object_name
        self.task_name = task_name
        self._current_table_urdf = new_table_urdf
        return new_start_pose

    # Recompute policy observations from live Isaac Lab tensors after an in-place goal swap.
    def _refresh_policy_obs_from_env(self) -> torch.Tensor:
        """Refresh cached policy observations without recreating the env."""
        if hasattr(self.env, "populate_sim_buffers"):
            self.env.populate_sim_buffers()
        elif hasattr(self.env, "_populate_sim_buffers"):
            self.env._populate_sim_buffers()
        if hasattr(self.env, "populate_obs_and_states_buffers"):
            self.env.populate_obs_and_states_buffers()
        if not hasattr(self.env, "obs_buf"):
            raise RuntimeError("Env does not expose obs_buf for live goal refresh.")
        raw_obs = self.env.obs_buf
        if isinstance(raw_obs, dict):
            if "policy" not in raw_obs:
                raise RuntimeError("Env obs_buf dict does not contain a 'policy' observation.")
            policy_obs = raw_obs["policy"]
        else:
            policy_obs = raw_obs
        policy_obs = policy_obs.to(self.device)
        self._legacy_env_wrapper._validate_policy_obs(policy_obs)
        return policy_obs

    # Apply a replacement fixed-goal sequence directly to the live env tensors.
    def _apply_goals_live(self, goals: List[List[float]]) -> None:
        """Swap the active goal list in-place without tearing down the env."""
        min_pose_z = TABLE_Z + float(getattr(self._eval_args, "z_offset", 0.03))
        clamped = [[g[0], g[1], max(g[2], min_pose_z)] + list(g[3:]) for g in goals]
        dev = self.env.goal_pos.device
        goal_tensor = torch.tensor(clamped, device=dev, dtype=torch.float32)
        self.env.goal_pos[0] = goal_tensor[0, :3]
        self.env.goal_rot[0] = goal_tensor[0, 3:7]
        self.env.goal_pose[0] = goal_tensor[0, :7]
        self.env.goal_states[0, :7] = goal_tensor[0, :7]
        self.env.cfg.fixed_goal_states = clamped
        self.env.cfg.max_consecutive_successes = len(clamped)
        if hasattr(self.env, "trajectory_states"):
            self.env.trajectory_states = goal_tensor.clone()
        self.env.max_consecutive_successes = len(clamped)
        if hasattr(self.env, "successes"):
            self.env.successes.fill_(0)
        if hasattr(self.env, "consecutive_successes"):
            self.env.consecutive_successes.fill_(0)
        if hasattr(self.env, "progress_buf"):
            self.env.progress_buf.fill_(0)
        self.obs = self._refresh_policy_obs_from_env()
        if self.viser is not None:
            self.viser.update_progress(0, self._success_target(), 0, self.control_hz)

    def _step(self, action) -> Tuple[torch.Tensor, bool, bool, bool]:
        """Step env and return (obs, done, terminated, truncated) via Isaac Lab wrapper."""
        obs, _, done_tensor, info = self._legacy_env_wrapper.step(action)
        terminated = info["terminated"]
        truncated = info["truncated"]
        done = done_tensor[0].item()
        return (
            obs,
            bool(done),
            bool(terminated[0].item()),
            bool(truncated[0].item()),
        )

    def _get_state(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract current state for visualization from Isaac Lab tensors."""
        if hasattr(self.env, "_populate_sim_buffers"):
            # Pull latest simulator tensors so visualization state is not stale
            # right after reset (prevents object/goal appearing under the table).
            self.env._populate_sim_buffers()

        obs_np = self.obs[0].cpu().numpy()
        joint_pos = (
            0.5 * (obs_np[:29] + 1.0) * (self.joint_upper - self.joint_lower) + self.joint_lower
        )
        object_pose = torch.cat([self.env.object_pos[0], self.env.object_rot[0]], dim=-1)

        return (
            joint_pos,
            object_pose.cpu().numpy(),
            self.env.goal_pose[0].cpu().numpy(),
            self.env.obj_keypoint_pos[0].cpu().numpy(),
            self.env.goal_keypoint_pos[0].cpu().numpy(),
            self.env.obj_keypoint_pos_fixed_size[0].cpu().numpy(),
            self.env.goal_keypoint_pos_fixed_size[0].cpu().numpy(),
        )

    def _success_count(self) -> int:
        """Return the current success count for env 0 across backends."""
        if hasattr(self.env, "successes"):
            return int(self.env.successes[0].item())
        return 0

    def _success_target(self) -> int:
        """Return target successes used for progress/percentage across backends."""
        if hasattr(self.env, "max_consecutive_successes"):
            return max(1, int(self.env.max_consecutive_successes))
        if hasattr(self.env, "cfg") and hasattr(self.env.cfg, "max_consecutive_successes"):
            return max(1, int(self.env.cfg.max_consecutive_successes))
        return 1

    # Derive passive reset signals and evaluate experiment-side success/timeout stopping.
    def _evaluate_trial_stop(self, step: int) -> TrialStopResult:
        """Return the active experiment trial-stop result for the current step."""
        reset_signals = self._collect_reset_signal_details(step)
        self._current_reset_signal_details = reset_signals
        self._record_reset_signal_details(reset_signals)
        signals = TrialStopSignals(
            dropped=bool(reset_signals["dropped"]),
            object_z_low=bool(reset_signals["object_z_low"]),
            success_count=self._success_count(),
            success_target=self._success_target(),
            sim_time_sec=float(step / max(self.control_hz, 1e-6)),
            signal_details=reset_signals,
        )
        return evaluate_trial_stop(signals, self.trial_stop_config)

    # Return the most recent experiment-side trial-stop payload for saved artifacts.
    def _trial_stop_payload(self) -> Optional[Dict[str, Any]]:
        """Return the latest trial-stop result as a JSON-compatible dictionary."""
        if self._last_trial_stop is None:
            return None
        return {
            "should_stop": bool(self._last_trial_stop.should_stop),
            "reason": self._last_trial_stop.reason,
            "is_failure": bool(self._last_trial_stop.is_failure),
            "details": dict(self._last_trial_stop.details),
        }

    def _robot_joint_names(self) -> List[str]:
        """Return policy-controlled robot joint names for replay metadata."""
        if hasattr(self.env, "robot") and hasattr(self.env.robot, "data"):
            joint_names = getattr(self.env.robot.data, "joint_names", None)
            if joint_names is not None:
                return [str(name) for name in list(joint_names)[: self.n_act]]
        if hasattr(self.env, "joint_names"):
            return [str(name) for name in list(self.env.joint_names)[: self.n_act]]
        return [f"joint_{index}" for index in range(self.n_act)]

    def _write_telemetry(
        self,
        status: str,
        step: int,
        sim_time_sec: float,
        wall_time_sec: float,
        realtime_factor: float,
        *,
        done: Optional[bool] = None,
        terminated: Optional[bool] = None,
        truncated: Optional[bool] = None,
        action: Optional[torch.Tensor] = None,
        policy_obs: Optional[torch.Tensor] = None,
    ) -> None:
        """Write structured runtime telemetry for e2e tests and debugging."""
        if self.telemetry_json_path is None:
            return
        if hasattr(self.env, "_populate_sim_buffers"):
            self.env._populate_sim_buffers()
        object_pose = torch.cat([self.env.object_pos[0], self.env.object_rot[0]], dim=-1).cpu()
        object_linvel = self.env.object_linvel[0].cpu()
        object_angvel = self.env.object_angvel[0].cpu()
        episode_length = int(self.env.episode_length_buf[0].item())

        goal_pose = self.env.goal_pose[0].cpu()
        # Record policy-side values to debug observation/action mismatches.
        action_first = (
            action[0].detach().cpu().tolist() if action is not None and action.ndim == 2 else None
        )
        obs_first = (
            policy_obs[0].detach().cpu().tolist()
            if policy_obs is not None and policy_obs.ndim == 2
            else None
        )
        policy_action_raw = (
            action[0].detach().cpu() if action is not None and action.ndim == 2 else None
        )
        policy_action_applied = (
            self.env.actions[0].detach().cpu()
            if hasattr(self.env, "actions") and self.env.actions.ndim == 2
            else None
        )
        if policy_action_raw is not None:
            raw_sat_ratio = float((policy_action_raw.abs() >= 0.999).float().mean().item())
            raw_abs_mean = float(policy_action_raw.abs().mean().item())
        else:
            raw_sat_ratio = 0.0
            raw_abs_mean = 0.0
        if policy_action_applied is not None:
            applied_sat_ratio = float((policy_action_applied.abs() >= 0.999).float().mean().item())
            applied_abs_mean = float(policy_action_applied.abs().mean().item())
        else:
            applied_sat_ratio = 0.0
            applied_abs_mean = 0.0
        action_delta_l2 = (
            float(torch.linalg.norm(policy_action_raw - policy_action_applied).item())
            if policy_action_raw is not None and policy_action_applied is not None
            else 0.0
        )
        obs_block_stats = self._obs_block_stats(policy_obs)
        debug_action_delay_idx = (
            int(self.env._debug_last_action_delay_idx[0].item())
            if hasattr(self.env, "_debug_last_action_delay_idx")
            else -1
        )
        debug_obs_delay_idx = (
            int(self.env._debug_last_obs_delay_idx[0].item())
            if hasattr(self.env, "_debug_last_obs_delay_idx")
            else -1
        )
        debug_object_state_delay_idx = (
            int(self.env._debug_last_object_state_delay_idx[0].item())
            if hasattr(self.env, "_debug_last_object_state_delay_idx")
            else -1
        )
        arm_clamp_ratio = (
            float(self.env._debug_last_arm_clamp_ratio[0].item())
            if hasattr(self.env, "_debug_last_arm_clamp_ratio")
            else 0.0
        )
        hand_clamp_ratio = (
            float(self.env._debug_last_hand_clamp_ratio[0].item())
            if hasattr(self.env, "_debug_last_hand_clamp_ratio")
            else 0.0
        )
        fingertip_min_dist = (
            float(self.env.curr_fingertip_distances[0].min().item())
            if hasattr(self.env, "curr_fingertip_distances")
            else 0.0
        )
        fingertip_max_dist = (
            float(self.env.curr_fingertip_distances[0].max().item())
            if hasattr(self.env, "curr_fingertip_distances")
            else 0.0
        )
        has_interacted = (
            bool(self.env.has_interacted_with_object[0].item())
            if hasattr(self.env, "has_interacted_with_object")
            else False
        )
        keypoint_max_dist = (
            float(self.env.keypoints_max_dist[0].item())
            if hasattr(self.env, "keypoints_max_dist")
            else 0.0
        )
        keypoint_max_dist_fixed_size = (
            float(self.env.keypoints_max_dist_fixed_size[0].item())
            if hasattr(self.env, "keypoints_max_dist_fixed_size")
            else 0.0
        )
        near_goal_steps = (
            int(self.env.near_goal_steps[0].item()) if hasattr(self.env, "near_goal_steps") else 0
        )
        success_threshold = (
            float(
                (
                    (
                        self.env.cfg.eval_success_tolerance
                        if self.env.cfg.eval_success_tolerance is not None
                        else self.env.cfg.success_tolerance
                    )
                    * self.env.cfg.keypoint_scale
                )
            )
            if hasattr(self.env, "cfg")
            else 0.0
        )
        done_reasons = {
            "object_z_low": (
                bool(self.env._debug_done_object_z_low[0].item())
                if hasattr(self.env, "_debug_done_object_z_low")
                else False
            ),
            "hand_far": (
                bool(self.env._debug_done_hand_far[0].item())
                if hasattr(self.env, "_debug_done_hand_far")
                else False
            ),
            "dropped": (
                bool(self.env._debug_done_dropped[0].item())
                if hasattr(self.env, "_debug_done_dropped")
                else False
            ),
            "max_success": (
                bool(self.env._debug_done_max_success[0].item())
                if hasattr(self.env, "_debug_done_max_success")
                else False
            ),
            "timeout": (
                bool(self.env._debug_done_timeout[0].item())
                if hasattr(self.env, "_debug_done_timeout")
                else False
            ),
        }
        payload = {
            "schema_version": 2,
            "status": status,
            "episode_count": self.episode_count,
            "step": step,
            "sim_time_sec": sim_time_sec,
            "wall_time_sec": wall_time_sec,
            "realtime_factor": realtime_factor,
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
            "object_pose": object_pose.tolist(),
            "object_linvel": object_linvel.tolist(),
            "object_angvel": object_angvel.tolist(),
            "goal_pose": goal_pose.tolist(),
            "object_z": float(object_pose[2].item()),
            "goal_z": float(goal_pose[2].item()),
            "episode_length_buf": episode_length,
            "success_count": self._success_count(),
            "success_target": self._success_target(),
            "trial_stop": self._trial_stop_payload(),
            "control_hz": self.control_hz,
            "policy_action_env0": action_first,
            "policy_obs_env0": obs_first,
            "policy_action_applied_env0": (
                policy_action_applied.tolist() if policy_action_applied is not None else None
            ),
            "policy_action_delta_l2": action_delta_l2,
            "action_raw_saturation_ratio": raw_sat_ratio,
            "action_applied_saturation_ratio": applied_sat_ratio,
            "action_raw_abs_mean": raw_abs_mean,
            "action_applied_abs_mean": applied_abs_mean,
            "obs_block_stats": obs_block_stats,
            "delay_indices": {
                "action_delay_idx": debug_action_delay_idx,
                "obs_delay_idx": debug_obs_delay_idx,
                "object_state_delay_idx": debug_object_state_delay_idx,
            },
            "joint_target_clamp_ratios": {
                "arm": arm_clamp_ratio,
                "hand": hand_clamp_ratio,
            },
            "contact_proxies": {
                "fingertip_min_dist": fingertip_min_dist,
                "fingertip_max_dist": fingertip_max_dist,
                "has_interacted_with_object": has_interacted,
            },
            "keypoint_max_dist": keypoint_max_dist,
            "keypoint_max_dist_fixed_size": keypoint_max_dist_fixed_size,
            "near_goal_steps": near_goal_steps,
            "success_threshold": success_threshold,
            "done_reasons": done_reasons,
            "material_bind_counts": (
                dict(self.env._debug_material_bind_counts)
                if hasattr(self.env, "_debug_material_bind_counts")
                else {}
            ),
            "runtime_snapshot_path": (
                str(self.output_dir / "runtime_snapshot.json")
                if self.output_dir is not None
                else None
            ),
        }
        self.telemetry_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.telemetry_json_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _sim_step(self, timestep: int) -> Tuple[bool, bool, bool, torch.Tensor]:
        """Execute one sim step and return done flags plus policy action."""
        t0 = time.time()
        if self.viser is not None:
            self.viser.update(*self._get_state())
        action = self.policy.get_normalized_action(self.obs, deterministic_actions=True)
        self.obs, done, terminated, truncated = self._step(action)
        if self.viser is not None:
            self.viser.update_progress(
                self._success_count(),
                self._success_target(),
                timestep,
                self.control_hz,
            )
            obj_state = (
                torch.cat([self.env.object_pos[0], self.env.object_rot[0]], dim=-1).cpu().numpy()
            )
            self.viser.update_object_state(obj_state)

        elapsed = time.time() - t0
        if (sleep_time := self.control_dt - elapsed) > 0:
            time.sleep(sleep_time)
        return done, terminated, truncated, action

    def _render_video(self, states: list, path: Path):
        """Render recorded states to video file."""
        if self.viser is None:
            raise RuntimeError("Video rendering requires Viser to be enabled.")
        log_info(f"Rendering {len(states)} frames...")
        frames = []
        for i, state in enumerate(states):
            self.viser.update(*state)
            time.sleep(0.05)
            frames.append(self.viser.get_frame())
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(states)}")
        imageio.mimsave(str(path), frames, fps=self.record_fps)
        log_success(f"Saved: {path}")

    # Execute one episode loop with support for GUI stop/reset/pause controls.
    def _run_episode(self):
        """Run a single evaluation episode."""

        # Don't let this function be called twice at the same time
        if not hasattr(self, "_run_episode_in_progress"):
            self._run_episode_in_progress = False
        if self._run_episode_in_progress:
            log_warn("Episode already in progress. Skipping...")
            return
        self._run_episode_in_progress = True
        previous_force_no_reset = None

        try:
            # Apply any pending GUI selection change (recreates env if needed).
            self._apply_pending_selection()

            self.policy.reset()
            log_info("Reset...")
            self.obs = self._reset()
            if self.enable_trial_stopping:
                previous_force_no_reset = self._set_env_force_no_reset(True)
            if self.viser is not None:
                self.viser.update(*self._get_state())

            log_success(f"Running{' (+ recording)' if self.record_video else ''}...")
            self._last_trial_stop = None
            self._current_reset_signal_summary = self._new_reset_signal_summary()
            self._current_reset_signal_details = {}
            states, step, done = [], 0, False
            episode_trace = [self._capture_execution_trace_step(step=0)]
            # Track per-episode peak successes before an env reset can zero out counters on done.
            peak_success_count = self._success_count()
            wall_start = time.time()
            realtime_violations = 0

            while not done:
                # Handle pause
                while self.viser is not None and self.viser.is_paused:
                    time.sleep(0.1)

                # Handle reset request: exit episode loop without recording stats.
                if self.viser is not None and self.viser.reset_requested:
                    log_info("Reset requested — aborting episode.")
                    return
                # Handle stop request: abort current episode and keep current state.
                if self.viser is not None and self.viser.stop_requested:
                    log_info("Stop requested — aborting episode without reset.")
                    self.viser.stop_requested = False
                    wall_elapsed = time.time() - wall_start
                    sim_elapsed = step / self.control_hz
                    rtf = sim_elapsed / max(wall_elapsed, 1e-6)
                    self._write_telemetry(
                        status="stopped",
                        step=step,
                        sim_time_sec=sim_elapsed,
                        wall_time_sec=wall_elapsed,
                        realtime_factor=rtf,
                    )
                    return

                if self.record_video and step % self.record_interval == 0:
                    states.append(tuple(x.copy() for x in self._get_state()))
                done, terminated, truncated, action = self._sim_step(step)
                step += 1
                peak_success_count = max(peak_success_count, self._success_count())
                if self.enable_trial_stopping:
                    trial_stop = self._evaluate_trial_stop(step)
                    self._last_trial_stop = trial_stop
                    if trial_stop.should_stop:
                        done = True
                        terminated = bool(trial_stop.is_failure)
                        truncated = trial_stop.reason == "experiment_timeout"
                episode_trace.append(self._capture_execution_trace_step(step=step))
                wall_elapsed = time.time() - wall_start
                sim_elapsed = step / self.control_hz
                rtf = sim_elapsed / max(wall_elapsed, 1e-6)
                self._write_telemetry(
                    status="running",
                    step=step,
                    sim_time_sec=sim_elapsed,
                    wall_time_sec=wall_elapsed,
                    realtime_factor=rtf,
                    done=done,
                    terminated=terminated,
                    truncated=truncated,
                    action=action,
                    policy_obs=self.obs,
                )
                if self.max_realtime_factor > 0 and step > 5 and rtf > self.max_realtime_factor:
                    realtime_violations += 1

            # Update stats
            goal_pct = 100 * peak_success_count / self._success_target()
            self.episode_goal_pcts.append(goal_pct)
            self.episode_lengths.append(step)
            self.episode_traces.append(episode_trace)
            self.episode_trial_stops.append(self._trial_stop_payload())
            self.episode_reset_signal_summaries.append(dict(self._current_reset_signal_summary))
            self.episode_count += 1
            avg_goal_pct = sum(self.episode_goal_pcts) / len(self.episode_goal_pcts)
            avg_time_sec = sum(self.episode_lengths) / len(self.episode_lengths) / self.control_hz
            if self.viser is not None:
                self.viser.update_stats(self.episode_count, avg_goal_pct, avg_time_sec)

            if states and self.record_video:
                self._render_video(states, self.session_dir / f"{self.episode_count}.mp4")

            wall_elapsed = time.time() - wall_start
            sim_elapsed = step / self.control_hz
            rtf = sim_elapsed / max(wall_elapsed, 1e-6)
            self._write_telemetry(
                status="done",
                step=step,
                sim_time_sec=sim_elapsed,
                wall_time_sec=wall_elapsed,
                realtime_factor=rtf,
            )
            if self.max_realtime_factor > 0 and realtime_violations > 3:
                raise RuntimeError(
                    f"Realtime factor exceeded threshold repeatedly: rtf={rtf:.2f}, "
                    f"max={self.max_realtime_factor:.2f}"
                )
            log_success(f"Done: {step / self.control_hz:.1f}s, {goal_pct:.0f}% goals")
        finally:
            self._restore_env_force_no_reset(previous_force_no_reset)
            self._run_episode_in_progress = False

    # Recreate env and scene when a new dropdown selection is applied.
    def _apply_pending_selection(self) -> bool:
        """Recreate the env for a new object/task if the GUI selection changed.

        Returns True if the env was recreated (caller can skip its own reset).
        Must only be called when no episode is in progress.
        """
        if self.viser is None or not self.viser.selection_changed:
            return False
        if self._eval_args is None:
            log_warn("Cannot switch selection: eval_args not stored.")
            return False

        new_obj = self.viser.pending_object_name
        new_task = self.viser.pending_task_name
        self.viser.selection_changed = False
        log_info(f"Applying selection: {new_obj} / {new_task} — recreating env...")

        self._recreate_environment(new_obj, new_task)
        self._reset_episode_tracking()
        self._reset_policy_and_env_state()
        self._refresh_viewer_state(rebuild_scene=True, reset_stats=True)

        log_success(f"Switched to {new_obj} / {new_task}.")
        return True

    # Reset current scene state and apply pending selection when requested.
    def _reset_scene(self):
        """Interrupt any running episode, reset env/policy, and refresh the viewer."""
        if self.viser is not None:
            # Signal the episode loop to exit and unblock a paused loop.
            self.viser.reset_requested = True
            self.viser._set_pause_state(False)
        # Busy-wait until the episode loop has exited.
        self._wait_for_episode_stop()
        # Perform the actual reset now that no episode is running.
        if self.viser is not None:
            self.viser.reset_requested = False
        # If selection changed, apply it (includes env recreation + hard reset + viser update).
        if self._apply_pending_selection():
            return
        log_info("Scene reset.")
        self._reset_policy_and_env_state()
        self._refresh_viewer_state(reset_progress=True, reset_stats=False)

    # Stop any running episode loop and keep the current env state unchanged.
    def _stop_episode(self):
        """Stop a running episode without resetting environment or policy state."""
        if self.viser is None:
            return
        if not getattr(self, "_run_episode_in_progress", False):
            log_info("No episode is running.")
            return
        self.viser.stop_requested = True
        self.viser._set_pause_state(False)
        while getattr(self, "_run_episode_in_progress", False):
            time.sleep(0.05)
        self.viser.stop_requested = False
        self.viser.update(*self._get_state())
        log_success("Episode stopped.")

    # Run the interactive UI loop and register button callbacks.
    def run_interactive_eval(self):
        """Start the interactive evaluation loop (click 'Run Episode' in GUI)."""
        if self.viser is None:
            raise RuntimeError("Interactive eval requires Viser to be enabled.")
        self.viser.add_controls(
            self._run_episode,
            self._reset_scene,
            self._stop_episode,
        )
        log_info(f"Open http://localhost:{self.viser.port}")
        log_info("Click 'Run Episode' to start.")
        if self.interactive_autorun:
            self._run_episode()
        while True:
            if self.exit_after_episodes > 0 and self.episode_count >= self.exit_after_episodes:
                return
            time.sleep(1.0)

    def run_eval(self, num_episodes: int):
        assert self.output_dir is not None, "Output directory must be provided"
        output_dir = self.output_dir
        output_json_file = self.output_dir / "eval.json"
        trace_json_file = self.output_dir / "trace.json"

        for i in range(num_episodes):
            self._run_episode()
        log_success(f"Done: {num_episodes} episodes")

        output_json_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_file, "w") as f:
            json.dump(
                {
                    "avg_goal_pct": np.mean(self.episode_goal_pcts),
                    "avg_time_sec": np.mean(self.episode_lengths) / self.control_hz,
                    "episode_goal_pcts": self.episode_goal_pcts,
                    "episode_lengths": self.episode_lengths,
                    "episode_trial_stops": self.episode_trial_stops,
                    "episode_reset_signal_summaries": self.episode_reset_signal_summaries,
                    "trace_path": str(trace_json_file),
                },
                f,
                indent=4,
            )

        with open(trace_json_file, "w") as f:
            json.dump(
                {
                    "schema_version": "execution_trace_v2",
                    "robot_joint_names": self._robot_joint_names(),
                    "episodes": self.episode_traces,
                },
                f,
                indent=2,
            )

        # Also need to save the policy config
        # And save the env cfg because of overrides
        from omegaconf import OmegaConf

        with open(output_dir / "policy_config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(self.policy.cfg))
        # Isaac Lab cfg dataclasses can include typing annotations OmegaConf
        # cannot always serialize (e.g., Literal unions). Fall back to repr.
        env_cfg_path = output_dir / "env_cfg.yaml"
        try:
            env_cfg_text = OmegaConf.to_yaml(self.env.cfg)
        except Exception as exc:
            log_warn(f"Failed to serialize env cfg with OmegaConf: {exc}")
            env_cfg_text = (
                "# Fallback serialization (repr) because OmegaConf export failed.\n"
                f"{repr(self.env.cfg)}\n"
            )
        with open(env_cfg_path, "w") as f:
            f.write(env_cfg_text)
        log_success(f"Saved: {output_json_file}")


@dataclass
class EvalArgs:
    config_path: Path
    """Path to the policy config YAML."""

    checkpoint_path: Path
    """Path to the policy checkpoint."""

    object_category: str = DEFAULT_OBJECT_CATEGORY
    """Object category (e.g. hammer, marker, spatula). Overridden at runtime by the GUI dropdowns."""

    object_name: str = DEFAULT_OBJECT_NAME
    """Object name within the category. Overridden at runtime by the GUI dropdowns."""

    task_name: str = DEFAULT_TASK_NAME
    """Task / trajectory name. Overridden at runtime by the GUI dropdowns."""

    output_dir: Optional[Path] = None
    """Directory to save evaluation results."""

    num_episodes: int = 1
    """Number of evaluation episodes to run."""

    downsample_factor: int = 1
    """Downsample factor for trajectory goals."""

    policy_name: Optional[str] = None
    """Name of the policy (for display)."""

    interactive: bool = False
    """If True, run interactive eval (GUI button to trigger episodes). Otherwise run all episodes automatically."""

    enable_viser: bool = False
    """If True, enable Viser visualization; default is headless eval without Viser."""

    force_table_urdf: bool = True
    """If True, always use the default table URDF regardless of object category."""

    use_task_env_urdf: bool = False
    """If True, prefer task-specific environment URDFs under assets/urdf/dextoolbench/environments."""

    z_offset: float = DEFAULT_Z_OFFSET_M
    """Z offset added to start pose to avoid the table."""

    custom_goals_json_path: Optional[Path] = None
    """Optional path to a JSON file {"goals": [[x,y,z,qx,qy,qz,qw], ...]} that replaces the
    trajectory goals from the standard trajectory file.  The start_pose is still loaded from the
    standard file so the object spawns at the correct initial position."""

    interactive_autorun: bool = False
    """If True, run one interactive episode immediately after server starts."""

    exit_after_episodes: int = 0
    """If > 0 in interactive mode, exit after this many completed episodes."""

    telemetry_json_path: Optional[Path] = None
    """Optional path for writing runtime telemetry JSON for e2e checks."""

    max_realtime_factor: float = DEFAULT_MAX_REALTIME_FACTOR
    """Fail if realtime factor repeatedly exceeds this threshold (>0 to enable)."""

    eval_success_tolerance: float = DEFAULT_EVAL_SUCCESS_TOLERANCE_M  # 0.014
    """Evaluation-time success tolerance override (default tuned for Lab parity)."""

    reset_time: float = -1.0
    """Optional episode timeout override in seconds."""

    enable_trial_stopping: bool = False
    """If True, stop eval episodes on success/timeout and log reset-condition signals."""

    trial_timeout_sec: Optional[float] = None
    """Optional sim-time timeout used by experiment-side trial stopping."""


# Resolve task-specific environment URDF if available (e.g. hammer nail task scenes).
def _get_task_env_table_urdf(
    object_category: str,
    object_name: str,
    task_name: str,
) -> Optional[str]:
    """Return relative assets path for a task-specific environment URDF, if present."""
    rel_path = f"urdf/dextoolbench/environments/{object_category}/{object_name}/{task_name}.urdf"
    abs_path = get_repo_root_dir() / "assets" / rel_path
    if abs_path.exists():
        return rel_path
    return None


def _build_eval_env(
    args: "EvalArgs",
    object_name: str,
    task_name: str,
    app_launcher=None,
    custom_goals: Optional[List[List[float]]] = None,
):
    """Load trajectory + build overrides + create env for the given object/task selection.

    Returns (env, table_urdf, start_pose). Reuses an existing app_launcher for Isaac Lab
    so the Omniverse runtime is not restarted on every selection change.
    """
    object_category = OBJECT_NAME_TO_CATEGORY[object_name]
    selected_table_urdf: str
    if getattr(args, "use_task_env_urdf", False):
        task_table_urdf = _get_task_env_table_urdf(object_category, object_name, task_name)
        if task_table_urdf is not None:
            selected_table_urdf = task_table_urdf
        else:
            log_warn(
                "Task-specific environment URDF not found for "
                f"{object_category}/{object_name}/{task_name}; falling back to table mapping."
            )
            selected_table_urdf = (
                TABLE_URDF
                if args.force_table_urdf
                else OBJECT_CATEGORY_TO_TABLE_URDF[object_category]
            )
    else:
        selected_table_urdf = (
            TABLE_URDF if args.force_table_urdf else OBJECT_CATEGORY_TO_TABLE_URDF[object_category]
        )

    # Load trajectory JSON for this object/task combination.
    trajectory_path = resolve_predefined_trajectory_path(object_name, task_name)
    assert trajectory_path.exists(), f"Trajectory file not found: {trajectory_path}"
    with open(trajectory_path) as f:
        traj_data = json.load(f)

    # Replace goals with custom JSON when provided (e.g. LLM-pre-computed goals from CLI).
    if custom_goals is not None:
        traj_data["goals"] = [list(goal) for goal in custom_goals]
    elif getattr(args, "custom_goals_json_path", None) is not None:
        with open(args.custom_goals_json_path) as f:
            traj_data["goals"] = json.load(f)["goals"]

    # Keep poses above the table surface.
    min_pose_z = TABLE_Z + args.z_offset
    traj_data["start_pose"][2] = max(traj_data["start_pose"][2] + args.z_offset, min_pose_z)
    for goal in traj_data["goals"]:
        goal[2] = max(goal[2], min_pose_z)
    traj_data["goals"] = traj_data["goals"][:: args.downsample_factor]

    eval_overrides = {
        # Turn off randomization noise
        "task.env.resetPositionNoiseX": 0.0,
        "task.env.resetPositionNoiseY": 0.0,
        "task.env.resetPositionNoiseZ": 0.0,
        "task.env.randomizeObjectRotation": False,
        "task.env.resetDofPosRandomIntervalFingers": 0.0,
        "task.env.resetDofPosRandomIntervalArm": 0.0,
        "task.env.resetDofVelRandomInterval": 0.0,
        "task.env.tableResetZRange": 0.0,
        # Object
        "task.env.objectName": object_name,
        # Environment parameters
        "task.env.numEnvs": 1,
        "task.env.envSpacing": 0.4,
        "task.env.capture_video": False,
        # Goal settings
        "task.env.useFixedGoalStates": True,
        "task.env.fixedGoalStates": traj_data["goals"],
        # Delays and noise
        "task.env.useActionDelay": False,
        "task.env.useObsDelay": False,
        "task.env.useObjectStateDelayNoise": False,
        # Parity mode
        "task.env.stabilizeObjectPreContact": False,
        "task.env.objectScaleNoiseMultiplierRange": [1.0, 1.0],
        # Keep interactive eval alive by default unless the caller explicitly requests reset_time.
        "task.env.episodeLength": _INTERACTIVE_EVAL_EPISODE_LENGTH_STEPS,
        # Reset
        "task.env.forceNoReset": False,
        "task.env.resetWhenDropped": False,
        # Moving average
        "task.env.armMovingAverage": 0.1,
        # Success criteria
        "task.env.evalSuccessTolerance": args.eval_success_tolerance,
        "task.env.successSteps": 1,
        "task.env.fixedSizeKeypointReward": True,
        # Table
        "task.env.asset.table": str(selected_table_urdf),
        "task.env.tableResetZ": TABLE_Z,
        # Initialization
        "task.env.useFixedInitObjectPose": True,
        "task.env.objectStartPose": traj_data["start_pose"],
        "task.env.startArmHigher": True,
        # Forces/torques (all zero for eval)
        "task.env.forceScale": 0.0,
        "task.env.torqueScale": 0.0,
        "task.env.linVelImpulseScale": 0.0,
        "task.env.angVelImpulseScale": 0.0,
        "task.env.forceOnlyWhenLifted": True,
        "task.env.torqueOnlyWhenLifted": True,
        "task.env.linVelImpulseOnlyWhenLifted": True,
        "task.env.angVelImpulseOnlyWhenLifted": True,
        "task.env.forceProbRange": [0.0001, 0.0001],
        "task.env.torqueProbRange": [0.0001, 0.0001],
        "task.env.linVelImpulseProbRange": [0.0001, 0.0001],
        "task.env.angVelImpulseProbRange": [0.0001, 0.0001],
    }
    if getattr(args, "reset_time", -1.0) > 0.0:
        eval_overrides["task.env.resetTime"] = float(args.reset_time)

    from deployment.isaac.isaac_env_lab import create_env_lab

    env = create_env_lab(
        config_path=str(args.config_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        headless=True,
        overrides=eval_overrides,
        physx_profile="eval",
    )

    return env, selected_table_urdf, list(traj_data["start_pose"])


def main():
    args: EvalArgs = tyro.cli(EvalArgs)

    # Launch Omniverse runtime once; reused for every selection change.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    env, selected_table_urdf, initial_start_pose = _build_eval_env(
        args, args.object_name, args.task_name, app_launcher
    )

    runner = EvalRunner(
        env=env,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        object_name=args.object_name,
        task_name=args.task_name,
        table_urdf=selected_table_urdf,
        output_dir=args.output_dir,
        policy_name=args.policy_name,
        enable_viser=(args.enable_viser or args.interactive),
        interactive_autorun=args.interactive_autorun,
        exit_after_episodes=args.exit_after_episodes,
        telemetry_json_path=args.telemetry_json_path,
        max_realtime_factor=args.max_realtime_factor,
        enable_trial_stopping=bool(args.enable_trial_stopping),
        trial_timeout_sec=args.trial_timeout_sec,
        eval_args=args,
        app_launcher=app_launcher,
        data_structure=DEXTOOLBENCH_DATA_STRUCTURE,
    )

    if args.interactive:
        runner.run_interactive_eval()
    else:
        runner.run_eval(num_episodes=args.num_episodes)

    # Isaac Sim teardown can hang in headless subprocess mode; bound close time for e2e reliability.
    if not args.interactive:
        close_simulation_app_with_timeout(
            simulation_app,
            timeout_sec=15.0,
            log_warn_fn=log_warn,
        )
    else:
        simulation_app.close()


if __name__ == "__main__":
    main()
