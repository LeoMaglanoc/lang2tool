"""LLM-augmented evaluation script: EvalRunner subclass with chat panel and live goal control.

Extends the classic eval (dextoolbench/eval.py) with:
  - LLM chat panel in the viser GUI (via ViserServer chat tab)
  - Tool-centered live commands (grasp, move, release) processed at safe step boundaries
  - In-place goal updates that write directly into Isaac Lab tensors without recreating the env
    (the bug-fixed _apply_goals_live — no teardown/rebuild on same object/task)

Classification: Refactor + Bugfix
"""

# NOTE: torch must be imported AFTER isaacgym imports
# isort: off
import torch

# isort: on

import importlib.util
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import tyro

# Support script-style execution (`python3 dextoolbench/eval_llm.py`) by ensuring
# the repo root is importable before any sibling packages are imported.
if importlib.util.find_spec("compat") is None:
    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from dextoolbench.eval import (
    EvalArgs,
    EvalRunner,
    _to_json_compatible,
    log_info,
    log_warn,
)
from dextoolbench.eval_config import (
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
    OBJECT_CATEGORY_TO_TABLE_URDF,
    TABLE_URDF,
    TABLE_Z,
)
from dextoolbench.llm_lie_trajectory import compile_llm_lie_trajectory
from dextoolbench.llm_supported_objects import (
    SUPPORTED_LLM_OBJECT_TASKS,
    supported_llm_data_structure,
    supported_llm_object_family,
    supported_llm_object_names,
    supported_llm_task_name,
)
from dextoolbench.metadata import OBJECT_NAME_TO_CATEGORY
from dextoolbench.object_start_poses import build_default_start_pose
from dextoolbench.shutdown_utils import close_simulation_app_with_timeout
from experiments.common import build_target_grid_xy
from geometric_tool_planning import (
    get_llm_static_strike_context,
    get_named_strike_point_payload,
    get_predefined_goal_sequence,
    is_target_a_strike_target_xy,
)
from geometric_tool_planning.viewer import TABLE_TOP_Z
from llm_runtime import (
    DEFAULT_RELEASE_OPEN_STEPS,
    DEFAULT_RELEASE_PREMOVE_TIMEOUT_STEPS,
    DEFAULT_RELEASE_TABLE_CLEARANCE_M,
    RuntimeMode,
    ToolCommand,
    ToolCommandIntent,
    ToolCommandQueue,
    ToolRuntimeState,
    apply_open_hand_override,
    enter_frozen,
    enter_policy,
)
from llm_runtime.chat.presentation import (
    DEFAULT_ASSISTANT_GREETING as _DEFAULT_ASSISTANT_GREETING,
)
from llm_runtime.chat.presentation import (
    format_chat_command_summary,
)
from llm_runtime.chat.service import ChatService
from llm_runtime.commands import ToolCommandExecutor
from llm_runtime.llm.chat_client import ChatResponse, build_chat_client
from llm_runtime.semantic_pose import get_object_pose_semantics_payload

_HOLD_LAST_GOAL_MAX_SUCCESSES = 1_000_000_000
_LLM_INTERACTIVE_EPISODE_LENGTH_STEPS = 1_000_000_000
_ACTIVE_TARGET_OVERLAY_EPSILON_M = 1e-4


@dataclass
class LLMEvalArgs(EvalArgs):
    """EvalArgs extended with LLM-specific fields. Defaults to use_llm=True."""

    object_name: Optional[str] = None
    """Optional object name. If omitted, startup prompts for tool selection in terminal."""

    task_name: tyro.conf.Suppress[str] = "grasp_hold"
    """Suppressed in LLM mode; trajectories/tasks are not used."""

    use_llm: bool = True
    """Enable LLM chat panel in the viser GUI (requires --interactive / --enable-viser)."""

    llm_backend: str = DEFAULT_LLM_BACKEND
    """LLM backend: 'mock' (offline keyword-based) or 'openai' (requires OPENAI_API_KEY)."""

    llm_debug_log_path: Optional[Path] = None
    """Optional JSONL path for LLM chat debug logs (exceptions + traceback per failed send)."""

    llm_startup_chat_message: Optional[str] = None
    """Optional startup chat message injected once before the eval loop begins."""

    llm_env_trace_path: Optional[Path] = None
    """Optional JSONL path for low-level env reset/goal-dispatch tracing."""

    llm_tool_control_enabled: bool = True
    """Enable tool-centered live command routing from chat messages."""

    llm_live_command_queue_enabled: bool = True
    """If True, apply queued commands at safe boundaries during live episode steps."""

    llm_release_table_clearance_m: float = DEFAULT_RELEASE_TABLE_CLEARANCE_M
    """Target clearance above table before opening the hand in release mode."""

    llm_lie_horizontal_strike_clearance_m: float = DEFAULT_LLM_LIE_HORIZONTAL_STRIKE_CLEARANCE_M
    """Execution-only clearance above the table during the horizontal Lie strike phase."""

    llm_lie_waypoint_table_clearance_m: float = DEFAULT_LLM_LIE_WAYPOINT_TABLE_CLEARANCE_M
    """Minimum clearance the hammer support points must maintain above the table."""

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

    llm_release_open_steps: int = DEFAULT_RELEASE_OPEN_STEPS
    """Number of control steps to hold hand-open override during release."""

    llm_release_pre_move_timeout_steps: int = DEFAULT_RELEASE_PREMOVE_TIMEOUT_STEPS
    """Max policy steps to wait for pre-release placement before forcing hand open."""

    llm_freeze_after_release: bool = True
    """Pause/freeze after release so policy does not immediately regrasp the tool."""

    llm_grasp_hover_offset_m: float = 0.03
    """Vertical hover offset for `grasp tool` goals above current tool pose."""

    llm_keep_episode_alive_interactive: bool = True
    """When interactive, ignore env done and keep episode running until reset."""

    llm_hold_last_goal_forever: bool = True
    """Keep the final waypoint active forever (until reset/selection change/new goals)."""

    llm_taskless_mode: bool = True
    """If True, remove task selection and use object-only grasp-hold bootstrap."""


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


# Return one validated startup object restricted to the supported LLM eval objects.
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
def _resolve_startup_object_name(args: LLMEvalArgs) -> str:
    """Return startup object name from args or interactive terminal selection."""
    if args.object_name:
        return _validate_supported_startup_object_name(args.object_name)
    return _prompt_for_object_name(
        data_structure=supported_llm_data_structure(), preferred_category=args.object_category
    )


# Build one default object start pose with the configured z-offset applied.
def _build_object_start_pose(args: LLMEvalArgs, object_name: str) -> List[float]:
    """Return one clamped default start pose for the selected object."""
    return build_default_start_pose(
        object_name,
        z_offset=float(getattr(args, "z_offset", 0.03)),
        table_z=float(TABLE_Z),
    )


# Resolve the table asset used by one supported object.
def _resolve_object_table_urdf(args: LLMEvalArgs, object_name: str) -> str:
    """Return the table URDF used for one supported object."""
    object_category = OBJECT_NAME_TO_CATEGORY[object_name]
    force_table_urdf = bool(getattr(args, "force_table_urdf", False))
    return TABLE_URDF if force_table_urdf else OBJECT_CATEGORY_TO_TABLE_URDF[object_category]


# Build single-object eval env for taskless LLM mode (no trajectory/task JSON dependency).
def _build_eval_env_llm(args: LLMEvalArgs, object_name: str, app_launcher=None):
    """Create eval env from object defaults only and return (env, table_urdf, start_pose)."""
    selected_table_urdf = _resolve_object_table_urdf(args, object_name)
    start_pose = _build_object_start_pose(args, object_name)
    min_pose_z = TABLE_Z + args.z_offset
    if getattr(args, "custom_goals_json_path", None) is not None:
        with open(args.custom_goals_json_path) as f:
            custom_goals = json.load(f)["goals"]
        bootstrap_goals = []
        for goal in custom_goals:
            g = list(goal)
            g[2] = max(g[2], min_pose_z)
            bootstrap_goals.append(g)
    else:
        bootstrap_goals = [list(start_pose)]

    eval_overrides = {
        "task.env.resetPositionNoiseX": 0.0,
        "task.env.resetPositionNoiseY": 0.0,
        "task.env.resetPositionNoiseZ": 0.0,
        "task.env.randomizeObjectRotation": False,
        "task.env.resetDofPosRandomIntervalFingers": 0.0,
        "task.env.resetDofPosRandomIntervalArm": 0.0,
        "task.env.resetDofVelRandomInterval": 0.0,
        "task.env.tableResetZRange": 0.0,
        "task.env.objectName": object_name,
        "task.env.numEnvs": 1,
        "task.env.envSpacing": 0.4,
        "task.env.capture_video": False,
        "task.env.useFixedGoalStates": True,
        "task.env.fixedGoalStates": bootstrap_goals,
        "task.env.useActionDelay": False,
        "task.env.useObsDelay": False,
        "task.env.useObjectStateDelayNoise": False,
        "task.env.stabilizeObjectPreContact": False,
        "task.env.objectScaleNoiseMultiplierRange": [1.0, 1.0],
        "task.env.episodeLength": _LLM_INTERACTIVE_EPISODE_LENGTH_STEPS,
        "task.env.forceNoReset": True,
        "task.env.resetWhenDropped": False,
        "task.env.armMovingAverage": 0.1,
        "task.env.evalSuccessTolerance": args.eval_success_tolerance,
        "task.env.successSteps": 1,
        "task.env.fixedSizeKeypointReward": True,
        "task.env.asset.table": str(selected_table_urdf),
        "task.env.tableResetZ": TABLE_Z,
        "task.env.useFixedInitObjectPose": True,
        "task.env.objectStartPose": start_pose,
        "task.env.startArmHigher": True,
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
    if getattr(args, "preloaded_object_names", None) is not None:
        eval_overrides["task.env.preloadedObjectNames"] = list(args.preloaded_object_names)
    if getattr(args, "preloaded_table_urdfs", None) is not None:
        eval_overrides["task.env.preloadedTableUrdfs"] = list(args.preloaded_table_urdfs)
    if getattr(args, "llm_env_trace_path", None) is not None:
        eval_overrides["task.env.debugTracePath"] = str(args.llm_env_trace_path)

    from deployment.isaac.isaac_env_lab import create_env_lab

    env = create_env_lab(
        config_path=str(args.config_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        headless=True,
        overrides=eval_overrides,
        physx_profile="eval",
    )
    return env, selected_table_urdf, start_pose


# Queue one optional startup chat message so browserless E2E runs can drive live commands.
def _maybe_send_startup_chat_message(runner, args: LLMEvalArgs) -> None:
    """Inject one startup chat message before the eval loop when requested."""
    message = getattr(args, "llm_startup_chat_message", None)
    if message:
        runner._handle_chat_send(message)


# Reuse the same startup-state restore path as the GUI Reset button before the loop begins.
def _finalize_startup_state(runner) -> None:
    """Restore startup state in-place so initial arm/hand pose matches manual Reset."""
    runner._restore_startup_state_in_place()


# Mirror one one-shot goal list into a down-then-up cycle without duplicating turnaround endpoints.
def _build_mirrored_cycle(goals: Sequence[Sequence[float]]) -> List[List[float]]:
    """Return one mirrored cycle for a forward-only goal list."""
    forward_goals = [list(goal) for goal in goals]
    if len(forward_goals) <= 1:
        return forward_goals
    return forward_goals + [list(goal) for goal in reversed(forward_goals[1:-1])]


class LLMEvalRunner(EvalRunner):
    """EvalRunner subclass that adds LLM chat + live tool-command control.

    The key behavioral differences from the base class are:
    - taskless object-only startup (no trajectory task dependency)
    - default grasp-hold bootstrap goals on init/reset/object-switch
    - in-place goal updates for live LLM control without env recreation.
    """

    def __init__(self, *args, **kwargs):
        self._active_goals: List[List[float]] = []
        self._cyclic_goal_chunk: Optional[List[List[float]]] = None
        self._pending_cyclic_goal_chunk: Optional[List[List[float]]] = None
        # Base class creates env, policy, viser, and takes the initial reset.
        super().__init__(*args, **kwargs)
        # Preserve startup selection so Reset can restore script-start state exactly.
        self._startup_object_name = self.object_name
        self._startup_task_name = self.task_name
        self._startup_table_urdf = self._current_table_urdf
        self._startup_data_structure = supported_llm_data_structure()

        eval_args = self._eval_args
        # --- LLM state ---
        self._use_llm = getattr(eval_args, "use_llm", True)
        self._llm_backend = getattr(eval_args, "llm_backend", "mock")
        configured_llm_debug_log_path = getattr(eval_args, "llm_debug_log_path", None)
        self._llm_debug_log_path = configured_llm_debug_log_path
        self._chat_history: List[Tuple[str, str]] = [
            ("assistant", _DEFAULT_ASSISTANT_GREETING)
        ]  # (role, text) pairs
        self._chat_client = build_chat_client(self._llm_backend) if self._use_llm else None
        # Start pose from the last loaded trajectory (set after __init__ via main()).
        self._current_start_pose: List[float] = [0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 1.0]
        # --- Tool-command state ---
        self._tool_command_queue = ToolCommandQueue()
        self._tool_runtime_state = ToolRuntimeState()
        self._command_executor = ToolCommandExecutor()
        self._chat_service = ChatService()
        self.TABLE_Z = TABLE_Z
        self._active_lie_strike_target_xyz: Optional[Tuple[float, float, float]] = None
        self._active_lie_strike_target_description: Optional[str] = None
        self._llm_tool_control_enabled = getattr(eval_args, "llm_tool_control_enabled", True)
        self._llm_live_command_queue_enabled = getattr(
            eval_args, "llm_live_command_queue_enabled", True
        )
        self._llm_release_table_clearance_m = float(
            getattr(eval_args, "llm_release_table_clearance_m", DEFAULT_RELEASE_TABLE_CLEARANCE_M)
        )
        self._llm_release_open_steps = int(
            getattr(eval_args, "llm_release_open_steps", DEFAULT_RELEASE_OPEN_STEPS)
        )
        self._llm_release_pre_move_timeout_steps = int(
            getattr(
                eval_args,
                "llm_release_pre_move_timeout_steps",
                DEFAULT_RELEASE_PREMOVE_TIMEOUT_STEPS,
            )
        )
        self._llm_freeze_after_release = bool(getattr(eval_args, "llm_freeze_after_release", True))
        self._llm_hold_last_goal_forever = bool(
            getattr(eval_args, "llm_hold_last_goal_forever", True)
        )
        self._has_custom_bootstrap_goals = bool(
            getattr(eval_args, "custom_goals_json_path", None) is not None
        )
        if self._llm_debug_log_path is not None:
            log_info(f"LLM debug logging enabled: {self._llm_debug_log_path}")

        # Wire up chat callback now that LLM state is initialised.
        if self.viser is not None and self._chat_client is not None:
            self.viser.register_chat_callback(self._handle_chat_send)
            self.viser.update_chat_history(self._chat_history)
        self._sync_active_goals_from_env()
        if not self._has_custom_bootstrap_goals:
            self._apply_default_grasp_hold_goal()
        self._refresh_named_strike_point_overlays()

    # -------------------------------------------------------------------------
    # LLM debug logging
    # -------------------------------------------------------------------------

    # Append one JSONL debug event for LLM chat diagnostics (best-effort).
    def _write_llm_debug_event(self, event: dict) -> None:
        """Write an LLM chat debug event to file when llm_debug_log_path is set."""
        if self._llm_debug_log_path is None:
            return
        event_with_ts = dict(event)
        event_with_ts["timestamp_utc"] = datetime.utcnow().isoformat() + "Z"
        try:
            self._llm_debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._llm_debug_log_path, "a") as f:
                f.write(json.dumps(_to_json_compatible(event_with_ts)) + "\n")
        except Exception as exc:
            log_warn(f"Failed to write llm debug log: {exc}")

    # Build one compact reset-debug payload from the env after a step.
    def _build_reset_debug_payload(self) -> dict:
        """Return current reset and goal-advance signals for live-step debugging."""
        env_reset_count = (
            int(self.env.reset_buf.sum().item()) if hasattr(self.env, "reset_buf") else 0
        )
        goal_reset_count = (
            int(self.env.reset_goal_buf.sum().item()) if hasattr(self.env, "reset_goal_buf") else 0
        )
        successes = int(self.env.successes[0].item()) if hasattr(self.env, "successes") else None
        max_consecutive_successes = (
            int(self.env.max_consecutive_successes)
            if hasattr(self.env, "max_consecutive_successes")
            else None
        )
        active_reset_reasons = (
            dict(getattr(self.env, "last_active_reset_reasons"))
            if hasattr(self.env, "last_active_reset_reasons")
            else {}
        )
        reset_reason_counts = (
            dict(getattr(self.env, "last_reset_reason_counts"))
            if hasattr(self.env, "last_reset_reason_counts")
            else {}
        )
        return {
            "env_reset_count": env_reset_count,
            "goal_reset_count": goal_reset_count,
            "successes": successes,
            "max_consecutive_successes": max_consecutive_successes,
            "active_reset_reasons": active_reset_reasons,
            "reset_reason_counts": reset_reason_counts,
        }

    # Write one reset-debug event when the env advances goals or triggers a full reset.
    def _maybe_log_live_reset_event(
        self, *, done: bool, terminated: bool, truncated: bool, timestep: int
    ) -> None:
        """Emit one debug event when a live step advances goals or resets the env."""
        payload = self._build_reset_debug_payload()
        if not done and payload["env_reset_count"] <= 0 and payload["goal_reset_count"] <= 0:
            return
        self._write_llm_debug_event(
            {
                "event": "live_step_reset_state",
                "timestep": int(timestep),
                "done": bool(done),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                **payload,
            }
        )
        if payload["env_reset_count"] > 0:
            log_warn(
                "Live env reset detected: "
                f"reasons={payload['active_reset_reasons'] or payload['reset_reason_counts']}, "
                f"successes={payload['successes']}, "
                f"max_consecutive_successes={payload['max_consecutive_successes']}"
            )
        elif payload["goal_reset_count"] > 0:
            log_info(
                "Live goal advance detected: "
                f"successes={payload['successes']}, "
                f"goal_reset_count={payload['goal_reset_count']}"
            )

    # -------------------------------------------------------------------------
    # Goal helpers
    # -------------------------------------------------------------------------

    # Read the active fixed-goal trajectory from Isaac Lab env tensors for live goal edits.
    def _get_current_goal_sequence(self) -> List[List[float]]:
        """Return current goal sequence as python lists from Isaac Lab env tensors.

        Prefers cfg.fixed_goal_states (full multi-goal sequence) and falls back to
        the single active goal_pose tensor.
        """
        if hasattr(self.env, "cfg") and hasattr(self.env.cfg, "fixed_goal_states"):
            fixed = self.env.cfg.fixed_goal_states
            if fixed:
                return [list(g) for g in fixed]
        if hasattr(self.env, "goal_pose"):
            pose = self.env.goal_pose
            if isinstance(pose, torch.Tensor) and pose.ndim == 2 and pose.shape[1] >= 7:
                return [pose[0, :7].detach().cpu().tolist()]
        return []

    # Keep runner-side active goals in sync with env state after reset/selection changes.
    def _sync_active_goals_from_env(self) -> None:
        """Refresh the active goal-list cache from env tensors/config when available."""
        current = self._get_current_goal_sequence()
        if current:
            self._active_goals = [list(g) for g in current]

    # Build a near-table release approach goal from the current object pose.
    def _build_pre_release_goals(self) -> List[List[float]]:
        """Create a short near-table goal sequence used before finger-open release."""
        _, object_pose, _, _, _, _, _ = self._get_state()
        target = [float(v) for v in object_pose[:7]]
        target[2] = max(target[2], TABLE_Z + self._llm_release_table_clearance_m)
        return [target, target.copy()]

    # Build a hover grasp goal from current tool pose while preserving tool orientation.
    def _build_grasp_hover_goals(self) -> List[List[float]]:
        """Create a single hover goal above the current tool pose for grasping."""
        _, object_pose, _, _, _, _, _ = self._get_state()
        target = [float(v) for v in object_pose[:7]]
        target[2] += float(getattr(self._eval_args, "llm_grasp_hover_offset_m", 0.03))
        return [target]

    # Apply default grasp-hold goal sequence used for taskless LLM bootstrapping.
    def _apply_default_grasp_hold_goal(self) -> None:
        """Set current active goals to one grasp-hover target above the current tool pose."""
        grasp_goals = self._build_grasp_hover_goals()
        self._apply_goals_live(grasp_goals)

    # Return the recorded predefined swing goals used for direct replay commands.
    def _load_predefined_swing_goals(self) -> List[List[float]]:
        """Return one cyclic playback of the recorded predefined motion for the active object."""
        predefined_goals = get_predefined_goal_sequence(
            object_name=self.object_name,
            task_name=self.task_name,
            include_start_pose=False,
        )
        target_a_payload = get_named_strike_point_payload(
            object_name=self.object_name,
            task_name=self.task_name,
        )
        target_a_world_xyz = (
            target_a_payload.get("points", {}).get("target_a", {}).get("world_xyz")
            if isinstance(target_a_payload, dict)
            else None
        )
        self._active_lie_strike_target_xyz = (
            tuple(float(value) for value in target_a_world_xyz)
            if isinstance(target_a_world_xyz, list) and len(target_a_world_xyz) == 3
            else None
        )
        self._active_lie_strike_target_description = "predefined swing"
        self._refresh_active_lie_strike_target_overlay()
        if supported_llm_object_family(self.object_name) == "screwdriver":
            self._pending_cyclic_goal_chunk = [list(goal) for goal in predefined_goals]
            return [list(goal) for goal in predefined_goals]
        mirrored_cycle = _build_mirrored_cycle(predefined_goals)
        self._pending_cyclic_goal_chunk = [list(goal) for goal in mirrored_cycle]
        return mirrored_cycle

    # Build the live Lie rollout sequence from the compiled semantic goals only.
    def _build_live_lie_goal_sequence(
        self,
        compiled_goals: List[List[float]],
        *,
        cycle_style: Optional[str],
    ) -> List[List[float]]:
        """Return one live Lie cycle, mirroring hammer swings and forward-repeating twists."""
        if cycle_style == "forward_repeat":
            self._pending_cyclic_goal_chunk = [list(goal) for goal in compiled_goals]
            return [list(goal) for goal in compiled_goals]
        mirrored_cycle = _build_mirrored_cycle(compiled_goals)
        self._pending_cyclic_goal_chunk = [list(goal) for goal in mirrored_cycle]
        return mirrored_cycle

    # Compile one coordinate-driven Lie swing trajectory from the startup/tool reference pose.
    # Compile one live Lie swing trajectory using the configured table-clearance constraints.
    def _compile_lie_trajectory_goals(
        self,
        strike_target_xy: List[float],
        target_description: Optional[str] = None,
    ) -> List[List[float]]:
        """Return compiled Lie swing goals for one explicit tabletop strike target."""
        clamped_target_xy, spec, compiled_goals, clamp_summary = compile_llm_lie_trajectory(
            object_name=self.object_name,
            task_name=self.task_name,
            pivot_point=self._current_start_pose[:3],
            strike_target_xy=strike_target_xy,
            horizontal_strike_clearance_m=float(
                getattr(
                    self._eval_args,
                    "llm_lie_horizontal_strike_clearance_m",
                    DEFAULT_LLM_LIE_HORIZONTAL_STRIKE_CLEARANCE_M,
                )
            ),
            waypoint_table_clearance_m=float(
                getattr(
                    self._eval_args,
                    "llm_lie_waypoint_table_clearance_m",
                    DEFAULT_LLM_LIE_WAYPOINT_TABLE_CLEARANCE_M,
                )
            ),
            screwdriver_twist_extra_hover_m=float(
                getattr(
                    self._eval_args,
                    "llm_lie_screwdriver_twist_extra_hover_m",
                    DEFAULT_LLM_LIE_SCREWDRIVER_TWIST_EXTRA_HOVER_M,
                )
            ),
            training_resampling_enabled=bool(
                getattr(self._eval_args, "llm_lie_training_resampling_enabled", True)
            ),
            training_resampling_pos_scale_m=float(
                getattr(
                    self._eval_args,
                    "llm_lie_training_resampling_pos_scale_m",
                    DEFAULT_LLM_LIE_TRAINING_RESAMPLING_POS_SCALE_M,
                )
            ),
            training_resampling_rot_scale_deg=float(
                getattr(
                    self._eval_args,
                    "llm_lie_training_resampling_rot_scale_deg",
                    DEFAULT_LLM_LIE_TRAINING_RESAMPLING_ROT_SCALE_DEG,
                )
            ),
            training_resampling_target_cost=float(
                getattr(
                    self._eval_args,
                    "llm_lie_training_resampling_target_cost",
                    DEFAULT_LLM_LIE_TRAINING_RESAMPLING_TARGET_COST,
                )
            ),
            training_resampling_min_waypoints=int(
                getattr(
                    self._eval_args,
                    "llm_lie_training_resampling_min_waypoints",
                    DEFAULT_LLM_LIE_TRAINING_RESAMPLING_MIN_WAYPOINTS,
                )
            ),
            training_volume_clamp_enabled=bool(
                getattr(self._eval_args, "llm_lie_training_volume_clamp_enabled", True)
            ),
            training_target_volume_mins=list(
                getattr(
                    self._eval_args,
                    "llm_lie_training_target_volume_mins",
                    DEFAULT_LLM_LIE_TRAINING_TARGET_VOLUME_MINS,
                )
            ),
            training_target_volume_maxs=list(
                getattr(
                    self._eval_args,
                    "llm_lie_training_target_volume_maxs",
                    DEFAULT_LLM_LIE_TRAINING_TARGET_VOLUME_MAXS,
                )
            ),
        )
        self._active_lie_strike_target_xyz = (
            float(clamped_target_xy[0]),
            float(clamped_target_xy[1]),
            float(TABLE_TOP_Z),
        )
        self._active_lie_strike_target_description = target_description
        self._refresh_active_lie_strike_target_overlay()
        resampling_summary = dict(spec.get("training_distribution_resampling", {}))
        if bool(resampling_summary.get("applied")):
            self._write_llm_debug_event(
                {
                    "event": "llm_lie_training_distribution_resampled",
                    "object_name": self.object_name,
                    "task_name": self.task_name,
                    "strike_target_xy": [float(value) for value in clamped_target_xy],
                    "resampling_summary": resampling_summary,
                    "llm_raw": spec,
                }
            )
        if bool(clamp_summary.get("applied")):
            self._write_llm_debug_event(
                {
                    "event": "llm_lie_training_volume_clamp_applied",
                    "object_name": self.object_name,
                    "task_name": self.task_name,
                    "strike_target_xy": [float(value) for value in clamped_target_xy],
                    "corrected_indices": list(clamp_summary.get("corrected_indices", [])),
                    "step_delta_summary": dict(clamp_summary.get("step_delta_summary", {})),
                    "llm_raw": spec,
                }
            )
        self._write_llm_debug_event(
            {
                "event": "llm_lie_compiled",
                "object_name": self.object_name,
                "task_name": self.task_name,
                "strike_target_xy": [float(value) for value in clamped_target_xy],
                "cycle_style": (
                    str(spec.get("cycle_style"))
                    if isinstance(spec, dict) and spec.get("cycle_style") is not None
                    else None
                ),
                "num_goals": len(compiled_goals),
                "llm_raw": spec,
            }
        )
        return self._build_live_lie_goal_sequence(
            compiled_goals,
            cycle_style=str(spec.get("cycle_style")) if isinstance(spec, dict) else None,
        )

    # Clear LLM runtime/session state so resets and object switches start from a clean slate.
    def _reset_llm_session_state(self) -> None:
        """Reset episode stats, chat history, queued commands, and runtime mode."""
        self._reset_episode_tracking()
        self._chat_history = [("assistant", _DEFAULT_ASSISTANT_GREETING)]
        self._active_goals = []
        self._cyclic_goal_chunk = None
        self._pending_cyclic_goal_chunk = None
        self._active_lie_strike_target_xyz = None
        self._active_lie_strike_target_description = None
        self._tool_command_queue.pop_all()
        self._tool_runtime_state = ToolRuntimeState()

    # Switch the active live object without creating a new env after startup.
    def _switch_active_object(self, object_name: str) -> None:
        """Switch the active live object in-place and refresh runtime/viewer state."""
        resolved_object_name = _validate_supported_startup_object_name(object_name)
        preserved_chat_history = list(getattr(self, "_chat_history", []))
        next_start_pose = _build_object_start_pose(self._eval_args, resolved_object_name)
        self.env.switch_active_object(
            resolved_object_name,
            object_start_pose=list(next_start_pose),
            reset=False,
        )
        from compat.legacy_env_wrapper import LegacyEnvWrapper

        self._legacy_env_wrapper = LegacyEnvWrapper(self.env)
        self.object_name = resolved_object_name
        self.task_name = supported_llm_task_name(resolved_object_name)
        self._current_table_urdf = _resolve_object_table_urdf(self._eval_args, resolved_object_name)
        self._current_start_pose = list(next_start_pose)
        self._reset_llm_session_state()
        if preserved_chat_history:
            self._chat_history = preserved_chat_history
        self._reset_policy_and_env_state()
        self._sync_active_goals_from_env()
        if not self._has_custom_bootstrap_goals:
            self._apply_default_grasp_hold_goal()
        self._refresh_llm_viewer_state(rebuild_scene=True)
        self._write_llm_debug_event(
            {
                "event": "active_object_switched",
                "object_name": self.object_name,
                "task_name": self.task_name,
                "table_urdf": self._current_table_urdf,
                "start_pose": list(self._current_start_pose),
            }
        )

    # Refresh the fixed named strike-point overlay set for LLM-mode hammer swings.
    def _refresh_named_strike_point_overlays(self) -> None:
        """Bind named strike-point overlays in the viewer when the active object supports them."""
        if self.viser is None:
            return
        payload = get_named_strike_point_payload(
            object_name=self.object_name,
            task_name=self.task_name,
        )
        if not payload.get("available"):
            self.viser.clear_named_strike_points_context()
            return
        points = payload.get("points", {})
        overlay_points = {
            name: tuple(point_payload["world_xyz"])
            for name, point_payload in points.items()
            if isinstance(point_payload, dict) and "world_xyz" in point_payload
        }
        self.viser.set_named_strike_points_context(overlay_points)

    # Refresh the active Lie strike-target marker from current runtime state.
    def _refresh_active_lie_strike_target_overlay(self) -> None:
        """Bind the active Lie strike-target overlay in the viewer when available."""
        if self.viser is None:
            return
        if self._active_lie_strike_target_xyz is None:
            self.viser.clear_active_lie_strike_target_context()
            return
        if is_target_a_strike_target_xy(
            self._active_lie_strike_target_xyz[:2],
            object_name=self.object_name,
            task_name=self.task_name,
            tolerance_m=_ACTIVE_TARGET_OVERLAY_EPSILON_M,
        ):
            self.viser.clear_active_lie_strike_target_context()
            return
        self.viser.set_active_lie_strike_target_context(self._active_lie_strike_target_xyz)

    # Refresh the taskless LLM viewer state after a reset, object switch, or startup restore.
    def _refresh_llm_viewer_state(self, *, rebuild_scene: bool = False) -> None:
        """Update the taskless viewer, preserving chat widgets and object-only scene rebuilds."""
        if self.viser is None:
            return
        self.viser.selection_changed = False
        self.viser.pending_object_name = self.object_name
        self.viser.pending_task_name = self.task_name
        if hasattr(self.viser, "_suppress_callbacks"):
            self.viser._suppress_callbacks = True
            try:
                if hasattr(self.viser, "category_dropdown") and hasattr(
                    self.viser, "object_dropdown"
                ):
                    category = OBJECT_NAME_TO_CATEGORY[self.object_name]
                    self.viser.category_dropdown.value = category
                    self.viser.object_dropdown.value = self.object_name
                if hasattr(self.viser, "task_dropdown"):
                    self.viser.task_dropdown.value = self.task_name
            finally:
                self.viser._suppress_callbacks = False
        if rebuild_scene:
            self.viser.rebuild_scene(self.object_name, None, self._current_table_urdf)
        self._refresh_named_strike_point_overlays()
        self._refresh_active_lie_strike_target_overlay()
        self.viser.update(*self._get_state())
        self.viser.update_progress(0, self._success_target(), 0, self.control_hz)
        self.viser.update_stats(0, 0.0, 0.0)
        self.viser.update_chat_history(self._chat_history)
        if hasattr(self.viser, "_chat_input"):
            self.viser._chat_input.value = ""

    # Override reset to always restore default grasp-hold goals in taskless mode.
    def _reset_scene(self):
        """Reset the active object/table bundle and restore its default grasp-hold bootstrap."""
        if self.viser is not None:
            # Signal episode loop to exit and unblock a paused loop.
            self.viser.reset_requested = True
            if self.viser.is_paused:
                self.viser.is_paused = False
                if hasattr(self.viser, "pause_button"):
                    self.viser.pause_button.name = "Pause"
                if hasattr(self.viser, "_chat_pause_button"):
                    self.viser._chat_pause_button.name = "Pause"
        # Wait for in-flight episode loop to stop.
        self._wait_for_episode_stop()
        if self.viser is not None:
            self.viser.reset_requested = False
        # Apply any buffered dropdown selection before considering startup restore behavior.
        if self._apply_pending_selection():
            return
        if (
            self.object_name == self._startup_object_name
            and self._current_table_urdf == self._startup_table_urdf
        ):
            self._restore_startup_state_in_place()
        else:
            self._restore_current_environment()

    # Fast path: restore startup state without recreating env when object/table already match.
    def _restore_startup_state_in_place(self) -> None:
        """Reset runtime/env state in-place to match fresh startup behavior."""
        log_info(
            "Scene fast-reset to startup state in-place: "
            f"{self._startup_object_name} / {self._startup_task_name}."
        )
        self.object_name = self._startup_object_name
        self.task_name = self._startup_task_name
        self._current_table_urdf = self._startup_table_urdf

        self._reset_llm_session_state()
        self._reset_policy_and_env_state()
        self._sync_active_goals_from_env()
        if not self._has_custom_bootstrap_goals:
            self._apply_default_grasp_hold_goal()

        self._refresh_llm_viewer_state(rebuild_scene=False)

    # Recreate env and runtime state for the original startup object/task.
    def _restore_startup_environment(self) -> None:
        """Rebuild env and runtime state to match script-start conditions."""
        log_info(
            "Scene reset to startup state: "
            f"{self._startup_object_name} / {self._startup_task_name}."
        )
        self._switch_active_object(self._startup_object_name)

    # Recreate runtime state for the current live object/task instead of restoring startup selection.
    def _restore_current_environment(self) -> None:
        """Rebuild env and runtime state for the currently active object/table bundle."""
        current_object_name = self.object_name
        current_task_name = self.task_name
        log_info("Scene reset to current state: " f"{current_object_name} / {current_task_name}.")
        self._switch_active_object(current_object_name)

    # Format one command as a concrete API call with parameters for chat transparency.
    def _format_api_call(self, command: ToolCommand) -> str:
        """Return an API-like method call string for a queued tool command."""
        if command.intent == ToolCommandIntent.GRASP_TOOL:
            hover = float(getattr(self._eval_args, "llm_grasp_hover_offset_m", 0.03))
            return f"api.grasp_tool(hover_offset_m={hover:.3f})"
        if command.intent == ToolCommandIntent.MOVE_TOOL:
            delta_t = command.delta_translation_m or [0.0, 0.0, 0.0]
            delta_r = command.delta_euler_rad or [0.0, 0.0, 0.0]
            frame = command.delta_frame or "camera_spawn"
            semantic_target = command.semantic_target
            semantic_preserve_position = command.semantic_preserve_position
            return (
                "api.apply_goal_delta("
                f"delta_translation_m={delta_t}, "
                f"delta_euler_rad={delta_r}, "
                f"frame='{frame}', "
                f"semantic_target={semantic_target!r}, "
                f"semantic_preserve_position={semantic_preserve_position!r}, "
                "target='all_active_goals'"
                ")"
            )
        if command.intent == ToolCommandIntent.RELEASE_TOOL:
            return (
                "api.release_tool("
                f"table_clearance_m={self._llm_release_table_clearance_m:.3f}, "
                f"open_steps={self._llm_release_open_steps}, "
                f"pre_move_timeout_steps={self._llm_release_pre_move_timeout_steps}"
                ")"
            )
        if command.intent == ToolCommandIntent.SET_GOALS:
            num_goals = len(command.goals or [])
            first_goal = (command.goals or [None])[0]
            return f"api.set_goals(num_goals={num_goals}, first_goal={first_goal})"
        if command.intent == ToolCommandIntent.EXECUTE_LIE_TRAJECTORY:
            return (
                "api.execute_lie_trajectory("
                f"strike_target_xy={command.strike_target_xy!r}, "
                f"target_description={command.target_description!r}, "
                f"replace_active_goals={command.replace_active_goals!r}"
                ")"
            )
        if command.intent == ToolCommandIntent.EXECUTE_PREDEFINED_SWING:
            return "api.execute_predefined_swing()"
        if command.intent == ToolCommandIntent.SWITCH_ACTIVE_OBJECT:
            return f"api.switch_active_object(object_name={command.object_name!r})"
        return "api.noop()"

    # Build one sim-context payload consumed by LLM chat clients/tool-calling.
    def _build_llm_sim_context(self) -> Dict[str, object]:
        """Return static strike geometry plus dynamic object/runtime state for LLM calls."""
        _, object_pose, goal_pose, _, _, _, _ = self._get_state()
        active_strike_target = {
            "available": self._active_lie_strike_target_xyz is not None,
            "world_xy": (
                [
                    float(self._active_lie_strike_target_xyz[0]),
                    float(self._active_lie_strike_target_xyz[1]),
                ]
                if self._active_lie_strike_target_xyz is not None
                else None
            ),
            "world_xyz": (
                [float(v) for v in self._active_lie_strike_target_xyz]
                if self._active_lie_strike_target_xyz is not None
                else None
            ),
            "description": self._active_lie_strike_target_description,
        }
        return {
            "current_object": self.object_name,
            "start_pose": self._current_start_pose,
            "episode_count": self.episode_count,
            "static_strike_context": get_llm_static_strike_context(
                object_name=self.object_name,
                task_name=self.task_name,
            ),
            "sim_state": {
                "object_name": self.object_name,
                "runtime_mode": self._tool_runtime_state.mode.value,
                "object_pose_xyzw": [float(v) for v in object_pose[:7]],
                "goal_pose_xyzw": [float(v) for v in goal_pose[:7]],
                "pose_semantics": get_object_pose_semantics_payload(self.object_name),
                "active_strike_target": active_strike_target,
            },
        }

    # Build a natural-language summary for one queued command.
    def _format_executed_command_summary(self, command: ToolCommand) -> str:
        """Return a user-facing natural-language summary of queued execution."""
        return format_chat_command_summary(command)

    # Apply object selection changes in taskless mode (object-only, no task trajectory).
    def _apply_pending_selection(self) -> bool:
        """Switch to one preloaded object selection and re-bootstrap grasp-hold goals."""
        if self.viser is None or not self.viser.selection_changed:
            return False
        if self._eval_args is None:
            log_warn("Cannot switch selection: eval_args not stored.")
            return False

        new_obj = self.viser.pending_object_name
        self.viser.selection_changed = False
        log_info(f"Applying selection: {new_obj} — switching preloaded bundle...")
        self._switch_active_object(str(new_obj))

        log_info(f"Switched to {new_obj}.")
        return True

    # Synchronize env-side fixed-goal buffers with the current runner-side active goals.
    def _sync_live_goal_config(self) -> None:
        """Synchronize fixed goals, trajectory states, and success caps from active goals."""
        active_goals = [list(goal) for goal in self._active_goals]
        self.env.cfg.fixed_goal_states = active_goals
        if hasattr(self.env, "trajectory_states"):
            self.env.trajectory_states = torch.tensor(
                active_goals,
                device=self.env.goal_pos.device,
                dtype=torch.float32,
            )
        if self._cyclic_goal_chunk is not None:
            success_cap = len(active_goals)
        elif bool(getattr(self._eval_args, "llm_hold_last_goal_forever", True)):
            success_cap = _HOLD_LAST_GOAL_MAX_SUCCESSES
        else:
            success_cap = len(active_goals)
        self.env.max_consecutive_successes = success_cap
        self.env.cfg.max_consecutive_successes = success_cap

    # Append one more mirrored cycle before a cyclic live-goal sequence reaches its current tail.
    def _maybe_extend_cyclic_goals(self) -> None:
        """Append another mirrored cycle when the active cyclic swing sequence nears exhaustion."""
        if self._cyclic_goal_chunk is None:
            return
        if self._success_count() + 2 < len(self._active_goals):
            return
        self._active_goals.extend([list(goal) for goal in self._cyclic_goal_chunk])
        self._sync_live_goal_config()

    # BUG FIX: write directly into Isaac Lab tensors so no physics teardown occurs.
    def _apply_goals_live(self, goals: List[List[float]]) -> None:
        """Apply new goals in-place for Isaac Lab env — robot keeps pose, no env recreation.

        Clamps each goal's z to stay above the table surface, writes the first goal
        into the live per-env tensors (goal_pos, goal_rot, goal_pose, goal_states) so
        the policy immediately starts chasing it, and stores the full sequence in
        cfg.fixed_goal_states so the env cycles through subsequent goals on success.
        """
        min_pose_z = TABLE_Z + float(getattr(self._eval_args, "z_offset", 0.03))
        clamped = [[g[0], g[1], max(g[2], min_pose_z)] + list(g[3:]) for g in goals]
        EvalRunner._apply_goals_live(self, clamped)
        # Keep a runner-side copy so cyclic extension and final-goal hold can inspect it.
        self._active_goals = [list(g) for g in clamped]
        pending_cyclic_chunk = getattr(self, "_pending_cyclic_goal_chunk", None)
        self._cyclic_goal_chunk = (
            [list(goal) for goal in pending_cyclic_chunk]
            if pending_cyclic_chunk is not None
            else None
        )
        self._pending_cyclic_goal_chunk = None
        self._sync_live_goal_config()
        self._write_llm_debug_event(
            {
                "event": "live_goals_applied",
                "num_goals": len(clamped),
                "first_goal": clamped[0] if clamped else None,
            }
        )
        log_info(f"Goals updated in-place ({len(clamped)} waypoints).")

    # Report UI progress target from active goals when hold-last mode is enabled.
    def _success_target(self) -> int:
        """Return displayed target count while keeping env reset threshold decoupled."""
        active_goals = getattr(self, "_active_goals", [])
        cyclic_goal_chunk = getattr(self, "_cyclic_goal_chunk", None)
        if cyclic_goal_chunk:
            return max(1, len(cyclic_goal_chunk))
        if bool(getattr(self._eval_args, "llm_hold_last_goal_forever", True)) and active_goals:
            return max(1, len(active_goals))
        return EvalRunner._success_target(self)

    # Enforce final-goal hold state to prevent wrap/reset after the last waypoint.
    def _enforce_hold_last_goal_if_reached(
        self, done: bool, terminated: bool, truncated: bool
    ) -> Tuple[bool, bool, bool]:
        """Pin the last goal and clear reset triggers once the final waypoint is reached."""
        if getattr(self, "_cyclic_goal_chunk", None):
            return done, terminated, truncated
        if not bool(getattr(self._eval_args, "llm_hold_last_goal_forever", True)):
            return done, terminated, truncated
        if not self._active_goals or not hasattr(self.env, "successes"):
            return done, terminated, truncated

        last_goal_idx = len(self._active_goals) - 1
        if int(self.env.successes[0].item()) < last_goal_idx:
            return done, terminated, truncated

        last_goal = self._active_goals[last_goal_idx]
        dev = self.env.goal_pos.device
        t = torch.tensor(last_goal, device=dev, dtype=torch.float32)
        self.env.goal_pos[0] = t[:3]
        self.env.goal_rot[0] = t[3:7]
        self.env.goal_pose[0] = t[:7]
        self.env.goal_states[0, :7] = t[:7]

        # Clamp counters at the final goal so env modulo indexing never wraps to goal 0.
        self.env.successes[0] = float(last_goal_idx)
        if hasattr(self.env, "consecutive_successes"):
            self.env.consecutive_successes[0] = float(last_goal_idx)

        # Clear deferred reset flags that would otherwise reset scene/goal on next step.
        if hasattr(self.env, "reset_goal_buf"):
            self.env.reset_goal_buf[0] = 0
        if hasattr(self.env, "reset_buf"):
            self.env.reset_buf[0] = 0
        if hasattr(self.env, "progress_buf"):
            self.env.progress_buf[0] = 0

        # Keep stepping indefinitely in final-goal hold mode until manual reset/retarget.
        return False, False, False

    # -------------------------------------------------------------------------
    # Release / freeze helpers
    # -------------------------------------------------------------------------

    # Set post-release freeze state and pause GUI so policy does not regrasp.
    def _freeze_after_release(self) -> None:
        """Freeze execution after release so policy control does not resume automatically."""
        enter_frozen(self._tool_runtime_state)

    # Build action that holds arm at frozen pose while forcing open-hand posture.
    def _build_frozen_release_action(self) -> torch.Tensor:
        """Return normalized action that freezes arm joints and keeps fingers open."""
        device = self.env.goal_pos.device
        action = torch.zeros((self.env.num_envs, self.n_act), device=device, dtype=torch.float32)
        if not hasattr(self.env, "arm_hand_dof_pos"):
            return apply_open_hand_override(action, self.env)

        lower = self.env.arm_hand_dof_lower_limits[: self.n_act]
        upper = self.env.arm_hand_dof_upper_limits[: self.n_act]
        denom = torch.clamp(upper - lower, min=1e-6)

        cached = self._tool_runtime_state.meta.get("frozen_arm_dof_pos")
        if cached is None:
            cached = self.env.arm_hand_dof_pos[0, :7].detach().clone()
            self._tool_runtime_state.meta["frozen_arm_dof_pos"] = cached
        arm_target = cached.to(device=device, dtype=torch.float32)
        arm_norm = torch.clamp((2.0 * (arm_target - lower[:7]) / denom[:7]) - 1.0, -1.0, 1.0)
        action[:, :7] = arm_norm.unsqueeze(0).repeat(self.env.num_envs, 1)
        return apply_open_hand_override(action, self.env)

    # -------------------------------------------------------------------------
    # Live command processing (runs at safe step boundaries)
    # -------------------------------------------------------------------------

    # Process queued chat commands at safe step boundaries in the simulation loop.
    def _process_live_commands(self) -> None:
        """Consume queued commands and apply policy-compatible runtime actions."""
        if not hasattr(self, "_command_executor"):
            self._command_executor = ToolCommandExecutor()
        if not hasattr(self, "TABLE_Z"):
            self.TABLE_Z = TABLE_Z
        self._command_executor.process_queued_commands(self)

    # -------------------------------------------------------------------------
    # Simulation step override — adds command processing + release state machine
    # -------------------------------------------------------------------------

    def _sim_step(self, timestep: int):
        """Execute one sim step: process live commands, run policy, apply release FSM."""
        t0 = time.time()
        self._process_live_commands()
        self._maybe_extend_cyclic_goals()
        if self.viser is not None:
            self.viser.update(*self._get_state())
        if self._tool_runtime_state.mode == RuntimeMode.FROZEN_AFTER_RELEASE:
            action = self._build_frozen_release_action()
        else:
            action = self.policy.get_normalized_action(self.obs, deterministic_actions=True)
        # Release mode is now a direct frozen state handled by _build_frozen_release_action.
        self.obs, done, terminated, truncated = self._step(action)
        self._maybe_log_live_reset_event(
            done=done,
            terminated=terminated,
            truncated=truncated,
            timestep=timestep,
        )
        done, terminated, truncated = self._enforce_hold_last_goal_if_reached(
            done, terminated, truncated
        )
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

    # -------------------------------------------------------------------------
    # Episode loop override — adds enter_policy and live-command pause hook
    # -------------------------------------------------------------------------

    # Decide whether episode loop should continue for current done state.
    def _should_continue_episode(self, done: bool) -> bool:
        """Return True while episode should keep stepping, including interactive keep-alive."""
        if not done:
            return True
        interactive_mode = bool(getattr(self._eval_args, "interactive", False))
        keep_alive = bool(getattr(self._eval_args, "llm_keep_episode_alive_interactive", True))
        if interactive_mode and keep_alive:
            return True
        return False

    def _run_episode(self):
        """Run a single evaluation episode with LLM runtime control."""
        if not hasattr(self, "_run_episode_in_progress"):
            self._run_episode_in_progress = False
        if self._run_episode_in_progress:
            log_warn("Episode already in progress. Skipping...")
            return
        self._run_episode_in_progress = True

        try:
            self._apply_pending_selection()

            enter_policy(self._tool_runtime_state)
            self.policy.reset()
            log_info("Reset...")
            self.obs = self._reset()
            self._sync_active_goals_from_env()
            if self.viser is not None:
                self.viser.update(*self._get_state())

            log_info(f"Running{' (+ recording)' if self.record_video else ''}...")
            states, step, done = [], 0, False
            peak_success_count = self._success_count()
            wall_start = time.time()
            realtime_violations = 0

            while self._should_continue_episode(done):
                # Handle pause — also process live commands while paused.
                while self.viser is not None and self.viser.is_paused:
                    self._process_live_commands()
                    time.sleep(0.1)

                if self.viser is not None and self.viser.reset_requested:
                    log_info("Reset requested — aborting episode.")
                    return

                if self.record_video and step % self.record_interval == 0:
                    states.append(tuple(x.copy() for x in self._get_state()))
                done, terminated, truncated, action = self._sim_step(step)
                step += 1
                peak_success_count = max(peak_success_count, self._success_count())
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

            goal_pct = 100 * peak_success_count / self._success_target()
            self.episode_goal_pcts.append(goal_pct)
            self.episode_lengths.append(step)
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
            from termcolor import colored

            print(colored(f"Done: {step / self.control_hz:.1f}s, {goal_pct:.0f}% goals", "green"))
        finally:
            self._run_episode_in_progress = False

    # -------------------------------------------------------------------------
    # Chat handler
    # -------------------------------------------------------------------------

    def _handle_chat_send(self, message: str) -> None:
        """Process a user chat message: call the LLM and apply any generated goals.

        Called by the ViserServer button callback on the viser background thread.
        Goal updates are applied in-place (no env recreation).
        """
        self._chat_history.append(("user", message))
        if self.viser is not None:
            self.viser.update_chat_history(self._chat_history + [("assistant", "typing...")])
        executed_lines: List[str] = []

        if self._chat_client is None:
            response = ChatResponse(text="Command queued.")
        else:
            sim_context = self._build_llm_sim_context()
            try:
                response = self._chat_service.send_to_llm(
                    self._chat_client, self._chat_history, sim_context
                )
            except Exception as exc:
                tb = traceback.format_exc()
                log_warn(f"LLM chat exception: {exc}")
                log_warn(tb)
                self._write_llm_debug_event(
                    {
                        "event": "llm_chat_exception",
                        "message": message,
                        "history": self._chat_history,
                        "sim_context": sim_context,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "traceback": tb,
                    }
                )
                response = ChatResponse(text=f"[Error: {exc}]")

        if response.goals is not None:
            set_goals_command = ToolCommand(
                intent=ToolCommandIntent.SET_GOALS, goals=response.goals, raw_text=message
            )
            self._tool_command_queue.push(set_goals_command)
            executed_lines.append(self._format_executed_command_summary(set_goals_command))
            self._write_llm_debug_event(
                {
                    "event": "tool_cmd_queued",
                    "intent": ToolCommandIntent.SET_GOALS.value,
                    "num_goals": len(response.goals),
                }
            )
            log_info(f"LLM generated {len(response.goals)} goals. Applying at next safe step.")
        if response.command is not None:
            self._tool_command_queue.push(response.command)
            self._write_llm_debug_event(
                {
                    "event": "tool_cmd_queued",
                    "intent": response.command.intent.value,
                    "delta_translation_m": response.command.delta_translation_m,
                    "delta_euler_rad": response.command.delta_euler_rad,
                    "delta_frame": response.command.delta_frame,
                    "semantic_target": response.command.semantic_target,
                    "semantic_preserve_position": response.command.semantic_preserve_position,
                    "strike_target_xy": response.command.strike_target_xy,
                    "target_description": response.command.target_description,
                    "replace_active_goals": response.command.replace_active_goals,
                }
            )
            executed_lines.append(self._format_executed_command_summary(response.command))
            log_info("LLM queued live tool command.")

        assistant_text = "\n".join(executed_lines) if executed_lines else response.text
        self._chat_history.append(("assistant", assistant_text))
        if self.viser is not None:
            self.viser.update_chat_history(self._chat_history)


def main():
    """Entry point: parse LLMEvalArgs and run LLMEvalRunner."""
    args: LLMEvalArgs = tyro.cli(LLMEvalArgs)
    selected_object_name = _resolve_startup_object_name(args)
    args.object_name = selected_object_name
    args.object_category = OBJECT_NAME_TO_CATEGORY[selected_object_name]
    args.preloaded_object_names = supported_llm_object_names()
    args.preloaded_table_urdfs = list(
        dict.fromkeys(
            OBJECT_CATEGORY_TO_TABLE_URDF[OBJECT_NAME_TO_CATEGORY[object_name]]
            for object_name in supported_llm_object_names()
        ).keys()
    )
    args.target_grid_xy = build_target_grid_xy(3, 3)

    # Launch Omniverse runtime once; reused for every selection change.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    env, selected_table_urdf, initial_start_pose = _build_eval_env_llm(
        args,
        selected_object_name,
        app_launcher,
    )

    runner = LLMEvalRunner(
        env=env,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        object_name=selected_object_name,
        task_name=supported_llm_task_name(selected_object_name),
        table_urdf=selected_table_urdf,
        output_dir=args.output_dir,
        policy_name=args.policy_name,
        enable_viser=(args.enable_viser or args.interactive),
        interactive_autorun=args.interactive_autorun,
        exit_after_episodes=args.exit_after_episodes,
        telemetry_json_path=args.telemetry_json_path,
        max_realtime_factor=args.max_realtime_factor,
        eval_args=args,
        app_launcher=app_launcher,
        data_structure=supported_llm_data_structure(),
    )
    # Store initial start pose so the chat client has the object reference position.
    runner._current_start_pose = initial_start_pose
    _finalize_startup_state(runner)
    _maybe_send_startup_chat_message(runner, args)

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
