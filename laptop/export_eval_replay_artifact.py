"""Export offline replay artifacts for laptop-friendly cached playback."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing
import os
import sys
import tempfile
import time
import types
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Optional

import torch
import tyro

# Support script-style execution by ensuring the repo root is importable.
if (
    importlib.util.find_spec("geometric_tool_planning") is None
    or importlib.util.find_spec("laptop") is None
):
    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from compat.legacy_env_wrapper import LegacyEnvWrapper
from deployment.rl_player import RlPlayer
from dextoolbench.eval import TABLE_Z, build_runtime_snapshot_payload
from dextoolbench.eval_config import (
    DEFAULT_CONTROL_HZ,
    DEFAULT_EVAL_SUCCESS_TOLERANCE_M,
    TABLE_URDF,
)
from dextoolbench.eval_goal_sources import (
    _build_mirrored_cycle,
    _build_policy_env,
    _policy_goals_for_artifact,
)
from dextoolbench.shutdown_utils import close_simulation_app_with_timeout
from geometric_tool_planning import build_goal_source_artifacts
from geometric_tool_planning.orchestrator_types import EvalGoalSourcesArgs
from laptop.utils import log_info, log_warn, to_json_compatible

_PENDING_SIMULATION_APP = None
_PREDEFINED_REPLAY_SOURCES = (
    ("predefined_swing", "claw_hammer", "swing_down"),
    ("predefined_twist", "long_screwdriver", "spin_vertical"),
)


class _MinimalPolicyReplayRunner:
    """Capture predefined policy replays without constructing the interactive eval runner."""

    # Initialize one reusable headless policy runner for predefined replay export.
    def __init__(self, export_args: ExportOfflineReplayArgs, *, app_launcher) -> None:
        """Create one reusable policy and store exporter/runtime dependencies."""
        self._export_args = export_args
        self._app_launcher = app_launcher
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = RlPlayer(
            140,
            29,
            export_args.config_path,
            export_args.checkpoint_path,
            self.device,
            1,
        )
        self.env = None
        self.wrapper = None
        self.obs = None
        self.control_hz = float(export_args.control_hz)
        self.control_dt = 1.0 / max(self.control_hz, 1e-6)
        self.joint_lower = None
        self.joint_upper = None
        self._goal_args = None
        self._artifact = None
        self._active_policy_goals: list[list[float]] = []
        self._cyclic_goal_chunk: list[list[float]] | None = None

    # Close the currently loaded Isaac env when switching sources or shutting down export.
    def close(self) -> None:
        """Close the currently active environment when one exists."""
        if self.env is None:
            return
        self.env.close()
        self.env = None
        self.wrapper = None
        self.obs = None

    # Apply one fixed-goal sequence directly to the current env tensors.
    def _apply_goals_live(self, goals: list[list[float]]) -> None:
        """Install one fixed-goal sequence into the active env without rebuilding it."""
        min_pose_z = TABLE_Z + float(self._export_args.z_offset)
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

    # Force the current env back to spawn state and rebuild the first policy observation.
    def _reset_policy_and_env_state(self) -> None:
        """Reset policy hidden state and active env to its initial state."""
        self.policy.reset()
        self.env.reset()
        self.obs = self.wrapper.reset(device=self.device)

    # Return the current env success count for source progress tracking.
    def _success_count(self) -> int:
        """Return the current success count for env 0."""
        return int(self.env.successes[0].item()) if hasattr(self.env, "successes") else 0

    # Extend cyclic predefined goals when replay approaches the current goal cap.
    def _maybe_extend_cyclic_goals(self) -> None:
        """Append another mirrored predefined cycle before the current cap is exhausted."""
        if self._cyclic_goal_chunk is None:
            return
        success_count = self._success_count()
        if success_count + 2 < len(self._active_policy_goals):
            return
        self._active_policy_goals.extend([list(goal) for goal in self._cyclic_goal_chunk])
        self.env.cfg.fixed_goal_states = [list(goal) for goal in self._active_policy_goals]
        self.env.cfg.max_consecutive_successes = len(self._active_policy_goals)
        self.env.max_consecutive_successes = len(self._active_policy_goals)

    # Return the current joint/object/goal state needed for cached replay export.
    def _state(self) -> tuple[list[float], list[float], list[float]]:
        """Return current joint positions, object pose, and goal pose."""
        if hasattr(self.env, "_populate_sim_buffers"):
            self.env._populate_sim_buffers()
        obs_np = self.obs[0].detach().cpu().numpy()
        joint_pos = (
            0.5 * (obs_np[:29] + 1.0) * (self.joint_upper - self.joint_lower) + self.joint_lower
        )
        object_pose = torch.cat([self.env.object_pos[0], self.env.object_rot[0]], dim=-1)
        goal_pose = self.env.goal_pose[0]
        return (
            [float(value) for value in joint_pos.tolist()],
            [float(value) for value in object_pose.detach().cpu().tolist()],
            [float(value) for value in goal_pose.detach().cpu().tolist()],
        )

    # Capture one replay frame from the current env state.
    def _capture_frame(self, *, step: int, event: str) -> dict[str, Any]:
        """Return one offline replay frame from the active env state."""
        joint_pos, object_pose, goal_pose = self._state()
        success_count = self._success_count()
        return {
            "tool_pose": object_pose,
            "goal_pose": goal_pose,
            "goal_index": int(success_count),
            "success_count": int(success_count),
            "sim_time_sec": float(step / max(self.control_hz, 1e-6)),
            "event": str(event),
            "robot_joint_positions": joint_pos,
        }

    # Step the active env once with the policy and return done flags.
    def _sim_step(self) -> tuple[bool, bool, bool]:
        """Execute one policy-controlled env step and return done flags."""
        action = self.policy.get_normalized_action(self.obs, deterministic_actions=True)
        self.obs, _, done_tensor, info = self.wrapper.step(action)
        terminated = bool(info["terminated"][0].item())
        truncated = bool(info["truncated"][0].item())
        return bool(done_tensor[0].item()), terminated, truncated

    # Load one predefined source by rebuilding only the env and active goal list.
    def load_source(
        self, *, object_name: str, task_name: str
    ) -> tuple[EvalGoalSourcesArgs, Any, str]:
        """Build and install one predefined source into the reusable export runner."""
        self.close()
        goal_args = _build_predefined_goal_args(
            self._export_args,
            object_name=object_name,
            task_name=task_name,
        )
        artifact = _load_predefined_artifact(goal_args)
        env, table_urdf, _ = _build_policy_env(
            goal_args,
            object_name=object_name,
            task_name=task_name,
            artifact=artifact,
            app_launcher=self._app_launcher,
        )
        self.env = env
        self.wrapper = LegacyEnvWrapper(env)
        self.joint_lower = self.wrapper.arm_hand_dof_lower_limits[:29].detach().cpu().numpy()
        self.joint_upper = self.wrapper.arm_hand_dof_upper_limits[:29].detach().cpu().numpy()
        dt = float(getattr(env.cfg.sim, "dt", 1.0 / 60.0))
        decimation = int(getattr(env.cfg, "decimation", 2))
        self.control_hz = 1.0 / (dt * decimation)
        self.control_dt = 1.0 / self.control_hz
        self._goal_args = goal_args
        self._artifact = artifact
        raw_goals = _policy_goals_for_artifact(artifact)
        mirrored_cycle = _build_mirrored_cycle(raw_goals)
        self._active_policy_goals = [list(goal) for goal in mirrored_cycle]
        self._cyclic_goal_chunk = [list(goal) for goal in mirrored_cycle]
        self._apply_goals_live(self._active_policy_goals)
        self._reset_policy_and_env_state()
        return goal_args, artifact, str(table_urdf)

    # Capture one capped replay for the currently loaded predefined source.
    def capture_loaded_source(
        self, *, mode: str, object_name: str, task_name: str
    ) -> dict[str, Any]:
        """Return one capped replay payload for the currently loaded predefined source."""
        max_steps = max(
            1,
            int(
                round(
                    _resolve_policy_replay_duration_sec(
                        self._export_args.policy_replay_duration_sec
                    )
                    * max(self.control_hz, 1e-6)
                )
            ),
        )
        frames = [self._capture_frame(step=0, event="startup")]
        step = 0
        done = False
        peak_success_count = self._success_count()
        duration_cap_reached = False
        while not done:
            self._maybe_extend_cyclic_goals()
            done, terminated, truncated = self._sim_step()
            del terminated, truncated
            step += 1
            duration_cap_reached = step >= max_steps and not done
            peak_success_count = max(peak_success_count, self._success_count())
            frames.append(
                self._capture_frame(
                    step=step,
                    event=(
                        "done"
                        if done
                        else "duration_cap_reached" if duration_cap_reached else "running"
                    ),
                )
            )
            if duration_cap_reached:
                done = True
        return {
            "mode": mode,
            "object_name": object_name,
            "task_name": task_name,
            "frames": frames,
            "summary": {
                "episode_goal_pct": float(
                    100.0
                    * min(peak_success_count, len(self._cyclic_goal_chunk))
                    / max(1, len(self._cyclic_goal_chunk))
                ),
                "episode_length_steps": int(step),
                "eval_success_tolerance": float(self._export_args.eval_success_tolerance),
                "avg_time_sec": float(step / max(self.control_hz, 1e-6)),
                "duration_cap_reached": bool(duration_cap_reached),
            },
            "artifact": self._artifact,
        }


@dataclass
class ExportOfflineReplayArgs:
    """CLI args for exporting one offline replay artifact."""

    output: Path
    """Output json path for the offline replay artifact."""

    object_name: str = "claw_hammer"
    """Object name for the exported replay artifact."""

    task_name: str = "swing_down"
    """Task name for the exported replay artifact."""

    control_hz: float = DEFAULT_CONTROL_HZ
    """Nominal control rate recorded in the exported artifact."""

    source_kind: str = "policy_goal_sources"
    """Artifact source kind: `predefined`, `policy_rollout`, or `policy_goal_sources`."""

    downsample_factor: int = 1
    """Optional stride for predefined replay export; defaults to the full cached path."""

    telemetry_history_path: Optional[Path] = None
    """JSONL trace sampled while `eval.py` runs when exporting cached policy rollouts."""

    eval_json_path: Optional[Path] = None
    """Optional `eval.json` summary emitted by `eval.py`."""

    runtime_snapshot_path: Optional[Path] = None
    """Optional `runtime_snapshot.json` emitted by `eval.py`."""

    llm_backend: str = "mock"
    """LLM backend used when generating deterministic goal-source artifacts."""

    config_path: Path = Path("pretrained_policy/config.yaml")
    """Policy config used for goal-source rollout export."""

    checkpoint_path: Path = Path("pretrained_policy/model.pth")
    """Policy checkpoint used for goal-source rollout export."""

    policy_name: Optional[str] = None
    """Optional display name for the cached policy replay metadata."""

    force_table_urdf: bool = True
    """Whether to force the shared narrow table URDF during goal-source export."""

    use_task_env_urdf: bool = False
    """Whether to use the task-specific env table URDF during goal-source export."""

    z_offset: float = 0.03
    """Vertical offset applied to goal-source policy env construction."""

    eval_success_tolerance: float = DEFAULT_EVAL_SUCCESS_TOLERANCE_M
    """Success tolerance to use for goal-source policy replay export."""

    policy_replay_duration_sec: float = 60.0
    """Per-trajectory policy rollout duration used when exporting cached replay artifacts."""

    single_predefined_mode: Optional[str] = None
    """Internal-only single-source export mode used to isolate Isaac app state per source."""


# Downsample one fixed pose list while preserving the final pose.
def _downsample_goals(goals: List[List[float]], downsample_factor: int) -> List[List[float]]:
    """Return one compact pose list with the last goal always preserved."""
    if downsample_factor <= 1 or len(goals) <= 2:
        return goals
    downsampled = [list(goal) for goal in goals[::downsample_factor]]
    if downsampled[-1] != goals[-1]:
        downsampled.append(list(goals[-1]))
    return downsampled


# Load one optional JSON payload and fall back to an empty dictionary when absent.
def _load_optional_json(path: Optional[Path]) -> dict[str, Any]:
    """Return one parsed JSON dictionary or an empty mapping."""
    if path is None:
        return {}
    return json.loads(path.read_text())


# Load one telemetry history JSONL file into a list of per-step dictionaries.
def _load_telemetry_history(path: Path) -> List[dict[str, Any]]:
    """Return all parsed telemetry-history samples from one JSONL file."""
    samples: List[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped:
            samples.append(json.loads(stripped))
    if not samples:
        raise ValueError("telemetry_history_path must contain at least one JSON object.")
    return samples


# Validate one requested policy replay duration before launching Isaac export.
def _resolve_policy_replay_duration_sec(duration_sec: float) -> float:
    """Return one validated policy replay duration in seconds."""
    if duration_sec <= 0.0:
        raise ValueError("policy_replay_duration_sec must be positive.")
    return float(duration_sec)


# Remove Isaac's prebundled torchvision path before AppLauncher import to avoid mixed torch stacks.
def _sanitize_isaac_python_path() -> None:
    """Strip Isaac's prebundled torchvision path before launching AppLauncher."""
    prebundle_marker = "omni.isaac.ml_archive/pip_prebundle"
    sys.path = [path for path in sys.path if prebundle_marker not in path]
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


# Install a small torchvision stub so Isaac task discovery avoids incompatible prebundled ops.
def _install_torchvision_stub() -> None:
    """Register a minimal torchvision stub before AppLauncher imports isaaclab_tasks."""
    if "torchvision" in sys.modules:
        return
    torchvision_module = types.ModuleType("torchvision")
    torchvision_utils = types.ModuleType("torchvision.utils")
    torchvision_transforms = types.ModuleType("torchvision.transforms")
    torchvision_models = types.ModuleType("torchvision.models")
    torchvision_datasets = types.ModuleType("torchvision.datasets")
    torchvision_io = types.ModuleType("torchvision.io")
    torchvision_ops = types.ModuleType("torchvision.ops")
    torchvision_utils.save_image = lambda *args, **kwargs: None
    torchvision_utils.make_grid = lambda *args, **kwargs: None
    torchvision_transforms.Compose = lambda transforms: transforms
    torchvision_transforms.Normalize = lambda *args, **kwargs: ("normalize", args, kwargs)
    torchvision_module.utils = torchvision_utils
    torchvision_module.transforms = torchvision_transforms
    torchvision_module.models = torchvision_models
    torchvision_module.datasets = torchvision_datasets
    torchvision_module.io = torchvision_io
    torchvision_module.ops = torchvision_ops
    sys.modules["torchvision"] = torchvision_module
    sys.modules["torchvision.utils"] = torchvision_utils
    sys.modules["torchvision.transforms"] = torchvision_transforms
    sys.modules["torchvision.models"] = torchvision_models
    sys.modules["torchvision.datasets"] = torchvision_datasets
    sys.modules["torchvision.io"] = torchvision_io
    sys.modules["torchvision.ops"] = torchvision_ops


# Build one compact single-source replay payload from the recorded reference path.
def _build_predefined_replay_payload(args: ExportOfflineReplayArgs) -> dict[str, Any]:
    """Return one legacy offline replay payload for the recorded predefined swing path."""
    goal_args = EvalGoalSourcesArgs(
        goal_source="predefined",
        object_name=args.object_name,
        task_name=args.task_name,
        llm_backend=args.llm_backend,
        enable_viser=False,
        control_hz=args.control_hz,
    )
    reference = build_goal_source_artifacts(goal_args)[0]
    sampled_goals = _downsample_goals(reference.goals, downsample_factor=args.downsample_factor)
    frames = []
    for index, tool_pose in enumerate(sampled_goals):
        frames.append(
            {
                "tool_pose": [float(value) for value in tool_pose],
                "goal_pose": [float(value) for value in tool_pose],
                "goal_index": int(index),
                "success_count": int(index),
                "sim_time_sec": float(index / max(args.control_hz, 1e-6)),
                "event": "goal_reached" if index > 0 else "startup",
                "robot_joint_positions": None,
            }
        )
    return {
        "schema_version": "offline_replay_v1",
        "object_name": args.object_name,
        "task_name": args.task_name,
        "source": "cached_predefined_swing",
        "control_hz": float(args.control_hz),
        "table_urdf": TABLE_URDF,
        "summary": {
            "episode_goal_pct": 100.0,
            "episode_length_steps": len(frames),
            "eval_success_tolerance": DEFAULT_EVAL_SUCCESS_TOLERANCE_M,
            "artifact_note": (
                "Offline replay artifact derived from the recorded predefined swing path. "
                "It is intended for laptop-side playback and predefined-path inspection."
            ),
        },
        "runtime_snapshot": {
            "backend": "offline_predefined_replay",
            "control": {"control_hz": float(args.control_hz)},
        },
        "frames": frames,
    }


# Build one legacy cached policy-rollout replay payload from existing eval outputs.
def _build_policy_rollout_payload(args: ExportOfflineReplayArgs) -> dict[str, Any]:
    """Return one legacy offline replay payload reconstructed from eval telemetry history."""
    if args.telemetry_history_path is None:
        raise ValueError("telemetry_history_path is required when source_kind=policy_rollout.")
    telemetry_history = _load_telemetry_history(args.telemetry_history_path)
    eval_payload = _load_optional_json(args.eval_json_path)
    runtime_snapshot = _load_optional_json(args.runtime_snapshot_path)
    snapshot_meta = runtime_snapshot.get("meta", {}) if isinstance(runtime_snapshot, dict) else {}
    object_name = str(snapshot_meta.get("object_name", args.object_name))
    task_name = str(snapshot_meta.get("task_name", args.task_name))
    control_hz = float(
        runtime_snapshot.get("control_hz", args.control_hz)
        if isinstance(runtime_snapshot, dict)
        else args.control_hz
    )
    frames = []
    for sample in telemetry_history:
        object_pose = sample.get("object_pose")
        goal_pose = sample.get("goal_pose")
        if not isinstance(object_pose, list) or len(object_pose) < 7:
            raise ValueError("telemetry history samples must contain object_pose with 7 values.")
        if not isinstance(goal_pose, list) or len(goal_pose) < 7:
            raise ValueError("telemetry history samples must contain goal_pose with 7 values.")
        frames.append(
            {
                "tool_pose": [float(value) for value in object_pose[:7]],
                "goal_pose": [float(value) for value in goal_pose[:7]],
                "goal_index": int(sample.get("success_count", 0)),
                "success_count": int(sample.get("success_count", 0)),
                "sim_time_sec": float(sample.get("sim_time_sec", 0.0)),
                "event": str(sample.get("status", "running")),
                "robot_joint_positions": (
                    [float(value) for value in sample.get("robot_joint_positions", [])]
                    if sample.get("robot_joint_positions") is not None
                    else None
                ),
            }
        )
    summary = {
        "episode_goal_pct": float(
            max(eval_payload.get("episode_goal_pcts", [eval_payload.get("avg_goal_pct", 0.0)]))
        ),
        "episode_length_steps": len(frames),
        "eval_success_tolerance": float(
            eval_payload.get("eval_success_tolerance", DEFAULT_EVAL_SUCCESS_TOLERANCE_M)
        ),
        "artifact_note": (
            "Offline replay artifact reconstructed from eval telemetry history, eval.json, "
            "and runtime_snapshot.json."
        ),
    }
    if "avg_time_sec" in eval_payload:
        summary["avg_time_sec"] = float(eval_payload["avg_time_sec"])
    return {
        "schema_version": "offline_replay_v1",
        "object_name": object_name,
        "task_name": task_name,
        "source": "cached_policy_rollout",
        "control_hz": control_hz,
        "table_urdf": (
            str(runtime_snapshot.get("table_urdf", TABLE_URDF))
            if isinstance(runtime_snapshot, dict)
            else TABLE_URDF
        ),
        "summary": summary,
        "runtime_snapshot": runtime_snapshot if isinstance(runtime_snapshot, dict) else {},
        "frames": frames,
    }


# Build one goal-source eval arg bundle for policy replay export.
def _build_policy_goal_source_eval_args(args: ExportOfflineReplayArgs) -> EvalGoalSourcesArgs:
    """Return policy goal-source eval args with the requested replay timeout override."""
    return EvalGoalSourcesArgs(
        goal_source="all",
        execution_backend="policy",
        object_name=args.object_name,
        task_name=args.task_name,
        llm_backend=args.llm_backend,
        enable_viser=False,
        control_hz=args.control_hz,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        policy_name=args.policy_name,
        force_table_urdf=args.force_table_urdf,
        use_task_env_urdf=args.use_task_env_urdf,
        z_offset=args.z_offset,
        interactive_autorun=False,
        exit_after_episodes=0,
        eval_success_tolerance=args.eval_success_tolerance,
        reset_time=_resolve_policy_replay_duration_sec(args.policy_replay_duration_sec),
    )


# Build one source entry for the unified multi-source offline replay payload.
def _build_replay_source_entry(captured_payload: dict, *, control_hz: float) -> dict[str, Any]:
    """Return one multi-source replay entry from a captured goal-source policy rollout."""
    artifact = captured_payload["artifact"]
    reference_goals = [list(goal) for goal in artifact.goals]
    reference_duration_sec = float(len(reference_goals) / max(control_hz, 1e-6))
    return {
        "object_name": str(captured_payload["object_name"]),
        "task_name": str(captured_payload["task_name"]),
        "reference_track": {
            "tool_poses": reference_goals,
            "duration_sec": reference_duration_sec,
            "sample_interval_sec": (
                reference_duration_sec / max(len(reference_goals) - 1, 1)
                if len(reference_goals) > 1
                else reference_duration_sec
            ),
        },
        "policy_track": {
            "frames": captured_payload["frames"],
            "summary": dict(captured_payload["summary"]),
        },
        "metrics": dict(artifact.metrics),
        "metadata": dict(artifact.metadata),
    }


# Capture one predefined policy replay source for a concrete object/task pair.
def _capture_predefined_policy_source(
    *,
    export_args: ExportOfflineReplayArgs,
    mode: str,
    object_name: str,
    task_name: str,
    runner: _MinimalPolicyReplayRunner,
) -> dict[str, Any]:
    """Return one captured predefined replay source using one reusable policy runner."""
    capture_start = time.monotonic()
    log_info(
        f"[export] Starting replay capture for `{mode}` with "
        f"{export_args.policy_replay_duration_sec:.1f}s hard cap."
    )
    captured_payload = runner.capture_loaded_source(
        mode=mode,
        object_name=object_name,
        task_name=task_name,
    )
    capture_elapsed = time.monotonic() - capture_start
    captured_payload["mode"] = mode
    captured_payload["object_name"] = object_name
    captured_payload["task_name"] = task_name
    log_info(
        f"[export] Finished `{mode}` in {capture_elapsed:.1f}s wall time; "
        f"{len(captured_payload['frames'])} frames, "
        f"{captured_payload['summary']['avg_time_sec']:.2f}s sim time."
    )
    return captured_payload


# Build one predefined-only eval args bundle for one concrete object/task pair.
def _build_predefined_goal_args(
    export_args: ExportOfflineReplayArgs,
    *,
    object_name: str,
    task_name: str,
) -> EvalGoalSourcesArgs:
    """Return predefined-only eval args for one object/task pair."""
    return replace(
        _build_policy_goal_source_eval_args(export_args),
        goal_source="predefined",
        object_name=object_name,
        task_name=task_name,
    )


# Load the single predefined artifact expected for one object/task pair.
def _load_predefined_artifact(goal_args: EvalGoalSourcesArgs):
    """Return the predefined artifact for one object/task pair."""
    artifacts = [
        artifact
        for artifact in build_goal_source_artifacts(goal_args)
        if artifact.mode == "predefined"
    ]
    if len(artifacts) != 1:
        raise RuntimeError(
            f"Expected exactly one predefined artifact for {goal_args.object_name}/{goal_args.task_name}, "
            f"got {len(artifacts)}."
        )
    return artifacts[0]


# Return the static predefined replay-source tuple for one requested mode.
def _predefined_source_spec(mode: str) -> tuple[str, str, str]:
    """Return the predefined replay source tuple for one mode name."""
    for source_mode, object_name, task_name in _PREDEFINED_REPLAY_SOURCES:
        if source_mode == mode:
            return source_mode, object_name, task_name
    raise ValueError(f"Unsupported predefined replay mode: {mode}")


# Capture one predefined replay source inside the current Python process.
def _capture_single_predefined_source_payload(
    export_args: ExportOfflineReplayArgs,
) -> tuple[dict[str, Any], dict[str, Any], float, str]:
    """Return one single-source replay entry plus runtime metadata for one predefined mode."""
    if export_args.single_predefined_mode is None:
        raise ValueError("single_predefined_mode is required for single-source capture.")
    mode, object_name, task_name = _predefined_source_spec(export_args.single_predefined_mode)
    _sanitize_isaac_python_path()
    _install_torchvision_stub()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    runner = _MinimalPolicyReplayRunner(export_args, app_launcher=app_launcher)
    try:
        log_info(f"[export] Initializing minimal runner for `{mode}` ({object_name}/{task_name}).")
        _, _, table_urdf = runner.load_source(
            object_name=object_name,
            task_name=task_name,
        )
        runtime_snapshot = build_runtime_snapshot_payload(
            runner.env,
            control_hz=runner.control_hz,
            control_dt=runner.control_dt,
        )
        captured_payload = _capture_predefined_policy_source(
            export_args=export_args,
            mode=mode,
            object_name=object_name,
            task_name=task_name,
            runner=runner,
        )
        source_entry = _build_replay_source_entry(
            captured_payload,
            control_hz=runner.control_hz,
        )
        return source_entry, runtime_snapshot, float(runner.control_hz), table_urdf
    finally:
        runner.close()
        close_simulation_app_with_timeout(
            simulation_app,
            timeout_sec=5.0,
            log_warn_fn=log_warn,
            force_exit_fn=lambda code: None,
        )


# Execute one isolated predefined-source export inside one spawned worker process.
def _single_source_export_worker(
    export_args: ExportOfflineReplayArgs,
    mode: str,
    output_path: Path,
) -> None:
    """Write one single-source predefined replay payload to the requested JSON path."""
    payload = build_offline_replay_payload(
        replace(
            export_args,
            output=output_path,
            single_predefined_mode=mode,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(to_json_compatible(payload), indent=2) + "\n")


# Capture the predefined-only replay sources through isolated spawned worker processes.
def _capture_predefined_policy_sources(
    *,
    export_args: ExportOfflineReplayArgs,
) -> tuple[dict[str, Any], dict[str, Any], float, str]:
    """Return replay sources, runtime snapshot, control rate, and last table URDF."""
    sources: dict[str, Any] = {}
    runtime_snapshot = None
    control_hz = float(export_args.control_hz)
    table_urdf = TABLE_URDF
    with tempfile.TemporaryDirectory(prefix="predefined_replay_export_") as temp_dir:
        temp_root = Path(temp_dir)
        spawn_context = multiprocessing.get_context("spawn")
        for mode, _, _ in _PREDEFINED_REPLAY_SOURCES:
            source_start = time.monotonic()
            source_path = temp_root / f"{mode}.json"
            worker = spawn_context.Process(
                target=_single_source_export_worker,
                args=(
                    export_args,
                    mode,
                    source_path,
                ),
            )
            log_info(f"[export] Launching isolated export worker for `{mode}`.")
            worker.start()
            worker.join()
            if worker.exitcode != 0:
                raise RuntimeError(
                    f"Isolated predefined replay worker for `{mode}` failed with exit code "
                    f"{worker.exitcode}."
                )
            if not source_path.exists():
                raise RuntimeError(
                    f"Isolated predefined replay worker for `{mode}` exited cleanly without "
                    f"writing {source_path}."
                )
            payload = json.loads(source_path.read_text())
            sources[mode] = payload["source_entry"]
            if runtime_snapshot is None:
                runtime_snapshot = payload["runtime_snapshot"]
            control_hz = float(payload["control_hz"])
            table_urdf = str(payload["table_urdf"])
            log_info(
                f"[export] Source `{mode}` total wall time: {time.monotonic() - source_start:.1f}s."
            )
    if runtime_snapshot is None:
        raise RuntimeError("Expected at least one predefined replay source to be captured.")
    return sources, runtime_snapshot, control_hz, table_urdf


# Export one combined predefined-only replay artifact directly from policy rollouts.
def _build_policy_goal_sources_payload(args: ExportOfflineReplayArgs) -> dict[str, Any]:
    """Return one unified offline replay payload for the predefined hammer and screwdriver motions."""
    export_start = time.monotonic()

    mode_order = [mode for mode, _, _ in _PREDEFINED_REPLAY_SOURCES]
    sources, runtime_snapshot, control_hz, table_urdf = _capture_predefined_policy_sources(
        export_args=args
    )
    log_info(
        f"[export] Completed predefined-only replay export in "
        f"{time.monotonic() - export_start:.1f}s."
    )
    return {
        "schema_version": "offline_replay_v3",
        "object_name": "",
        "task_name": "",
        "control_hz": control_hz,
        "table_urdf": str(table_urdf),
        "summary": {
            "artifact_note": (
                "Unified offline replay artifact exported from policy rollouts for the "
                "predefined hammer swing and predefined screwdriver twist motions."
            ),
            "exported_sources": mode_order,
            "policy_replay_duration_sec": float(args.policy_replay_duration_sec),
        },
        "runtime_snapshot": runtime_snapshot,
        "mode_order": mode_order,
        "sources": sources,
    }


# Close one deferred Isaac simulation app after the export payload has been written to disk.
def _close_pending_simulation_app() -> None:
    """Close any deferred Isaac simulation app created for policy replay export."""
    global _PENDING_SIMULATION_APP
    if _PENDING_SIMULATION_APP is None:
        return
    close_simulation_app_with_timeout(
        _PENDING_SIMULATION_APP,
        timeout_sec=5.0,
        log_warn_fn=log_warn,
        force_exit_fn=lambda code: None,
    )
    _PENDING_SIMULATION_APP = None


# Build one offline replay payload from the requested laptop export source.
def build_offline_replay_payload(args: ExportOfflineReplayArgs) -> dict:
    """Return one cached replay artifact payload for the selected export mode."""
    if args.single_predefined_mode is not None:
        source_entry, runtime_snapshot, control_hz, table_urdf = (
            _capture_single_predefined_source_payload(args)
        )
        return {
            "single_predefined_mode": args.single_predefined_mode,
            "source_entry": source_entry,
            "runtime_snapshot": runtime_snapshot,
            "control_hz": control_hz,
            "table_urdf": str(table_urdf),
        }
    if args.source_kind == "predefined":
        return _build_predefined_replay_payload(args)
    if args.source_kind == "policy_rollout":
        return _build_policy_rollout_payload(args)
    if args.source_kind == "policy_goal_sources":
        return _build_policy_goal_sources_payload(args)
    raise ValueError(f"Unsupported source_kind: {args.source_kind}")


# Execute the offline replay export command and persist the JSON artifact.
def main() -> None:
    """Entry point for offline replay artifact export."""
    args = tyro.cli(ExportOfflineReplayArgs)
    try:
        payload = build_offline_replay_payload(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(to_json_compatible(payload), indent=2) + "\n")
        log_info(f"Saved offline replay artifact: {args.output}")
    finally:
        _close_pending_simulation_app()


if __name__ == "__main__":
    main()
