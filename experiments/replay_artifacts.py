"""Build laptop-runnable replay artifacts from experiment benchmark outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dextoolbench.eval_config import TABLE_URDF
from experiments.common import write_json
from geometric_tool_planning import sample_interval_sec

REPLAY_SCHEMA_VERSION = "offline_replay_v4"


# Return one 7D pose list in the replay schema's float format.
def _pose(values: Sequence[float], *, field_name: str) -> List[float]:
    """Validate and normalize one 7D pose list."""
    if len(values) != 7:
        raise ValueError(f"{field_name} must contain exactly 7 pose values.")
    return [float(value) for value in values]


# Return control rate from runtime metadata, falling back to the benchmark default.
def _control_hz(runtime_snapshot: Optional[Dict[str, Any]]) -> float:
    """Return the replay control frequency from runtime metadata."""
    if isinstance(runtime_snapshot, dict):
        control = runtime_snapshot.get("control", {})
        if isinstance(control, dict) and control.get("control_hz") is not None:
            return float(control["control_hz"])
        if runtime_snapshot.get("control_hz") is not None:
            return float(runtime_snapshot["control_hz"])
    return 60.0


# Return a replay sample interval while accepting single-frame debug artifacts.
def _sample_interval(duration_sec: float, num_samples: int) -> float:
    """Return nominal sample spacing with a single-frame fallback."""
    if num_samples < 2:
        return float(duration_sec)
    return sample_interval_sec(duration_sec, num_samples)


# Return one source label that is stable and readable inside the replay dropdown.
def _execution_source_label(execution_payload: Dict[str, Any]) -> str:
    """Return one replay source label for an execution result."""
    benchmark_cell = execution_payload.get("benchmark_cell")
    if benchmark_cell:
        return str(benchmark_cell)
    return f"{execution_payload.get('mode', '')} + {execution_payload.get('policy_variant', '')}"


# Build manifest entry metadata common to geometry and execution replay browsers.
def _base_manifest_entry(
    *,
    trial_id: str,
    object_name: str,
    task_name: str,
    mode: str,
    replay_artifact_path: Path,
    experiment_dir: Path,
    target_xy: Optional[Sequence[float]],
    resolved_baseline_object: Optional[str],
) -> Dict[str, Any]:
    """Return one manifest entry with fields used by the replay UI."""
    return {
        "trial_id": str(trial_id),
        "object_name": str(object_name),
        "task_name": str(task_name),
        "mode": str(mode),
        "target_xy": ([float(value) for value in target_xy] if target_xy is not None else None),
        "resolved_baseline_object": resolved_baseline_object,
        "replay_artifact_path": str(replay_artifact_path.relative_to(experiment_dir)),
    }


# Build one laptop replay artifact for a frozen geometry trajectory.
def build_geometry_replay_payload(geometry_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a robotless replay payload for one commanded geometry trajectory."""
    goals = [
        _pose(goal, field_name=f"goals[{index}]")
        for index, goal in enumerate(geometry_payload.get("goals", []))
    ]
    if not goals:
        raise ValueError("Geometry replay requires at least one saved goal pose.")
    control_hz = 60.0
    duration_sec = float(len(goals) / control_hz)
    mode = str(geometry_payload["mode"])
    frames = [
        {
            "tool_pose": list(goal),
            "goal_pose": list(goal),
            "goal_index": int(index),
            "success_count": int(index),
            "sim_time_sec": float(index / control_hz),
            "event": "geometry_commanded",
            "robot_joint_positions": None,
        }
        for index, goal in enumerate(goals)
    ]
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "object_name": str(geometry_payload["object_name"]),
        "task_name": str(geometry_payload["task_name"]),
        "control_hz": control_hz,
        "table_urdf": TABLE_URDF,
        "summary": {
            "replay_kind": "geometry_commanded",
            "geometry_trial_id": str(geometry_payload["trial_id"]),
            "num_frames": len(frames),
        },
        "runtime_snapshot": {},
        "mode_order": [mode],
        "sources": {
            mode: {
                "object_name": str(geometry_payload["object_name"]),
                "task_name": str(geometry_payload["task_name"]),
                "reference_track": {
                    "tool_poses": goals,
                    "duration_sec": duration_sec,
                    "sample_interval_sec": _sample_interval(duration_sec, len(goals)),
                },
                "policy_track": {
                    "frames": frames,
                    "summary": {
                        "episode_goal_pct": 0.0,
                        "episode_length_steps": len(frames),
                        "replay_kind": "geometry_commanded",
                    },
                },
                "metrics": dict(geometry_payload.get("metrics", {})),
                "metadata": {
                    "replay_kind": "geometry_commanded",
                    "geometry_trial_id": str(geometry_payload["trial_id"]),
                    "target_xy": geometry_payload.get("target_xy"),
                    "resolved_baseline_object": geometry_payload.get("resolved_baseline_object"),
                    "generation_context": dict(geometry_payload.get("generation_context", {})),
                },
            }
        },
    }


# Build one laptop replay artifact for an execution rollout trace.
def build_execution_replay_payload(
    *,
    geometry_payload: Dict[str, Any],
    execution_payload: Dict[str, Any],
    trace_payload: Dict[str, Any],
    runtime_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a robot-capable replay payload for one executed policy rollout."""
    reference_goals = [
        _pose(goal, field_name=f"goals[{index}]")
        for index, goal in enumerate(geometry_payload.get("goals", []))
    ]
    if not reference_goals:
        raise ValueError("Execution replay requires at least one reference goal pose.")
    episodes = trace_payload.get("episodes", [])
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Execution replay requires at least one trace episode.")
    first_episode = episodes[0]
    if not isinstance(first_episode, list) or not first_episode:
        raise ValueError("Execution replay requires at least one trace sample.")
    control_hz = _control_hz(runtime_snapshot)
    frames = []
    for index, sample in enumerate(first_episode):
        if not isinstance(sample, dict):
            continue
        object_pose = sample.get("object_pose")
        goal_pose = sample.get("goal_pose")
        if not isinstance(object_pose, list) or not isinstance(goal_pose, list):
            continue
        frame = {
            "tool_pose": _pose(object_pose[:7], field_name=f"episodes[0][{index}].object_pose"),
            "goal_pose": _pose(goal_pose[:7], field_name=f"episodes[0][{index}].goal_pose"),
            "goal_index": int(sample.get("success_count", 0)),
            "success_count": int(sample.get("success_count", 0)),
            "sim_time_sec": float(
                sample.get("sim_time_sec", sample.get("step", index) / control_hz)
            ),
            "event": str(sample.get("status", "running")),
            "robot_joint_positions": (
                [float(value) for value in sample.get("robot_joint_positions", [])]
                if sample.get("robot_joint_positions") is not None
                else None
            ),
        }
        frames.append(frame)
    if not frames:
        raise ValueError("Execution replay trace did not contain any valid pose samples.")
    source_label = _execution_source_label(execution_payload)
    reference_duration_sec = float(len(reference_goals) / control_hz)
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "object_name": str(geometry_payload["object_name"]),
        "task_name": str(geometry_payload["task_name"]),
        "control_hz": control_hz,
        "table_urdf": (
            str(runtime_snapshot.get("table_urdf", TABLE_URDF))
            if isinstance(runtime_snapshot, dict)
            else TABLE_URDF
        ),
        "summary": {
            "replay_kind": "execution_actual",
            "geometry_trial_id": str(geometry_payload["trial_id"]),
            "execution_trial_id": str(execution_payload["trial_id"]),
            "policy_variant": str(execution_payload.get("policy_variant", "")),
            "num_frames": len(frames),
        },
        "runtime_snapshot": dict(runtime_snapshot or {}),
        "mode_order": [source_label],
        "sources": {
            source_label: {
                "object_name": str(geometry_payload["object_name"]),
                "task_name": str(geometry_payload["task_name"]),
                "reference_track": {
                    "tool_poses": reference_goals,
                    "duration_sec": reference_duration_sec,
                    "sample_interval_sec": _sample_interval(
                        reference_duration_sec, len(reference_goals)
                    ),
                },
                "policy_track": {
                    "frames": frames,
                    "summary": {
                        "episode_goal_pct": float(
                            execution_payload.get("goal_completion_pct", 0.0)
                        ),
                        "episode_length_steps": len(frames),
                        "failure_category": execution_payload.get("failure_category"),
                        "replay_kind": "execution_actual",
                    },
                },
                "metrics": dict(execution_payload.get("metrics", {})),
                "metadata": {
                    "replay_kind": "execution_actual",
                    "geometry_trial_id": str(geometry_payload["trial_id"]),
                    "execution_trial_id": str(execution_payload["trial_id"]),
                    "policy_variant": execution_payload.get("policy_variant"),
                    "target_xy": geometry_payload.get("target_xy"),
                    "resolved_baseline_object": geometry_payload.get("resolved_baseline_object"),
                    "robot_joint_names": trace_payload.get("robot_joint_names", []),
                    "geometry_goal_count": execution_payload.get("geometry_goal_count"),
                    "execution_goal_count": execution_payload.get("execution_goal_count"),
                    "reference_goal_count": execution_payload.get("reference_goal_count"),
                    "execution_goal_transform": execution_payload.get("execution_goal_transform"),
                    "execution_goal_cycle_style": execution_payload.get(
                        "execution_goal_cycle_style"
                    ),
                },
            }
        },
    }


# Persist one geometry replay artifact and return its manifest entry.
def write_geometry_replay_artifact(
    *,
    experiment_dir: Path,
    geometry_payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Write one geometry replay JSON and return its manifest entry when possible."""
    if not geometry_payload.get("compile_success") or not geometry_payload.get(
        "validation_success"
    ):
        return None
    replay_path = (
        experiment_dir / "geometry" / "replay" / "trials" / f"{geometry_payload['trial_id']}.json"
    )
    payload = build_geometry_replay_payload(geometry_payload)
    write_json(replay_path, payload)
    return {
        **_base_manifest_entry(
            trial_id=str(geometry_payload["trial_id"]),
            object_name=str(geometry_payload["object_name"]),
            task_name=str(geometry_payload["task_name"]),
            mode=str(geometry_payload["mode"]),
            replay_artifact_path=replay_path,
            experiment_dir=experiment_dir,
            target_xy=geometry_payload.get("target_xy"),
            resolved_baseline_object=geometry_payload.get("resolved_baseline_object"),
        ),
        "replay_kind": "geometry_commanded",
        "compile_success": bool(geometry_payload.get("compile_success")),
        "validation_success": bool(geometry_payload.get("validation_success")),
    }


# Persist one execution replay artifact and return its manifest entry.
def write_execution_replay_artifact(
    *,
    experiment_dir: Path,
    geometry_payload: Dict[str, Any],
    execution_payload: Dict[str, Any],
    trace_payload: Optional[Dict[str, Any]],
    runtime_snapshot: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Write one execution replay JSON and return its manifest entry when trace data exists."""
    if not isinstance(trace_payload, dict):
        return None
    replay_path = (
        experiment_dir / "execution" / "replay" / "trials" / f"{execution_payload['trial_id']}.json"
    )
    payload = build_execution_replay_payload(
        geometry_payload=geometry_payload,
        execution_payload=execution_payload,
        trace_payload=trace_payload,
        runtime_snapshot=runtime_snapshot,
    )
    write_json(replay_path, payload)
    return {
        **_base_manifest_entry(
            trial_id=str(execution_payload["trial_id"]),
            object_name=str(execution_payload["object_name"]),
            task_name=str(execution_payload["task_name"]),
            mode=str(execution_payload["mode"]),
            replay_artifact_path=replay_path,
            experiment_dir=experiment_dir,
            target_xy=geometry_payload.get("target_xy"),
            resolved_baseline_object=execution_payload.get("resolved_baseline_object"),
        ),
        "replay_kind": "execution_actual",
        "geometry_trial_id": str(execution_payload["geometry_trial_id"]),
        "policy_variant": str(execution_payload.get("policy_variant", "")),
        "benchmark_cell": str(execution_payload.get("benchmark_cell", "")),
        "execution_success": bool(execution_payload.get("execution_success", False)),
        "goal_completion_pct": float(execution_payload.get("goal_completion_pct", 0.0)),
        "failure_category": execution_payload.get("failure_category"),
    }


# Persist one replay manifest for a benchmark stage.
def write_replay_manifest(
    *,
    experiment_dir: Path,
    stage: str,
    entries: Sequence[Dict[str, Any]],
) -> Path:
    """Write one stage-level replay manifest and return its path."""
    manifest_path = experiment_dir / stage / "replay" / "manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": "experiment_replay_manifest_v1",
            "stage": stage,
            "num_trials": len(entries),
            "entries": list(entries),
        },
    )
    return manifest_path
