"""Offline replay artifact helpers for laptop-friendly cached rollout viewing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from geometric_tool_planning import GoalSourceArtifact, sample_interval_sec


@dataclass
class OfflineReplayFrame:
    """One cached replay frame with tool pose and rollout progress metadata."""

    tool_pose: List[float]
    goal_pose: List[float]
    goal_index: int
    success_count: int
    sim_time_sec: float
    event: Optional[str] = None
    robot_joint_positions: Optional[List[float]] = None


@dataclass
class OfflineReplaySource:
    """One replayable source entry inside a multi-source offline replay artifact."""

    mode: str
    object_name: str
    task_name: str
    reference_tool_poses: List[List[float]]
    reference_duration_sec: float
    reference_sample_interval_sec: float
    policy_frames: List[OfflineReplayFrame]
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    summary: Dict[str, Any]


@dataclass
class OfflineReplayArtifact:
    """One CPU-friendly cached replay artifact for offline viewer playback."""

    schema_version: str
    object_name: str
    task_name: str
    control_hz: float
    table_urdf: str
    summary: Dict[str, Any]
    runtime_snapshot: Dict[str, Any]
    mode_order: List[str]
    sources: Dict[str, OfflineReplaySource]


# Validate one 7D pose list in `[x, y, z, qx, qy, qz, qw]` format.
def _validate_pose_list(values: Any, *, field_name: str) -> List[float]:
    """Return one validated 7D pose list of floats."""
    if not isinstance(values, list) or len(values) != 7:
        raise ValueError(f"{field_name} must be a 7-element list.")
    return [float(value) for value in values]


# Parse one replay frame dictionary into the typed offline replay dataclass.
def _parse_frame(payload: Dict[str, Any], *, index: int) -> OfflineReplayFrame:
    """Return one validated offline replay frame."""
    return OfflineReplayFrame(
        tool_pose=_validate_pose_list(
            payload.get("tool_pose"), field_name=f"frames[{index}].tool_pose"
        ),
        goal_pose=_validate_pose_list(
            payload.get("goal_pose"), field_name=f"frames[{index}].goal_pose"
        ),
        goal_index=int(payload.get("goal_index", 0)),
        success_count=int(payload.get("success_count", 0)),
        sim_time_sec=float(payload.get("sim_time_sec", 0.0)),
        event=str(payload["event"]) if payload.get("event") is not None else None,
        robot_joint_positions=(
            [float(value) for value in payload.get("robot_joint_positions", [])]
            if payload.get("robot_joint_positions") is not None
            else None
        ),
    )


# Parse one replay-source dictionary into the typed multi-source replay dataclass.
def _parse_source(payload: Dict[str, Any], *, mode: str) -> OfflineReplaySource:
    """Return one validated replay-source payload."""
    reference_track = dict(payload.get("reference_track", {}))
    reference_tool_poses_payload = reference_track.get("tool_poses")
    if not isinstance(reference_tool_poses_payload, list) or not reference_tool_poses_payload:
        raise ValueError(f"sources[{mode!r}].reference_track.tool_poses must be a non-empty list.")
    policy_track = dict(payload.get("policy_track", {}))
    policy_frames_payload = policy_track.get("frames")
    if not isinstance(policy_frames_payload, list) or not policy_frames_payload:
        raise ValueError(f"sources[{mode!r}].policy_track.frames must be a non-empty list.")
    return OfflineReplaySource(
        mode=str(mode),
        object_name=str(payload.get("object_name", "")),
        task_name=str(payload.get("task_name", "")),
        reference_tool_poses=[
            _validate_pose_list(
                tool_pose,
                field_name=f"sources[{mode!r}].reference_track.tool_poses[{index}]",
            )
            for index, tool_pose in enumerate(reference_tool_poses_payload)
        ],
        reference_duration_sec=float(reference_track.get("duration_sec", 0.0)),
        reference_sample_interval_sec=float(reference_track.get("sample_interval_sec", 0.0)),
        policy_frames=[
            _parse_frame(frame_payload, index=index)
            for index, frame_payload in enumerate(policy_frames_payload)
        ],
        metrics=dict(payload.get("metrics", {})),
        metadata=dict(payload.get("metadata", {})),
        summary=dict(policy_track.get("summary", payload.get("summary", {}))),
    )


# Build one single-source replay artifact view for backward compatibility with v1 artifacts.
def _load_legacy_offline_replay_artifact(payload: Dict[str, Any]) -> OfflineReplayArtifact:
    """Return one v2-style replay artifact converted from the legacy v1 schema."""
    frames_payload = payload.get("frames")
    if not isinstance(frames_payload, list) or not frames_payload:
        raise ValueError("Offline replay artifact must contain at least one frame.")
    source_mode = "predefined" if payload.get("source") == "cached_predefined_swing" else "policy"
    frames = [
        _parse_frame(frame_payload, index=index)
        for index, frame_payload in enumerate(frames_payload)
    ]
    reference_tool_poses = [list(frame.tool_pose) for frame in frames]
    duration_sec = float(
        len(reference_tool_poses) / max(float(payload.get("control_hz", 0.0)), 1e-6)
    )
    source = OfflineReplaySource(
        mode=source_mode,
        object_name=str(payload.get("object_name", "")),
        task_name=str(payload.get("task_name", "")),
        reference_tool_poses=reference_tool_poses,
        reference_duration_sec=duration_sec,
        reference_sample_interval_sec=sample_interval_sec(duration_sec, len(reference_tool_poses)),
        policy_frames=frames,
        metrics={},
        metadata={"legacy_source": str(payload.get("source", ""))},
        summary=dict(payload.get("summary", {})),
    )
    return OfflineReplayArtifact(
        schema_version="offline_replay_v2",
        object_name=str(payload.get("object_name", "")),
        task_name=str(payload.get("task_name", "")),
        control_hz=float(payload.get("control_hz", 0.0)),
        table_urdf=str(payload.get("table_urdf", "")),
        summary=dict(payload.get("summary", {})),
        runtime_snapshot=dict(payload.get("runtime_snapshot", {})),
        mode_order=[source_mode],
        sources={source_mode: source},
    )


# Load one offline replay artifact json file and validate its required fields.
def load_offline_replay_artifact(path: Path) -> OfflineReplayArtifact:
    """Return one parsed offline replay artifact from json."""
    payload = json.loads(path.read_text())
    schema_version = str(payload.get("schema_version", ""))
    if schema_version == "offline_replay_v1":
        return _load_legacy_offline_replay_artifact(payload)
    sources_payload = payload.get("sources")
    if not isinstance(sources_payload, dict) or not sources_payload:
        raise ValueError("Offline replay artifact must contain at least one replay source.")
    mode_order_payload = payload.get("mode_order")
    if not isinstance(mode_order_payload, list) or not mode_order_payload:
        mode_order = list(sources_payload.keys())
    else:
        mode_order = [str(mode) for mode in mode_order_payload]
    sources = {
        str(mode): _parse_source(source_payload, mode=str(mode))
        for mode, source_payload in sources_payload.items()
    }
    filtered_mode_order = [mode for mode in mode_order if mode in sources]
    if not filtered_mode_order:
        filtered_mode_order = list(sources.keys())
    artifact = OfflineReplayArtifact(
        schema_version=schema_version,
        object_name=str(payload.get("object_name", "")),
        task_name=str(payload.get("task_name", "")),
        control_hz=float(payload.get("control_hz", 0.0)),
        table_urdf=str(payload.get("table_urdf", "")),
        summary=dict(payload.get("summary", {})),
        runtime_snapshot=dict(payload.get("runtime_snapshot", {})),
        mode_order=filtered_mode_order,
        sources=sources,
    )
    for source in artifact.sources.values():
        if not source.object_name:
            source.object_name = artifact.object_name
        if not source.task_name:
            source.task_name = artifact.task_name
    return artifact


# Resolve the replay duration used for recorded-speed viewer playback.
def _policy_duration_sec(source: OfflineReplaySource, *, control_hz: float) -> float:
    """Return recorded policy duration from frame sim time when available."""
    if len(source.policy_frames) >= 2:
        start_time_sec = float(source.policy_frames[0].sim_time_sec)
        end_time_sec = float(source.policy_frames[-1].sim_time_sec)
        recorded_duration_sec = end_time_sec - start_time_sec
        if recorded_duration_sec > 0.0:
            return recorded_duration_sec
    return float(len(source.policy_frames) / max(control_hz, 1e-6))


# Convert one replay source's policy track into the shared viewer artifact type.
def replay_source_to_goal_source_artifact(
    artifact: OfflineReplayArtifact,
    *,
    mode: str,
) -> GoalSourceArtifact:
    """Return one GoalSourceArtifact for the selected replay source's policy track."""
    source = artifact.sources[mode]
    goals = [list(frame.tool_pose) for frame in source.policy_frames]
    duration_sec = _policy_duration_sec(source, control_hz=artifact.control_hz)
    goal_poses_by_frame = [
        [float(value) for value in frame.goal_pose] for frame in source.policy_frames
    ]
    robot_joint_positions_by_frame = [
        (
            [float(value) for value in frame.robot_joint_positions]
            if frame.robot_joint_positions
            else None
        )
        for frame in source.policy_frames
    ]
    return GoalSourceArtifact(
        mode=mode,
        goals=goals,
        duration_sec=duration_sec,
        sample_interval_sec=sample_interval_sec(duration_sec, len(goals)),
        metrics=dict(source.metrics),
        metadata={
            "source": mode,
            "summary": dict(source.summary),
            "runtime_snapshot": dict(artifact.runtime_snapshot),
            "table_urdf": artifact.table_urdf,
            "metadata": dict(source.metadata),
            "object_name": source.object_name or artifact.object_name,
            "task_name": source.task_name or artifact.task_name,
            "goal_poses_by_frame": goal_poses_by_frame,
            "robot_joint_positions_by_frame": robot_joint_positions_by_frame,
        },
        llm_raw=None,
        execution_metrics={},
    )


# Return the selected replay source entry from one multi-source artifact.
def replay_source(artifact: OfflineReplayArtifact, mode: str) -> OfflineReplaySource:
    """Return one replay source by mode or raise when it is unavailable."""
    if mode not in artifact.sources:
        raise KeyError(f"Replay source '{mode}' is unavailable.")
    return artifact.sources[mode]


# Build one compact markdown summary for the current replay frame and source metadata.
def replay_frame_markdown(artifact: OfflineReplayArtifact, mode: str, frame_index: int) -> str:
    """Return markdown summary for one active replay source and frame."""
    source = replay_source(artifact, mode)
    clamped_index = max(0, min(frame_index, len(source.policy_frames) - 1))
    frame = source.policy_frames[clamped_index]
    lines = [
        f"**Replay Source:** `{mode}`",
        (
            f"- object/task: `{source.object_name or artifact.object_name}` / "
            f"`{source.task_name or artifact.task_name}`"
        ),
        f"- frame: `{clamped_index}` / `{len(source.policy_frames) - 1}`",
        f"- goal_index: `{frame.goal_index}`",
        f"- success_count: `{frame.success_count}`",
        f"- sim_time_sec: `{frame.sim_time_sec:.3f}`",
        f"- reference_samples: `{len(source.reference_tool_poses)}`",
    ]
    if frame.robot_joint_positions is not None:
        lines.append(f"- robot_joint_count: `{len(frame.robot_joint_positions)}`")
    if frame.event:
        lines.append(f"- event: `{frame.event}`")
    if source.summary:
        for key in ("episode_goal_pct", "episode_length_steps", "eval_success_tolerance"):
            if key in source.summary:
                lines.append(f"- {key}: `{source.summary[key]}`")
    return "\n".join(lines)
