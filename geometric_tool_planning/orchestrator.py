"""Shared goal-source orchestration for predefined, llm_lie, and llm_only trajectories."""

from __future__ import annotations

import json
import math
from typing import Dict, List, Sequence

from dextoolbench.predefined_baselines import resolve_predefined_trajectory_path

from .llm import generate_llm_lie_specs, generate_llm_only_payload
from .orchestrator_types import EvalGoalSourcesArgs, GoalSourceArtifact
from .swing import compile_llm_lie_goals, validate_llm_lie_spec
from .trajectory import path_length_m, resample_goals, rotation_distance_deg, validate_pose_sequence
from .viewer import ToolTrajectoryViewer

DEFAULT_CONTROL_HZ = 60.0
SUPPORTED_GOAL_SOURCES = ("predefined", "llm_lie", "llm_only", "all")


# Validate the requested goal-source selector and normalize it.
def validate_goal_source(goal_source: str) -> str:
    """Return the selected goal source when it is supported."""
    if goal_source not in SUPPORTED_GOAL_SOURCES:
        raise ValueError(
            f"Unsupported goal source '{goal_source}'. Supported: {SUPPORTED_GOAL_SOURCES}."
        )
    return goal_source


# Expand a goal-source selector into the concrete modes to run.
def goal_source_modes(goal_source: str) -> List[str]:
    """Return a concrete list of goal-source modes for one comparison run."""
    selected = validate_goal_source(goal_source)
    if selected == "all":
        return ["predefined", "llm_lie", "llm_only"]
    return [selected]


# Load the stored DexToolBench trajectory payload for the requested object and task.
def load_predefined_trajectory_payload(object_name: str, task_name: str) -> Dict[str, object]:
    """Return the stored DexToolBench trajectory payload for one object/task pair."""
    trajectory_path = resolve_predefined_trajectory_path(object_name, task_name)
    with open(trajectory_path) as file_obj:
        return json.load(file_obj)


# Load the stored DexToolBench kinematic path with start pose prepended.
def load_predefined_goals(object_name: str, task_name: str) -> List[List[float]]:
    """Return the stored start pose followed by goal poses for one object/task trajectory."""
    payload = load_predefined_trajectory_payload(object_name, task_name)
    return validate_pose_sequence([payload["start_pose"], *payload["goals"]])


# Convert duration and sample count into the nominal time step for playback.
def sample_interval_sec(duration_sec: float, num_samples: int) -> float:
    """Return the nominal playback spacing between adjacent samples."""
    if duration_sec <= 0.0:
        raise ValueError("duration_sec must be positive.")
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2.")
    return duration_sec / float(num_samples - 1)


# Build the recorded path artifact used as the reference for all comparisons.
def build_reference_path(
    object_name: str,
    task_name: str,
    control_hz: float = DEFAULT_CONTROL_HZ,
) -> GoalSourceArtifact:
    """Return the predefined reference path and derived timing metadata."""
    payload = load_predefined_trajectory_payload(object_name, task_name)
    goals = validate_pose_sequence([payload["start_pose"], *payload["goals"]])
    duration_sec = len(goals) / float(control_hz)
    return GoalSourceArtifact(
        mode="predefined",
        goals=goals,
        duration_sec=duration_sec,
        sample_interval_sec=sample_interval_sec(duration_sec, len(goals)),
        metrics={},
        metadata={
            "object_name": object_name,
            "task_name": task_name,
            "start_pose": payload["start_pose"],
            "recorded_goals": payload["goals"],
        },
        llm_raw=None,
    )


# Compare a candidate path against the predefined reference path after progress resampling.
def compute_path_metrics(
    reference_goals: Sequence[Sequence[float]],
    candidate_goals: Sequence[Sequence[float]],
) -> Dict[str, float]:
    """Return translation and rotation deviation metrics between two pose paths."""
    count = max(len(reference_goals), len(candidate_goals))
    reference = resample_goals(reference_goals, count)
    candidate = resample_goals(candidate_goals, count)

    translation_errors: List[float] = []
    rotation_errors: List[float] = []
    for index in range(count):
        dx = reference[index][0] - candidate[index][0]
        dy = reference[index][1] - candidate[index][1]
        dz = reference[index][2] - candidate[index][2]
        translation_errors.append(math.sqrt(dx * dx + dy * dy + dz * dz))
        rotation_errors.append(rotation_distance_deg(reference[index][3:], candidate[index][3:]))

    reference_path_length = path_length_m(reference)
    candidate_path_length = path_length_m(candidate)
    return {
        "mean_translation_error_m": sum(translation_errors) / float(len(translation_errors)),
        "max_translation_error_m": max(translation_errors),
        "mean_rotation_error_deg": sum(rotation_errors) / float(len(rotation_errors)),
        "max_rotation_error_deg": max(rotation_errors),
        "reference_path_length_m": reference_path_length,
        "candidate_path_length_m": candidate_path_length,
        "path_length_ratio": (
            candidate_path_length / reference_path_length if reference_path_length > 1e-8 else 1.0
        ),
        "sample_count": float(count),
    }


# Build the kinematics artifact for one selected goal-source mode.
def build_artifact_for_mode(
    mode: str,
    args: EvalGoalSourcesArgs,
    reference: GoalSourceArtifact,
) -> GoalSourceArtifact:
    """Return a generated artifact for predefined, llm_lie, or llm_only."""
    if mode == "predefined":
        return reference
    if mode == "llm_only":
        raw_payload = generate_llm_only_payload(args, reference)
        goals = validate_pose_sequence(raw_payload["goals"])
        if len(goals) != len(reference.goals):
            raise ValueError(
                "llm_only output must contain exactly "
                f"{len(reference.goals)} goals, got {len(goals)}."
            )
        duration_sec = float(raw_payload.get("duration_sec", len(goals) / float(args.control_hz)))
        return GoalSourceArtifact(
            mode=mode,
            goals=goals,
            duration_sec=duration_sec,
            sample_interval_sec=sample_interval_sec(duration_sec, len(goals)),
            metrics=compute_path_metrics(reference.goals, goals),
            metadata={"prompt_context": raw_payload.get("prompt_context", {})},
            llm_raw=raw_payload,
            execution_metrics={},
        )
    raise ValueError(f"Unsupported mode '{mode}'.")


# Build the full set of llm_lie target-conditioned artifacts for one comparison run.
def build_llm_lie_artifacts(
    args: EvalGoalSourcesArgs,
    reference: GoalSourceArtifact,
) -> List[GoalSourceArtifact]:
    """Return the target-conditioned llm_lie artifacts used for debugging and visualization."""
    artifacts: List[GoalSourceArtifact] = []
    raw_specs = generate_llm_lie_specs(args, reference)
    for variant_name, raw_spec in raw_specs.items():
        spec = validate_llm_lie_spec(raw_spec)
        goals = validate_pose_sequence(
            [
                reference.metadata["start_pose"],
                *compile_llm_lie_goals(spec, object_name=args.object_name),
            ]
        )
        duration_sec = float(spec["duration_sec"])
        artifacts.append(
            GoalSourceArtifact(
                mode=f"llm_lie[{variant_name}]",
                goals=goals,
                duration_sec=duration_sec,
                sample_interval_sec=sample_interval_sec(duration_sec, len(goals)),
                metrics=compute_path_metrics(reference.goals, goals),
                metadata={"spec": spec, "variant_name": variant_name},
                llm_raw=raw_spec,
                execution_metrics={},
            )
        )
    return artifacts


# Render a compact markdown summary for a list of compared artifacts.
def summary_markdown(artifacts: Sequence[GoalSourceArtifact]) -> str:
    """Return a short markdown summary for console or Viser display."""
    lines = ["# Goal Source Comparison"]
    for artifact in artifacts:
        headline = f"- `{artifact.mode}`: duration_sec={artifact.duration_sec:.3f}"
        if artifact.metrics:
            if "mean_translation_error_m" in artifact.metrics:
                headline += (
                    f", mean_translation_error_m="
                    f"{float(artifact.metrics['mean_translation_error_m']):.4f}"
                )
            if "mean_rotation_error_deg" in artifact.metrics:
                headline += (
                    f", mean_rotation_error_deg="
                    f"{float(artifact.metrics['mean_rotation_error_deg']):.2f}"
                )
        if artifact.execution_metrics:
            if "latest_goal_pct" in artifact.execution_metrics:
                headline += (
                    f", latest_goal_pct="
                    f"{float(artifact.execution_metrics['latest_goal_pct']):.1f}"
                )
            if "episodes" in artifact.execution_metrics:
                headline += f", episodes={int(artifact.execution_metrics['episodes'])}"
        lines.append(headline)
    return "\n".join(lines)


# Launch an optional live Viser tool viewer for compared trajectories.
def maybe_visualize(
    object_name: str,
    artifacts: Sequence[GoalSourceArtifact],
) -> None:
    """Render the actual tool mesh moving through one selected trajectory source."""
    viewer = ToolTrajectoryViewer(object_name=object_name, artifacts=artifacts)
    viewer.run_forever()


# Build the in-memory comparison artifacts for the selected goal sources.
def build_goal_source_artifacts(args: EvalGoalSourcesArgs) -> List[GoalSourceArtifact]:
    """Generate compared goal-source artifacts without persisting run directories."""
    reference = build_reference_path(args.object_name, args.task_name, control_hz=args.control_hz)
    artifacts = [reference]
    for mode in goal_source_modes(args.goal_source):
        if mode == "predefined":
            continue
        if mode == "llm_lie":
            artifacts.extend(build_llm_lie_artifacts(args, reference))
            continue
        artifacts.append(build_artifact_for_mode(mode, args, reference))
    return artifacts


# Run the end-to-end in-memory kinematics comparison and optionally visualize it.
def run_goal_source_comparison(args: EvalGoalSourcesArgs) -> List[GoalSourceArtifact]:
    """Generate and optionally visualize compared goal-source artifacts."""
    artifacts = build_goal_source_artifacts(args)
    if args.enable_viser:
        maybe_visualize(args.object_name, artifacts)
    return artifacts
