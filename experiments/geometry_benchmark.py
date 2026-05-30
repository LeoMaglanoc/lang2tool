"""Run the thesis geometry benchmark and persist frozen trajectory artifacts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import tyro

from dextoolbench.eval_config import (
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
    DEFAULT_Z_OFFSET_M,
    TABLE_Z,
)
from dextoolbench.llm_lie_trajectory import compile_llm_lie_trajectory
from dextoolbench.llm_supported_objects import supported_llm_object_names, supported_llm_task_name
from dextoolbench.object_start_poses import build_default_start_pose
from dextoolbench.predefined_baselines import resolve_predefined_baseline_object_name
from experiments.common import (
    build_target_grid_xy,
    default_experiment_name,
    ensure_experiment_dirs,
    ensure_experiment_metadata,
    parse_csv_tokens,
    read_json,
    save_trial_summaries,
    write_json,
)
from experiments.replay_artifacts import write_geometry_replay_artifact, write_replay_manifest
from experiments.result_schema import GeometryTrialResult, to_dict
from geometric_tool_planning import (
    EvalGoalSourcesArgs,
    GoalSourceArtifact,
    build_artifact_for_mode,
    build_reference_path,
    compute_path_metrics,
    get_predefined_goal_sequence,
    get_predefined_strike_target_xy,
    validate_pose_sequence,
)
from geometric_tool_planning.viewer import (
    semantic_strike_axis_table_intersection,
    target_error_frame_index,
)
from llm_runtime.semantic_pose import get_object_pose_semantics, quat_rotate_xyzw


# Configure one geometry benchmark run from the CLI.
@dataclass
class GeometryBenchmarkArgs:
    """CLI arguments for the thesis geometry benchmark."""

    results_dir: Path = Path("experiments/results")
    """Root directory under which experiment folders are created."""

    experiment_name: Optional[str] = None
    """Optional stable experiment name. If omitted, one timestamped name is generated."""

    object_names_csv: str = ",".join(supported_llm_object_names())
    """Comma-separated object names to benchmark."""

    modes_csv: str = "predefined,llm_lie,llm_only"
    """Comma-separated geometry modes to benchmark."""

    seed: int = 0
    """Seed recorded in saved artifacts for reproducibility metadata."""

    llm_backend: str = "openai"
    """Backend used by the direct llm_only geometry baseline."""

    llm_model: Optional[str] = None
    """Optional OpenAI model override for LLM-backed geometry generation."""

    llm_instruction: str = "Swing the hammer down toward the right."
    """Legacy non-canonical instruction for direct llm_only helpers outside the benchmark path."""

    target_grid_x: int = 3
    """Number of evenly spaced target X positions for Lie-generated trials."""

    target_grid_y: int = 3
    """Number of evenly spaced target Y positions for Lie-generated trials."""

    max_trials: Optional[int] = None
    """Optional hard cap on the number of generated trials."""

    z_offset: float = DEFAULT_Z_OFFSET_M
    """Safety Z offset used when building canonical object start poses."""

    llm_lie_horizontal_strike_clearance_m: float = DEFAULT_LLM_LIE_HORIZONTAL_STRIKE_CLEARANCE_M
    """Clearance above the table during horizontal Lie strike execution."""

    llm_lie_waypoint_table_clearance_m: float = DEFAULT_LLM_LIE_WAYPOINT_TABLE_CLEARANCE_M
    """Minimum waypoint clearance above the table for Lie generation."""

    llm_lie_screwdriver_twist_extra_hover_m: float = DEFAULT_LLM_LIE_SCREWDRIVER_TWIST_EXTRA_HOVER_M
    """Extra hover margin used when compiling screwdriver Lie twist trajectories."""

    llm_lie_training_resampling_enabled: bool = DEFAULT_LLM_LIE_TRAINING_RESAMPLING_ENABLED
    """Whether Lie trajectories are resampled toward RL-like step scales."""

    llm_lie_training_resampling_pos_scale_m: float = DEFAULT_LLM_LIE_TRAINING_RESAMPLING_POS_SCALE_M
    """Translation scale for training-distribution Lie resampling."""

    llm_lie_training_resampling_rot_scale_deg: float = (
        DEFAULT_LLM_LIE_TRAINING_RESAMPLING_ROT_SCALE_DEG
    )
    """Rotation scale for training-distribution Lie resampling."""

    llm_lie_training_resampling_target_cost: float = DEFAULT_LLM_LIE_TRAINING_RESAMPLING_TARGET_COST
    """Target combined progress per resampled Lie step."""

    llm_lie_training_resampling_min_waypoints: int = (
        DEFAULT_LLM_LIE_TRAINING_RESAMPLING_MIN_WAYPOINTS
    )
    """Minimum waypoint count retained after Lie resampling."""

    llm_lie_training_volume_clamp_enabled: bool = DEFAULT_LLM_LIE_TRAINING_VOLUME_CLAMP_ENABLED
    """Whether Lie waypoints are clamped into the RL training target volume."""

    llm_lie_training_target_volume_mins: List[float] = None
    """Inclusive XYZ lower bounds for the Lie training target volume."""

    llm_lie_training_target_volume_maxs: List[float] = None
    """Inclusive XYZ upper bounds for the Lie training target volume."""

    recompute_from_raw: bool = False
    """Whether to recompute derived summaries from frozen raw geometry artifacts only."""

    # Normalize default mutable list fields after dataclass construction.
    def __post_init__(self) -> None:
        """Populate mutable defaults for target volume bounds."""
        if self.llm_lie_training_target_volume_mins is None:
            self.llm_lie_training_target_volume_mins = list(
                DEFAULT_LLM_LIE_TRAINING_TARGET_VOLUME_MINS
            )
        if self.llm_lie_training_target_volume_maxs is None:
            self.llm_lie_training_target_volume_maxs = list(
                DEFAULT_LLM_LIE_TRAINING_TARGET_VOLUME_MAXS
            )


# Build one stable raw-trial identifier from the current object/mode/target tuple.
def _trial_id(
    *,
    object_name: str,
    mode: str,
    target_xy: Optional[Sequence[float]],
    index: int,
) -> str:
    """Return one human-readable stable trial identifier."""
    if target_xy is None:
        return f"{index:04d}_{object_name}_{mode}"
    return (
        f"{index:04d}_{object_name}_{mode}_"
        f"x{target_xy[0]:+.3f}_y{target_xy[1]:+.3f}".replace(".", "p")
    )


# Validate one saved geometry goal list using the shared benchmark criterion.
def _validate_saved_geometry_goals(goals: Sequence[Sequence[float]]) -> List[List[float]]:
    """Return one validated saved geometry pose list for any benchmark mode."""
    return validate_pose_sequence([list(goal) for goal in goals])


# Return the coarse tool family used by thesis-facing geometry summaries.
def _tool_family(object_name: str) -> str:
    """Return the semantic tool family for one benchmark object."""
    if "hammer" in str(object_name):
        return "hammer"
    if "screwdriver" in str(object_name):
        return "screwdriver"
    return "other"


# Return the selected semantic target-error frame for one saved goal sequence.
def _semantic_target_frame_index(
    object_name: str, goals: Sequence[Sequence[float]]
) -> Optional[int]:
    """Return the semantic target-error frame index used by geometry metrics."""
    if len(goals) == 0:
        return None
    artifact = GoalSourceArtifact(
        mode="semantic_metric",
        goals=[list(goal) for goal in goals],
        duration_sec=0.0,
        sample_interval_sec=0.0,
        metrics={},
    )
    return int(target_error_frame_index(artifact, object_name))


# Return the selected-frame blue-axis table-intersection XY error for one saved goal sequence.
def _semantic_contact_point_xy_error_m(
    object_name: str,
    goals: Sequence[Sequence[float]],
    target_xy: Optional[Sequence[float]],
) -> Optional[float]:
    """Return semantic-frame blue-axis table-intersection XY error when available."""
    if target_xy is None or len(goals) == 0:
        return None
    frame_index = _semantic_target_frame_index(object_name, goals)
    if frame_index is None:
        return None
    selected_goal = [float(value) for value in goals[frame_index]]
    implied_target_world = semantic_strike_axis_table_intersection(selected_goal, object_name)
    if implied_target_world is None:
        return None
    delta_x = float(implied_target_world[0]) - float(target_xy[0])
    delta_y = float(implied_target_world[1]) - float(target_xy[1])
    return float((delta_x * delta_x + delta_y * delta_y) ** 0.5)


# Return the worst primary-axis tilt from vertical for one screwdriver trajectory.
def _screwdriver_max_primary_axis_tilt_deg(
    object_name: str, goals: Sequence[Sequence[float]]
) -> Optional[float]:
    """Return max screwdriver tilt from vertical in degrees or None for non-screwdrivers."""
    if _tool_family(object_name) != "screwdriver" or len(goals) == 0:
        return None
    semantics = get_object_pose_semantics(object_name)
    tilt_values: List[float] = []
    for goal in goals:
        primary_axis_world = quat_rotate_xyzw(goal[3:7], semantics.primary_axis_local)
        vertical_alignment = max(-1.0, min(1.0, abs(float(primary_axis_world[2]))))
        tilt_values.append(float(math.degrees(math.acos(vertical_alignment))))
    return float(max(tilt_values)) if tilt_values else None


# Return the achieved face-normal twist span for one screwdriver trajectory.
def _screwdriver_twist_angle_span_deg(
    object_name: str, goals: Sequence[Sequence[float]]
) -> Optional[float]:
    """Return unwrapped screwdriver twist-angle span in degrees or None for non-screwdrivers."""
    if _tool_family(object_name) != "screwdriver" or len(goals) == 0:
        return None
    semantics = get_object_pose_semantics(object_name)
    azimuths_rad: List[float] = []
    for goal in goals:
        face_normal_world = quat_rotate_xyzw(goal[3:7], semantics.face_normal_local)
        azimuths_rad.append(float(math.atan2(face_normal_world[1], face_normal_world[0])))
    if not azimuths_rad:
        return None
    unwrapped = [azimuths_rad[0]]
    for value in azimuths_rad[1:]:
        candidate = float(value)
        while candidate - unwrapped[-1] > math.pi:
            candidate -= 2.0 * math.pi
        while candidate - unwrapped[-1] < -math.pi:
            candidate += 2.0 * math.pi
        unwrapped.append(candidate)
    span_rad = max(unwrapped) - min(unwrapped)
    return float(math.degrees(span_rad))


# Build semantic geometry metrics computed directly from the saved goal poses.
def _build_semantic_geometry_metrics(
    *,
    object_name: str,
    goals: Sequence[Sequence[float]],
    target_xy: Optional[Sequence[float]],
) -> Dict[str, Any]:
    """Return thesis-facing semantic geometry metrics for one saved trajectory."""
    contact_point_error_m = _semantic_contact_point_xy_error_m(object_name, goals, target_xy)
    semantic_frame_index = _semantic_target_frame_index(object_name, goals)
    max_tilt_deg = _screwdriver_max_primary_axis_tilt_deg(object_name, goals)
    twist_span_deg = _screwdriver_twist_angle_span_deg(object_name, goals)
    return {
        "tool_family": _tool_family(object_name),
        "semantic_contact_point_xy_error_m": contact_point_error_m,
        "semantic_target_frame_index": semantic_frame_index,
        "screwdriver_max_primary_axis_tilt_deg": max_tilt_deg,
        "screwdriver_twist_angle_span_deg": twist_span_deg,
    }


# Build one predefined-motion geometry artifact for the requested object/task pair.
def _run_predefined_trial(
    object_name: str, task_name: str, seed: int, trial_id: str
) -> GeometryTrialResult:
    """Return one saved-trajectory geometry result without Lie compilation."""
    goals = [
        list(goal)
        for goal in get_predefined_goal_sequence(
            object_name=object_name,
            task_name=task_name,
            include_start_pose=False,
        )
    ]
    validated_goals = _validate_saved_geometry_goals(goals)
    target_xy = get_predefined_strike_target_xy(object_name=object_name, task_name=task_name)
    semantic_metrics = _build_semantic_geometry_metrics(
        object_name=object_name,
        goals=validated_goals,
        target_xy=target_xy,
    )
    return GeometryTrialResult(
        trial_id=trial_id,
        object_name=object_name,
        task_name=task_name,
        mode="predefined",
        seed=seed,
        target_xy=[float(value) for value in target_xy] if target_xy is not None else None,
        resolved_baseline_object=resolve_predefined_baseline_object_name(object_name),
        pivot_point=None,
        compile_success=True,
        validation_success=True,
        num_waypoints=len(validated_goals),
        goals=[list(goal) for goal in validated_goals],
        metrics={
            "target_xy_available": target_xy is not None,
            "num_waypoints": len(validated_goals),
            **semantic_metrics,
        },
    )


# Build one Lie-generated geometry artifact for the requested object/task/target tuple.
def _run_llm_lie_trial(
    args: GeometryBenchmarkArgs,
    *,
    object_name: str,
    task_name: str,
    target_xy: Sequence[float],
    seed: int,
    trial_id: str,
) -> GeometryTrialResult:
    """Return one Lie-generated geometry result and embedded compile summaries."""
    pivot_point = build_default_start_pose(
        object_name,
        z_offset=float(args.z_offset),
        table_z=float(TABLE_Z),
    )[:3]
    clamped_target_xy, spec, compiled_goals, clamp_summary = compile_llm_lie_trajectory(
        object_name=object_name,
        task_name=task_name,
        pivot_point=pivot_point,
        strike_target_xy=target_xy,
        horizontal_strike_clearance_m=float(args.llm_lie_horizontal_strike_clearance_m),
        waypoint_table_clearance_m=float(args.llm_lie_waypoint_table_clearance_m),
        screwdriver_twist_extra_hover_m=float(args.llm_lie_screwdriver_twist_extra_hover_m),
        training_resampling_enabled=bool(args.llm_lie_training_resampling_enabled),
        training_resampling_pos_scale_m=float(args.llm_lie_training_resampling_pos_scale_m),
        training_resampling_rot_scale_deg=float(args.llm_lie_training_resampling_rot_scale_deg),
        training_resampling_target_cost=float(args.llm_lie_training_resampling_target_cost),
        training_resampling_min_waypoints=int(args.llm_lie_training_resampling_min_waypoints),
        training_volume_clamp_enabled=bool(args.llm_lie_training_volume_clamp_enabled),
        training_target_volume_mins=list(args.llm_lie_training_target_volume_mins),
        training_target_volume_maxs=list(args.llm_lie_training_target_volume_maxs),
    )
    validated_goals = _validate_saved_geometry_goals(compiled_goals)
    semantic_metrics = _build_semantic_geometry_metrics(
        object_name=object_name,
        goals=validated_goals,
        target_xy=clamped_target_xy,
    )
    return GeometryTrialResult(
        trial_id=trial_id,
        object_name=object_name,
        task_name=task_name,
        mode="llm_lie",
        seed=seed,
        target_xy=[float(value) for value in clamped_target_xy],
        resolved_baseline_object=resolve_predefined_baseline_object_name(object_name),
        pivot_point=[float(value) for value in pivot_point],
        compile_success=True,
        validation_success=True,
        num_waypoints=len(validated_goals),
        goals=[list(goal) for goal in validated_goals],
        generation_context={"spec": dict(spec)},
        clamp_summary=dict(clamp_summary),
        resampling_summary=dict(spec.get("training_distribution_resampling", {})),
        metrics={
            "requested_target_xy": [float(value) for value in target_xy],
            "hammer_swing_solver": spec.get("hammer_swing_solver"),
            "step_delta_summary": dict(spec.get("step_delta_summary", {})),
            **semantic_metrics,
        },
    )


# Build one direct-pose llm_only geometry artifact for the requested object/task pair.
def _run_llm_only_trial(
    args: GeometryBenchmarkArgs,
    *,
    object_name: str,
    task_name: str,
    target_xy: Sequence[float],
    seed: int,
    trial_id: str,
) -> GeometryTrialResult:
    """Return one direct-pose geometry result plus reference-relative path metrics."""
    reference = build_reference_path(object_name=object_name, task_name=task_name)
    pivot_point = build_default_start_pose(
        object_name,
        z_offset=float(args.z_offset),
        table_z=float(TABLE_Z),
    )[:3]
    resolved_baseline_object = resolve_predefined_baseline_object_name(object_name)
    artifact = build_artifact_for_mode(
        "llm_only",
        EvalGoalSourcesArgs(
            goal_source="llm_only",
            object_name=object_name,
            task_name=task_name,
            instruction="",
            target_xy=[float(target_xy[0]), float(target_xy[1])],
            resolved_baseline_object=resolved_baseline_object,
            pivot_point=[float(value) for value in pivot_point],
            llm_backend=str(args.llm_backend),
            llm_model=args.llm_model,
            enable_viser=False,
            seed=int(seed),
            z_offset=float(args.z_offset),
        ),
        reference,
    )
    saved_goals = _validate_saved_geometry_goals(artifact.goals[1:])
    metrics = compute_path_metrics(reference.goals, artifact.goals)
    semantic_metrics = _build_semantic_geometry_metrics(
        object_name=object_name,
        goals=saved_goals,
        target_xy=target_xy,
    )
    return GeometryTrialResult(
        trial_id=trial_id,
        object_name=object_name,
        task_name=task_name,
        mode="llm_only",
        seed=seed,
        target_xy=[float(target_xy[0]), float(target_xy[1])],
        resolved_baseline_object=resolved_baseline_object,
        pivot_point=[float(value) for value in pivot_point],
        compile_success=True,
        validation_success=True,
        num_waypoints=len(saved_goals),
        goals=[list(goal) for goal in saved_goals],
        generation_context=dict(artifact.metadata.get("prompt_context", {})),
        metrics={**metrics, **semantic_metrics},
    )


# Run the full geometry benchmark and save raw trials plus summary tables.
def run_geometry_benchmark(args: GeometryBenchmarkArgs) -> Path:
    """Execute the geometry benchmark and return the saved experiment root."""
    experiment_name = args.experiment_name or default_experiment_name("geometry_benchmark")
    experiment_dir = ensure_experiment_dirs(args.results_dir, experiment_name)
    ensure_experiment_metadata(experiment_dir, experiment_name)
    write_json(experiment_dir / "config.json", asdict(args))

    object_names = parse_csv_tokens(args.object_names_csv)
    modes = parse_csv_tokens(args.modes_csv)
    target_grid_xy = build_target_grid_xy(args.target_grid_x, args.target_grid_y)

    rows: List[Dict[str, Any]] = []
    replay_entries: List[Dict[str, Any]] = []
    trial_index = 0
    for object_name in object_names:
        task_name = supported_llm_task_name(object_name)
        for mode in modes:
            if args.max_trials is not None and trial_index >= int(args.max_trials):
                break
            if mode == "predefined":
                trial_id = _trial_id(
                    object_name=object_name,
                    mode=mode,
                    target_xy=None,
                    index=trial_index,
                )
                try:
                    result = _run_predefined_trial(object_name, task_name, args.seed, trial_id)
                except Exception as exc:
                    result = GeometryTrialResult(
                        trial_id=trial_id,
                        object_name=object_name,
                        task_name=task_name,
                        mode=mode,
                        seed=args.seed,
                        target_xy=None,
                        resolved_baseline_object=resolve_predefined_baseline_object_name(
                            object_name
                        ),
                        pivot_point=None,
                        compile_success=False,
                        validation_success=False,
                        num_waypoints=0,
                        error=str(exc),
                    )
                replay_entry = write_geometry_replay_artifact(
                    experiment_dir=experiment_dir,
                    geometry_payload=to_dict(result),
                )
                if replay_entry is not None:
                    result.replay_artifact_path = str(replay_entry["replay_artifact_path"])
                    replay_entries.append(replay_entry)
                write_json(
                    experiment_dir / "geometry" / "raw" / f"{trial_id}.json", to_dict(result)
                )
                rows.append(_flatten_geometry_result(result))
                trial_index += 1
                continue

            if mode == "llm_only":
                for target_xy in target_grid_xy:
                    if args.max_trials is not None and trial_index >= int(args.max_trials):
                        break
                    trial_id = _trial_id(
                        object_name=object_name,
                        mode=mode,
                        target_xy=target_xy,
                        index=trial_index,
                    )
                    try:
                        result = _run_llm_only_trial(
                            args,
                            object_name=object_name,
                            task_name=task_name,
                            target_xy=target_xy,
                            seed=args.seed,
                            trial_id=trial_id,
                        )
                    except Exception as exc:
                        result = GeometryTrialResult(
                            trial_id=trial_id,
                            object_name=object_name,
                            task_name=task_name,
                            mode=mode,
                            seed=args.seed,
                            target_xy=[float(target_xy[0]), float(target_xy[1])],
                            resolved_baseline_object=resolve_predefined_baseline_object_name(
                                object_name
                            ),
                            pivot_point=build_default_start_pose(
                                object_name,
                                z_offset=float(args.z_offset),
                                table_z=float(TABLE_Z),
                            )[:3],
                            compile_success=False,
                            validation_success=False,
                            num_waypoints=0,
                            error=str(exc),
                        )
                    replay_entry = write_geometry_replay_artifact(
                        experiment_dir=experiment_dir,
                        geometry_payload=to_dict(result),
                    )
                    if replay_entry is not None:
                        result.replay_artifact_path = str(replay_entry["replay_artifact_path"])
                        replay_entries.append(replay_entry)
                    write_json(
                        experiment_dir / "geometry" / "raw" / f"{trial_id}.json",
                        to_dict(result),
                    )
                    rows.append(_flatten_geometry_result(result))
                    trial_index += 1
                continue

            if mode != "llm_lie":
                raise ValueError(f"Unsupported geometry mode: {mode}")
            for target_xy in target_grid_xy:
                if args.max_trials is not None and trial_index >= int(args.max_trials):
                    break
                trial_id = _trial_id(
                    object_name=object_name,
                    mode=mode,
                    target_xy=target_xy,
                    index=trial_index,
                )
                try:
                    result = _run_llm_lie_trial(
                        args,
                        object_name=object_name,
                        task_name=task_name,
                        target_xy=target_xy,
                        seed=args.seed,
                        trial_id=trial_id,
                    )
                except Exception as exc:
                    result = GeometryTrialResult(
                        trial_id=trial_id,
                        object_name=object_name,
                        task_name=task_name,
                        mode=mode,
                        seed=args.seed,
                        target_xy=[float(target_xy[0]), float(target_xy[1])],
                        resolved_baseline_object=resolve_predefined_baseline_object_name(
                            object_name
                        ),
                        pivot_point=build_default_start_pose(
                            object_name,
                            z_offset=float(args.z_offset),
                            table_z=float(TABLE_Z),
                        )[:3],
                        compile_success=False,
                        validation_success=False,
                        num_waypoints=0,
                        error=str(exc),
                    )
                replay_entry = write_geometry_replay_artifact(
                    experiment_dir=experiment_dir,
                    geometry_payload=to_dict(result),
                )
                if replay_entry is not None:
                    result.replay_artifact_path = str(replay_entry["replay_artifact_path"])
                    replay_entries.append(replay_entry)
                write_json(
                    experiment_dir / "geometry" / "raw" / f"{trial_id}.json", to_dict(result)
                )
                rows.append(_flatten_geometry_result(result))
                trial_index += 1
        if args.max_trials is not None and trial_index >= int(args.max_trials):
            break

    aggregate_payload = {
        "objects": object_names,
        "modes": modes,
        "target_grid_xy": [[float(x_value), float(y_value)] for x_value, y_value in target_grid_xy],
        **_build_geometry_aggregate(rows),
    }
    save_trial_summaries(experiment_dir, "geometry", rows, aggregate_payload)
    write_replay_manifest(experiment_dir=experiment_dir, stage="geometry", entries=replay_entries)
    return experiment_dir


# Flatten one raw geometry trial so it can be stored in the summary CSV.
def _flatten_geometry_result(result: GeometryTrialResult) -> Dict[str, Any]:
    """Return one summary row for the geometry trials table."""
    target_x = None if result.target_xy is None else float(result.target_xy[0])
    target_y = None if result.target_xy is None else float(result.target_xy[1])
    corrected_indices = result.clamp_summary.get("corrected_indices", [])
    metrics = dict(result.metrics)
    generation_context = dict(result.generation_context)
    spec = generation_context.get("spec", {})
    hammer_swing_solver = (
        spec.get("hammer_swing_solver") if isinstance(spec, dict) else None
    ) or metrics.get("hammer_swing_solver")
    return {
        "trial_id": result.trial_id,
        "object_name": result.object_name,
        "tool_family": str(metrics.get("tool_family", _tool_family(result.object_name))),
        "task_name": result.task_name,
        "mode": result.mode,
        "seed": int(result.seed),
        "resolved_baseline_object": result.resolved_baseline_object,
        "pivot_x": None if result.pivot_point is None else float(result.pivot_point[0]),
        "pivot_y": None if result.pivot_point is None else float(result.pivot_point[1]),
        "pivot_z": None if result.pivot_point is None else float(result.pivot_point[2]),
        "target_x": target_x,
        "target_y": target_y,
        "target_conditioned": bool(
            result.mode == "llm_only" and isinstance(result.target_xy, list)
        ),
        "target_conditioning_schema": generation_context.get("schema_version"),
        "hammer_swing_solver": hammer_swing_solver,
        "compile_success": bool(result.compile_success),
        "validation_success": bool(result.validation_success),
        "num_waypoints": int(result.num_waypoints),
        "num_clamped_waypoints": (
            int(len(corrected_indices)) if isinstance(corrected_indices, list) else 0
        ),
        "mean_translation_error_m": float(metrics.get("mean_translation_error_m", 0.0)),
        "max_translation_error_m": float(metrics.get("max_translation_error_m", 0.0)),
        "mean_rotation_error_deg": float(metrics.get("mean_rotation_error_deg", 0.0)),
        "max_rotation_error_deg": float(metrics.get("max_rotation_error_deg", 0.0)),
        "reference_path_length_m": float(metrics.get("reference_path_length_m", 0.0)),
        "candidate_path_length_m": float(metrics.get("candidate_path_length_m", 0.0)),
        "path_length_ratio": float(metrics.get("path_length_ratio", 0.0)),
        "sample_count": float(metrics.get("sample_count", 0.0)),
        "semantic_contact_point_xy_error_m": metrics.get("semantic_contact_point_xy_error_m"),
        "semantic_target_frame_index": metrics.get("semantic_target_frame_index"),
        "screwdriver_max_primary_axis_tilt_deg": metrics.get(
            "screwdriver_max_primary_axis_tilt_deg"
        ),
        "screwdriver_twist_angle_span_deg": metrics.get("screwdriver_twist_angle_span_deg"),
        "error": result.error,
        "replay_artifact_path": result.replay_artifact_path,
    }


# Build one geometry trial result from a saved raw payload with fresh semantic metrics.
def _geometry_result_from_raw_payload(payload: Dict[str, Any]) -> GeometryTrialResult:
    """Return one result object reconstructed from raw JSON with recomputed semantic metrics."""
    object_name = str(payload.get("object_name", ""))
    goals = [list(goal) for goal in payload.get("goals", []) if isinstance(goal, list)]
    target_xy = payload.get("target_xy")
    metrics = dict(payload.get("metrics") or {})
    if bool(payload.get("compile_success")) and bool(payload.get("validation_success")):
        semantic_metrics = _build_semantic_geometry_metrics(
            object_name=object_name,
            goals=goals,
            target_xy=target_xy if isinstance(target_xy, list) else None,
        )
        metrics.update(semantic_metrics)
    return GeometryTrialResult(
        trial_id=str(payload.get("trial_id", "")),
        object_name=object_name,
        task_name=str(payload.get("task_name", "")),
        mode=str(payload.get("mode", "")),
        seed=int(payload.get("seed", 0)),
        target_xy=target_xy if isinstance(target_xy, list) else None,
        resolved_baseline_object=payload.get("resolved_baseline_object"),
        pivot_point=(
            payload.get("pivot_point") if isinstance(payload.get("pivot_point"), list) else None
        ),
        compile_success=bool(payload.get("compile_success")),
        validation_success=bool(payload.get("validation_success")),
        num_waypoints=int(payload.get("num_waypoints", len(goals))),
        goals=goals,
        generation_context=dict(payload.get("generation_context") or {}),
        clamp_summary=dict(payload.get("clamp_summary") or {}),
        resampling_summary=dict(payload.get("resampling_summary") or {}),
        metrics=metrics,
        error=payload.get("error"),
        replay_artifact_path=payload.get("replay_artifact_path"),
    )


# Recompute derived geometry summaries from existing raw trajectory artifacts.
def recompute_geometry_summaries_from_raw(experiment_dir: Path) -> None:
    """Rewrite geometry summary files from frozen raw artifacts without changing raw JSON."""
    raw_dir = experiment_dir / "geometry" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing geometry raw directory: {raw_dir}")
    (experiment_dir / "geometry" / "summaries").mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for raw_path in sorted(raw_dir.glob("*.json")):
        result = _geometry_result_from_raw_payload(read_json(raw_path))
        rows.append(_flatten_geometry_result(result))
    config_path = experiment_dir / "config.json"
    if config_path.exists():
        config = read_json(config_path)
        target_grid_xy = build_target_grid_xy(
            int(config.get("target_grid_x", 3)),
            int(config.get("target_grid_y", 3)),
        )
    else:
        target_grid_xy = _target_grid_xy_from_generated_rows(rows)
    aggregate_payload = {
        "objects": list(dict.fromkeys(str(row["object_name"]) for row in rows)),
        "modes": list(dict.fromkeys(str(row["mode"]) for row in rows)),
        "target_grid_xy": target_grid_xy,
        **_build_geometry_aggregate(rows),
    }
    save_trial_summaries(experiment_dir, "geometry", rows, aggregate_payload)


# Return the target grid implied by generated-mode rows when no config is available.
def _target_grid_xy_from_generated_rows(
    rows: Sequence[Dict[str, Any]],
) -> List[tuple[float, float]]:
    """Return unique target XY values from generated geometry rows."""
    target_grid_xy = []
    seen_targets = set()
    for row in rows:
        if str(row.get("mode")) == "predefined":
            continue
        if row.get("target_x") is None or row.get("target_y") is None:
            continue
        target_tuple = (float(row["target_x"]), float(row["target_y"]))
        if target_tuple in seen_targets:
            continue
        seen_targets.add(target_tuple)
        target_grid_xy.append(target_tuple)
    return target_grid_xy


# Compute compact aggregate statistics for the geometry benchmark section.
def _build_geometry_aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Return aggregate metrics for geometry summary persistence."""
    num_trials = len(rows)
    if num_trials == 0:
        return {
            "num_trials": 0,
            "compile_success_rate": 0.0,
            "validation_success_rate": 0.0,
            "by_tool_family_mode": [],
        }
    compile_success_count = sum(1 for row in rows if bool(row["compile_success"]))
    validation_success_count = sum(1 for row in rows if bool(row["validation_success"]))
    grouped_rows: List[Dict[str, Any]] = []
    grouping_keys = sorted({(str(row["tool_family"]), str(row["mode"])) for row in rows})
    for tool_family, mode in grouping_keys:
        subset = [
            row
            for row in rows
            if str(row["tool_family"]) == tool_family and str(row["mode"]) == mode
        ]
        if not subset:
            continue
        semantic_errors = [
            float(row["semantic_contact_point_xy_error_m"])
            for row in subset
            if row.get("semantic_contact_point_xy_error_m") is not None
        ]
        grouped_rows.append(
            {
                "tool_family": tool_family,
                "mode": mode,
                "num_trials": int(len(subset)),
                "accepted_share": float(
                    sum(
                        bool(row["compile_success"]) and bool(row["validation_success"])
                        for row in subset
                    )
                    / len(subset)
                ),
                "mean_semantic_contact_point_xy_error_m": (
                    float(sum(semantic_errors) / len(semantic_errors)) if semantic_errors else None
                ),
            }
        )
    return {
        "num_trials": int(num_trials),
        "compile_success_rate": float(compile_success_count / num_trials),
        "validation_success_rate": float(validation_success_count / num_trials),
        "by_tool_family_mode": grouped_rows,
    }


# Parse CLI arguments and execute the geometry benchmark.
def main() -> None:
    """Entry point for the geometry benchmark CLI."""
    args = tyro.cli(GeometryBenchmarkArgs)
    if bool(args.recompute_from_raw):
        if args.experiment_name is None:
            raise ValueError("--experiment-name is required with --recompute-from-raw.")
        experiment_dir = args.results_dir / args.experiment_name
        recompute_geometry_summaries_from_raw(experiment_dir)
        print(f"Recomputed geometry summaries from raw artifacts in {experiment_dir}")
        return
    experiment_dir = run_geometry_benchmark(args)
    print(f"Saved geometry benchmark to {experiment_dir}")


if __name__ == "__main__":
    main()
