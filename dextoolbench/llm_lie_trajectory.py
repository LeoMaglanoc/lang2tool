"""Shared helpers for LLM-built Lie swing/twist trajectory compilation."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch

from dextoolbench.eval_config import DEFAULT_LLM_LIE_SEMANTIC_CONTACT_CLEARANCE_M
from dextoolbench.llm_supported_objects import supported_llm_object_family
from geometric_tool_planning import (
    build_llm_lie_spec_from_target_xy,
    clamp_llm_strike_target_xy,
    compile_llm_lie_goals,
)
from geometric_tool_planning.trajectory import normalize_quaternion_xyzw, validate_pose_sequence
from geometric_tool_planning.viewer import TABLE_TOP_Z
from isaacgymenvs.utils.torch_jit_utils import quat_conjugate, quat_mul
from isaacgymenvs.utils.torch_jit_utils import slerp as torch_quaternion_slerp
from llm_runtime.semantic_pose import get_object_pose_semantics, quat_mul_xyzw, quat_rotate_xyzw


# Resolve the task name used for LLM Lie compilation so all eval entrypoints share the same logic.
def resolve_llm_lie_task_name(object_name: str, task_name: str) -> str:
    """Return the task name used when compiling one LLM Lie trajectory."""
    try:
        if supported_llm_object_family(object_name) == "hammer":
            return "swing_down"
    except ValueError:
        pass
    return task_name


# Build one local-axis angle quaternion in XYZW convention.
def _axis_angle_quaternion_xyzw(axis: Sequence[float], angle_rad: float) -> List[float]:
    """Return one XYZW quaternion for a rotation about the given local axis."""
    axis_norm = math.sqrt(sum(float(value) * float(value) for value in axis))
    if axis_norm <= 1e-8:
        raise ValueError("Rotation axis must be non-zero.")
    half_angle = 0.5 * float(angle_rad)
    sin_half = math.sin(half_angle)
    return [
        float(axis[0]) / axis_norm * sin_half,
        float(axis[1]) / axis_norm * sin_half,
        float(axis[2]) / axis_norm * sin_half,
        math.cos(half_angle),
    ]


# Build the screwdriver-specific upright twist path around one hover target.
def _compile_screwdriver_spin_vertical_goals(
    *,
    object_name: str,
    strike_target_xy: Sequence[float],
    waypoint_table_clearance_m: float,
    screwdriver_twist_extra_hover_m: float,
    semantic_contact_clearance_m: float,
) -> Tuple[Dict[str, object], List[List[float]]]:
    """Return one upright in-place twist sequence solved directly from semantic geometry."""
    semantics = get_object_pose_semantics(object_name)
    requested_target_xy = [float(strike_target_xy[0]), float(strike_target_xy[1])]
    upright_quaternion = _quat_from_two_unit_vectors(
        semantics.primary_axis_local,
        (0.0, 0.0, -1.0),
    )
    rotated_tip_offset = quat_rotate_xyzw(upright_quaternion, semantics.strike_point_local)
    rotated_support_offset = quat_rotate_xyzw(
        upright_quaternion, semantics.swing_support_point_local
    )
    execution_target_z = float(TABLE_TOP_Z) + float(semantic_contact_clearance_m)
    target_position = [
        float(requested_target_xy[0]),
        float(requested_target_xy[1]),
        float(execution_target_z),
    ]
    anchor_position = [
        float(requested_target_xy[0]) - float(rotated_tip_offset[0]),
        float(requested_target_xy[1]) - float(rotated_tip_offset[1]),
        float(execution_target_z) - float(rotated_support_offset[2]),
    ]
    support_position = [
        float(anchor_position[axis]) + float(rotated_support_offset[axis]) for axis in range(3)
    ]

    # Use a full turn so forward repetition creates a continuous spin without a mirrored unwind.
    twist_angles_deg = [45.0 * float(index) for index in range(9)]
    compiled_goals: List[List[float]] = []
    for twist_angle_deg in twist_angles_deg:
        local_delta = _axis_angle_quaternion_xyzw(
            semantics.primary_axis_local,
            math.radians(twist_angle_deg),
        )
        twisted_quat = normalize_quaternion_xyzw(quat_mul_xyzw(upright_quaternion, local_delta))
        rotated_twisted_tip_offset = quat_rotate_xyzw(twisted_quat, semantics.strike_point_local)
        waypoint_position = [
            float(target_position[axis]) - float(rotated_twisted_tip_offset[axis])
            for axis in range(3)
        ]
        compiled_goals.append(
            [
                *waypoint_position,
                *twisted_quat,
            ]
        )
    spec = {
        "schema_version": "twist_v1",
        "verb": "twist",
        "task_frame": "world_table_xy",
        "pivot_point": [float(value) for value in anchor_position],
        "strike_target_xy": requested_target_xy,
        "object_name": object_name,
        "task_name": "spin_vertical",
        "aligned_target_xyz": [float(value) for value in target_position],
        "aligned_support_xyz": [float(value) for value in support_position],
        "semantic_tip_point_local": [float(value) for value in semantics.strike_point_local],
        "semantic_support_point_local": [
            float(value) for value in semantics.swing_support_point_local
        ],
        "twist_axis_local": [float(value) for value in semantics.primary_axis_local],
        "upright_quaternion_xyzw": [float(value) for value in upright_quaternion],
        "support_point_z_offset_m": float(rotated_support_offset[2]),
        "execution_target_z": float(execution_target_z),
        "semantic_contact_clearance_m": float(semantic_contact_clearance_m),
        "waypoint_table_clearance_m": float(waypoint_table_clearance_m),
        "screwdriver_twist_extra_hover_m": float(screwdriver_twist_extra_hover_m),
        "twist_angle_deg": 360.0,
        "cycle_style": "forward_repeat",
        "waypoint_count": len(compiled_goals),
    }
    return spec, compiled_goals


# Build the shortest-arc quaternion rotating one source axis onto one target axis.
def _quat_from_two_unit_vectors(
    source_axis: Sequence[float],
    target_axis: Sequence[float],
) -> List[float]:
    """Return one deterministic XYZW quaternion that rotates source_axis onto target_axis."""
    source_norm = math.sqrt(sum(float(value) * float(value) for value in source_axis))
    target_norm = math.sqrt(sum(float(value) * float(value) for value in target_axis))
    if source_norm <= 1e-8 or target_norm <= 1e-8:
        raise ValueError("Rotation axes must be non-zero.")
    source = [float(value) / source_norm for value in source_axis]
    target = [float(value) / target_norm for value in target_axis]
    dot = max(-1.0, min(1.0, sum(source[idx] * target[idx] for idx in range(3))))
    if dot > 1.0 - 1e-6:
        return [0.0, 0.0, 0.0, 1.0]
    if dot < -1.0 + 1e-6:
        orthogonal = [0.0, -source[2], source[1]]
        orthogonal_norm = math.sqrt(sum(value * value for value in orthogonal))
        if orthogonal_norm <= 1e-8:
            orthogonal = [-source[2], 0.0, source[0]]
            orthogonal_norm = math.sqrt(sum(value * value for value in orthogonal))
        return [float(value) / orthogonal_norm for value in orthogonal] + [0.0]
    cross = [
        source[1] * target[2] - source[2] * target[1],
        source[2] * target[0] - source[0] * target[2],
        source[0] * target[1] - source[1] * target[0],
    ]
    quaternion = [cross[0], cross[1], cross[2], 1.0 + dot]
    quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
    return [float(value) / quaternion_norm for value in quaternion]


# Measure per-step translation and rotation changes for one pose sequence.
def _summarize_goal_step_deltas(goals: Sequence[Sequence[float]]) -> Dict[str, float]:
    """Return max step translation and rotation magnitudes for one goal sequence."""
    if len(goals) <= 1:
        return {
            "max_step_translation_m": 0.0,
            "mean_step_translation_m": 0.0,
            "max_step_rotation_deg": 0.0,
            "mean_step_rotation_deg": 0.0,
        }

    positions = torch.tensor([goal[:3] for goal in goals], dtype=torch.float32)
    quats = torch.tensor([goal[3:7] for goal in goals], dtype=torch.float32)
    step_translations = torch.linalg.norm(positions[1:] - positions[:-1], dim=1).tolist()

    step_rotations_deg: List[float] = []
    for idx in range(len(goals) - 1):
        qrel = quat_mul(quats[idx + 1 : idx + 2], quat_conjugate(quats[idx : idx + 1]))[0]
        quat_w = max(-1.0, min(1.0, abs(float(qrel[3]))))
        step_rotations_deg.append(math.degrees(2.0 * math.acos(quat_w)))

    return {
        "max_step_translation_m": float(max(step_translations)),
        "mean_step_translation_m": float(sum(step_translations) / len(step_translations)),
        "max_step_rotation_deg": float(max(step_rotations_deg)),
        "mean_step_rotation_deg": float(sum(step_rotations_deg) / len(step_rotations_deg)),
    }


# Interpolate one XYZW quaternion pair with spherical interpolation.
def _slerp_quaternion_xyzw(
    quaternion_start: Sequence[float],
    quaternion_end: Sequence[float],
    fraction: float,
) -> List[float]:
    """Return one normalized XYZW quaternion interpolated at the requested fraction."""
    start_tensor = torch.tensor([normalize_quaternion_xyzw(quaternion_start)], dtype=torch.float32)
    end_tensor = torch.tensor([normalize_quaternion_xyzw(quaternion_end)], dtype=torch.float32)
    fraction_tensor = torch.tensor([[float(fraction)]], dtype=torch.float32)
    interpolated = torch_quaternion_slerp(start_tensor, end_tensor, fraction_tensor)[0].tolist()
    return normalize_quaternion_xyzw(interpolated)


# Resample one dense Lie pose sequence toward training-like step scales with a combined SE(3) metric.
def resample_lie_goals_for_training_distribution(
    goals: Sequence[Sequence[float]],
    *,
    pos_scale_m: float,
    rot_scale_deg: float,
    target_cost: float,
    min_waypoints: int,
) -> Tuple[List[List[float]], Dict[str, object]]:
    """Return resampled goals and metadata for one deterministic training-like discretization."""
    validated_goals = validate_pose_sequence(goals)
    if pos_scale_m <= 0.0 or rot_scale_deg <= 0.0 or target_cost <= 0.0:
        raise ValueError(
            "Training-distribution resampling scales and target cost must be positive."
        )
    if min_waypoints < 2:
        raise ValueError("Training-distribution resampling needs at least two waypoints.")
    if len(validated_goals) <= min_waypoints:
        summary = {
            "enabled": True,
            "applied": False,
            "pos_scale_m": float(pos_scale_m),
            "rot_scale_deg": float(rot_scale_deg),
            "target_cost": float(target_cost),
            "min_waypoints": int(min_waypoints),
            "original_waypoint_count": len(validated_goals),
            "resampled_waypoint_count": len(validated_goals),
            "step_delta_summary": _summarize_goal_step_deltas(validated_goals),
        }
        return [list(goal) for goal in validated_goals], summary

    cumulative_costs: List[float] = [0.0]
    for index in range(len(validated_goals) - 1):
        segment_translation_m = math.dist(
            validated_goals[index][:3], validated_goals[index + 1][:3]
        )
        segment_rotation_deg = _summarize_goal_step_deltas(
            [validated_goals[index], validated_goals[index + 1]]
        )["max_step_rotation_deg"]
        segment_cost = (segment_translation_m / float(pos_scale_m)) + (
            segment_rotation_deg / float(rot_scale_deg)
        )
        cumulative_costs.append(cumulative_costs[-1] + float(segment_cost))

    total_cost = cumulative_costs[-1]
    target_waypoint_count = max(
        int(min_waypoints),
        int(math.ceil(total_cost / float(target_cost))) + 1,
    )
    if total_cost <= 1e-8 or target_waypoint_count >= len(validated_goals):
        summary = {
            "enabled": True,
            "applied": False,
            "pos_scale_m": float(pos_scale_m),
            "rot_scale_deg": float(rot_scale_deg),
            "target_cost": float(target_cost),
            "min_waypoints": int(min_waypoints),
            "original_waypoint_count": len(validated_goals),
            "resampled_waypoint_count": len(validated_goals),
            "step_delta_summary": _summarize_goal_step_deltas(validated_goals),
        }
        return [list(goal) for goal in validated_goals], summary

    target_costs = torch.linspace(0.0, float(total_cost), steps=target_waypoint_count).tolist()
    resampled_goals: List[List[float]] = []
    last_segment_index = len(validated_goals) - 2
    for sample_cost in target_costs:
        if sample_cost <= 0.0:
            resampled_goals.append(list(validated_goals[0]))
            continue
        if sample_cost >= total_cost:
            resampled_goals.append(list(validated_goals[-1]))
            continue

        upper_index = next(
            index
            for index, cumulative_cost in enumerate(cumulative_costs)
            if cumulative_cost >= sample_cost
        )
        lower_index = max(0, upper_index - 1)
        lower_cost = cumulative_costs[lower_index]
        upper_cost = cumulative_costs[upper_index]
        segment_index = min(lower_index, last_segment_index)
        if upper_cost - lower_cost <= 1e-8:
            fraction = 0.0
        else:
            fraction = (float(sample_cost) - lower_cost) / (upper_cost - lower_cost)
        lower_goal = validated_goals[segment_index]
        upper_goal = validated_goals[segment_index + 1]
        interpolated_position = [
            float(lower_goal[axis]) + (float(upper_goal[axis]) - float(lower_goal[axis])) * fraction
            for axis in range(3)
        ]
        interpolated_quaternion = _slerp_quaternion_xyzw(
            lower_goal[3:7],
            upper_goal[3:7],
            fraction,
        )
        resampled_goals.append(interpolated_position + interpolated_quaternion)

    validated_resampled_goals = validate_pose_sequence(resampled_goals)
    summary = {
        "enabled": True,
        "applied": True,
        "pos_scale_m": float(pos_scale_m),
        "rot_scale_deg": float(rot_scale_deg),
        "target_cost": float(target_cost),
        "min_waypoints": int(min_waypoints),
        "original_waypoint_count": len(validated_goals),
        "resampled_waypoint_count": len(validated_resampled_goals),
        "step_delta_summary": _summarize_goal_step_deltas(validated_resampled_goals),
    }
    return validated_resampled_goals, summary


# Clamp one pose sequence into the RL training target-volume box and summarize corrections.
def clamp_lie_goals_to_training_volume(
    goals: Sequence[Sequence[float]],
    *,
    target_volume_mins: Sequence[float],
    target_volume_maxs: Sequence[float],
    clamp_z: bool = True,
) -> Tuple[List[List[float]], Dict[str, object]]:
    """Return clamped goals and one correction summary for the training target volume."""
    if len(target_volume_mins) != 3 or len(target_volume_maxs) != 3:
        raise ValueError("Training target volume bounds must each contain exactly three values.")

    min_bounds = [float(value) for value in target_volume_mins]
    max_bounds = [float(value) for value in target_volume_maxs]
    corrected_goals: List[List[float]] = []
    corrected_indices: List[int] = []
    original_positions: List[List[float]] = []
    corrected_positions: List[List[float]] = []

    for index, goal in enumerate(goals):
        corrected_goal = list(goal)
        original_position = [float(value) for value in goal[:3]]
        clamped_position = [
            min(max(original_position[axis], min_bounds[axis]), max_bounds[axis])
            for axis in range(3)
        ]
        if not clamp_z:
            clamped_position[2] = original_position[2]
        corrected_goal[:3] = clamped_position
        corrected_goals.append(corrected_goal)
        if any(abs(clamped_position[axis] - original_position[axis]) > 1e-9 for axis in range(3)):
            corrected_indices.append(index)
            original_positions.append(original_position)
            corrected_positions.append([float(value) for value in clamped_position])

    summary = {
        "enabled": True,
        "applied": bool(corrected_indices),
        "clamp_z": bool(clamp_z),
        "target_volume_mins": min_bounds,
        "target_volume_maxs": max_bounds,
        "corrected_indices": corrected_indices,
        "original_positions": original_positions,
        "corrected_positions": corrected_positions,
        "step_delta_summary": _summarize_goal_step_deltas(corrected_goals),
    }
    return corrected_goals, summary


# Compile one clamped LLM Lie trajectory from a pivot point and explicit tabletop strike target.
def compile_llm_lie_trajectory(
    *,
    object_name: str,
    task_name: str,
    pivot_point: Sequence[float],
    strike_target_xy: Sequence[float],
    horizontal_strike_clearance_m: float,
    waypoint_table_clearance_m: float,
    screwdriver_twist_extra_hover_m: float,
    training_resampling_enabled: bool,
    training_resampling_pos_scale_m: float,
    training_resampling_rot_scale_deg: float,
    training_resampling_target_cost: float,
    training_resampling_min_waypoints: int,
    training_volume_clamp_enabled: bool,
    training_target_volume_mins: Sequence[float],
    training_target_volume_maxs: Sequence[float],
) -> Tuple[List[float], Dict[str, object], List[List[float]], Dict[str, object]]:
    """Return clamped target, validated raw spec, corrected Lie goals, and clamp metadata."""
    resolved_task_name = resolve_llm_lie_task_name(object_name, task_name)
    object_family = supported_llm_object_family(object_name)
    semantic_contact_clearance_m = float(DEFAULT_LLM_LIE_SEMANTIC_CONTACT_CLEARANCE_M)
    clamped_target_xy = list(
        clamp_llm_strike_target_xy(
            strike_target_xy,
            object_name=object_name,
            task_name=resolved_task_name,
        )
    )
    if object_family == "screwdriver" and resolved_task_name == "spin_vertical":
        spec, compiled_goals = _compile_screwdriver_spin_vertical_goals(
            object_name=object_name,
            strike_target_xy=clamped_target_xy,
            waypoint_table_clearance_m=float(waypoint_table_clearance_m),
            screwdriver_twist_extra_hover_m=float(screwdriver_twist_extra_hover_m),
            semantic_contact_clearance_m=semantic_contact_clearance_m,
        )
    else:
        spec = build_llm_lie_spec_from_target_xy(
            object_name=object_name,
            task_name=resolved_task_name,
            pivot_point=[float(value) for value in pivot_point[:3]],
            strike_target_xy=clamped_target_xy,
            horizontal_strike_clearance_m=semantic_contact_clearance_m,
            waypoint_table_clearance_m=float(waypoint_table_clearance_m),
        )
        spec["semantic_contact_clearance_m"] = semantic_contact_clearance_m
        spec["configured_horizontal_strike_clearance_m"] = float(horizontal_strike_clearance_m)
        spec["hammer_swing_solver"] = "horizontal_head"
        compiled_goals = compile_llm_lie_goals(spec, object_name=object_name)
    if training_resampling_enabled:
        resampled_goals, resampling_summary = resample_lie_goals_for_training_distribution(
            compiled_goals,
            pos_scale_m=float(training_resampling_pos_scale_m),
            rot_scale_deg=float(training_resampling_rot_scale_deg),
            target_cost=float(training_resampling_target_cost),
            min_waypoints=int(training_resampling_min_waypoints),
        )
    else:
        resampled_goals = [list(goal) for goal in compiled_goals]
        resampling_summary = {
            "enabled": False,
            "applied": False,
            "pos_scale_m": float(training_resampling_pos_scale_m),
            "rot_scale_deg": float(training_resampling_rot_scale_deg),
            "target_cost": float(training_resampling_target_cost),
            "min_waypoints": int(training_resampling_min_waypoints),
            "original_waypoint_count": len(resampled_goals),
            "resampled_waypoint_count": len(resampled_goals),
            "step_delta_summary": _summarize_goal_step_deltas(resampled_goals),
        }
    if training_volume_clamp_enabled:
        corrected_goals, clamp_summary = clamp_lie_goals_to_training_volume(
            resampled_goals,
            target_volume_mins=training_target_volume_mins,
            target_volume_maxs=training_target_volume_maxs,
            clamp_z=not (
                (object_family == "hammer" and resolved_task_name == "swing_down")
                or (object_family == "screwdriver" and resolved_task_name == "spin_vertical")
            ),
        )
    else:
        corrected_goals = [list(goal) for goal in resampled_goals]
        clamp_summary = {
            "enabled": False,
            "applied": False,
            "clamp_z": not (
                (object_family == "hammer" and resolved_task_name == "swing_down")
                or (object_family == "screwdriver" and resolved_task_name == "spin_vertical")
            ),
            "target_volume_mins": [float(value) for value in training_target_volume_mins],
            "target_volume_maxs": [float(value) for value in training_target_volume_maxs],
            "corrected_indices": [],
            "original_positions": [],
            "corrected_positions": [],
            "step_delta_summary": _summarize_goal_step_deltas(corrected_goals),
        }
    spec["training_distribution_resampling"] = resampling_summary
    spec["training_distribution_clamp"] = clamp_summary
    return clamped_target_xy, spec, corrected_goals, clamp_summary
