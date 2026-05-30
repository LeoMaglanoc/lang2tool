"""LLM and mock backends for timed swing specs and direct pose lists."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, Optional

from dextoolbench.llm_supported_objects import supported_llm_object_family
from dextoolbench.predefined_baselines import (
    resolve_predefined_baseline_object_name,
    resolve_predefined_trajectory_path,
)
from llm_runtime.semantic_pose import get_object_pose_semantics, quat_rotate_xyzw

from .orchestrator_types import EvalGoalSourcesArgs, GoalSourceArtifact
from .viewer import TABLE_TOP_Z

_TABLE_HALF_EXTENT_X_M = 0.475 / 2.0
_TABLE_HALF_EXTENT_Y_M = 0.4 / 2.0
_TABLE_TARGET_MARGIN_M = 0.04
_DEFAULT_SWING_DURATION_SEC = 0.62
_DEFAULT_SWING_NUM_SAMPLES = 37
_DEFAULT_SWING_ANGLE_RAD = 1.6
DEFAULT_OPENAI_MODEL = "gpt-5.5"
_DEFAULT_LIE_HORIZONTAL_STRIKE_CLEARANCE_M = 0.05
_DEFAULT_LIE_WAYPOINT_TABLE_CLEARANCE_M = 0.05
_PREDEFINED_NAMED_STRIKE_REFERENCE_FRAME_INDEX = 50
_NAMED_STRIKE_POINT_ALIASES: Dict[str, tuple[str, ...]] = {
    "target_a": ("a", "strike point a", "striking point a"),
    "target_b": ("b", "strike point b", "striking point b", "front of the table"),
    "target_c": ("c", "strike point c", "striking point c", "back of the table"),
}


# Query the configured OpenAI backend for a JSON response.
def openai_json_response(system_prompt: str, user_prompt: str, model: str) -> Dict[str, Any]:
    """Return a parsed JSON object from an OpenAI chat completion."""
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - depends on optional package/runtime
        raise RuntimeError(
            "OpenAI backend requested but the openai package is unavailable."
        ) from exc

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise RuntimeError("OpenAI backend did not return JSON text content.")
    return json.loads(content)


# Build a deterministic mock Lie-group swing spec from the reference trajectory.
def generate_mock_llm_lie_spec(reference: GoalSourceArtifact) -> Dict[str, Any]:
    """Return a deterministic semantic `swing_v1` spec anchored to the recorded swing path."""
    recorded_goals = reference.metadata.get("recorded_goals", reference.goals[1:])
    if not recorded_goals:
        raise ValueError("Reference artifact must include recorded swing goals.")
    first = recorded_goals[0]
    last = recorded_goals[-1]
    object_name = str(reference.metadata["object_name"])
    strike_point_local = get_object_pose_semantics(object_name).strike_point_local
    final_strike_point = [
        float(last[axis]) + quat_rotate_xyzw(last[3:], strike_point_local)[axis]
        for axis in range(3)
    ]
    final_head_direction = quat_rotate_xyzw(last[3:], (1.0, 0.0, 0.0))
    horizontal = [float(final_head_direction[0]), float(final_head_direction[1]), 0.0]
    if math.sqrt(sum(value * value for value in horizontal)) <= 1e-8:
        horizontal = [float(last[0]) - float(first[0]), float(last[1]) - float(first[1]), 0.0]
    swing_angle_rad = math.acos(max(-1.0, min(1.0, float(final_head_direction[2]))))
    return {
        "schema_version": "swing_v1",
        "verb": "swing_down",
        "task_frame": "world",
        "pivot_point": [first[0], first[1], first[2]],
        "strike_target_xy": [final_strike_point[0], final_strike_point[1]],
        "swing_angle_rad": max(math.pi / 2.0, min(2.4, swing_angle_rad)),
        "duration_sec": reference.duration_sec,
        "num_samples": max(2, len(reference.goals) - 1),
    }


# Return the fixed named strike targets exposed to both overlays and the LLM API.
def get_named_strike_points(
    *, object_name: str, task_name: Optional[str] = None
) -> Dict[str, list[float]]:
    """Return fixed world-frame tabletop strike targets for one supported tool."""
    target_a_xy = get_predefined_strike_target_xy(object_name=object_name, task_name=task_name)
    if target_a_xy is None:
        return {}
    target_b_xy = [0.0, _clamp_table_target(_TABLE_HALF_EXTENT_Y_M, axis="y")]
    target_c_xy = [0.0, _clamp_table_target(-_TABLE_HALF_EXTENT_Y_M, axis="y")]
    return {
        "target_a": [float(target_a_xy[0]), float(target_a_xy[1])],
        "target_b": [float(target_b_xy[0]), float(target_b_xy[1])],
        "target_c": [float(target_c_xy[0]), float(target_c_xy[1])],
    }


# Load one recorded predefined trajectory payload for one supported object/task pair.
@lru_cache(maxsize=None)
def _resolve_predefined_payload_object_name(object_name: str) -> str:
    """Return the object name whose prerecorded trajectory should back predefined playback."""
    return resolve_predefined_baseline_object_name(object_name)


# Load one recorded predefined trajectory payload for one supported object/task pair.
@lru_cache(maxsize=None)
def load_predefined_trajectory_payload(*, object_name: str, task_name: str) -> Dict[str, Any]:
    """Return the stored DexToolBench trajectory payload for one object/task pair."""
    trajectory_path = resolve_predefined_trajectory_path(object_name, task_name)
    with open(trajectory_path) as file_obj:
        return json.load(file_obj)


# Return the recorded predefined swing goals for one object/task pair.
def get_predefined_goal_sequence(
    *, object_name: str, task_name: str, include_start_pose: bool = False
) -> list[list[float]]:
    """Return recorded predefined goals, optionally prepending the stored start pose."""
    payload = load_predefined_trajectory_payload(object_name=object_name, task_name=task_name)
    recorded_goals = [[float(value) for value in goal] for goal in payload["goals"]]
    if not include_start_pose:
        return recorded_goals
    return [[float(value) for value in payload["start_pose"]], *recorded_goals]


# Mirror one forward-only predefined swing into the same cycle shown by the viewers.
def _mirrored_predefined_cycle(goals: list[list[float]]) -> list[list[float]]:
    """Return the down-then-up predefined cycle without duplicated turnaround endpoints."""
    if len(goals) <= 1:
        return [list(goal) for goal in goals]
    return [list(goal) for goal in goals] + [list(goal) for goal in reversed(goals[1:-1])]


# Intersect one semantic strike-face axis ray with the tabletop plane for a prerecorded pose.
def _table_intersection_from_semantic_axis(
    pose: list[float],
    *,
    object_name: str,
) -> list[float]:
    """Return the tabletop XY hit point of the blue semantic strike-face axis for one pose."""
    semantics = get_object_pose_semantics(object_name)
    axis_origin = [
        float(pose[axis]) + quat_rotate_xyzw(pose[3:], semantics.strike_point_local)[axis]
        for axis in range(3)
    ]
    axis_direction = quat_rotate_xyzw(pose[3:], semantics.strike_face_normal_local)
    if abs(float(axis_direction[2])) <= 1e-8:
        raise RuntimeError("Predefined strike-face axis is parallel to the tabletop plane.")
    ray_scale = (float(TABLE_TOP_Z) - float(axis_origin[2])) / float(axis_direction[2])
    if ray_scale < 0.0:
        raise RuntimeError("Predefined strike-face axis points away from the tabletop plane.")
    return [
        float(axis_origin[0]) + float(ray_scale) * float(axis_direction[0]),
        float(axis_origin[1]) + float(ray_scale) * float(axis_direction[1]),
    ]


# Return the recorded predefined strike target on the table when supported.
def get_predefined_strike_target_xy(
    *, object_name: str, task_name: Optional[str] = None
) -> Optional[list[float]]:
    """Return the frame-50 semantic strike-face axis intersection with the tabletop plane."""
    resolved_task_name = "swing_down" if task_name in (None, "grasp_hold") else str(task_name)
    try:
        object_family = supported_llm_object_family(object_name)
    except ValueError:
        object_family = ""
    if object_family == "screwdriver" and resolved_task_name == "spin_vertical":
        goals = get_predefined_goal_sequence(
            object_name=object_name,
            task_name=resolved_task_name,
            include_start_pose=False,
        )
        if not goals:
            return None
        mean_x = sum(float(goal[0]) for goal in goals) / len(goals)
        mean_y = sum(float(goal[1]) for goal in goals) / len(goals)
        return [mean_x, mean_y]
    if object_family != "hammer" or resolved_task_name != "swing_down":
        return None
    mirrored_goals = _mirrored_predefined_cycle(
        get_predefined_goal_sequence(
            object_name=object_name,
            task_name=resolved_task_name,
            include_start_pose=False,
        )
    )
    if _PREDEFINED_NAMED_STRIKE_REFERENCE_FRAME_INDEX >= len(mirrored_goals):
        raise RuntimeError(
            "Named strike reference frame is outside the prerecorded predefined swing cycle."
        )
    return _table_intersection_from_semantic_axis(
        mirrored_goals[_PREDEFINED_NAMED_STRIKE_REFERENCE_FRAME_INDEX],
        object_name=object_name,
    )


# Return static strike-target geometry shared by the prompt and runtime validation.
def get_llm_static_strike_context(
    *, object_name: str, task_name: Optional[str] = None
) -> Dict[str, Any]:
    """Return immutable world-table targeting geometry for one supported tool."""
    named_points = get_named_strike_points(object_name=object_name)
    x_limit = _TABLE_HALF_EXTENT_X_M - _TABLE_TARGET_MARGIN_M
    y_limit = _TABLE_HALF_EXTENT_Y_M - _TABLE_TARGET_MARGIN_M
    return {
        "available": bool(named_points),
        "frame": "world_table_xy",
        "frame_description": (
            "Use world-table XY coordinates on the tabletop plane. X increases toward the "
            "camera-right side of the table, and Y increases away from the robot base. "
            "When the user says front of the table, interpret that as farther away from the "
            "camera. When the user says back of the table, interpret that as closer to the camera."
        ),
        "table_target_region": {
            "x_min": -float(x_limit),
            "x_max": float(x_limit),
            "y_min": -float(y_limit),
            "y_max": float(y_limit),
        },
        "named_strike_points": get_named_strike_point_payload(
            object_name=object_name,
            task_name=task_name,
        ),
    }


# Return the structured named strike-point payload exposed through get_sim_state.
def get_named_strike_point_payload(
    *, object_name: str, task_name: Optional[str] = None
) -> Dict[str, Any]:
    """Return chat/viewer metadata for named strike points when available."""
    points = get_named_strike_points(object_name=object_name, task_name=task_name)
    if not points:
        return {"available": False, "frame": "world_table_xy", "points": {}}
    return {
        "available": True,
        "frame": "world_table_xy",
        "points": {
            name: {
                "aliases": list(_NAMED_STRIKE_POINT_ALIASES.get(name, ())),
                "world_xy": [float(value) for value in xy],
                "world_xyz": [float(xy[0]), float(xy[1]), float(TABLE_TOP_Z)],
            }
            for name, xy in points.items()
        },
    }


# Return whether one strike-target XY duplicates the fixed strike point a within overlay tolerance.
def is_target_a_strike_target_xy(
    strike_target_xy: list[float] | tuple[float, float],
    *,
    object_name: str,
    task_name: Optional[str] = None,
    tolerance_m: float = 1e-4,
) -> bool:
    """Return True when one strike-target XY matches named strike point a."""
    target_a_xy = get_predefined_strike_target_xy(object_name=object_name, task_name=task_name)
    if target_a_xy is None or len(strike_target_xy) != 2:
        return False
    dx = float(strike_target_xy[0]) - float(target_a_xy[0])
    dy = float(strike_target_xy[1]) - float(target_a_xy[1])
    return math.sqrt(dx * dx + dy * dy) <= float(tolerance_m)


# Clamp one world-table XY target into the valid tabletop targeting region.
def clamp_llm_strike_target_xy(
    strike_target_xy: list[float],
    *,
    object_name: str,
    task_name: Optional[str] = None,
) -> list[float]:
    """Return one validated world-table XY strike target within the static tabletop bounds."""
    del task_name
    if len(strike_target_xy) != 2:
        raise ValueError("strike_target_xy must contain exactly two coordinates.")
    return [
        _clamp_table_target(float(strike_target_xy[0]), axis="x"),
        _clamp_table_target(float(strike_target_xy[1]), axis="y"),
    ]


# Build one semantic Lie swing spec toward an explicit tabletop XY target.
# Build the deterministic swing_v1 Lie spec from one requested tabletop strike target.
def build_llm_lie_spec_from_target_xy(
    *,
    object_name: str,
    task_name: Optional[str],
    pivot_point: list[float],
    strike_target_xy: list[float],
    horizontal_strike_clearance_m: float = _DEFAULT_LIE_HORIZONTAL_STRIKE_CLEARANCE_M,
    waypoint_table_clearance_m: float = _DEFAULT_LIE_WAYPOINT_TABLE_CLEARANCE_M,
) -> Dict[str, Any]:
    """Return a deterministic swing_v1 spec for one explicit tabletop strike target."""
    clamped_target_xy = clamp_llm_strike_target_xy(
        strike_target_xy,
        object_name=object_name,
        task_name=task_name,
    )
    return {
        "schema_version": "swing_v1",
        "verb": "swing_down",
        "task_frame": "world",
        "pivot_point": [float(value) for value in pivot_point],
        "strike_target_xy": clamped_target_xy,
        "execution_target_z": float(TABLE_TOP_Z + horizontal_strike_clearance_m),
        "swing_angle_rad": _DEFAULT_SWING_ANGLE_RAD,
        "duration_sec": _DEFAULT_SWING_DURATION_SEC,
        "num_samples": _DEFAULT_SWING_NUM_SAMPLES,
        "waypoint_table_clearance_m": float(waypoint_table_clearance_m),
        "hammer_swing_solver": "horizontal_head",
    }


# Build one semantic Lie swing spec toward a fixed named strike point.
def build_named_llm_lie_spec(
    *,
    object_name: str,
    task_name: Optional[str],
    pivot_point: list[float],
    target_name: str,
) -> Dict[str, Any]:
    """Return a deterministic swing_v1 spec for one named strike point."""
    points = get_named_strike_points(object_name=object_name, task_name=task_name)
    if target_name not in points:
        raise ValueError(
            f"Named strike point '{target_name}' is unavailable for {object_name}/{task_name}."
        )
    return build_llm_lie_spec_from_target_xy(
        object_name=object_name,
        task_name=task_name,
        pivot_point=pivot_point,
        strike_target_xy=points[target_name],
    )


# Build a deterministic mock direct-pose trajectory near the recorded reference.
def generate_mock_llm_only_goals(reference: GoalSourceArtifact) -> Dict[str, Any]:
    """Return a deterministic direct pose list for the llm_only baseline."""
    shifted = []
    for index, goal in enumerate(reference.goals):
        offset = 0.004 * __import__("math").sin(
            index / max(1.0, len(reference.goals) - 1.0) * __import__("math").pi
        )
        shifted.append([goal[0] + offset, goal[1], goal[2], goal[3], goal[4], goal[5], goal[6]])
    return {"goals": shifted, "duration_sec": reference.duration_sec}


# Build the structured direct-generation input context shared by OpenAI and saved artifacts.
def build_llm_only_prompt_context(
    args: EvalGoalSourcesArgs,
    reference: GoalSourceArtifact,
) -> Dict[str, Any]:
    """Return the target-conditioned context allowed for direct llm_only generation."""
    target_xy = (
        None if args.target_xy is None else [float(args.target_xy[0]), float(args.target_xy[1])]
    )
    pivot_point = (
        None
        if args.pivot_point is None
        else [float(args.pivot_point[0]), float(args.pivot_point[1]), float(args.pivot_point[2])]
    )
    return {
        "schema_version": "llm_only_target_conditioned_v1",
        "object_name": str(args.object_name),
        "task_name": str(args.task_name),
        "resolved_baseline_object": str(
            args.resolved_baseline_object
            or resolve_predefined_baseline_object_name(str(args.object_name))
        ),
        "start_pose": [float(value) for value in reference.metadata["start_pose"]],
        "start_context": {
            "pivot_point": pivot_point,
            "start_position": [float(value) for value in reference.metadata["start_pose"][:3]],
        },
        "target_xy": target_xy,
        "reference_duration_sec": float(reference.duration_sec),
        "reference_num_goals": len(reference.goals),
        "reference_goals": [[float(value) for value in goal] for goal in reference.goals],
        "static_strike_context": get_llm_static_strike_context(
            object_name=str(args.object_name),
            task_name=str(args.task_name),
        ),
    }


# Generate the timed swing spec for llm_lie mode from the selected backend.
def generate_llm_lie_spec(
    args: EvalGoalSourcesArgs,
    reference: GoalSourceArtifact,
) -> Dict[str, Any]:
    """Return an `swing_v1` spec from mock or OpenAI generation."""
    if args.llm_backend == "mock":
        return generate_mock_llm_lie_spec(reference)
    if args.llm_backend == "openai":
        model = args.llm_model or DEFAULT_OPENAI_MODEL
        return openai_json_response(
            system_prompt=(
                "Return JSON only. Emit schema swing_v1 for a hammer swing_down motion with keys: "
                "schema_version, verb, task_frame, pivot_point, strike_target_xy, "
                "swing_angle_rad, duration_sec, num_samples. "
                "Set schema_version='swing_v1', verb='swing_down', task_frame='world'. "
                "Do not emit object_name, start semantic pose, or quaternion fields."
            ),
            user_prompt=args.instruction,
            model=model,
        )
    raise ValueError(f"Unsupported llm backend '{args.llm_backend}'.")


# Expand one base swing spec into deterministic strike-target variants for debugging.
def generate_llm_lie_specs(
    args: EvalGoalSourcesArgs,
    reference: GoalSourceArtifact,
) -> Dict[str, Dict[str, Any]]:
    """Return the deterministic `swing_v1` specs for supported llm_lie target variants."""
    base_spec = generate_llm_lie_spec(args, reference)
    variants: Dict[str, Dict[str, Any]] = {}
    for variant_name, strike_target_xy in _variant_strike_targets(
        base_spec, object_name=args.object_name, task_name=args.task_name
    ).items():
        variant_spec = deepcopy(base_spec)
        variant_spec["strike_target_xy"] = strike_target_xy
        variants[variant_name] = variant_spec
    return variants


# Clamp one table target coordinate into the usable tabletop bounds.
def _clamp_table_target(value: float, *, axis: str) -> float:
    """Return one tabletop coordinate clamped to the visible narrow-table bounds."""
    half_extent = _TABLE_HALF_EXTENT_X_M if axis == "x" else _TABLE_HALF_EXTENT_Y_M
    limit = half_extent - _TABLE_TARGET_MARGIN_M
    return max(-limit, min(limit, float(value)))


# Build the fixed debug strike targets used for llm_lie visualization variants.
def _variant_strike_targets(
    base_spec: Dict[str, Any],
    *,
    object_name: str = "claw_hammer",
    task_name: Optional[str] = "swing_down",
) -> Dict[str, list[float]]:
    """Return widely separated tabletop targets that induce distinct strike directions."""
    named_points = get_named_strike_points(object_name=object_name, task_name=task_name)
    if named_points:
        return named_points
    base_target = [float(value) for value in base_spec["strike_target_xy"]]
    shared_target_y = _clamp_table_target(base_target[1], axis="y")
    return {"target_a": [_clamp_table_target(base_target[0], axis="x"), shared_target_y]}


# Generate the direct pose-list payload for llm_only mode from the selected backend.
def generate_llm_only_payload(
    args: EvalGoalSourcesArgs,
    reference: GoalSourceArtifact,
) -> Dict[str, Any]:
    """Return a direct pose-list JSON payload from mock or OpenAI generation."""
    prompt_context = build_llm_only_prompt_context(args, reference)
    user_prompt = f"Context JSON: {json.dumps(prompt_context, sort_keys=True)}"
    if args.target_xy is None:
        user_prompt = f"Instruction: {args.instruction}\n{user_prompt}"
    if args.llm_backend == "mock":
        raw_payload = generate_mock_llm_only_goals(reference)
        raw_payload["prompt_context"] = prompt_context
        return raw_payload
    if args.llm_backend == "openai":
        model = args.llm_model or DEFAULT_OPENAI_MODEL
        raw_payload = openai_json_response(
            system_prompt=(
                "Return JSON only with keys: goals and duration_sec. goals must be a list of "
                "poses [x,y,z,qx,qy,qz,qw] in world frame. The first goal pose must equal the "
                "provided start_pose. Use reference_goals as a demonstration of good timing, "
                "clearance, smoothness, and orientation evolution, then directly produce a new "
                "goal pose list adapted to the explicit target_xy. Output exactly "
                "reference_num_goals poses and set duration_sec to reference_duration_sec. "
                "Use only the structured context JSON. Do not emit analytic Lie outputs, "
                "compiled goals, repaired paths, or clamped trajectories."
            ),
            user_prompt=user_prompt,
            model=model,
        )
        raw_payload["prompt_context"] = prompt_context
        return raw_payload
    raise ValueError(f"Unsupported llm backend '{args.llm_backend}'.")
