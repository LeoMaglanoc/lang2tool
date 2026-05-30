"""Semantic swing spec validation and Lie-group compilation."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from llm_runtime.semantic_pose import get_object_pose_semantics

from .trajectory import matrix_to_quaternion_xyzw, validate_pose_sequence

_WORLD_UP = (0.0, 0.0, 1.0)
_WORLD_DOWN = (0.0, 0.0, -1.0)
_TABLE_SURFACE_Z = 0.53
_DEFAULT_EXECUTION_TARGET_Z = _TABLE_SURFACE_Z + 0.05
_DEFAULT_WAYPOINT_TABLE_CLEARANCE_M = 0.05
_HAMMER_SWING_SOLVER = "horizontal_head"


# Normalize a 3D vector and reject degenerate inputs.
def normalize3(vector: Sequence[float], *, name: str) -> List[float]:
    """Return a normalized 3D vector."""
    if len(vector) != 3:
        raise ValueError(f"{name} must be a 3-element vector.")
    values = [float(value) for value in vector]
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite numbers.")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-8:
        raise ValueError(f"{name} must be non-zero.")
    return [value / norm for value in values]


# Compute a 3D dot product.
def dot3(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the dot product of two 3D vectors."""
    return sum(float(left[index]) * float(right[index]) for index in range(3))


# Compute a 3D cross product.
def cross3(left: Sequence[float], right: Sequence[float]) -> List[float]:
    """Return the cross product of two 3D vectors."""
    return [
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    ]


# Project a vector onto the plane orthogonal to the provided normal.
def project_to_plane(vector: Sequence[float], plane_normal: Sequence[float]) -> List[float]:
    """Return the component of vector orthogonal to plane_normal."""
    scale = dot3(vector, plane_normal)
    return [float(vector[index]) - scale * float(plane_normal[index]) for index in range(3)]


# Convert a basis with column vectors into a 3x3 matrix.
def basis_matrix(
    first: Sequence[float], second: Sequence[float], third: Sequence[float]
) -> List[List[float]]:
    """Return a column-basis matrix from three basis vectors."""
    return [
        [float(first[0]), float(second[0]), float(third[0])],
        [float(first[1]), float(second[1]), float(third[1])],
        [float(first[2]), float(second[2]), float(third[2])],
    ]


# Transpose a 3x3 matrix.
def transpose3(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    """Return the transpose of a 3x3 matrix."""
    return [[float(matrix[column][row]) for column in range(3)] for row in range(3)]


# Multiply two 3x3 matrices.
def matrix_multiply(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> List[List[float]]:
    """Return the matrix product of two 3x3 matrices."""
    return [
        [
            sum(float(left[row][k]) * float(right[k][column]) for k in range(3))
            for column in range(3)
        ]
        for row in range(3)
    ]


# Rotate a 3D vector by a 3x3 matrix.
def matrix_vector_multiply(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> List[float]:
    """Return the rotated 3D vector."""
    return [
        sum(float(matrix[row][column]) * float(vector[column]) for column in range(3))
        for row in range(3)
    ]


# Create an SO(3) rotation matrix from an axis-angle specification.
def rotation_from_axis_angle(axis: Sequence[float], angle_rad: float) -> List[List[float]]:
    """Return the 3x3 rotation matrix for an axis-angle pair."""
    x_axis, y_axis, z_axis = axis
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    one_minus_cosine = 1.0 - cosine
    return [
        [
            cosine + x_axis * x_axis * one_minus_cosine,
            x_axis * y_axis * one_minus_cosine - z_axis * sine,
            x_axis * z_axis * one_minus_cosine + y_axis * sine,
        ],
        [
            y_axis * x_axis * one_minus_cosine + z_axis * sine,
            cosine + y_axis * y_axis * one_minus_cosine,
            y_axis * z_axis * one_minus_cosine - x_axis * sine,
        ],
        [
            z_axis * x_axis * one_minus_cosine - y_axis * sine,
            z_axis * y_axis * one_minus_cosine + x_axis * sine,
            cosine + z_axis * z_axis * one_minus_cosine,
        ],
    ]


# Interpolate linearly between two scalar values.
def lerp(start: float, end: float, fraction: float) -> float:
    """Return the linear interpolation between two scalars."""
    return float(start) + (float(end) - float(start)) * float(fraction)


# Build the hammer's canonical upright orientation from semantic axes and strike target geometry.
def canonical_upright_rotation(
    object_name: str,
    strike_direction: Sequence[float],
) -> List[List[float]]:
    """Return the world rotation matrix for the primitive's canonical upright pose."""
    semantics = get_object_pose_semantics(object_name)
    head_axis_local = normalize3(semantics.head_axis_local, name="head_axis_local")
    strike_face_seed_local = project_to_plane(semantics.strike_face_normal_local, head_axis_local)
    strike_face_axis_local = normalize3(strike_face_seed_local, name="strike_face_normal_local")
    local_binormal = normalize3(
        cross3(head_axis_local, strike_face_axis_local), name="local_binormal"
    )

    strike_axis_world = normalize3(strike_direction, name="strike_direction")
    strike_face_axis_world = normalize3(strike_axis_world, name="strike_face_axis_world")
    world_binormal = normalize3(cross3(_WORLD_UP, strike_face_axis_world), name="world_binormal")

    local_basis = basis_matrix(head_axis_local, strike_face_axis_local, local_binormal)
    world_basis = basis_matrix(_WORLD_UP, strike_face_axis_world, world_binormal)
    return matrix_multiply(world_basis, transpose3(local_basis))


# Derive the strike direction from a pivot point and table target coordinate.
def strike_direction_from_target(
    pivot_point: Sequence[float], strike_target_xy: Sequence[float]
) -> List[float]:
    """Return the normalized horizontal direction from pivot to the strike target."""
    return normalize3(
        [
            float(strike_target_xy[0]) - float(pivot_point[0]),
            float(strike_target_xy[1]) - float(pivot_point[1]),
            0.0,
        ],
        name="strike_target_xy",
    )


# Return the semantic strike point defined in the object's local frame.
def semantic_strike_point_local(object_name: str) -> List[float]:
    """Return the object's semantic strike point in local coordinates."""
    semantics = get_object_pose_semantics(object_name)
    return validate_vector3(list(semantics.strike_point_local), name="strike_point_local")


# Return the semantic swing support point defined in the object's local frame.
def semantic_swing_support_point_local(object_name: str) -> List[float]:
    """Return the object's semantic swing support point in local coordinates."""
    semantics = get_object_pose_semantics(object_name)
    return validate_vector3(
        list(semantics.swing_support_point_local), name="swing_support_point_local"
    )


# Return the local support points that must stay above the table during a Lie swing.
def semantic_swing_clearance_points_local(object_name: str) -> List[List[float]]:
    """Return the object's clearance support points in local coordinates."""
    semantics = get_object_pose_semantics(object_name)
    return [
        validate_vector3(list(point), name="swing_clearance_points_local")
        for point in semantics.swing_clearance_points_local
    ]


# Solve the upright tool-origin offset that makes the final strike point hit the table target.
# Solve the upright object-origin offset that lands the strike point on the lifted target.
def solve_upright_offset_from_strike_target(
    *,
    object_name: str,
    pivot_point: Sequence[float],
    strike_target_xy: Sequence[float],
    execution_target_z: float,
    swing_axis: Sequence[float],
    swing_angle_rad: float,
    base_rotation: Sequence[Sequence[float]],
) -> List[float]:
    """Return the upright pivot-to-tool offset implied by the semantic strike point target."""
    target_world = [
        float(strike_target_xy[0]),
        float(strike_target_xy[1]),
        float(execution_target_z),
    ]
    strike_point_world_upright = matrix_vector_multiply(
        base_rotation, semantic_strike_point_local(object_name)
    )
    inverse_final_rotation = rotation_from_axis_angle(swing_axis, -float(swing_angle_rad))
    rotated_target_offset = matrix_vector_multiply(
        inverse_final_rotation,
        [float(target_world[index]) - float(pivot_point[index]) for index in range(3)],
    )
    return [
        float(rotated_target_offset[index]) - float(strike_point_world_upright[index])
        for index in range(3)
    ]


# Validate the minimal semantic `swing_v1` spec.
# Validate and normalize one swing_v1 Lie spec before compilation.
def validate_llm_lie_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized `swing_v1` spec for semantic swing compilation."""
    if spec.get("schema_version") != "swing_v1":
        raise ValueError("schema_version must equal 'swing_v1'.")
    if spec.get("verb") != "swing_down":
        raise ValueError("verb must equal 'swing_down'.")
    if spec.get("task_frame") != "world":
        raise ValueError("task_frame must equal 'world' in the MVP.")

    num_samples = int(spec.get("num_samples", 0))
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2.")
    duration_sec = float(spec.get("duration_sec", 0.0))
    if duration_sec <= 0.0:
        raise ValueError("duration_sec must be positive.")
    swing_angle_rad = float(spec.get("swing_angle_rad", 0.0))
    if not math.isfinite(swing_angle_rad) or swing_angle_rad <= 0.0:
        raise ValueError("swing_angle_rad must be positive.")
    execution_target_z = float(spec.get("execution_target_z", _DEFAULT_EXECUTION_TARGET_Z))
    if not math.isfinite(execution_target_z):
        raise ValueError("execution_target_z must be finite.")
    waypoint_table_clearance_m = float(
        spec.get("waypoint_table_clearance_m", _DEFAULT_WAYPOINT_TABLE_CLEARANCE_M)
    )
    if not math.isfinite(waypoint_table_clearance_m) or waypoint_table_clearance_m < 0.0:
        raise ValueError("waypoint_table_clearance_m must be finite and non-negative.")
    return {
        "schema_version": "swing_v1",
        "verb": "swing_down",
        "task_frame": "world",
        "pivot_point": validate_vector3(spec.get("pivot_point"), name="pivot_point"),
        "strike_target_xy": validate_vector2(spec.get("strike_target_xy"), name="strike_target_xy"),
        "execution_target_z": execution_target_z,
        "swing_angle_rad": swing_angle_rad,
        "duration_sec": duration_sec,
        "num_samples": num_samples,
        "waypoint_table_clearance_m": waypoint_table_clearance_m,
        "hammer_swing_solver": _HAMMER_SWING_SOLVER,
    }


# Validate 3D point or offset fields without forcing unit length.
def validate_vector3(values: Any, *, name: str) -> List[float]:
    """Return a finite 3D vector."""
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"{name} must be a 3-element list.")
    vector = [float(value) for value in values]
    if any(not math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain only finite numbers.")
    return vector


# Validate 2D point fields used for table targets.
def validate_vector2(values: Any, *, name: str) -> List[float]:
    """Return a finite 2D vector."""
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"{name} must be a 2-element list.")
    vector = [float(value) for value in values]
    if any(not math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain only finite numbers.")
    return vector


# Lift one world-frame pose so the hammer's clearance support points stay above the table.
def lift_pose_above_table_clearance(
    *,
    object_name: str,
    position: Sequence[float],
    rotation_matrix: Sequence[Sequence[float]],
    minimum_clearance_m: float,
) -> List[float]:
    """Return a lifted world position whose support points all clear the table."""
    clearance_points_local = semantic_swing_clearance_points_local(object_name)
    min_support_z = min(
        float(position[2]) + matrix_vector_multiply(rotation_matrix, point_local)[2]
        for point_local in clearance_points_local
    )
    required_min_z = float(_TABLE_SURFACE_Z + minimum_clearance_m)
    lift_delta_z = max(0.0, required_min_z - min_support_z)
    return [float(position[0]), float(position[1]), float(position[2] + lift_delta_z)]


# Solve a horizontal-head final pose with strike contact and face-down hammer orientation.
def solve_horizontal_head_swing(
    *,
    object_name: str,
    strike_target_xy: Sequence[float],
    execution_target_z: float,
    requested_pivot_point: Sequence[float],
) -> Dict[str, object]:
    """Return horizontal-head swing geometry with face-down strike semantics."""
    requested_direction = strike_direction_from_target(requested_pivot_point, strike_target_xy)
    semantics = get_object_pose_semantics(object_name)
    head_axis_local = normalize3(semantics.head_axis_local, name="head_axis_local")
    strike_face_axis_local = normalize3(
        project_to_plane(semantics.strike_face_normal_local, head_axis_local),
        name="strike_face_normal_local",
    )
    strike_point_local = semantic_strike_point_local(object_name)
    support_point_local = semantic_swing_support_point_local(object_name)

    local_third = normalize3(cross3(head_axis_local, strike_face_axis_local), name="local_third")
    world_third = normalize3(cross3(requested_direction, _WORLD_DOWN), name="world_third")
    local_basis = basis_matrix(head_axis_local, strike_face_axis_local, local_third)
    world_basis = basis_matrix(requested_direction, _WORLD_DOWN, world_third)
    final_rotation = matrix_multiply(world_basis, transpose3(local_basis))

    strike_target_world = [
        float(strike_target_xy[axis]) if axis < 2 else float(execution_target_z)
        for axis in range(3)
    ]
    rotated_strike = matrix_vector_multiply(final_rotation, strike_point_local)
    final_position = [strike_target_world[axis] - rotated_strike[axis] for axis in range(3)]
    support_point_world = [
        final_position[axis] + matrix_vector_multiply(final_rotation, support_point_local)[axis]
        for axis in range(3)
    ]

    return {
        "support_point_world": support_point_world,
        "swing_axis": normalize3(cross3(_WORLD_UP, requested_direction), name="swing_axis"),
        "final_rotation": final_rotation,
        "support_point_local": support_point_local,
    }


# Compile the semantic swing spec into a pose list for one object.
# Compile one semantic Lie swing into table-safe world-frame pose goals.
def compile_llm_lie_goals(spec: Dict[str, Any], *, object_name: str) -> List[List[float]]:
    """Return a world-frame pose list for the provided semantic `swing_v1` swing spec."""
    validated = validate_llm_lie_spec(spec)
    return compile_horizontal_head_llm_lie_goals(validated, object_name=object_name)


# Compile the horizontal-head swing around the lifted semantic support point.
def compile_horizontal_head_llm_lie_goals(
    spec: Dict[str, Any], *, object_name: str
) -> List[List[float]]:
    """Return pose goals for the horizontal-head hammer swing solver."""
    solved_geometry = solve_horizontal_head_swing(
        object_name=object_name,
        strike_target_xy=spec["strike_target_xy"],
        execution_target_z=float(spec["execution_target_z"]),
        requested_pivot_point=spec["pivot_point"],
    )
    support_point_world = solved_geometry["support_point_world"]
    support_point_local = solved_geometry["support_point_local"]
    swing_axis = solved_geometry["swing_axis"]
    final_rotation = solved_geometry["final_rotation"]
    num_samples = int(spec["num_samples"])

    goals: List[List[float]] = []
    for sample_index in range(num_samples):
        fraction = sample_index / float(num_samples - 1)
        swing_angle = lerp(-0.5 * math.pi, 0.0, fraction)
        rotation_matrix = matrix_multiply(
            rotation_from_axis_angle(swing_axis, swing_angle),
            final_rotation,
        )
        quaternion_xyzw = matrix_to_quaternion_xyzw(rotation_matrix)
        rotated_support = matrix_vector_multiply(rotation_matrix, support_point_local)
        position = [float(support_point_world[axis]) - rotated_support[axis] for axis in range(3)]
        goals.append(position + quaternion_xyzw)
    return validate_pose_sequence(goals)
