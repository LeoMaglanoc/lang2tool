"""Trajectory validation, interpolation, and quaternion helpers."""

from __future__ import annotations

import math
from typing import List, Sequence


# Normalize a quaternion from XYZW into unit XYZW form.
def normalize_quaternion_xyzw(quaternion_xyzw: Sequence[float]) -> List[float]:
    """Return a normalized XYZW quaternion."""
    if len(quaternion_xyzw) != 4:
        raise ValueError("Quaternion must contain exactly 4 elements.")
    qx, qy, qz, qw = [float(value) for value in quaternion_xyzw]
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-8:
        raise ValueError("Quaternion norm must be positive.")
    return [qx / norm, qy / norm, qz / norm, qw / norm]


# Validate and normalize an SE(3) pose list in `[x, y, z, qx, qy, qz, qw]` format.
def validate_pose_sequence(goals: Sequence[Sequence[float]]) -> List[List[float]]:
    """Return a normalized Nx7 pose list with unit quaternions."""
    if not goals:
        raise ValueError("Pose sequence must be non-empty.")
    validated: List[List[float]] = []
    for index, goal in enumerate(goals):
        if len(goal) != 7:
            raise ValueError(f"Goal {index} must have 7 elements.")
        pose = [float(value) for value in goal]
        if any(not math.isfinite(value) for value in pose):
            raise ValueError(f"Goal {index} contains non-finite values.")
        pose[3:] = normalize_quaternion_xyzw(pose[3:])
        validated.append(pose)
    return validated


# Convert a normalized XYZW quaternion into a 3x3 rotation matrix.
def quaternion_to_matrix_xyzw(quaternion_xyzw: Sequence[float]) -> List[List[float]]:
    """Return a rotation matrix for the given XYZW quaternion."""
    qx, qy, qz, qw = normalize_quaternion_xyzw(quaternion_xyzw)
    return [
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ],
        [
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ],
        [
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ]


# Convert a rotation matrix into a normalized XYZW quaternion.
def matrix_to_quaternion_xyzw(matrix: Sequence[Sequence[float]]) -> List[float]:
    """Return a normalized XYZW quaternion for a 3x3 rotation matrix."""
    trace = float(matrix[0][0] + matrix[1][1] + matrix[2][2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2][1] - matrix[1][2]) / scale
        qy = (matrix[0][2] - matrix[2][0]) / scale
        qz = (matrix[1][0] - matrix[0][1]) / scale
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        qw = (matrix[2][1] - matrix[1][2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0][1] + matrix[1][0]) / scale
        qz = (matrix[0][2] + matrix[2][0]) / scale
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        qw = (matrix[0][2] - matrix[2][0]) / scale
        qx = (matrix[0][1] + matrix[1][0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1][2] + matrix[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        qw = (matrix[1][0] - matrix[0][1]) / scale
        qx = (matrix[0][2] + matrix[2][0]) / scale
        qy = (matrix[1][2] + matrix[2][1]) / scale
        qz = 0.25 * scale
    return normalize_quaternion_xyzw([qx, qy, qz, qw])


# Interpolate linearly between two scalar values.
def lerp(start: float, end: float, fraction: float) -> float:
    """Return the linear interpolation between two scalars."""
    return float(start) + (float(end) - float(start)) * float(fraction)


# Compute Euclidean path length over a pose sequence using only translation.
def path_length_m(goals: Sequence[Sequence[float]]) -> float:
    """Return the total translational path length of a pose sequence."""
    total = 0.0
    for index in range(1, len(goals)):
        dx = float(goals[index][0]) - float(goals[index - 1][0])
        dy = float(goals[index][1]) - float(goals[index - 1][1])
        dz = float(goals[index][2]) - float(goals[index - 1][2])
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


# Interpolate one pose list to a fixed sample count using translation lerp and quaternion nlerp.
def resample_goals(goals: Sequence[Sequence[float]], target_count: int) -> List[List[float]]:
    """Return a pose sequence resampled to the requested sample count."""
    if target_count < 1:
        raise ValueError("target_count must be positive.")
    normalized = validate_pose_sequence(goals)
    if len(normalized) == target_count:
        return normalized
    if len(normalized) == 1:
        return [list(normalized[0]) for _ in range(target_count)]

    resampled: List[List[float]] = []
    max_index = len(normalized) - 1
    for sample_index in range(target_count):
        progress = 0.0 if target_count == 1 else sample_index / float(target_count - 1)
        position = progress * max_index
        lower_index = min(int(math.floor(position)), max_index - 1)
        upper_index = lower_index + 1
        fraction = position - lower_index
        lower = normalized[lower_index]
        upper = normalized[upper_index]
        pose = [lerp(lower[axis], upper[axis], fraction) for axis in range(3)]
        q0 = lower[3:]
        q1 = upper[3:]
        dot = sum(q0[i] * q1[i] for i in range(4))
        if dot < 0.0:
            q1 = [-value for value in q1]
        pose.extend(normalize_quaternion_xyzw([lerp(q0[i], q1[i], fraction) for i in range(4)]))
        resampled.append(pose)
    return resampled


# Compute the rotational distance in degrees between two XYZW quaternions.
def rotation_distance_deg(quaternion_a: Sequence[float], quaternion_b: Sequence[float]) -> float:
    """Return the absolute geodesic angle between two XYZW quaternions."""
    qa = normalize_quaternion_xyzw(quaternion_a)
    qb = normalize_quaternion_xyzw(quaternion_b)
    dot = abs(sum(qa[index] * qb[index] for index in range(4)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))
