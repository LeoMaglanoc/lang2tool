"""Concrete SE(3) pose sequence converter using pure-numpy rotation math."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..errors import ConverterError
from .schema import GeometricGoalV1, SE3PoseSequence

# Default object reference pose: center of target volume, identity rotation.
_DEFAULT_REF_POSE: Tuple[float, ...] = (0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 1.0)

# Numerical tolerance for degeneracy detection.
_EPS = 1e-6


# Normalize a vector, raising ConverterError if near-zero.
def _normalize(v: np.ndarray, name: str) -> np.ndarray:
    """Return unit vector or raise ConverterError if the input is near-zero."""
    norm = float(np.linalg.norm(v))
    if norm < _EPS:
        raise ConverterError(f"Vector '{name}' is near-zero and cannot be normalized.")
    return v / norm


# Convert a 3×3 rotation matrix to a unit quaternion via Shepperd's method.
def _rot_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (qx, qy, qz, qw) from a 3×3 rotation matrix using Shepperd's method."""
    # Shepperd's method: pick the largest diagonal trace case for numerical stability.
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    # Normalize to guard against floating-point drift.
    quat = np.array([x, y, z, w])
    quat /= np.linalg.norm(quat)
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


# Build rotation matrix mapping object-frame directions to canonical world axes.
def _build_rotation(approach: np.ndarray, tool_axis: np.ndarray) -> np.ndarray:
    """Return rotation R such that R@approach=[0,0,-1] and R@tool_ortho=[1,0,0].

    Raises ConverterError if approach and tool_axis are parallel (degenerate frame).
    """
    a = _normalize(approach, "approach_direction_object")
    t = _normalize(tool_axis, "tool_axis_object")

    # Orthogonalize tool axis against approach direction.
    t_ortho = t - float(np.dot(t, a)) * a
    ortho_norm = float(np.linalg.norm(t_ortho))
    if ortho_norm < _EPS:
        raise ConverterError(
            "approach_direction_object and tool_axis_object are parallel; "
            "cannot construct a valid rotation frame."
        )
    t_ortho = t_ortho / ortho_norm

    # Side axis closes the right-hand coordinate system.
    s = np.cross(-a, t_ortho)

    # C maps canonical axes to object-frame: C = [t_ortho | s | -a].
    # R = C^T maps object-frame to canonical: R@a = [0,0,-1], R@t_ortho = [1,0,0].
    C = np.column_stack([t_ortho, s, -a])
    return C.T


class GeometricPoseConverter:
    """Convert validated GeometricGoalV1 parameters to a two-waypoint SE(3) sequence."""

    # Accept an optional 7-element world reference pose for the object.
    def __init__(
        self,
        object_ref_pose: Optional[Tuple[float, ...]] = None,
    ) -> None:
        """Initialize with an object reference pose (x,y,z,qx,qy,qz,qw).

        Defaults to (0,0,0.75,0,0,0,1) — center of target volume, identity rotation.
        """
        ref = object_ref_pose if object_ref_pose is not None else _DEFAULT_REF_POSE
        if len(ref) != 7:
            raise ValueError(f"object_ref_pose must have 7 elements, got {len(ref)}.")
        self._ref_pos = np.array(ref[:3], dtype=float)
        # Quaternion portion is stored for future extension; rotation is identity here.

    # Produce two waypoints: grasp orientation and lifted pose.
    def to_pose_sequence(self, params: GeometricGoalV1) -> SE3PoseSequence:
        """Return a 2-element SE(3) list: [grasp_pose, lifted_pose].

        Waypoint 1: object at ref_pos with orientation aligning approach→[0,0,-1].
        Waypoint 2: object lifted by lift_height_m along world +Z, same orientation.

        Raises ConverterError if approach and tool_axis are parallel.
        """
        approach = np.array(params.approach_direction_object, dtype=float)
        tool_axis = np.array(params.tool_axis_object, dtype=float)

        R = _build_rotation(approach, tool_axis)
        qx, qy, qz, qw = _rot_to_quat(R)

        # Waypoint 1: grasp orientation at reference position.
        pos1 = self._ref_pos.copy()
        wp1 = (
            float(pos1[0]),
            float(pos1[1]),
            float(pos1[2]),
            qx,
            qy,
            qz,
            qw,
        )

        # Waypoint 2: lifted by lift_height_m along world Z.
        pos2 = pos1 + np.array([0.0, 0.0, params.lift_height_m])
        wp2 = (
            float(pos2[0]),
            float(pos2[1]),
            float(pos2[2]),
            qx,
            qy,
            qz,
            qw,
        )

        return [wp1, wp2]
