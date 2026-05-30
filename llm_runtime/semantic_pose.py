"""Semantic pose ontology, per-object pose meaning, and deterministic pose math."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

_EPS = 1e-8
_WORLD_UP = (0.0, 0.0, 1.0)
_WORLD_DOWN = (0.0, 0.0, -1.0)
_WORLD_FORWARD = (0.0, 1.0, 0.0)


@dataclass(frozen=True)
class ObjectPoseSemantics:
    """Define semantic axes in the object's local frame."""

    primary_axis_local: Tuple[float, float, float]
    head_axis_local: Tuple[float, float, float]
    tip_axis_local: Tuple[float, float, float]
    face_normal_local: Tuple[float, float, float]
    strike_face_normal_local: Tuple[float, float, float]
    head_center_local: Tuple[float, float, float]
    strike_point_local: Tuple[float, float, float]
    swing_support_point_local: Tuple[float, float, float]
    swing_clearance_points_local: Tuple[Tuple[float, float, float], ...]


_DEFAULT_SEMANTICS = ObjectPoseSemantics(
    primary_axis_local=(1.0, 0.0, 0.0),
    head_axis_local=(1.0, 0.0, 0.0),
    tip_axis_local=(1.0, 0.0, 0.0),
    face_normal_local=(0.0, 0.0, 1.0),
    strike_face_normal_local=(0.0, 1.0, 0.0),
    head_center_local=(0.0, 0.0, 0.0),
    strike_point_local=(0.0, 0.0, 0.0),
    swing_support_point_local=(0.0, 0.0, 0.0),
    swing_clearance_points_local=((0.0, 0.0, 0.0),),
)

_CLAW_HAMMER_SEMANTICS = ObjectPoseSemantics(
    primary_axis_local=(1.0, 0.0, 0.0),
    head_axis_local=(1.0, 0.0, 0.0),
    tip_axis_local=(-1.0, 0.0, 0.0),
    face_normal_local=(0.0, 0.0, 1.0),
    strike_face_normal_local=(0.0, -1.0, 0.0),
    head_center_local=(0.123122, 0.0, 0.001475),
    strike_point_local=(0.123122, -0.038279, 0.001475),
    swing_support_point_local=(-0.0523742, 0.00921726, -0.00364191),
    swing_clearance_points_local=(
        (0.123122, -0.038279, 0.001475),
        (0.123122, 0.038279, 0.001475),
        (0.14262493, 0.0, -0.01443837),
        (-0.0523742, 0.0, -0.01443837),
    ),
)

_LONG_SCREWDRIVER_SEMANTICS = ObjectPoseSemantics(
    primary_axis_local=(1.0, 0.0, 0.0),
    head_axis_local=(-1.0, 0.0, 0.0),
    tip_axis_local=(1.0, 0.0, 0.0),
    face_normal_local=(0.0, 0.0, 1.0),
    strike_face_normal_local=(1.0, 0.0, 0.0),
    head_center_local=(-0.045, 0.0, 0.0),
    strike_point_local=(0.19718152, -0.00140855, -0.0027986),
    swing_support_point_local=(0.19718152, -0.00140855, -0.0027986),
    swing_clearance_points_local=(
        (-0.045, 0.0, 0.0),
        (0.0, 0.0, 0.012),
        (0.045, 0.0, 0.0),
        (0.19718152, -0.00140855, -0.0027986),
    ),
)

_MALLET_HAMMER_SEMANTICS = ObjectPoseSemantics(
    primary_axis_local=(1.0, 0.0, 0.0),
    head_axis_local=(1.0, 0.0, 0.0),
    tip_axis_local=(-1.0, 0.0, 0.0),
    face_normal_local=(0.0, 0.0, 1.0),
    strike_face_normal_local=(0.0, -1.0, 0.0),
    head_center_local=(0.105, 0.0, 0.0),
    strike_point_local=(0.19395672, -0.0462501, -0.02061698),
    swing_support_point_local=(-0.06529985, 0.00319191, -0.0122853),
    swing_clearance_points_local=(
        (0.19395672, -0.0462501, -0.02061698),
        (0.105, 0.032, 0.0),
        (0.132, 0.0, -0.018),
        (-0.062, 0.0, -0.018),
    ),
)

_CUBOID_HAMMER_V014_SEMANTICS = ObjectPoseSemantics(
    primary_axis_local=(1.0, 0.0, 0.0),
    head_axis_local=(1.0, 0.0, 0.0),
    tip_axis_local=(-1.0, 0.0, 0.0),
    face_normal_local=(0.0, 0.0, 1.0),
    strike_face_normal_local=(0.0, -1.0, 0.0),
    head_center_local=(0.09594095009006728, 0.0, 0.0),
    strike_point_local=(0.09594095009006728, -0.030644512266176062, 0.0),
    swing_support_point_local=(-0.07670341, 0.0, 0.0),
    swing_clearance_points_local=(
        (0.09594095009006728, -0.030644512266176062, 0.0),
        (0.09594095009006728, 0.030644512266176062, 0.0),
        (0.11517849469509452, 0.0, -0.02002089550206727),
        (-0.07670340548503954, 0.0, -0.02002089550206727),
    ),
)

_SHORT_SCREWDRIVER_SEMANTICS = ObjectPoseSemantics(
    primary_axis_local=(1.0, 0.0, 0.0),
    head_axis_local=(-1.0, 0.0, 0.0),
    tip_axis_local=(1.0, 0.0, 0.0),
    face_normal_local=(0.0, 0.0, 1.0),
    strike_face_normal_local=(1.0, 0.0, 0.0),
    head_center_local=(-0.034, 0.0, 0.0),
    strike_point_local=(0.13108414, 0.00201369, -0.00028784),
    swing_support_point_local=(0.13108414, 0.00201369, -0.00028784),
    swing_clearance_points_local=(
        (-0.034, 0.0, 0.0),
        (0.0, 0.0, 0.010),
        (0.034, 0.0, 0.0),
        (0.13108414, 0.00201369, -0.00028784),
    ),
)

_CYLINDER_SCREWDRIVER_V3009_SEMANTICS = ObjectPoseSemantics(
    primary_axis_local=(1.0, 0.0, 0.0),
    head_axis_local=(-1.0, 0.0, 0.0),
    tip_axis_local=(1.0, 0.0, 0.0),
    face_normal_local=(0.0, 0.0, 1.0),
    strike_face_normal_local=(1.0, 0.0, 0.0),
    head_center_local=(-0.0383025709325386, 0.0, 0.0),
    strike_point_local=(0.11591023, 0.0, 0.0),
    swing_support_point_local=(0.11591023, 0.0, 0.0),
    swing_clearance_points_local=(
        (-0.0383025709325386, 0.0, 0.0),
        (0.0, 0.0, 0.0342156961081288),
        (0.07710640086173598, 0.0, 0.0),
        (0.11591023, 0.0, 0.0),
    ),
)

OBJECT_POSE_SEMANTICS: Dict[str, ObjectPoseSemantics] = {
    "claw_hammer": _CLAW_HAMMER_SEMANTICS,
    "mallet_hammer": _MALLET_HAMMER_SEMANTICS,
    "cuboid_hammer_v014": _CUBOID_HAMMER_V014_SEMANTICS,
    "flat_spatula": _DEFAULT_SEMANTICS,
    "spoon_spatula": _DEFAULT_SEMANTICS,
    "handle_eraser": _DEFAULT_SEMANTICS,
    "flat_eraser": _DEFAULT_SEMANTICS,
    "red_brush": _DEFAULT_SEMANTICS,
    "blue_brush": _DEFAULT_SEMANTICS,
    "sharpie_marker": _DEFAULT_SEMANTICS,
    "staples_marker": _DEFAULT_SEMANTICS,
    "long_screwdriver": _LONG_SCREWDRIVER_SEMANTICS,
    "short_screwdriver": _SHORT_SCREWDRIVER_SEMANTICS,
    "cylinder_screwdriver_v3009": _CYLINDER_SCREWDRIVER_V3009_SEMANTICS,
}


# Normalize a 3D vector and reject near-zero magnitudes.
def _normalize3(v: Sequence[float]) -> Tuple[float, float, float]:
    """Return normalized vector for 3D inputs."""
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < _EPS:
        raise ValueError("Cannot normalize near-zero 3D vector.")
    return (float(v[0]) / n, float(v[1]) / n, float(v[2]) / n)


# Compute dot product for two 3D vectors.
def _dot3(a: Sequence[float], b: Sequence[float]) -> float:
    """Return dot product a.b for 3D vectors."""
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


# Compute cross product for two 3D vectors.
def _cross3(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """Return 3D cross product a x b."""
    return (
        float(a[1] * b[2] - a[2] * b[1]),
        float(a[2] * b[0] - a[0] * b[2]),
        float(a[0] * b[1] - a[1] * b[0]),
    )


# Multiply two xyzw quaternions with left-composition semantics.
def quat_mul_xyzw(qa: Sequence[float], qb: Sequence[float]) -> Tuple[float, float, float, float]:
    """Return quaternion product qa * qb in xyzw convention."""
    ax, ay, az, aw = [float(v) for v in qa]
    bx, by, bz, bw = [float(v) for v in qb]
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


# Return the quaternion conjugate for xyzw input.
def _quat_conjugate_xyzw(q: Sequence[float]) -> Tuple[float, float, float, float]:
    """Return conjugate quaternion for xyzw q."""
    return (-float(q[0]), -float(q[1]), -float(q[2]), float(q[3]))


# Rotate a 3D vector by an xyzw quaternion.
def quat_rotate_xyzw(q: Sequence[float], v: Sequence[float]) -> Tuple[float, float, float]:
    """Rotate vector v by unit quaternion q using q*v*q_conj."""
    vq = (float(v[0]), float(v[1]), float(v[2]), 0.0)
    rotated = quat_mul_xyzw(quat_mul_xyzw(q, vq), _quat_conjugate_xyzw(q))
    return (rotated[0], rotated[1], rotated[2])


# Construct the shortest-arc quaternion mapping unit vector a to unit vector b.
def _quat_from_two_unit_vectors(
    a: Sequence[float], b: Sequence[float]
) -> Tuple[float, float, float, float]:
    """Return delta quaternion rotating a onto b."""
    dot = max(-1.0, min(1.0, _dot3(a, b)))
    if dot > 1.0 - 1e-6:
        return (0.0, 0.0, 0.0, 1.0)
    if dot < -1.0 + 1e-6:
        # Pick a stable orthogonal axis when vectors are opposite.
        ortho = _cross3((1.0, 0.0, 0.0), a)
        if math.sqrt(_dot3(ortho, ortho)) < _EPS:
            ortho = _cross3((0.0, 1.0, 0.0), a)
        axis = _normalize3(ortho)
        return (axis[0], axis[1], axis[2], 0.0)
    cross = _cross3(a, b)
    q = (cross[0], cross[1], cross[2], 1.0 + dot)
    qn = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if qn < _EPS:
        return (0.0, 0.0, 0.0, 1.0)
    return (q[0] / qn, q[1] / qn, q[2] / qn, q[3] / qn)


# Return object pose semantics for a known object or a deterministic default.
def get_object_pose_semantics(object_name: str) -> ObjectPoseSemantics:
    """Return per-object semantic axis definitions."""
    return OBJECT_POSE_SEMANTICS.get(str(object_name), _DEFAULT_SEMANTICS)


# Build a JSON-safe payload describing semantic pose definitions for one object.
# Serialize the semantic object metadata that the LLM runtime exposes to tools.
def get_object_pose_semantics_payload(object_name: str) -> dict:
    """Return object semantic axis metadata for tool-calling context."""
    semantics = get_object_pose_semantics(object_name)
    return {
        "object_name": str(object_name),
        "axis_convention": "object_local_xyz",
        "quaternion_convention": "xyzw",
        "semantic_targets": [
            "upright",
            "flat",
            "head_down",
            "tip_forward",
            "face_table",
        ],
        "axes_local": {
            "primary_axis": list(semantics.primary_axis_local),
            "head_axis": list(semantics.head_axis_local),
            "tip_axis": list(semantics.tip_axis_local),
            "face_normal": list(semantics.face_normal_local),
            "strike_face_normal": list(semantics.strike_face_normal_local),
        },
        "points_local": {
            "head_center": list(semantics.head_center_local),
            "strike_point": list(semantics.strike_point_local),
            "swing_support_point": list(semantics.swing_support_point_local),
            "swing_clearance_points": [
                list(point) for point in semantics.swing_clearance_points_local
            ],
        },
    }


# Resolve the target world direction for the requested semantic pose target.
def _resolve_target_world_direction(
    semantic_target: str, current_axis_world: Sequence[float]
) -> Tuple[float, float, float]:
    """Return normalized world direction used by semantic target alignment."""
    if semantic_target == "upright":
        return _WORLD_UP
    if semantic_target == "head_down":
        return _WORLD_DOWN
    if semantic_target == "tip_forward":
        return _WORLD_FORWARD
    if semantic_target == "face_table":
        return _WORLD_DOWN
    if semantic_target == "flat":
        projected = (float(current_axis_world[0]), float(current_axis_world[1]), 0.0)
        if math.sqrt(_dot3(projected, projected)) < _EPS:
            projected = _WORLD_FORWARD
        return _normalize3(projected)
    raise ValueError(f"Unsupported semantic target: {semantic_target!r}")


# Build a quaternion delta that aligns the semantic axis for the requested target.
def compute_semantic_quat_delta(
    object_name: str, object_quat_xyzw: Sequence[float], semantic_target: str
) -> Tuple[float, float, float, float]:
    """Return delta quaternion to satisfy one ontology semantic target."""
    semantics = get_object_pose_semantics(object_name)
    if semantic_target in ("upright", "flat"):
        axis_local = semantics.primary_axis_local
    elif semantic_target == "head_down":
        axis_local = semantics.head_axis_local
    elif semantic_target == "tip_forward":
        axis_local = semantics.tip_axis_local
    elif semantic_target == "face_table":
        axis_local = semantics.face_normal_local
    else:
        raise ValueError(f"Unsupported semantic target: {semantic_target!r}")

    axis_local_n = _normalize3(axis_local)
    axis_world_n = _normalize3(quat_rotate_xyzw(object_quat_xyzw, axis_local_n))
    target_world = _resolve_target_world_direction(semantic_target, axis_world_n)
    return _quat_from_two_unit_vectors(axis_world_n, target_world)
