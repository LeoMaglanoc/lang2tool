"""Goal trajectory transforms for delta-based tool command execution."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

# Fixed startup camera used by viser/eval runtime.
_CAMERA_SPAWN_POS = (0.0, -1.0, 1.0)
_CAMERA_SPAWN_LOOK_AT = (0.0, 0.0, 0.5)
_WORLD_UP = (0.0, 0.0, 1.0)
_EPS = 1e-8


# Normalize a 3D vector with explicit degeneracy guard.
def _normalize3(v: Sequence[float]) -> Tuple[float, float, float]:
    """Return normalized 3D vector and raise ValueError when near-zero."""
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < _EPS:
        raise ValueError("Cannot normalize near-zero vector.")
    return (v[0] / n, v[1] / n, v[2] / n)


# Compute cross product for two 3D vectors.
def _cross(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """Return 3D cross product a x b."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


# Quaternion multiplication for xyzw quaternions.
def _quat_mul_xyzw(qa: Sequence[float], qb: Sequence[float]) -> Tuple[float, float, float, float]:
    """Return qa * qb with inputs in (qx,qy,qz,qw) convention."""
    ax, ay, az, aw = [float(v) for v in qa]
    bx, by, bz, bw = [float(v) for v in qb]
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


# Build delta quaternion from intrinsic XYZ Euler deltas.
def _quat_from_euler_xyz(
    droll: float, dpitch: float, dyaw: float
) -> Tuple[float, float, float, float]:
    """Return xyzw quaternion for intrinsic XYZ rotation delta."""
    hx = 0.5 * float(droll)
    hy = 0.5 * float(dpitch)
    hz = 0.5 * float(dyaw)
    sx, cx = math.sin(hx), math.cos(hx)
    sy, cy = math.sin(hy), math.cos(hy)
    sz, cz = math.sin(hz), math.cos(hz)
    # Intrinsic XYZ == qx * qy * qz composition.
    qx = (sx, 0.0, 0.0, cx)
    qy = (0.0, sy, 0.0, cy)
    qz = (0.0, 0.0, sz, cz)
    q = _quat_mul_xyzw(_quat_mul_xyzw(qx, qy), qz)
    qn = math.sqrt(sum(v * v for v in q))
    if qn < _EPS:
        return (0.0, 0.0, 0.0, 1.0)
    return (q[0] / qn, q[1] / qn, q[2] / qn, q[3] / qn)


# Public wrapper so semantic runtime paths can build deterministic euler deltas.
def quat_from_euler_xyz(
    droll: float, dpitch: float, dyaw: float
) -> Tuple[float, float, float, float]:
    """Return xyzw quaternion for intrinsic XYZ rotation deltas."""
    return _quat_from_euler_xyz(droll, dpitch, dyaw)


# Build fixed camera-spawn basis vectors in world frame.
def _camera_spawn_basis() -> (
    Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]
):
    """Return (right, up, forward) unit basis vectors for startup camera frame."""
    fwd = (
        _CAMERA_SPAWN_LOOK_AT[0] - _CAMERA_SPAWN_POS[0],
        _CAMERA_SPAWN_LOOK_AT[1] - _CAMERA_SPAWN_POS[1],
        _CAMERA_SPAWN_LOOK_AT[2] - _CAMERA_SPAWN_POS[2],
    )
    forward = _normalize3(fwd)
    right = _normalize3(_cross(forward, _WORLD_UP))
    up = _normalize3(_cross(right, forward))
    return right, up, forward


# Map camera-spawn local [right, up, forward] delta to world-frame XYZ delta.
def camera_spawn_delta_to_world(delta_translation_m: Sequence[float]) -> List[float]:
    """Convert local camera_spawn translation delta to world-frame [dx,dy,dz]."""
    right_m, up_m, forward_m = [float(v) for v in delta_translation_m]
    right, up, forward = _camera_spawn_basis()
    return [
        right_m * right[0] + up_m * up[0] + forward_m * forward[0],
        right_m * right[1] + up_m * up[1] + forward_m * forward[1],
        right_m * right[2] + up_m * up[2] + forward_m * forward[2],
    ]


# Apply a pose delta to all current goals in world frame with z safety clamp.
def shift_goals_pose_delta(
    goals: Iterable[Iterable[float]],
    delta_translation_m: Sequence[float],
    delta_euler_rad: Sequence[float],
    min_pose_z: float,
    delta_frame: str,
) -> List[List[float]]:
    """Return transformed 7D goals after translation+rotation delta composition."""
    if delta_frame != "camera_spawn":
        raise ValueError(f"Unsupported delta frame: {delta_frame}")
    world_delta = camera_spawn_delta_to_world(delta_translation_m)
    dq = _quat_from_euler_xyz(
        float(delta_euler_rad[0]),
        float(delta_euler_rad[1]),
        float(delta_euler_rad[2]),
    )
    return shift_goals_pose_quat_delta(
        goals=goals,
        world_delta_translation_m=world_delta,
        delta_quat_xyzw=dq,
        min_pose_z=min_pose_z,
    )


# Apply a world-frame translation and quaternion delta to all active goals.
def shift_goals_pose_quat_delta(
    goals: Iterable[Iterable[float]],
    world_delta_translation_m: Sequence[float],
    delta_quat_xyzw: Sequence[float],
    min_pose_z: float,
) -> List[List[float]]:
    """Return transformed 7D goals after world-translation and quaternion composition."""
    world_delta = [float(v) for v in world_delta_translation_m]
    dq = [float(v) for v in delta_quat_xyzw]

    shifted: List[List[float]] = []
    for goal in goals:
        g = [float(v) for v in goal]
        if len(g) != 7:
            continue
        g[0] += world_delta[0]
        g[1] += world_delta[1]
        g[2] = max(g[2] + world_delta[2], min_pose_z)
        qn = _quat_mul_xyzw(dq, g[3:7])
        qnorm = math.sqrt(sum(v * v for v in qn))
        if qnorm < _EPS:
            g[3:7] = [0.0, 0.0, 0.0, 1.0]
        else:
            g[3:7] = [qn[0] / qnorm, qn[1] / qnorm, qn[2] / qnorm, qn[3] / qnorm]
        shifted.append(g)
    return shifted
