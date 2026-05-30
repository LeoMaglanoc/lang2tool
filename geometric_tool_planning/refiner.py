"""Deterministic MVP swing refiner for typed intent contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .contracts import IntentContractV0

_AMPLITUDE_TO_RADIUS_M = {
    "small": 0.10,
    "medium": 0.14,
    "large": 0.18,
}
_FIXED_ORIENTATION_XYZW = (0.0, 0.0, 0.0, 1.0)
_SWING_SAMPLE_COUNT = 8


# Store the deterministic waypoint plan returned by the MVP refiner.
@dataclass(frozen=True)
class SwingRefinerPlanV0:
    """Deterministic swing-refiner output for one typed intent contract."""

    goals: List[List[float]]
    metadata: Dict[str, float]


# Clamp one 3D point into the provided workspace bounds.
def _clamp_workspace_point(
    point_xyz: Tuple[float, float, float],
    workspace_bounds_xyz: Sequence[Sequence[float]],
) -> Tuple[float, float, float]:
    """Return a 3D point clamped to the configured workspace bounds."""
    lower = workspace_bounds_xyz[0]
    upper = workspace_bounds_xyz[1]
    return (
        max(float(lower[0]), min(float(upper[0]), float(point_xyz[0]))),
        max(float(lower[1]), min(float(upper[1]), float(point_xyz[1]))),
        max(float(lower[2]), min(float(upper[2]), float(point_xyz[2]))),
    )


# Convert normalized table UV coordinates into a world-frame tabletop target.
def _target_world_xy(
    target_hint_uv: Tuple[float, float],
    table_bounds_xy: Sequence[Sequence[float]],
) -> Tuple[float, float]:
    """Return a tabletop XY target interpolated from normalized UV coordinates."""
    lower = table_bounds_xy[0]
    upper = table_bounds_xy[1]
    return (
        float(lower[0]) + (float(upper[0]) - float(lower[0])) * float(target_hint_uv[0]),
        float(lower[1]) + (float(upper[1]) - float(lower[1])) * float(target_hint_uv[1]),
    )


# Build a deterministic swing trajectory centered on the repaired tabletop target.
def refine_swing_down(
    intent: IntentContractV0,
    runtime_context: dict,
) -> SwingRefinerPlanV0:
    """Return a deterministic fixed-orientation swing plan for one intent contract."""
    start_pose = [float(value) for value in runtime_context["start_pose"]]
    table_z = float(runtime_context["table_z"])
    table_bounds_xy = runtime_context["table_bounds_xy"]
    workspace_bounds_xyz = runtime_context["workspace_bounds_xyz"]

    target_x, target_y = _target_world_xy(intent.target_hint_uv, table_bounds_xy)
    floor_z = table_z + float(intent.clearance_m)
    radius_m = _AMPLITUDE_TO_RADIUS_M[intent.amplitude]
    direction_x = float(intent.direction_xy[0])

    goals: List[List[float]] = []
    for sample_index in range(_SWING_SAMPLE_COUNT):
        fraction = sample_index / float(_SWING_SAMPLE_COUNT - 1)
        sweep_x = -radius_m + (2.0 * radius_m * fraction)
        raw_point = (
            target_x + direction_x * sweep_x,
            target_y,
            max(floor_z, start_pose[2]),
        )
        clamped_point = _clamp_workspace_point(raw_point, workspace_bounds_xyz)
        goals.append(list(clamped_point) + list(_FIXED_ORIENTATION_XYZW))

    metadata = {
        "floor_z": floor_z,
        "radius_m": radius_m,
    }
    return SwingRefinerPlanV0(goals=goals, metadata=metadata)
