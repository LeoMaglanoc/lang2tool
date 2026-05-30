"""Shared camera controls for Viser-based viewers."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_CAMERA_ZOOM_FRACTION = 0.1
DEFAULT_CAMERA_MIN_DISTANCE_M = 0.05


# Return one normalized camera position update that moves toward or away from the current look-at target.
def zoom_camera_position(
    position: Any,
    look_at: Any,
    *,
    zoom_in: bool,
    step_fraction: float = DEFAULT_CAMERA_ZOOM_FRACTION,
    min_distance_m: float = DEFAULT_CAMERA_MIN_DISTANCE_M,
) -> tuple[float, float, float] | None:
    """Return the next camera position for one zoom step, or `None` when unchanged."""
    position_array = np.asarray(position, dtype=float)
    look_at_array = np.asarray(look_at, dtype=float)
    offset = position_array - look_at_array
    distance = float(np.linalg.norm(offset))
    if distance <= 1e-9:
        return None
    clamped_step_fraction = max(0.0, min(0.95, float(step_fraction)))
    target_distance = (
        max(float(min_distance_m), distance * (1.0 - clamped_step_fraction))
        if zoom_in
        else distance * (1.0 + clamped_step_fraction)
    )
    direction = offset / distance
    next_position = look_at_array + direction * target_distance
    return tuple(float(value) for value in next_position)


# Apply one zoom step to all currently connected Viser clients.
def zoom_connected_viser_clients(
    server: Any,
    *,
    zoom_in: bool,
    step_fraction: float = DEFAULT_CAMERA_ZOOM_FRACTION,
    min_distance_m: float = DEFAULT_CAMERA_MIN_DISTANCE_M,
) -> None:
    """Move every connected Viser client camera closer to or farther from its look-at target."""
    for client in server.get_clients().values():
        next_position = zoom_camera_position(
            client.camera.position,
            client.camera.look_at,
            zoom_in=zoom_in,
            step_fraction=step_fraction,
            min_distance_m=min_distance_m,
        )
        if next_position is not None:
            client.camera.position = next_position
