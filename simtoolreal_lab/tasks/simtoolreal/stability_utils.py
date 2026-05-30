"""Utilities for stabilizing object state before hand-object interaction."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_pre_contact_stabilization_mask(
    *,
    has_interacted: Tensor,
    curr_fingertip_distances: Tensor,
    object_pos: Tensor,
    object_init_pos: Tensor,
    object_linvel: Tensor,
    object_angvel: Tensor,
    distance_threshold: float,
    max_drift: float,
    max_speed: float,
    min_allowed_z: float,
) -> tuple[Tensor, Tensor]:
    """Return (unstable_mask, updated_has_interacted) for pre-contact object stabilization."""
    min_dist = curr_fingertip_distances.min(dim=-1).values
    updated_has_interacted = has_interacted | (min_dist < distance_threshold)
    drift = torch.norm(object_pos - object_init_pos, dim=-1)
    speed = torch.norm(object_linvel, dim=-1) + torch.norm(object_angvel, dim=-1)
    below_allowed_z = object_pos[:, 2] < min_allowed_z
    unstable = (~updated_has_interacted) & (
        (drift > max_drift) | (speed > max_speed) | below_allowed_z
    )
    return unstable, updated_has_interacted
