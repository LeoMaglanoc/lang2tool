"""Pure-torch termination helpers for Isaac Lab/Gym parity."""

from __future__ import annotations

import torch
from torch import Tensor


# Compute terminated/truncated tensors to mirror legacy Gym reset semantics.
def compute_termination_and_truncation(
    object_pos_z: Tensor,
    curr_fingertip_distances: Tensor,
    lifted_object: Tensor,
    object_init_z: Tensor,
    episode_length_buf: Tensor,
    max_episode_length: int,
    successes: Tensor,
    is_success: Tensor,
    max_consecutive_successes: int | Tensor,
    reset_when_dropped: bool,
    hand_far_threshold: float = 1.5,
    object_fall_threshold_z: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Return (terminated, truncated) following legacy reset conditions."""
    device = object_pos_z.device
    ones = torch.ones_like(object_pos_z, dtype=torch.bool, device=device)
    zeros = torch.zeros_like(object_pos_z, dtype=torch.bool, device=device)

    object_z_low = torch.where(object_pos_z < object_fall_threshold_z, ones, zeros)
    hand_far_from_object = torch.where(
        curr_fingertip_distances.max(dim=-1).values > hand_far_threshold,
        ones,
        zeros,
    )

    if reset_when_dropped:
        dropped = torch.where(object_pos_z < object_init_z, ones, zeros) & lifted_object
    else:
        dropped = zeros

    if isinstance(max_consecutive_successes, Tensor):
        max_success_reached = torch.where(
            max_consecutive_successes > 0,
            successes >= max_consecutive_successes.float(),
            zeros,
        )
    elif max_consecutive_successes > 0:
        # Legacy Gym only terminates when accumulated successes reach the cap.
        # Per-step success (`is_success`) resets goal/progress, not the episode.
        max_success_reached = torch.where(successes >= max_consecutive_successes, ones, zeros)
    else:
        max_success_reached = zeros

    max_episode_length_reached = episode_length_buf >= max_episode_length

    terminated = object_z_low | hand_far_from_object | dropped | max_success_reached
    truncated = max_episode_length_reached
    return terminated, truncated
