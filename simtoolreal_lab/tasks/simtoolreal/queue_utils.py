"""Pure-torch queue helpers for delayed observations/actions."""

from __future__ import annotations

import torch
from torch import Tensor


# Update delay queue and fill at episode start to mirror legacy Gym behavior.
def update_delay_queue(queue: Tensor, current: Tensor, is_episode_start: Tensor) -> Tensor:
    """Return updated queue with index 0 set to current values."""
    queue = torch.where(
        is_episode_start[:, None, None],
        current[:, None, :].expand(-1, queue.shape[1], -1),
        queue,
    )
    queue = torch.roll(queue, 1, dims=1)
    queue[:, 0] = current
    return queue
