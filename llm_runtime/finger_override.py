"""Finger-action override helpers for reliable tool release."""

from __future__ import annotations

import torch


# Convert desired open-hand DOF targets into normalized action values and override hand dims.
def apply_open_hand_override(action: torch.Tensor, env) -> torch.Tensor:
    """Return action with hand dimensions overridden to open-hand normalized commands."""
    out = action.clone()
    if not hasattr(env, "num_hand_arm_dofs"):
        return out
    hand_start = 7
    hand_end = int(env.num_hand_arm_dofs)
    if hand_end <= hand_start:
        return out

    lower = env.arm_hand_dof_lower_limits[hand_start:hand_end]
    upper = env.arm_hand_dof_upper_limits[hand_start:hand_end]
    # Use lower-limit posture as the release/open target.
    target = lower
    denom = torch.clamp(upper - lower, min=1e-6)
    normalized = torch.clamp((2.0 * (target - lower) / denom) - 1.0, -1.0, 1.0)
    out[:, hand_start:hand_end] = normalized.unsqueeze(0).repeat(out.shape[0], 1)
    return out
