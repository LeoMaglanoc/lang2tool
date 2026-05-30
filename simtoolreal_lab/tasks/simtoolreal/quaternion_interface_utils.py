"""Quaternion conversion helpers for policy/sim interface parity."""

from __future__ import annotations

import torch
from torch import Tensor


# Convert quaternion tensor from Isaac Lab/world convention (wxyz) to policy convention (xyzw).
def quat_wxyz_to_xyzw(quat_wxyz: Tensor) -> Tensor:
    """Return quaternion with components reordered from wxyz -> xyzw."""
    return torch.cat([quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]], dim=-1)


# Convert quaternion tensor from policy convention (xyzw) to Isaac Lab/world convention (wxyz).
def quat_xyzw_to_wxyz(quat_xyzw: Tensor) -> Tensor:
    """Return quaternion with components reordered from xyzw -> wxyz."""
    return torch.cat([quat_xyzw[..., 3:4], quat_xyzw[..., 0:3]], dim=-1)
