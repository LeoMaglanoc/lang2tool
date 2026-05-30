"""Pure-torch reward helper functions for SimToolReal.

These functions are extracted from the monolithic env.py so that they can be
unit-tested independently of Isaac Lab or Isaac Gym.  All inputs/outputs are
plain ``torch.Tensor`` objects; no simulator state is accessed here.
"""

from __future__ import annotations

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Lifting reward
# ---------------------------------------------------------------------------


def compute_lifting_reward(
    object_pos_z: Tensor,
    object_init_z: Tensor,
    lifted_object: Tensor,
    lifting_bonus_threshold: float,
    lifting_bonus: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reward for lifting the object off the table.

    Args:
        object_pos_z: Current object z-position, shape (N,).
        object_init_z: Initial (table) object z-position, shape (N,).
        lifted_object: Boolean flag, True if object was already lifted, shape (N,).
        lifting_bonus_threshold: Height in metres above init_z to trigger bonus.
        lifting_bonus: One-time bonus reward for crossing the threshold.

    Returns:
        lifting_rew: Continuous height reward (zeroed after threshold), shape (N,).
        lift_bonus_rew: One-time bonus when threshold first crossed, shape (N,).
        lifted_object: Updated lifted flag, shape (N,).
    """
    # Offset by 0.05 so small lifts register immediately (matches legacy env)
    z_lift = 0.05 + object_pos_z - object_init_z
    lifting_rew = torch.clip(z_lift, 0, 0.5)

    new_lifted = (z_lift > lifting_bonus_threshold) | lifted_object
    just_lifted = new_lifted & ~lifted_object
    lift_bonus_rew = lifting_bonus * just_lifted.float()

    # Stop height reward once threshold crossed
    lifting_rew = lifting_rew * (~new_lifted).float()

    return lifting_rew, lift_bonus_rew, new_lifted


# ---------------------------------------------------------------------------
# Keypoint reward
# ---------------------------------------------------------------------------


def compute_keypoint_reward(
    keypoints_max_dist: Tensor,
    closest_keypoint_max_dist: Tensor,
    lifted_object: Tensor,
) -> tuple[Tensor, Tensor]:
    """Delta reward for reducing the maximum keypoint distance to the goal.

    Args:
        keypoints_max_dist: Current max keypoint dist to goal, shape (N,).
        closest_keypoint_max_dist: Best (smallest) max dist achieved so far, shape (N,).
        lifted_object: Boolean, whether object has been lifted, shape (N,).

    Returns:
        keypoint_rew: Reward tensor, shape (N,).
        updated_closest: Updated closest_keypoint_max_dist, shape (N,).
    """
    # Positive if we improved over the best, 0 otherwise
    delta = closest_keypoint_max_dist - keypoints_max_dist
    updated_closest = torch.minimum(closest_keypoint_max_dist, keypoints_max_dist)

    # Only give keypoint reward after the object is lifted
    keypoint_rew = torch.clip(delta, 0, 10) * lifted_object.float()

    return keypoint_rew, updated_closest


# ---------------------------------------------------------------------------
# Near-goal / success
# ---------------------------------------------------------------------------


def compute_near_goal_success(
    keypoints_max_dist: Tensor,
    near_goal_steps: Tensor,
    success_tolerance: float,
    keypoint_scale: float,
    success_steps: int,
    force_consecutive: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Determine near-goal steps and success flags.

    Args:
        keypoints_max_dist: Max keypoint distance to goal, shape (N,).
        near_goal_steps: Accumulated near-goal step counts, shape (N,).
        success_tolerance: Distance threshold for success.
        keypoint_scale: Scale applied to success tolerance.
        success_steps: Number of steps needed to declare success.
        force_consecutive: If True, steps must be consecutive.

    Returns:
        near_goal: Boolean whether currently near goal, shape (N,).
        updated_steps: Updated near_goal_steps, shape (N,).
        is_success: Boolean episode success flag, shape (N,).
    """
    threshold = success_tolerance * keypoint_scale
    near_goal = keypoints_max_dist <= threshold

    if force_consecutive:
        # Reset to 0 if not near goal, else increment
        updated_steps = (near_goal_steps + near_goal.int()) * near_goal.int()
    else:
        updated_steps = near_goal_steps + near_goal.int()

    is_success = updated_steps >= success_steps

    return near_goal, updated_steps, is_success


# ---------------------------------------------------------------------------
# Action penalties
# ---------------------------------------------------------------------------


def compute_action_penalties(
    dof_vel: Tensor,
    num_arm_dofs: int,
    kuka_actions_penalty_scale: float,
    hand_actions_penalty_scale: float,
) -> tuple[Tensor, Tensor]:
    """Penalty terms on arm/hand joint velocities (legacy Gym parity).

    Args:
        dof_vel: Joint velocities, shape (N, num_hand_arm_dofs).
        num_arm_dofs: Number of arm DOFs (7 for iiwa14).
        kuka_actions_penalty_scale: Penalty coefficient for arm.
        hand_actions_penalty_scale: Penalty coefficient for hand.

    Returns:
        kuka_penalty: Arm action penalty, shape (N,).  (negative)
        hand_penalty: Hand action penalty, shape (N,).  (negative)
    """
    # Match Isaac Gym env.py: L1 penalty on absolute joint velocities.
    kuka_penalty = -kuka_actions_penalty_scale * torch.sum(
        torch.abs(dof_vel[:, :num_arm_dofs]), dim=-1
    )
    hand_penalty = -hand_actions_penalty_scale * torch.sum(
        torch.abs(dof_vel[:, num_arm_dofs:]), dim=-1
    )
    return kuka_penalty, hand_penalty
