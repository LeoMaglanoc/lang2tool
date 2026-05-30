"""Small helpers for runtime mode transitions."""

from __future__ import annotations

from .types import RuntimeMode, ToolRuntimeState


# Enter pre-release waiting mode with a bounded timeout before finger-open override.
def enter_pre_release(state: ToolRuntimeState, timeout_steps: int) -> None:
    """Transition to WAIT_PRE_RELEASE mode and initialize countdown."""
    state.mode = RuntimeMode.WAIT_PRE_RELEASE
    state.pre_release_steps_left = int(max(timeout_steps, 1))
    state.open_steps_left = 0


# Enter hand-opening mode for a fixed number of steps.
def enter_open_hand(state: ToolRuntimeState, open_steps: int) -> None:
    """Transition to OPEN_HAND mode and initialize step budget."""
    state.mode = RuntimeMode.OPEN_HAND
    state.open_steps_left = int(max(open_steps, 1))


# Enter frozen mode after release so policy does not immediately regrasp.
def enter_frozen(state: ToolRuntimeState) -> None:
    """Transition to FROZEN_AFTER_RELEASE mode."""
    state.mode = RuntimeMode.FROZEN_AFTER_RELEASE
    state.pre_release_steps_left = 0
    state.open_steps_left = 0
    state.meta.clear()


# Return runtime control to normal policy stepping.
def enter_policy(state: ToolRuntimeState) -> None:
    """Transition to POLICY mode and clear release counters."""
    state.mode = RuntimeMode.POLICY
    state.pre_release_steps_left = 0
    state.open_steps_left = 0
    state.meta.clear()
