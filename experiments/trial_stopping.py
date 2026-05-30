"""Experiment-side trial stopping helpers for execution rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrialStopConfig:
    """Configuration for execution-trial stop evaluation."""

    timeout_sec: Optional[float]


@dataclass(frozen=True)
class TrialStopSignals:
    """Per-step signals consumed by execution-trial stop evaluation."""

    dropped: bool
    object_z_low: bool
    success_count: int
    success_target: int
    sim_time_sec: float
    signal_details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialStopResult:
    """Outcome of one execution-trial stop evaluation."""

    should_stop: bool
    reason: Optional[str] = None
    is_failure: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


# Decide whether one execution trial should terminate while reset signals stay passive.
def evaluate_trial_stop(
    signals: TrialStopSignals,
    config: TrialStopConfig,
) -> TrialStopResult:
    """Return terminal success/timeout state while carrying passive reset diagnostics."""
    details = {
        "dropped": bool(signals.dropped),
        "object_z_low": bool(signals.object_z_low),
        "success_count": int(signals.success_count),
        "success_target": int(signals.success_target),
        "sim_time_sec": float(signals.sim_time_sec),
        "timeout_sec": None if config.timeout_sec is None else float(config.timeout_sec),
    }
    details.update(dict(signals.signal_details))
    if signals.success_target > 0 and signals.success_count >= signals.success_target:
        return TrialStopResult(True, "success_complete", False, details)
    if config.timeout_sec is not None and signals.sim_time_sec >= float(config.timeout_sec):
        return TrialStopResult(True, "experiment_timeout", True, details)
    return TrialStopResult(False, None, False, details)
