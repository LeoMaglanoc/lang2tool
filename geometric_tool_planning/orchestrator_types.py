"""Shared dataclasses for goal-source orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dextoolbench.eval_config import (
    DEFAULT_CONTROL_HZ,
    DEFAULT_EVAL_SUCCESS_TOLERANCE_M,
    DEFAULT_LLM_BACKEND,
    DEFAULT_OBJECT_NAME,
    DEFAULT_TASK_NAME,
    DEFAULT_Z_OFFSET_M,
)


@dataclass
class EvalGoalSourcesArgs:
    """CLI arguments for interactive goal-source comparison."""

    goal_source: str = "all"
    execution_backend: str = "kinematics"
    object_name: str = DEFAULT_OBJECT_NAME
    task_name: str = DEFAULT_TASK_NAME
    instruction: str = "Swing the hammer down toward the right."
    target_xy: Optional[List[float]] = None
    resolved_baseline_object: Optional[str] = None
    pivot_point: Optional[List[float]] = None
    llm_backend: str = DEFAULT_LLM_BACKEND
    llm_model: Optional[str] = None
    enable_viser: bool = True
    seed: int = 0
    control_hz: float = DEFAULT_CONTROL_HZ
    config_path: Path = Path("pretrained_policy/config.yaml")
    checkpoint_path: Path = Path("pretrained_policy/model.pth")
    policy_name: Optional[str] = None
    force_table_urdf: bool = True
    use_task_env_urdf: bool = False
    z_offset: float = DEFAULT_Z_OFFSET_M
    interactive_autorun: bool = False
    exit_after_episodes: int = 0
    eval_success_tolerance: float = DEFAULT_EVAL_SUCCESS_TOLERANCE_M
    reset_time: float = -1.0


@dataclass
class GoalSourceArtifact:
    """Saved artifact for one compared goal-source mode."""

    mode: str
    goals: List[List[float]]
    duration_sec: float
    sample_interval_sec: float
    metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    llm_raw: Optional[Dict[str, Any]] = None
    execution_metrics: Dict[str, Any] = field(default_factory=dict)
