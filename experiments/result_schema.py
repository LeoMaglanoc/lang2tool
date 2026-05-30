"""Shared result dataclasses and serialization helpers for thesis experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Store top-level experiment metadata persisted under one experiment root.
@dataclass
class ExperimentMetadata:
    """Serializable metadata for one saved experiment directory."""

    experiment_name: str
    created_at_utc: str
    benchmark_version: str = "v3"
    git_commit: Optional[str] = None


# Store one language-grounding benchmark trial artifact.
@dataclass
class LanguageTrialResult:
    """Serializable raw result for one language-grounding benchmark trial."""

    trial_id: str
    prompt_id: str
    prompt_family: str
    prompt_variant: str
    prompt_text: str
    backend: str
    active_object_context: str
    expected_outcome_type: str
    expected_intent: Optional[str]
    expected_object_name: Optional[str]
    expected_target_label: Optional[str]
    expected_clarification: bool
    predicted_outcome_type: str
    predicted_intent: Optional[str]
    predicted_object_name: Optional[str]
    predicted_target_label: Optional[str]
    predicted_clarification: bool
    assistant_text: str
    exact_match: bool
    object_match: bool
    intent_match: bool
    target_match: bool
    clarification_match: bool
    expected_tool_call: Optional[Dict[str, Any]] = None
    predicted_tool_call: Optional[Dict[str, Any]] = None
    target_evaluation: Dict[str, Any] = field(default_factory=dict)
    predicted_tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# Store one geometry-benchmark trial artifact.
@dataclass
class GeometryTrialResult:
    """Serializable raw result for one geometry benchmark trial."""

    trial_id: str
    object_name: str
    task_name: str
    mode: str
    seed: int
    target_xy: Optional[List[float]]
    resolved_baseline_object: Optional[str]
    pivot_point: Optional[List[float]]
    compile_success: bool
    validation_success: bool
    num_waypoints: int
    goals: List[List[float]] = field(default_factory=list)
    generation_context: Dict[str, Any] = field(default_factory=dict)
    clamp_summary: Dict[str, Any] = field(default_factory=dict)
    resampling_summary: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    replay_artifact_path: Optional[str] = None


# Store one execution-benchmark trial artifact.
@dataclass
class ExecutionTrialResult:
    """Serializable raw result for one execution benchmark trial."""

    trial_id: str
    geometry_trial_id: str
    object_name: str
    task_name: str
    mode: str
    policy_variant: str
    benchmark_cell: str
    seed: int
    execution_success: bool
    goal_completion_pct: float
    peak_success_count: int
    failure_stage: Optional[str]
    failure_category: Optional[str]
    resolved_baseline_object: Optional[str]
    eval_returncode: Optional[int]
    produced_eval_json: bool = False
    trace_path: Optional[str] = None
    tracking_success_rate: float = 0.0
    translation_rmse_m: float = 0.0
    rotation_rmse_deg: float = 0.0
    mean_translation_error_m: float = 0.0
    mean_rotation_error_deg: float = 0.0
    final_translation_error_m: float = 0.0
    final_rotation_error_deg: float = 0.0
    dropped_count: int = 0
    dropped_first_step: Optional[int] = None
    object_z_low_count: int = 0
    object_z_low_first_step: Optional[int] = None
    geometry_goal_count: int = 0
    execution_goal_count: int = 0
    reference_goal_count: Optional[int] = None
    execution_goal_transform: str = "unknown"
    execution_goal_cycle_style: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    artifact_dir: Optional[str] = None
    replay_artifact_path: Optional[str] = None


# Convert one dataclass or mapping into a JSON-compatible dictionary.
def to_dict(payload: Any) -> Dict[str, Any]:
    """Return a JSON-serializable dictionary for one supported payload."""
    if hasattr(payload, "__dataclass_fields__"):
        return asdict(payload)
    if isinstance(payload, dict):
        return dict(payload)
    raise TypeError(f"Unsupported payload type: {type(payload).__name__}")


# Return the expected summary file path for one experiment section.
def summary_path(experiment_dir: Path, section: str, filename: str) -> Path:
    """Return one summary path under the requested experiment section."""
    return experiment_dir / section / "summaries" / filename
