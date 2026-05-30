"""Run the fixed-policy execution benchmark against frozen geometry artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import acos, degrees
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import tyro

from experiments.common import load_geometry_trials_df, read_json, save_trial_summaries, write_json
from experiments.replay_artifacts import write_execution_replay_artifact, write_replay_manifest
from experiments.result_schema import ExecutionTrialResult, to_dict


# Configure one execution benchmark run from the CLI.
@dataclass
class ExecutionBenchmarkArgs:
    """CLI arguments for the thesis execution benchmark."""

    experiment_dir: Path
    """Existing experiment directory produced by the geometry benchmark."""

    config_path: Path = Path("pretrained_policy/config.yaml")
    """Path to the fixed RL policy config used for execution trials."""

    checkpoint_path: Path = Path("pretrained_policy/model.pth")
    """Path to the fixed RL policy checkpoint used for execution trials."""

    finetuned_config_path: Optional[Path] = None
    """Optional config path for the llm_lie finetuned execution cell."""

    finetuned_checkpoint_path: Optional[Path] = None
    """Optional checkpoint path for the llm_lie finetuned execution cell."""

    finetuned_policy_path: Optional[Path] = None
    """Optional convenience alias for finetuned_checkpoint_path."""

    num_episodes: int = 1
    """Number of policy-eval episodes to run per geometry artifact."""

    max_trials: Optional[int] = None
    """Optional hard cap on the number of geometry artifacts to execute."""

    overwrite: bool = False
    """Whether to rerun trials that already have saved execution outputs."""

    timeout_sec: float = 600.0
    """Subprocess timeout for one policy evaluation trial."""

    trial_timeout_sec: float = 180.0
    """Sim-time timeout for one policy execution trial."""

    success_goal_pct_threshold: float = 50.0
    """Goal-percentage threshold used to mark execution_success in saved summaries."""

    tracking_translation_tolerance_m: float = 0.03
    """Translation tolerance used when counting a trace step as tracked successfully."""

    tracking_rotation_tolerance_deg: float = 15.0
    """Rotation tolerance used when counting a trace step as tracked successfully."""

    # Normalize convenience aliases after CLI parsing.
    def __post_init__(self) -> None:
        """Resolve convenience aliases for the finetuned execution cell."""
        if self.finetuned_checkpoint_path is None and self.finetuned_policy_path is not None:
            self.finetuned_checkpoint_path = Path(self.finetuned_policy_path)
        if self.finetuned_checkpoint_path is not None and self.finetuned_config_path is None:
            self.finetuned_config_path = Path(self.config_path)


# Store execution goals plus audit metadata for one geometry artifact.
@dataclass
class PreparedExecutionGoals:
    """Prepared goal sequence and transform metadata for one execution trial."""

    goals: List[List[float]]
    metadata: Dict[str, Any]


# Return the active Python interpreter for subprocess benchmark calls.
def _python_cmd() -> str:
    """Return the current Python executable used by the benchmark process."""
    return sys.executable


# Return one timestamped UTC string for benchmark event logging.
def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# Compute Euclidean translation error for one object-pose / goal-pose pair.
def _translation_error_m(object_pose: List[float], goal_pose: List[float]) -> float:
    """Return translation error in metres for one 7D pose pair."""
    deltas = [float(object_pose[index]) - float(goal_pose[index]) for index in range(3)]
    return float(sum(delta * delta for delta in deltas) ** 0.5)


# Compute quaternion angular error in degrees for one object-pose / goal-pose pair.
def _rotation_error_deg(object_pose: List[float], goal_pose: List[float]) -> float:
    """Return shortest-angle quaternion error in degrees for one 7D pose pair."""
    object_quat = [float(value) for value in object_pose[3:7]]
    goal_quat = [float(value) for value in goal_pose[3:7]]
    if len(object_quat) != 4 or len(goal_quat) != 4:
        return 0.0
    dot = sum(object_quat[index] * goal_quat[index] for index in range(4))
    dot = max(-1.0, min(1.0, abs(dot)))
    return float(degrees(2.0 * acos(dot)))


# Return the stable key used to match predefined reference lengths.
def _geometry_reference_key(geometry_payload: Dict[str, Any]) -> Tuple[str, str, int]:
    """Return the object/task/seed key for one geometry payload."""
    return (
        str(geometry_payload["object_name"]),
        str(geometry_payload["task_name"]),
        int(geometry_payload.get("seed", 0)),
    )


# Build the predefined trajectory-length lookup used by execution goal cycling.
def _build_predefined_goal_counts(
    *,
    experiment_dir: Path,
    geometry_df: pd.DataFrame,
) -> Dict[Tuple[str, str, int], int]:
    """Return predefined goal counts keyed by object, task, and seed."""
    counts: Dict[Tuple[str, str, int], int] = {}
    for _, row in geometry_df.iterrows():
        if str(row.get("mode", "")) != "predefined":
            continue
        trial_id = str(row["trial_id"])
        raw_geometry_path = experiment_dir / "geometry" / "raw" / f"{trial_id}.json"
        geometry_payload = read_json(raw_geometry_path)
        goals = geometry_payload.get("goals", [])
        if isinstance(goals, list):
            counts[_geometry_reference_key(geometry_payload)] = len(goals)
    return counts


# Mirror one forward-only goal list into a down/up cycle.
def _build_mirrored_cycle(goals: Sequence[Sequence[float]]) -> List[List[float]]:
    """Return one mirrored cycle without duplicating turnaround endpoints."""
    forward_goals = [list(goal) for goal in goals]
    if len(forward_goals) <= 1:
        return forward_goals
    return forward_goals + [list(goal) for goal in reversed(forward_goals[1:-1])]


# Repeat one cycle and crop it to an exact target length.
def _repeat_and_crop_cycle(
    cycle_goals: Sequence[Sequence[float]],
    *,
    target_count: int,
) -> List[List[float]]:
    """Return a repeated cycle cropped to exactly target_count goals."""
    if target_count <= 0:
        return []
    cycle = [list(goal) for goal in cycle_goals]
    if not cycle:
        return []
    repeated: List[List[float]] = []
    while len(repeated) < target_count:
        repeated.extend([list(goal) for goal in cycle])
    return repeated[:target_count]


# Return the execution-cycle style for one LLM Lie trajectory.
def _execution_goal_cycle_style(geometry_payload: Dict[str, Any]) -> str:
    """Return the cycle style used for execution-only Lie goal expansion."""
    if "screwdriver" in str(geometry_payload["object_name"]):
        return "forward_repeat"
    return "mirrored"


# Prepare the actual goal list consumed by eval.py for one geometry artifact.
def _prepare_execution_goals(
    *,
    geometry_payload: Dict[str, Any],
    predefined_goal_counts: Dict[Tuple[str, str, int], int],
) -> PreparedExecutionGoals:
    """Return execution goals with llm_lie cycled to the predefined length."""
    geometry_goals = [list(goal) for goal in geometry_payload.get("goals", [])]
    reference_count = predefined_goal_counts.get(_geometry_reference_key(geometry_payload))
    metadata: Dict[str, Any] = {
        "geometry_goal_count": len(geometry_goals),
        "execution_goal_count": len(geometry_goals),
        "reference_goal_count": reference_count,
        "execution_goal_transform": "identity",
        "execution_goal_cycle_style": None,
    }
    if str(geometry_payload.get("mode", "")) != "llm_lie":
        return PreparedExecutionGoals(goals=geometry_goals, metadata=metadata)
    if reference_count is None:
        metadata["execution_goal_transform"] = "identity_missing_reference"
        return PreparedExecutionGoals(goals=geometry_goals, metadata=metadata)

    cycle_style = _execution_goal_cycle_style(geometry_payload)
    if cycle_style == "forward_repeat":
        cycle_goals = [list(goal) for goal in geometry_goals]
    else:
        cycle_goals = _build_mirrored_cycle(geometry_goals)
    execution_goals = _repeat_and_crop_cycle(cycle_goals, target_count=int(reference_count))
    metadata.update(
        {
            "execution_goal_count": len(execution_goals),
            "execution_goal_transform": "cycle_to_predefined_length",
            "execution_goal_cycle_style": cycle_style,
        }
    )
    return PreparedExecutionGoals(goals=execution_goals, metadata=metadata)


# Return whether one saved raw result matches the current execution-goal transform.
def _existing_result_matches_goal_transform(
    *,
    existing_payload: Dict[str, Any],
    goal_metadata: Dict[str, Any],
) -> bool:
    """Return whether a previous raw result can be reused safely."""
    if str(existing_payload.get("mode", "")) != "llm_lie":
        return True
    expected_transform = str(goal_metadata.get("execution_goal_transform", "unknown"))
    expected_goal_count = int(goal_metadata.get("execution_goal_count", 0))
    return (
        str(existing_payload.get("execution_goal_transform", "")) == expected_transform
        and int(existing_payload.get("execution_goal_count", -1)) == expected_goal_count
    )


# Load one saved execution trace JSON payload when present.
def _load_trace_payload(trace_path: Path) -> Optional[Dict[str, Any]]:
    """Return parsed trace payload or None when no trace file exists."""
    if not trace_path.exists():
        return None
    return read_json(trace_path)


# Load optional JSON diagnostics without treating absence as a benchmark failure.
def _load_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    """Return parsed JSON when present, otherwise None."""
    if not path.exists():
        return None
    return read_json(path)


# Return whether a passive reset diagnostic should end metric accumulation.
def _has_passive_reset_metric_cutoff(sample: Dict[str, Any]) -> bool:
    """Return True when object tracking is no longer physically meaningful."""
    reset_signals = sample.get("reset_signals", {})
    if not isinstance(reset_signals, dict):
        return False
    return bool(reset_signals.get("object_z_low", False)) or bool(
        reset_signals.get("dropped", False)
    )


# Return the earliest non-null step from one candidate pair.
def _earliest_step(current_step: Optional[int], candidate_step: Any) -> Optional[int]:
    """Return the earliest integer step while preserving None for missing data."""
    if candidate_step is None:
        return current_step
    candidate = int(candidate_step)
    if current_step is None:
        return candidate
    return min(current_step, candidate)


# Summarize passive reset diagnostics from eval output or trace samples.
def _reset_signal_summary(
    eval_json: Optional[Dict[str, Any]],
    trace_payload: Optional[Dict[str, Any]],
) -> Dict[str, Optional[int]]:
    """Return dropped/object-z-low counts and first observed steps."""
    summary: Dict[str, Optional[int]] = {
        "dropped_count": 0,
        "dropped_first_step": None,
        "object_z_low_count": 0,
        "object_z_low_first_step": None,
    }
    if isinstance(eval_json, dict):
        summaries = eval_json.get("episode_reset_signal_summaries", [])
        if isinstance(summaries, list) and summaries:
            for episode_summary in summaries:
                if not isinstance(episode_summary, dict):
                    continue
                summary["dropped_count"] = int(summary["dropped_count"] or 0) + int(
                    episode_summary.get("dropped_count", 0) or 0
                )
                summary["object_z_low_count"] = int(summary["object_z_low_count"] or 0) + int(
                    episode_summary.get("object_z_low_count", 0) or 0
                )
                summary["dropped_first_step"] = _earliest_step(
                    summary["dropped_first_step"],
                    episode_summary.get("dropped_first_step"),
                )
                summary["object_z_low_first_step"] = _earliest_step(
                    summary["object_z_low_first_step"],
                    episode_summary.get("object_z_low_first_step"),
                )
            return summary

    episodes = trace_payload.get("episodes", []) if isinstance(trace_payload, dict) else []
    for episode in episodes if isinstance(episodes, list) else []:
        if not isinstance(episode, list):
            continue
        for index, sample in enumerate(episode):
            if not isinstance(sample, dict):
                continue
            reset_signals = sample.get("reset_signals", {})
            if not isinstance(reset_signals, dict):
                continue
            step = int(sample.get("step", index))
            if bool(reset_signals.get("dropped", False)):
                summary["dropped_count"] = int(summary["dropped_count"] or 0) + 1
                summary["dropped_first_step"] = _earliest_step(
                    summary["dropped_first_step"],
                    step,
                )
            if bool(reset_signals.get("object_z_low", False)):
                summary["object_z_low_count"] = int(summary["object_z_low_count"] or 0) + 1
                summary["object_z_low_first_step"] = _earliest_step(
                    summary["object_z_low_first_step"],
                    step,
                )
    return summary


# Compute tracking-fidelity metrics from one saved execution trace payload.
def _compute_tracking_metrics(
    trace_payload: Optional[Dict[str, Any]],
    *,
    translation_tolerance_m: float,
    rotation_tolerance_deg: float,
) -> Dict[str, float]:
    """Return compact tracking-fidelity metrics for one execution trace."""
    if not isinstance(trace_payload, dict):
        return {
            "tracking_success_rate": 0.0,
            "translation_rmse_m": 0.0,
            "rotation_rmse_deg": 0.0,
            "mean_translation_error_m": 0.0,
            "mean_rotation_error_deg": 0.0,
            "final_translation_error_m": 0.0,
            "final_rotation_error_deg": 0.0,
            "trace_num_samples": 0.0,
        }

    episodes = trace_payload.get("episodes", [])
    translation_errors: List[float] = []
    rotation_errors: List[float] = []
    translation_squared_errors: List[float] = []
    rotation_squared_errors: List[float] = []
    tracked_steps = 0
    total_steps = 0
    final_translation_error_m = 0.0
    final_rotation_error_deg = 0.0

    for episode in episodes if isinstance(episodes, list) else []:
        if not isinstance(episode, list):
            continue
        for sample in episode:
            if not isinstance(sample, dict):
                continue
            if _has_passive_reset_metric_cutoff(sample):
                break
            object_pose = sample.get("object_pose", [])
            goal_pose = sample.get("goal_pose", [])
            if not isinstance(object_pose, list) or not isinstance(goal_pose, list):
                continue
            if len(object_pose) != 7 or len(goal_pose) != 7:
                continue
            translation_error = _translation_error_m(object_pose, goal_pose)
            rotation_error = _rotation_error_deg(object_pose, goal_pose)
            translation_errors.append(translation_error)
            rotation_errors.append(rotation_error)
            translation_squared_errors.append(translation_error * translation_error)
            rotation_squared_errors.append(rotation_error * rotation_error)
            total_steps += 1
            if translation_error <= float(translation_tolerance_m) and rotation_error <= float(
                rotation_tolerance_deg
            ):
                tracked_steps += 1
            final_translation_error_m = translation_error
            final_rotation_error_deg = rotation_error

    if total_steps == 0:
        return {
            "tracking_success_rate": 0.0,
            "translation_rmse_m": 0.0,
            "rotation_rmse_deg": 0.0,
            "mean_translation_error_m": 0.0,
            "mean_rotation_error_deg": 0.0,
            "final_translation_error_m": 0.0,
            "final_rotation_error_deg": 0.0,
            "trace_num_samples": 0.0,
        }

    return {
        "tracking_success_rate": float(tracked_steps / total_steps),
        "translation_rmse_m": float((sum(translation_squared_errors) / total_steps) ** 0.5),
        "rotation_rmse_deg": float((sum(rotation_squared_errors) / total_steps) ** 0.5),
        "mean_translation_error_m": float(sum(translation_errors) / total_steps),
        "mean_rotation_error_deg": float(sum(rotation_errors) / total_steps),
        "final_translation_error_m": float(final_translation_error_m),
        "final_rotation_error_deg": float(final_rotation_error_deg),
        "trace_num_samples": float(total_steps),
    }


# Return experiment-control stop failures recorded by eval.py, if any.
def _trial_stop_failure_category(eval_json: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return the first non-passive trial-stop failure reason saved in eval output."""
    if not isinstance(eval_json, dict):
        return None
    for trial_stop in eval_json.get("episode_trial_stops", []):
        if not isinstance(trial_stop, dict):
            continue
        if bool(trial_stop.get("is_failure", False)):
            reason = trial_stop.get("reason")
            if reason in {"dropped", "object_z_low"}:
                continue
            return str(reason) if reason is not None else "execution_failure"
    return None


# Run one `dextoolbench/eval.py` subprocess for the provided geometry artifact.
def _run_eval_subprocess(
    *,
    args: ExecutionBenchmarkArgs,
    artifact_dir: Path,
    object_name: str,
    task_name: str,
    goals_json_path: Path,
    config_path: Path,
    checkpoint_path: Path,
) -> subprocess.CompletedProcess[str]:
    """Execute one fixed-policy eval subprocess for one frozen goal artifact."""
    cmd = [
        _python_cmd(),
        "dextoolbench/eval.py",
        "--object_name",
        object_name,
        "--task_name",
        task_name,
        "--num_episodes",
        str(args.num_episodes),
        "--config_path",
        str(config_path),
        "--checkpoint_path",
        str(checkpoint_path),
        "--output_dir",
        str(artifact_dir),
        "--custom_goals_json_path",
        str(goals_json_path),
        "--max_realtime_factor",
        "0",
        "--enable_trial_stopping",
        "--trial_timeout_sec",
        str(args.trial_timeout_sec),
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=float(args.timeout_sec),
        check=False,
    )


# Append one human-readable line to the benchmark execution log.
def _append_benchmark_log(log_path: Path, message: str) -> None:
    """Append one timestamped line to the benchmark text log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as file_obj:
        file_obj.write(f"{_utc_now_iso()} {message}\n")


# Append one structured event record to the benchmark JSONL log.
def _append_benchmark_event(events_path: Path, payload: Dict[str, Any]) -> None:
    """Append one JSON event line to the benchmark JSONL log."""
    events_path.parent.mkdir(parents=True, exist_ok=True)
    event_payload = {"timestamp_utc": _utc_now_iso(), **payload}
    with open(events_path, "a") as file_obj:
        file_obj.write(json.dumps(event_payload) + "\n")


# Return which expected execution artifacts currently exist for one trial.
def _artifact_presence_snapshot(artifact_dir: Path) -> Dict[str, bool]:
    """Return one boolean snapshot of key artifact files for a trial."""
    return {
        "has_goals_json": (artifact_dir / "goals.json").exists(),
        "has_runtime_snapshot": (artifact_dir / "runtime_snapshot.json").exists(),
        "has_env_cfg": (artifact_dir / "env_cfg.yaml").exists(),
        "has_policy_config": (artifact_dir / "policy_config.yaml").exists(),
        "has_eval_json": (artifact_dir / "eval.json").exists(),
        "has_trace_json": (artifact_dir / "trace.json").exists(),
        "has_stdout_txt": (artifact_dir / "stdout.txt").exists(),
        "has_stderr_txt": (artifact_dir / "stderr.txt").exists(),
    }


# Persist one stdout/stderr pair into the artifact directory when available.
def _write_subprocess_streams(
    artifact_dir: Path,
    *,
    stdout_text: Optional[str],
    stderr_text: Optional[str],
) -> None:
    """Write subprocess stdout/stderr snapshots when text is available."""
    if stdout_text is not None:
        (artifact_dir / "stdout.txt").write_text(stdout_text)
    if stderr_text is not None:
        (artifact_dir / "stderr.txt").write_text(stderr_text)


# Build one stable benchmark-cell label from geometry mode and policy variant.
def _benchmark_cell(mode: str, policy_variant: str) -> str:
    """Return one human-readable benchmark-cell label."""
    return f"{mode} + {policy_variant}"


# Return the policy variants that should execute for one geometry mode.
def _policy_variants_for_mode(
    args: ExecutionBenchmarkArgs,
    *,
    mode: str,
) -> List[Dict[str, Any]]:
    """Return ordered execution policy variants for one geometry mode."""
    variants = [
        {
            "policy_variant": "pretrained",
            "config_path": Path(args.config_path),
            "checkpoint_path": Path(args.checkpoint_path),
        }
    ]
    if (
        mode == "llm_lie"
        and args.finetuned_config_path is not None
        and args.finetuned_checkpoint_path is not None
    ):
        variants.append(
            {
                "policy_variant": "finetuned",
                "config_path": Path(args.finetuned_config_path),
                "checkpoint_path": Path(args.finetuned_checkpoint_path),
            }
        )
    return variants


# Convert one eval subprocess output into the persisted execution summary format.
def _build_execution_result(
    *,
    args: ExecutionBenchmarkArgs,
    trial_id: str,
    geometry_payload: Dict[str, Any],
    policy_variant: str,
    artifact_dir: Path,
    eval_json: Optional[Dict[str, Any]],
    eval_returncode: Optional[int],
    produced_eval_json: bool,
    failure_category: Optional[str],
    error: Optional[str],
    trace_path: Optional[Path],
    tracking_metrics: Dict[str, float],
    reset_signal_summary: Dict[str, Optional[int]],
    goal_metadata: Dict[str, Any],
) -> ExecutionTrialResult:
    """Return one execution summary artifact from eval outputs and error state."""
    episode_goal_pcts = eval_json.get("episode_goal_pcts", []) if eval_json is not None else []
    peak_goal_pct = max([float(value) for value in episode_goal_pcts], default=0.0)
    trial_stop_failure_category = _trial_stop_failure_category(eval_json)
    effective_failure_category = failure_category or trial_stop_failure_category
    return ExecutionTrialResult(
        trial_id=trial_id,
        geometry_trial_id=str(geometry_payload["trial_id"]),
        object_name=str(geometry_payload["object_name"]),
        task_name=str(geometry_payload["task_name"]),
        mode=str(geometry_payload["mode"]),
        policy_variant=str(policy_variant),
        benchmark_cell=_benchmark_cell(str(geometry_payload["mode"]), str(policy_variant)),
        seed=int(geometry_payload.get("seed", 0)),
        execution_success=bool(
            error is None
            and trial_stop_failure_category is None
            and peak_goal_pct >= float(args.success_goal_pct_threshold)
        ),
        goal_completion_pct=float(peak_goal_pct),
        peak_success_count=int(round(peak_goal_pct)),
        failure_stage=(
            None if error is None and effective_failure_category is None else "execution_failure"
        ),
        failure_category=effective_failure_category,
        resolved_baseline_object=geometry_payload.get("resolved_baseline_object"),
        eval_returncode=eval_returncode,
        produced_eval_json=bool(produced_eval_json),
        trace_path=(str(trace_path) if trace_path is not None else None),
        tracking_success_rate=float(tracking_metrics.get("tracking_success_rate", 0.0)),
        translation_rmse_m=float(tracking_metrics.get("translation_rmse_m", 0.0)),
        rotation_rmse_deg=float(tracking_metrics.get("rotation_rmse_deg", 0.0)),
        mean_translation_error_m=float(tracking_metrics.get("mean_translation_error_m", 0.0)),
        mean_rotation_error_deg=float(tracking_metrics.get("mean_rotation_error_deg", 0.0)),
        final_translation_error_m=float(tracking_metrics.get("final_translation_error_m", 0.0)),
        final_rotation_error_deg=float(tracking_metrics.get("final_rotation_error_deg", 0.0)),
        dropped_count=int(reset_signal_summary.get("dropped_count") or 0),
        dropped_first_step=reset_signal_summary.get("dropped_first_step"),
        object_z_low_count=int(reset_signal_summary.get("object_z_low_count") or 0),
        object_z_low_first_step=reset_signal_summary.get("object_z_low_first_step"),
        geometry_goal_count=int(goal_metadata.get("geometry_goal_count", 0)),
        execution_goal_count=int(goal_metadata.get("execution_goal_count", 0)),
        reference_goal_count=goal_metadata.get("reference_goal_count"),
        execution_goal_transform=str(goal_metadata.get("execution_goal_transform", "unknown")),
        execution_goal_cycle_style=goal_metadata.get("execution_goal_cycle_style"),
        metrics=eval_json or {},
        error=error,
        artifact_dir=str(artifact_dir),
    )


# Return one classified failure category from the eval subprocess outcome.
def _classify_execution_failure(
    *,
    completed: Optional[subprocess.CompletedProcess[str]],
    produced_eval_json: bool,
    error: Optional[str],
) -> Optional[str]:
    """Return one benchmark failure category derived from eval subprocess outputs."""
    if error is None:
        return None
    stderr_text = "" if completed is None else str(completed.stderr)
    if "timed out" in str(error).lower():
        return "eval_timeout"
    if "Trajectory file not found" in stderr_text:
        return "missing_predefined_baseline"
    if "Failed to import python module isaaclab_tasks" in stderr_text:
        return "env_bootstrap_failure"
    if completed is not None and completed.returncode != 0:
        return "eval_runtime_failure"
    if not produced_eval_json:
        return "policy_execution_failure"
    return "execution_failure"


# Build one failed execution result row for timeout or runner-side exceptions.
def _build_execution_failure_result(
    *,
    args: ExecutionBenchmarkArgs,
    trial_id: str,
    geometry_payload: Dict[str, Any],
    policy_variant: str,
    artifact_dir: Path,
    failure_category: str,
    error: str,
    eval_returncode: Optional[int],
    produced_eval_json: bool,
    trace_path: Optional[Path],
    tracking_metrics: Dict[str, float],
    reset_signal_summary: Optional[Dict[str, Optional[int]]],
    goal_metadata: Dict[str, Any],
    diagnostics: Optional[Dict[str, Any]] = None,
) -> ExecutionTrialResult:
    """Return one failed execution summary row for timeout or runner-side errors."""
    metrics: Dict[str, Any] = {}
    if diagnostics is not None:
        metrics["diagnostics"] = dict(diagnostics)
    return ExecutionTrialResult(
        trial_id=trial_id,
        geometry_trial_id=str(geometry_payload["trial_id"]),
        object_name=str(geometry_payload["object_name"]),
        task_name=str(geometry_payload["task_name"]),
        mode=str(geometry_payload["mode"]),
        policy_variant=str(policy_variant),
        benchmark_cell=_benchmark_cell(str(geometry_payload["mode"]), str(policy_variant)),
        seed=int(geometry_payload.get("seed", 0)),
        execution_success=False,
        goal_completion_pct=0.0,
        peak_success_count=0,
        failure_stage="execution_failure",
        failure_category=failure_category,
        resolved_baseline_object=geometry_payload.get("resolved_baseline_object"),
        eval_returncode=eval_returncode,
        produced_eval_json=bool(produced_eval_json),
        trace_path=(str(trace_path) if trace_path is not None else None),
        tracking_success_rate=float(tracking_metrics.get("tracking_success_rate", 0.0)),
        translation_rmse_m=float(tracking_metrics.get("translation_rmse_m", 0.0)),
        rotation_rmse_deg=float(tracking_metrics.get("rotation_rmse_deg", 0.0)),
        mean_translation_error_m=float(tracking_metrics.get("mean_translation_error_m", 0.0)),
        mean_rotation_error_deg=float(tracking_metrics.get("mean_rotation_error_deg", 0.0)),
        final_translation_error_m=float(tracking_metrics.get("final_translation_error_m", 0.0)),
        final_rotation_error_deg=float(tracking_metrics.get("final_rotation_error_deg", 0.0)),
        dropped_count=int((reset_signal_summary or {}).get("dropped_count") or 0),
        dropped_first_step=(reset_signal_summary or {}).get("dropped_first_step"),
        object_z_low_count=int((reset_signal_summary or {}).get("object_z_low_count") or 0),
        object_z_low_first_step=(reset_signal_summary or {}).get("object_z_low_first_step"),
        geometry_goal_count=int(goal_metadata.get("geometry_goal_count", 0)),
        execution_goal_count=int(goal_metadata.get("execution_goal_count", 0)),
        reference_goal_count=goal_metadata.get("reference_goal_count"),
        execution_goal_transform=str(goal_metadata.get("execution_goal_transform", "unknown")),
        execution_goal_cycle_style=goal_metadata.get("execution_goal_cycle_style"),
        metrics=metrics,
        error=error,
        artifact_dir=str(artifact_dir),
    )


# Convert one invalid geometry artifact into a skipped execution result row.
def _build_skipped_geometry_result(
    *,
    trial_id: str,
    geometry_payload: Dict[str, Any],
    policy_variant: str,
    artifact_dir: Path,
    goal_metadata: Optional[Dict[str, Any]] = None,
) -> ExecutionTrialResult:
    """Return one execution summary row for geometry artifacts that never reach eval.py."""
    compile_success = bool(geometry_payload.get("compile_success", False))
    validation_success = bool(geometry_payload.get("validation_success", False))
    if not compile_success:
        failure_category = "geometry_compile_failure"
    elif not validation_success:
        failure_category = "geometry_validation_failure"
    else:
        failure_category = "geometry_failure"
    metadata = goal_metadata or {
        "geometry_goal_count": len(geometry_payload.get("goals", [])),
        "execution_goal_count": 0,
        "reference_goal_count": None,
        "execution_goal_transform": "skipped_geometry_failure",
        "execution_goal_cycle_style": None,
    }
    return ExecutionTrialResult(
        trial_id=trial_id,
        geometry_trial_id=str(geometry_payload["trial_id"]),
        object_name=str(geometry_payload["object_name"]),
        task_name=str(geometry_payload["task_name"]),
        mode=str(geometry_payload["mode"]),
        policy_variant=str(policy_variant),
        benchmark_cell=_benchmark_cell(str(geometry_payload["mode"]), str(policy_variant)),
        seed=int(geometry_payload.get("seed", 0)),
        execution_success=False,
        goal_completion_pct=0.0,
        peak_success_count=0,
        failure_stage="geometry_failure",
        failure_category=failure_category,
        resolved_baseline_object=geometry_payload.get("resolved_baseline_object"),
        eval_returncode=None,
        produced_eval_json=False,
        trace_path=None,
        tracking_success_rate=0.0,
        translation_rmse_m=0.0,
        rotation_rmse_deg=0.0,
        mean_translation_error_m=0.0,
        mean_rotation_error_deg=0.0,
        final_translation_error_m=0.0,
        final_rotation_error_deg=0.0,
        geometry_goal_count=int(metadata.get("geometry_goal_count", 0)),
        execution_goal_count=int(metadata.get("execution_goal_count", 0)),
        reference_goal_count=metadata.get("reference_goal_count"),
        execution_goal_transform=str(metadata.get("execution_goal_transform", "unknown")),
        execution_goal_cycle_style=metadata.get("execution_goal_cycle_style"),
        metrics={},
        error=geometry_payload.get("error"),
        artifact_dir=str(artifact_dir),
    )


# Flatten one execution result so it can be stored in the summary CSV.
def _flatten_execution_result(result: ExecutionTrialResult) -> Dict[str, Any]:
    """Return one summary row for the execution trials table."""
    return {
        "trial_id": result.trial_id,
        "geometry_trial_id": result.geometry_trial_id,
        "object_name": result.object_name,
        "task_name": result.task_name,
        "mode": result.mode,
        "policy_variant": result.policy_variant,
        "benchmark_cell": result.benchmark_cell,
        "seed": int(result.seed),
        "execution_success": bool(result.execution_success),
        "goal_completion_pct": float(result.goal_completion_pct),
        "peak_success_count": int(result.peak_success_count),
        "tracking_success_rate": float(result.tracking_success_rate),
        "translation_rmse_m": float(result.translation_rmse_m),
        "rotation_rmse_deg": float(result.rotation_rmse_deg),
        "mean_translation_error_m": float(result.mean_translation_error_m),
        "mean_rotation_error_deg": float(result.mean_rotation_error_deg),
        "final_translation_error_m": float(result.final_translation_error_m),
        "final_rotation_error_deg": float(result.final_rotation_error_deg),
        "dropped_count": int(result.dropped_count),
        "dropped_first_step": result.dropped_first_step,
        "object_z_low_count": int(result.object_z_low_count),
        "object_z_low_first_step": result.object_z_low_first_step,
        "geometry_goal_count": int(result.geometry_goal_count),
        "execution_goal_count": int(result.execution_goal_count),
        "reference_goal_count": result.reference_goal_count,
        "execution_goal_transform": result.execution_goal_transform,
        "execution_goal_cycle_style": result.execution_goal_cycle_style,
        "failure_stage": result.failure_stage,
        "failure_category": result.failure_category,
        "error": result.error,
        "artifact_dir": result.artifact_dir,
        "trace_path": result.trace_path,
        "resolved_baseline_object": result.resolved_baseline_object,
        "eval_returncode": result.eval_returncode,
        "produced_eval_json": bool(result.produced_eval_json),
        "replay_artifact_path": result.replay_artifact_path,
    }


# Persist one execution replay artifact when enough trace data is available.
def _write_execution_replay_if_available(
    *,
    experiment_dir: Path,
    geometry_payload: Dict[str, Any],
    result: ExecutionTrialResult,
    artifact_dir: Path,
    execution_goals: Optional[List[List[float]]] = None,
) -> Optional[Dict[str, Any]]:
    """Write one execution replay artifact and attach its path to the result."""
    trace_payload = _load_trace_payload(artifact_dir / "trace.json")
    runtime_snapshot = _load_optional_json(artifact_dir / "runtime_snapshot.json")
    replay_geometry_payload = dict(geometry_payload)
    if execution_goals is not None:
        replay_geometry_payload["goals"] = [list(goal) for goal in execution_goals]
    try:
        replay_entry = write_execution_replay_artifact(
            experiment_dir=experiment_dir,
            geometry_payload=replay_geometry_payload,
            execution_payload=to_dict(result),
            trace_payload=trace_payload,
            runtime_snapshot=runtime_snapshot,
        )
    except (KeyError, TypeError, ValueError) as exc:
        result.metrics["replay_export_error"] = f"{type(exc).__name__}: {exc}"
        return None
    if replay_entry is not None:
        result.replay_artifact_path = str(replay_entry["replay_artifact_path"])
    return replay_entry


# Run the execution benchmark for one existing experiment root.
def run_execution_benchmark(args: ExecutionBenchmarkArgs) -> Path:
    """Execute the fixed-policy benchmark against frozen geometry artifacts."""
    geometry_df = load_geometry_trials_df(args.experiment_dir)
    predefined_goal_counts = _build_predefined_goal_counts(
        experiment_dir=args.experiment_dir,
        geometry_df=geometry_df,
    )
    rows: List[Dict[str, Any]] = []
    replay_entries: List[Dict[str, Any]] = []
    trials_run = 0
    benchmark_log_path = args.experiment_dir / "execution" / "benchmark.log"
    benchmark_events_path = args.experiment_dir / "execution" / "benchmark_events.jsonl"

    write_json(args.experiment_dir / "execution" / "config.json", asdict(args))
    _append_benchmark_log(
        benchmark_log_path,
        (
            f"benchmark start geometry_trials={len(geometry_df)} timeout_sec={float(args.timeout_sec)} "
            f"pretrained_checkpoint={args.checkpoint_path} finetuned_checkpoint={args.finetuned_checkpoint_path}"
        ),
    )
    _append_benchmark_event(
        benchmark_events_path,
        {
            "event": "benchmark_start",
            "geometry_trials": int(len(geometry_df)),
            "timeout_sec": float(args.timeout_sec),
            "pretrained_config_path": str(args.config_path),
            "pretrained_checkpoint_path": str(args.checkpoint_path),
            "finetuned_config_path": (
                str(args.finetuned_config_path) if args.finetuned_config_path is not None else None
            ),
            "finetuned_checkpoint_path": (
                str(args.finetuned_checkpoint_path)
                if args.finetuned_checkpoint_path is not None
                else None
            ),
        },
    )

    for _, geometry_row in geometry_df.iterrows():
        geometry_trial_id = str(geometry_row["trial_id"])
        raw_geometry_path = args.experiment_dir / "geometry" / "raw" / f"{geometry_trial_id}.json"
        geometry_payload = read_json(raw_geometry_path)
        prepared_goals = _prepare_execution_goals(
            geometry_payload=geometry_payload,
            predefined_goal_counts=predefined_goal_counts,
        )
        for policy_spec in _policy_variants_for_mode(
            args,
            mode=str(geometry_payload["mode"]),
        ):
            if args.max_trials is not None and trials_run >= int(args.max_trials):
                break
            policy_variant = str(policy_spec["policy_variant"])
            trial_id = f"{geometry_trial_id}_{policy_variant}"
            artifact_dir = args.experiment_dir / "execution" / "artifacts" / trial_id
            result_json_path = args.experiment_dir / "execution" / "raw" / f"{trial_id}.json"
            if result_json_path.exists() and not args.overwrite:
                existing_payload = read_json(result_json_path)
                if not _existing_result_matches_goal_transform(
                    existing_payload=existing_payload,
                    goal_metadata=prepared_goals.metadata,
                ):
                    raise RuntimeError(
                        f"Existing execution result {result_json_path} was produced with "
                        "incompatible goal-transform metadata. Rerun execution_benchmark with "
                        "--overwrite to regenerate cyclic llm_lie trials."
                    )
                _append_benchmark_log(
                    benchmark_log_path,
                    f"trial {trial_id} reused existing raw result",
                )
                _append_benchmark_event(
                    benchmark_events_path,
                    {
                        "event": "trial_reused",
                        "trial_id": trial_id,
                        "geometry_trial_id": geometry_trial_id,
                        "policy_variant": policy_variant,
                        "artifact_dir": str(artifact_dir),
                        **prepared_goals.metadata,
                    },
                )
                trace_path = artifact_dir / "trace.json"
                trace_payload = _load_trace_payload(trace_path)
                eval_json = _load_optional_json(artifact_dir / "eval.json")
                tracking_metrics = _compute_tracking_metrics(
                    trace_payload,
                    translation_tolerance_m=float(args.tracking_translation_tolerance_m),
                    rotation_tolerance_deg=float(args.tracking_rotation_tolerance_deg),
                )
                reset_summary = _reset_signal_summary(eval_json, trace_payload)
                existing_result = ExecutionTrialResult(
                    trial_id=str(existing_payload["trial_id"]),
                    geometry_trial_id=str(existing_payload["geometry_trial_id"]),
                    object_name=str(existing_payload["object_name"]),
                    task_name=str(existing_payload["task_name"]),
                    mode=str(existing_payload["mode"]),
                    policy_variant=str(existing_payload.get("policy_variant", "pretrained")),
                    benchmark_cell=str(
                        existing_payload.get(
                            "benchmark_cell",
                            _benchmark_cell(
                                str(existing_payload["mode"]),
                                str(existing_payload.get("policy_variant", "pretrained")),
                            ),
                        )
                    ),
                    seed=int(existing_payload.get("seed", 0)),
                    execution_success=bool(existing_payload["execution_success"]),
                    goal_completion_pct=float(existing_payload["goal_completion_pct"]),
                    peak_success_count=int(existing_payload["peak_success_count"]),
                    failure_stage=existing_payload.get("failure_stage"),
                    failure_category=existing_payload.get("failure_category"),
                    resolved_baseline_object=existing_payload.get("resolved_baseline_object"),
                    eval_returncode=existing_payload.get("eval_returncode"),
                    produced_eval_json=bool(existing_payload.get("produced_eval_json", False)),
                    trace_path=str(trace_path) if trace_path.exists() else None,
                    tracking_success_rate=float(tracking_metrics["tracking_success_rate"]),
                    translation_rmse_m=float(tracking_metrics["translation_rmse_m"]),
                    rotation_rmse_deg=float(tracking_metrics["rotation_rmse_deg"]),
                    mean_translation_error_m=float(tracking_metrics["mean_translation_error_m"]),
                    mean_rotation_error_deg=float(tracking_metrics["mean_rotation_error_deg"]),
                    final_translation_error_m=float(tracking_metrics["final_translation_error_m"]),
                    final_rotation_error_deg=float(tracking_metrics["final_rotation_error_deg"]),
                    dropped_count=int(reset_summary.get("dropped_count") or 0),
                    dropped_first_step=reset_summary.get("dropped_first_step"),
                    object_z_low_count=int(reset_summary.get("object_z_low_count") or 0),
                    object_z_low_first_step=reset_summary.get("object_z_low_first_step"),
                    geometry_goal_count=int(
                        existing_payload.get(
                            "geometry_goal_count",
                            prepared_goals.metadata.get("geometry_goal_count", 0),
                        )
                    ),
                    execution_goal_count=int(
                        existing_payload.get(
                            "execution_goal_count",
                            prepared_goals.metadata.get("execution_goal_count", 0),
                        )
                    ),
                    reference_goal_count=existing_payload.get(
                        "reference_goal_count",
                        prepared_goals.metadata.get("reference_goal_count"),
                    ),
                    execution_goal_transform=str(
                        existing_payload.get(
                            "execution_goal_transform",
                            prepared_goals.metadata.get("execution_goal_transform", "unknown"),
                        )
                    ),
                    execution_goal_cycle_style=existing_payload.get(
                        "execution_goal_cycle_style",
                        prepared_goals.metadata.get("execution_goal_cycle_style"),
                    ),
                    metrics=dict(existing_payload.get("metrics", {})),
                    error=existing_payload.get("error"),
                    artifact_dir=existing_payload.get("artifact_dir"),
                    replay_artifact_path=existing_payload.get("replay_artifact_path"),
                )
                replay_entry = _write_execution_replay_if_available(
                    experiment_dir=args.experiment_dir,
                    geometry_payload=geometry_payload,
                    result=existing_result,
                    artifact_dir=artifact_dir,
                    execution_goals=prepared_goals.goals,
                )
                if replay_entry is not None:
                    replay_entries.append(replay_entry)
                write_json(result_json_path, to_dict(existing_result))
                rows.append(_flatten_execution_result(existing_result))
                trials_run += 1
                continue

            artifact_dir.mkdir(parents=True, exist_ok=True)
            _append_benchmark_log(
                benchmark_log_path,
                (
                    f"trial {trial_id} start object={geometry_payload['object_name']} "
                    f"mode={geometry_payload['mode']} policy_variant={policy_variant}"
                ),
            )
            _append_benchmark_event(
                benchmark_events_path,
                {
                    "event": "trial_start",
                    "trial_id": trial_id,
                    "geometry_trial_id": geometry_trial_id,
                    "object_name": str(geometry_payload["object_name"]),
                    "task_name": str(geometry_payload["task_name"]),
                    "mode": str(geometry_payload["mode"]),
                    "policy_variant": policy_variant,
                    "benchmark_cell": _benchmark_cell(
                        str(geometry_payload["mode"]), policy_variant
                    ),
                    "artifact_dir": str(artifact_dir),
                    "config_path": str(policy_spec["config_path"]),
                    "checkpoint_path": str(policy_spec["checkpoint_path"]),
                    **prepared_goals.metadata,
                },
            )
            if not bool(geometry_payload.get("compile_success", False)) or not bool(
                geometry_payload.get("validation_success", False)
            ):
                result = _build_skipped_geometry_result(
                    trial_id=trial_id,
                    geometry_payload=geometry_payload,
                    policy_variant=policy_variant,
                    artifact_dir=artifact_dir,
                    goal_metadata=prepared_goals.metadata,
                )
                write_json(result_json_path, to_dict(result))
                rows.append(_flatten_execution_result(result))
                _append_benchmark_event(
                    benchmark_events_path,
                    {
                        "event": "trial_end",
                        "trial_id": trial_id,
                        "failure_category": result.failure_category,
                        **prepared_goals.metadata,
                        "artifact_presence": _artifact_presence_snapshot(artifact_dir),
                    },
                )
                trials_run += 1
                continue

            goals_json_path = artifact_dir / "goals.json"
            write_json(
                goals_json_path,
                {
                    "goals": prepared_goals.goals,
                    "goal_metadata": prepared_goals.metadata,
                },
            )

            eval_json = None
            error = None
            eval_returncode = None
            produced_eval_json = False
            trace_path = artifact_dir / "trace.json"
            completed: Optional[subprocess.CompletedProcess[str]] = None
            result: Optional[ExecutionTrialResult] = None
            try:
                completed = _run_eval_subprocess(
                    args=args,
                    artifact_dir=artifact_dir,
                    object_name=str(geometry_payload["object_name"]),
                    task_name=str(geometry_payload["task_name"]),
                    goals_json_path=goals_json_path,
                    config_path=Path(policy_spec["config_path"]),
                    checkpoint_path=Path(policy_spec["checkpoint_path"]),
                )
                eval_returncode = int(completed.returncode)
                _write_subprocess_streams(
                    artifact_dir,
                    stdout_text=completed.stdout,
                    stderr_text=completed.stderr,
                )

                if completed.returncode != 0:
                    error = (
                        f"eval.py exited with code {completed.returncode}. "
                        f"See {artifact_dir / 'stderr.txt'}"
                    )
                else:
                    eval_json_path = artifact_dir / "eval.json"
                    if eval_json_path.exists():
                        eval_json = read_json(eval_json_path)
                        produced_eval_json = True
                    else:
                        error = f"Missing eval.json under {artifact_dir}"

                trace_payload = _load_trace_payload(trace_path)
                tracking_metrics = _compute_tracking_metrics(
                    trace_payload,
                    translation_tolerance_m=float(args.tracking_translation_tolerance_m),
                    rotation_tolerance_deg=float(args.tracking_rotation_tolerance_deg),
                )
                reset_summary = _reset_signal_summary(eval_json, trace_payload)
                failure_category = _classify_execution_failure(
                    completed=completed,
                    produced_eval_json=produced_eval_json,
                    error=error,
                )
                result = _build_execution_result(
                    args=args,
                    trial_id=trial_id,
                    geometry_payload=geometry_payload,
                    policy_variant=policy_variant,
                    artifact_dir=artifact_dir,
                    eval_json=eval_json,
                    eval_returncode=eval_returncode,
                    produced_eval_json=produced_eval_json,
                    failure_category=failure_category,
                    error=error,
                    trace_path=(trace_path if trace_path.exists() else None),
                    tracking_metrics=tracking_metrics,
                    reset_signal_summary=reset_summary,
                    goal_metadata=prepared_goals.metadata,
                )
            except subprocess.TimeoutExpired as exc:
                _write_subprocess_streams(
                    artifact_dir,
                    stdout_text=exc.output if isinstance(exc.output, str) else None,
                    stderr_text=exc.stderr if isinstance(exc.stderr, str) else None,
                )
                artifact_presence = _artifact_presence_snapshot(artifact_dir)
                trace_payload = _load_trace_payload(trace_path)
                tracking_metrics = _compute_tracking_metrics(
                    trace_payload,
                    translation_tolerance_m=float(args.tracking_translation_tolerance_m),
                    rotation_tolerance_deg=float(args.tracking_rotation_tolerance_deg),
                )
                reset_summary = _reset_signal_summary(
                    _load_optional_json(artifact_dir / "eval.json"),
                    trace_payload,
                )
                error = (
                    f"eval.py timed out after {float(args.timeout_sec):.1f}s. "
                    f"See {artifact_dir / 'stderr.txt'}"
                )
                result = _build_execution_failure_result(
                    args=args,
                    trial_id=trial_id,
                    geometry_payload=geometry_payload,
                    policy_variant=policy_variant,
                    artifact_dir=artifact_dir,
                    failure_category="eval_timeout",
                    error=error,
                    eval_returncode=None,
                    produced_eval_json=(artifact_dir / "eval.json").exists(),
                    trace_path=(trace_path if trace_path.exists() else None),
                    tracking_metrics=tracking_metrics,
                    reset_signal_summary=reset_summary,
                    goal_metadata=prepared_goals.metadata,
                    diagnostics={
                        "exception_type": "TimeoutExpired",
                        "timeout_sec": float(args.timeout_sec),
                        "artifact_presence": artifact_presence,
                    },
                )
                _append_benchmark_log(
                    benchmark_log_path,
                    f"trial {trial_id} timeout after {float(args.timeout_sec):.1f}s",
                )
                _append_benchmark_event(
                    benchmark_events_path,
                    {
                        "event": "trial_timeout",
                        "trial_id": trial_id,
                        "geometry_trial_id": geometry_trial_id,
                        "policy_variant": policy_variant,
                        "timeout_sec": float(args.timeout_sec),
                        "artifact_presence": artifact_presence,
                    },
                )
            except Exception as exc:
                artifact_presence = _artifact_presence_snapshot(artifact_dir)
                trace_payload = _load_trace_payload(trace_path)
                tracking_metrics = _compute_tracking_metrics(
                    trace_payload,
                    translation_tolerance_m=float(args.tracking_translation_tolerance_m),
                    rotation_tolerance_deg=float(args.tracking_rotation_tolerance_deg),
                )
                reset_summary = _reset_signal_summary(
                    _load_optional_json(artifact_dir / "eval.json"),
                    trace_payload,
                )
                result = _build_execution_failure_result(
                    args=args,
                    trial_id=trial_id,
                    geometry_payload=geometry_payload,
                    policy_variant=policy_variant,
                    artifact_dir=artifact_dir,
                    failure_category="eval_launch_failure",
                    error=f"{type(exc).__name__}: {exc}",
                    eval_returncode=None,
                    produced_eval_json=(artifact_dir / "eval.json").exists(),
                    trace_path=(trace_path if trace_path.exists() else None),
                    tracking_metrics=tracking_metrics,
                    reset_signal_summary=reset_summary,
                    goal_metadata=prepared_goals.metadata,
                    diagnostics={
                        "exception_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                        "artifact_presence": artifact_presence,
                    },
                )
                _append_benchmark_log(
                    benchmark_log_path,
                    f"trial {trial_id} exception {type(exc).__name__}: {exc}",
                )
                _append_benchmark_event(
                    benchmark_events_path,
                    {
                        "event": "trial_exception",
                        "trial_id": trial_id,
                        "geometry_trial_id": geometry_trial_id,
                        "policy_variant": policy_variant,
                        "exception_type": type(exc).__name__,
                        "artifact_presence": artifact_presence,
                    },
                )

            assert result is not None
            replay_entry = _write_execution_replay_if_available(
                experiment_dir=args.experiment_dir,
                geometry_payload=geometry_payload,
                result=result,
                artifact_dir=artifact_dir,
                execution_goals=prepared_goals.goals,
            )
            if replay_entry is not None:
                replay_entries.append(replay_entry)
            raw_row = to_dict(result)
            write_json(result_json_path, raw_row)
            rows.append(_flatten_execution_result(result))
            _append_benchmark_log(
                benchmark_log_path,
                (
                    f"trial {trial_id} end failure_category={result.failure_category} "
                    f"execution_success={bool(result.execution_success)}"
                ),
            )
            _append_benchmark_event(
                benchmark_events_path,
                {
                    "event": "trial_end",
                    "trial_id": trial_id,
                    "geometry_trial_id": geometry_trial_id,
                    "policy_variant": policy_variant,
                    "failure_category": result.failure_category,
                    "execution_success": bool(result.execution_success),
                    "produced_eval_json": bool(result.produced_eval_json),
                    "geometry_goal_count": int(result.geometry_goal_count),
                    "execution_goal_count": int(result.execution_goal_count),
                    "reference_goal_count": result.reference_goal_count,
                    "execution_goal_transform": result.execution_goal_transform,
                    "execution_goal_cycle_style": result.execution_goal_cycle_style,
                    "artifact_presence": _artifact_presence_snapshot(artifact_dir),
                },
            )
            trials_run += 1
        if args.max_trials is not None and trials_run >= int(args.max_trials):
            break

    aggregate_payload = _build_execution_aggregate(rows)
    save_trial_summaries(args.experiment_dir, "execution", rows, aggregate_payload)
    write_replay_manifest(
        experiment_dir=args.experiment_dir,
        stage="execution",
        entries=replay_entries,
    )
    _append_benchmark_log(
        benchmark_log_path,
        f"benchmark end num_trials={len(rows)} execution_success_rate={aggregate_payload.get('execution_success_rate', 0.0)}",
    )
    _append_benchmark_event(
        benchmark_events_path,
        {
            "event": "benchmark_end",
            "num_trials": int(len(rows)),
            "execution_success_rate": float(aggregate_payload.get("execution_success_rate", 0.0)),
        },
    )
    return args.experiment_dir


# Compute aggregate execution metrics for the saved execution summary.
def _build_execution_aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return compact execution aggregate metrics for one benchmark run."""
    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "num_trials": 0,
            "execution_success_rate": 0.0,
            "failure_attribution_by_mode": [],
        }
    grouped = (
        df.groupby(
            ["object_name", "mode", "policy_variant", "benchmark_cell"],
            dropna=False,
        )
        .agg(
            num_trials=("trial_id", "count"),
            execution_success_rate=("execution_success", "mean"),
            mean_goal_completion_pct=("goal_completion_pct", "mean"),
            mean_translation_rmse_m=("translation_rmse_m", "mean"),
            mean_tracking_success_rate=("tracking_success_rate", "mean"),
            dropped_trial_rate=("dropped_count", lambda values: (values.astype(float) > 0).mean()),
            object_z_low_trial_rate=(
                "object_z_low_count",
                lambda values: (values.astype(float) > 0).mean(),
            ),
        )
        .reset_index()
    )
    attribution_rows: List[Dict[str, Any]] = []
    for (mode, policy_variant, benchmark_cell), subset_df in df.groupby(
        ["mode", "policy_variant", "benchmark_cell"], dropna=False
    ):
        num_trials = int(len(subset_df))
        geometry_failures = int(
            subset_df["failure_category"]
            .fillna("")
            .astype(str)
            .isin(
                [
                    "geometry_compile_failure",
                    "geometry_validation_failure",
                    "geometry_failure",
                    "missing_predefined_baseline",
                ]
            )
            .sum()
        )
        execution_failures = int(
            subset_df["failure_category"]
            .fillna("")
            .astype(str)
            .isin(
                [
                    "env_bootstrap_failure",
                    "eval_timeout",
                    "eval_runtime_failure",
                    "eval_launch_failure",
                    "policy_execution_failure",
                    "execution_failure",
                ]
            )
            .sum()
        )
        successful_trials = int(subset_df["execution_success"].astype(bool).sum())
        attribution_rows.append(
            {
                "mode": str(mode),
                "policy_variant": str(policy_variant),
                "benchmark_cell": str(benchmark_cell),
                "num_trials": num_trials,
                "successful_trials": successful_trials,
                "geometry_originated_failures": geometry_failures,
                "execution_runtime_failures": execution_failures,
                "successful_share": float(successful_trials / num_trials),
                "geometry_originated_failure_share": float(geometry_failures / num_trials),
                "execution_runtime_failure_share": float(execution_failures / num_trials),
            }
        )
    return {
        "num_trials": int(len(df)),
        "execution_success_rate": float(df["execution_success"].mean()),
        "mean_translation_rmse_m": float(df["translation_rmse_m"].mean()),
        "mean_tracking_success_rate": float(df["tracking_success_rate"].mean()),
        "dropped_trial_rate": float((df["dropped_count"].astype(float) > 0).mean()),
        "object_z_low_trial_rate": float((df["object_z_low_count"].astype(float) > 0).mean()),
        "by_object_mode": grouped.to_dict(orient="records"),
        "failure_attribution_by_mode": attribution_rows,
    }


# Parse CLI arguments and execute the execution benchmark.
def main() -> None:
    """Entry point for the execution benchmark CLI."""
    args = tyro.cli(ExecutionBenchmarkArgs)
    experiment_dir = run_execution_benchmark(args)
    print(f"Saved execution benchmark to {experiment_dir}")


if __name__ == "__main__":
    main()
