"""Shared helpers for experiment runners and the results dashboard."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from experiments.result_schema import ExperimentMetadata, summary_path, to_dict

CANONICAL_TARGET_GRID_X_BOUNDS_M = (-0.15, 0.15)
CANONICAL_TARGET_GRID_Y_BOUNDS_M = (-0.13, -0.03)


# Return the current UTC timestamp formatted for metadata persistence.
def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# Parse a comma-separated CLI string into a non-empty ordered list of tokens.
def parse_csv_tokens(raw_value: str) -> List[str]:
    """Return normalized non-empty comma-separated tokens."""
    return [token.strip() for token in raw_value.split(",") if token.strip()]


# Build one stable experiment directory and create its required child folders.
def ensure_experiment_dirs(results_dir: Path, experiment_name: str) -> Path:
    """Create one experiment root and the standard geometry/execution subdirectories."""
    experiment_dir = results_dir / experiment_name
    for rel_path in (
        "language/raw",
        "language/summaries",
        "geometry/raw",
        "geometry/replay/trials",
        "geometry/summaries",
        "execution/raw",
        "execution/replay/trials",
        "execution/summaries",
        "execution/artifacts",
        "dashboard",
    ):
        (experiment_dir / rel_path).mkdir(parents=True, exist_ok=True)
    return experiment_dir


# Persist one JSON payload with parent-directory creation.
def write_json(path: Path, payload: Any) -> None:
    """Write one JSON file with stable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file_obj:
        json.dump(_json_safe(payload), file_obj, indent=2)


# Read one JSON payload from disk.
def read_json(path: Path) -> Dict[str, Any]:
    """Return one parsed JSON object from disk."""
    with open(path) as file_obj:
        return json.load(file_obj)


# Convert nested payloads into JSON-safe primitives before persistence.
def _json_safe(payload: Any) -> Any:
    """Return one JSON-serializable version of the provided payload."""
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {str(key): _json_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_json_safe(value) for value in payload]
    return payload


# Write one top-level experiment metadata file if it does not exist yet.
def ensure_experiment_metadata(experiment_dir: Path, experiment_name: str) -> None:
    """Persist one metadata.json file for the experiment root when absent."""
    metadata_path = experiment_dir / "metadata.json"
    if metadata_path.exists():
        return
    git_commit = None
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=experiment_dir.parent.parent,
        ).stdout.strip()
    except Exception:
        git_commit = None
    metadata = ExperimentMetadata(
        experiment_name=experiment_name,
        created_at_utc=utc_now_iso(),
        git_commit=git_commit,
    )
    write_json(metadata_path, to_dict(metadata))


# Save one flat trial table and aggregate summary for a benchmark section.
def save_trial_summaries(
    experiment_dir: Path,
    section: str,
    rows: Sequence[Dict[str, Any]],
    aggregate_payload: Dict[str, Any],
) -> None:
    """Write one section's trials CSV and aggregate JSON."""
    trials_df = pd.DataFrame(list(rows))
    trials_csv_path = summary_path(experiment_dir, section, "trials.csv")
    trials_df.to_csv(trials_csv_path, index=False)
    write_json(summary_path(experiment_dir, section, "aggregate.json"), aggregate_payload)


# Build one deterministic back-table rectangular XY target grid.
def build_target_grid_xy(num_x: int, num_y: int) -> List[Tuple[float, float]]:
    """Return one evenly spaced XY target grid inside the back tabletop region."""
    if num_x < 1 or num_y < 1:
        raise ValueError("Target grid dimensions must be positive.")
    min_x, max_x = CANONICAL_TARGET_GRID_X_BOUNDS_M
    min_y, max_y = CANONICAL_TARGET_GRID_Y_BOUNDS_M
    x_values = _linspace(min_x, max_x, num_x)
    y_values = _linspace(min_y, max_y, num_y)
    return [(float(x_value), float(y_value)) for y_value in y_values for x_value in x_values]


# Build one evenly spaced list that keeps endpoints when count is one.
def _linspace(start: float, stop: float, count: int) -> List[float]:
    """Return one inclusive float sequence with special handling for count=1."""
    if count == 1:
        return [float((start + stop) * 0.5)]
    step = (float(stop) - float(start)) / float(count - 1)
    return [float(start) + step * float(index) for index in range(count)]


# Return one flat list of geometry trial rows loaded from the experiment summaries.
def load_geometry_trials_df(experiment_dir: Path) -> pd.DataFrame:
    """Return the geometry trials summary table for one experiment root."""
    return pd.read_csv(summary_path(experiment_dir, "geometry", "trials.csv"))


# Return one flat list of language trial rows loaded from the experiment summaries.
def load_language_trials_df(experiment_dir: Path) -> pd.DataFrame:
    """Return the language trials summary table for one experiment root."""
    return pd.read_csv(summary_path(experiment_dir, "language", "trials.csv"))


# Return one flat list of execution trial rows loaded from the experiment summaries.
def load_execution_trials_df(experiment_dir: Path) -> pd.DataFrame:
    """Return the execution trials summary table for one experiment root."""
    return pd.read_csv(summary_path(experiment_dir, "execution", "trials.csv"))


# Compute compact object/mode aggregates for the geometry benchmark.
def summarize_geometry_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Return aggregate summary payload for geometry trial rows."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return {"num_trials": 0, "by_object_mode": []}
    grouped = (
        df.groupby(["object_name", "mode"], dropna=False)
        .agg(
            num_trials=("trial_id", "count"),
            compile_success_rate=("compile_success", "mean"),
            validation_success_rate=("validation_success", "mean"),
            mean_num_waypoints=("num_waypoints", "mean"),
        )
        .reset_index()
    )
    return {
        "num_trials": int(len(df)),
        "by_object_mode": grouped.to_dict(orient="records"),
    }


# Compute compact object/mode aggregates for the execution benchmark.
def summarize_execution_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Return aggregate summary payload for execution trial rows."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return {"num_trials": 0, "by_object_mode": []}
    grouped = (
        df.groupby(
            ["object_name", "mode", "policy_variant", "benchmark_cell"],
            dropna=False,
        )
        .agg(
            num_trials=("trial_id", "count"),
            execution_success_rate=("execution_success", "mean"),
            mean_goal_completion_pct=("goal_completion_pct", "mean"),
        )
        .reset_index()
    )
    return {
        "num_trials": int(len(df)),
        "by_object_mode": grouped.to_dict(orient="records"),
    }


# Return one safe experiment name when the caller does not provide one.
def default_experiment_name(prefix: str) -> str:
    """Return one timestamped experiment name with the requested prefix."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{timestamp}"
