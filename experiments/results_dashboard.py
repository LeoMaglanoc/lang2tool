"""Matplotlib-backed thesis website for saved language, geometry, and execution benchmark results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro
from dash import Dash, dash_table, html
from flask import send_from_directory
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from experiments.common import read_json
from experiments.geometry_benchmark import _semantic_target_frame_index
from experiments.replay_language_offline import default_language_viewer_path
from geometric_tool_planning import get_named_strike_points
from geometric_tool_planning.viewer import semantic_strike_axis_table_intersection
from llm_runtime.semantic_pose import get_object_pose_semantics, quat_rotate_xyzw

MODE_COLORS = {
    "predefined": "#52606d",
    "llm_lie": "#2f80ed",
    "llm_only": "#f2994a",
}

SECTION_COLORS = {
    "language": "#264653",
    "geometry": "#2a9d8f",
    "execution": "#e76f51",
}

LANGUAGE_METRIC_LABELS = {
    "exact_match_accuracy": "Exact Match",
    "intent_accuracy": "Intent",
    "object_accuracy": "Object",
    "target_accuracy": "Target",
    "clarification_accuracy": "Clarification",
}

LANGUAGE_FAMILY_LABELS = {
    "clarification_or_ambiguity": "Ambiguity",
    "explicit_object_switching": "Object Switch",
    "paraphrase_robustness": "Paraphrase",
    "default_family_selection": "Default Choice",
    "target_grounding": "Target Grounding",
    "lifecycle_and_predefined_commands": "Lifecycle",
    "semantic_pose_edits": "Pose Edits",
    "unsupported_or_out_of_scope": "Unsupported",
}

MODE_LABELS = {
    "predefined": "Predefined",
    "llm_lie": "LLM SE(3)",
    "llm_only": "LLM Direct",
}

THESIS_MODE_ORDER = ["predefined", "llm_lie", "llm_only"]
GEOMETRY_GENERATED_MODE_ORDER = ["llm_lie", "llm_only"]

OBJECT_DISPLAY_LABELS = {
    "cuboid_hammer_v014": "primitive hammer",
    "cylinder_screwdriver_v3009": "primitive screwdriver",
}

OBJECT_FAMILY_ORDER = {
    "claw_hammer": ("hammer", 0),
    "mallet_hammer": ("hammer", 1),
    "cuboid_hammer_v014": ("hammer", 2),
    "long_screwdriver": ("screwdriver", 0),
    "short_screwdriver": ("screwdriver", 1),
    "cylinder_screwdriver_v3009": ("screwdriver", 2),
}

ASSET_FILENAMES = {
    "language_prompt_family_accuracy": "language_prompt_family_accuracy.svg",
    "language_prompt_family_pass_fail": "language_prompt_family_pass_fail.svg",
    "geometry_semantic_by_object": "geometry_semantic_by_object.svg",
    "geometry_implied_target_hammer": "geometry_implied_target_hammer.svg",
    "geometry_implied_target_screwdriver": "geometry_implied_target_screwdriver.svg",
    "execution_strict_success": "execution_strict_success.svg",
    "execution_goal_completion": "execution_goal_completion.svg",
    "execution_goal_completion_by_family": "execution_goal_completion_by_family.svg",
    "execution_strict_success_by_family": "execution_strict_success_by_family.svg",
    "execution_translation_rmse_by_family": "execution_translation_rmse_by_family.svg",
}

STALE_ASSET_FILENAMES = (
    "language_accuracy.svg",
    "geometry_validity.svg",
    "geometry_semantic_summary.svg",
    "geometry_target_a_comparison.svg",
    "geometry_target_a_exemplar.svg",
    "geometry_implied_target_topdown.svg",
    "execution_failure_overview.svg",
    "execution_failure_by_cell.svg",
    "execution_pairwise_delta.svg",
    "execution_success_progress.svg",
    "execution_success_progress_by_family.svg",
    "execution_translation_rmse.svg",
)


# Configure one dashboard server instance from the CLI.
@dataclass
class ResultsDashboardArgs:
    """CLI arguments for serving the thesis website."""

    results_dir: Path
    """Experiment directory containing language/, geometry/, and execution/ summaries."""

    host: str = "127.0.0.1"
    """Host interface for the Dash server."""

    port: int = 8080
    """TCP port for the Dash server."""

    strict_success_threshold_pct: float = 100.0
    """Goal-completion threshold used for committee-facing strict success."""

    partial_success_threshold_pct: float = 50.0
    """Goal-completion threshold used for committee-facing partial success."""


# Load metadata and the three benchmark summary tables from one experiment root.
def load_dashboard_data(results_dir: Path) -> Tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return metadata plus language, geometry, and execution summary DataFrames."""
    metadata = read_json(results_dir / "metadata.json")
    language_df = _load_optional_trials_df(results_dir / "language" / "summaries" / "trials.csv")
    geometry_df = _load_optional_trials_df(results_dir / "geometry" / "summaries" / "trials.csv")
    execution_df = _load_optional_trials_df(results_dir / "execution" / "summaries" / "trials.csv")
    return metadata, language_df, geometry_df, execution_df


# Load one trial summary CSV when present, else return an empty DataFrame.
def _load_optional_trials_df(path: Path) -> pd.DataFrame:
    """Return summary rows from disk or an empty DataFrame when the file is absent."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# Return one execution table augmented with committee-facing derived metrics.
def _augment_execution_metrics(
    execution_df: pd.DataFrame,
    *,
    strict_success_threshold_pct: float,
    partial_success_threshold_pct: float,
) -> pd.DataFrame:
    """Return one execution DataFrame with strict and partial success columns added."""
    augmented_df = execution_df.copy()
    if augmented_df.empty:
        augmented_df["strict_success"] = []
        augmented_df["partial_success"] = []
        augmented_df["failure_group"] = []
        augmented_df["is_clean_trial"] = []
        augmented_df["is_infra_failure"] = []
        augmented_df["is_geometry_failure"] = []
        augmented_df["is_policy_failure"] = []
        return augmented_df
    if "policy_variant" not in augmented_df.columns:
        augmented_df["policy_variant"] = "pretrained"
    if "benchmark_cell" not in augmented_df.columns:
        augmented_df["benchmark_cell"] = (
            augmented_df["mode"].astype(str) + " + " + augmented_df["policy_variant"].astype(str)
        )
    for column_name in (
        "translation_rmse_m",
        "rotation_rmse_deg",
        "tracking_success_rate",
        "dropped_count",
        "object_z_low_count",
    ):
        if column_name not in augmented_df.columns:
            augmented_df[column_name] = 0.0
    augmented_df["goal_completion_pct"] = augmented_df["goal_completion_pct"].astype(float)
    augmented_df["translation_rmse_m"] = augmented_df["translation_rmse_m"].astype(float)
    augmented_df["rotation_rmse_deg"] = augmented_df["rotation_rmse_deg"].astype(float)
    augmented_df["tracking_success_rate"] = augmented_df["tracking_success_rate"].astype(float)
    augmented_df["dropped_count"] = augmented_df["dropped_count"].fillna(0).astype(int)
    augmented_df["object_z_low_count"] = augmented_df["object_z_low_count"].fillna(0).astype(int)
    augmented_df["policy_variant"] = augmented_df["policy_variant"].astype(str)
    augmented_df["benchmark_cell"] = augmented_df["benchmark_cell"].astype(str)
    augmented_df["failure_category"] = augmented_df.get(
        "failure_category", pd.Series(index=augmented_df.index, dtype=object)
    ).fillna("")
    augmented_df["failure_group"] = augmented_df["failure_category"].apply(_failure_group)
    augmented_df["strict_success"] = (
        augmented_df["goal_completion_pct"] >= float(strict_success_threshold_pct)
    ) & (augmented_df["dropped_count"] == 0)
    augmented_df["partial_success"] = augmented_df["goal_completion_pct"] >= float(
        partial_success_threshold_pct
    )
    augmented_df["execution_success"] = augmented_df["execution_success"].astype(bool)
    augmented_df["is_infra_failure"] = augmented_df["failure_group"] == "infrastructure"
    augmented_df["is_geometry_failure"] = augmented_df["failure_group"] == "geometry"
    augmented_df["is_policy_failure"] = augmented_df["failure_group"] == "policy_or_outcome"
    augmented_df["is_clean_trial"] = augmented_df["failure_group"] == "clean"
    return augmented_df


# Collapse raw execution failures into committee-facing failure groups.
def _failure_group(failure_category: object) -> str:
    """Return one higher-level failure group for one raw failure category string."""
    normalized_category = str(failure_category or "").strip()
    if normalized_category == "":
        return "clean"
    if normalized_category in {
        "env_bootstrap_failure",
        "eval_timeout",
        "eval_runtime_failure",
        "eval_launch_failure",
        "execution_failure",
    }:
        return "infrastructure"
    if normalized_category in {
        "geometry_compile_failure",
        "geometry_validation_failure",
        "missing_predefined_baseline",
    }:
        return "geometry"
    if normalized_category in {
        "dropped",
        "object_z_low",
        "experiment_timeout",
        "policy_execution_failure",
    }:
        return "policy_or_outcome"
    return "policy_or_outcome"


# Build one aggregate language summary table grouped by outcome and backend.
def _build_language_summary_df(language_df: pd.DataFrame) -> pd.DataFrame:
    """Return one prompt-family language summary table for thesis figures."""
    if language_df.empty:
        return pd.DataFrame(
            columns=[
                "prompt_family",
                "num_trials",
                "num_exact_match",
                "num_exact_mismatch",
                "exact_match_accuracy",
                "intent_accuracy",
                "object_accuracy",
                "target_accuracy",
                "clarification_accuracy",
            ]
        )
    return (
        language_df.groupby(["prompt_family"], dropna=False)
        .agg(
            num_trials=("trial_id", "count"),
            num_exact_match=("exact_match", "sum"),
            exact_match_accuracy=("exact_match", "mean"),
            intent_accuracy=("intent_match", "mean"),
            object_accuracy=("object_match", "mean"),
            target_accuracy=("target_match", "mean"),
            clarification_accuracy=("clarification_match", "mean"),
        )
        .reset_index()
        .assign(
            num_exact_mismatch=lambda df: df["num_trials"].astype(int)
            - df["num_exact_match"].astype(int)
        )
    )


# Build one compact language component summary table over all prompts.
def _build_language_component_summary_df(language_df: pd.DataFrame) -> pd.DataFrame:
    """Return one single-row language component summary for the thesis website."""
    if language_df.empty:
        return pd.DataFrame(
            [
                {
                    "exact_match_accuracy": 0.0,
                    "intent_accuracy": 0.0,
                    "object_accuracy": 0.0,
                    "target_accuracy": 0.0,
                    "clarification_accuracy": 0.0,
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "exact_match_accuracy": float(language_df["exact_match"].astype(float).mean()),
                "intent_accuracy": float(language_df["intent_match"].astype(float).mean()),
                "object_accuracy": float(language_df["object_match"].astype(float).mean()),
                "target_accuracy": float(language_df["target_match"].astype(float).mean()),
                "clarification_accuracy": float(
                    language_df["clarification_match"].astype(float).mean()
                ),
            }
        ]
    )


# Build one aggregate geometry summary table grouped by object and mode.
def _build_geometry_summary_df(geometry_df: pd.DataFrame) -> pd.DataFrame:
    """Return one object/mode geometry summary table for thesis figures."""
    if geometry_df.empty:
        return pd.DataFrame(
            columns=[
                "object_name",
                "tool_family",
                "mode",
                "valid_trajectory_rate",
                "compile_success_rate",
                "validation_success_rate",
                "mean_num_waypoints",
                "mean_num_clamped_waypoints",
                "mean_semantic_contact_point_xy_error_m",
                "mean_screwdriver_max_primary_axis_tilt_deg",
                "mean_screwdriver_twist_angle_span_deg",
                "mean_translation_error_m",
                "mean_rotation_error_deg",
                "mean_path_length_ratio",
            ]
        )
    geometry_df = geometry_df.copy()
    geometry_df["valid_trajectory"] = geometry_df["compile_success"].astype(bool) & geometry_df[
        "validation_success"
    ].astype(bool)
    for column_name in (
        "semantic_contact_point_xy_error_m",
        "screwdriver_max_primary_axis_tilt_deg",
        "screwdriver_twist_angle_span_deg",
        "mean_translation_error_m",
        "mean_rotation_error_deg",
        "path_length_ratio",
    ):
        if column_name not in geometry_df.columns:
            geometry_df[column_name] = 0.0
    return (
        geometry_df.groupby(["object_name", "mode"], dropna=False)
        .agg(
            tool_family=("tool_family", "first"),
            valid_trajectory_rate=("valid_trajectory", "mean"),
            compile_success_rate=("compile_success", "mean"),
            validation_success_rate=("validation_success", "mean"),
            mean_num_waypoints=("num_waypoints", "mean"),
            mean_num_clamped_waypoints=("num_clamped_waypoints", "mean"),
            mean_semantic_contact_point_xy_error_m=(
                "semantic_contact_point_xy_error_m",
                "mean",
            ),
            mean_screwdriver_max_primary_axis_tilt_deg=(
                "screwdriver_max_primary_axis_tilt_deg",
                "mean",
            ),
            mean_screwdriver_twist_angle_span_deg=(
                "screwdriver_twist_angle_span_deg",
                "mean",
            ),
            mean_translation_error_m=("mean_translation_error_m", "mean"),
            mean_rotation_error_deg=("mean_rotation_error_deg", "mean"),
            mean_path_length_ratio=("path_length_ratio", "mean"),
        )
        .reset_index()
    )


# Build one aggregate execution summary table grouped by object and mode.
def _build_execution_summary_df(execution_df: pd.DataFrame) -> pd.DataFrame:
    """Return one object/mode execution summary table for thesis figures."""
    if execution_df.empty:
        return pd.DataFrame(
            columns=[
                "object_name",
                "mode",
                "policy_variant",
                "benchmark_cell",
                "resolved_baseline_object",
                "num_trials",
                "trajectory_rmse_m",
                "rotation_rmse_deg",
                "tracking_fidelity",
                "strict_success_rate",
                "partial_success_rate",
                "mean_goal_completion_pct",
            ]
        )
    execution_df = execution_df.copy()
    for column_name in (
        "translation_rmse_m",
        "rotation_rmse_deg",
        "tracking_success_rate",
        "mean_translation_error_m",
        "mean_rotation_error_deg",
        "strict_success",
        "partial_success",
        "goal_completion_pct",
    ):
        if column_name not in execution_df.columns:
            execution_df[column_name] = 0.0
    return (
        execution_df.groupby(
            [
                "object_name",
                "mode",
                "policy_variant",
                "benchmark_cell",
                "resolved_baseline_object",
            ],
            dropna=False,
        )
        .agg(
            num_trials=("trial_id", "count"),
            trajectory_rmse_m=("translation_rmse_m", "mean"),
            rotation_rmse_deg=("rotation_rmse_deg", "mean"),
            tracking_fidelity=("tracking_success_rate", "mean"),
            mean_translation_error_m=("mean_translation_error_m", "mean"),
            mean_rotation_error_deg=("mean_rotation_error_deg", "mean"),
            strict_success_rate=("strict_success", "mean"),
            partial_success_rate=("partial_success", "mean"),
            mean_goal_completion_pct=("goal_completion_pct", "mean"),
        )
        .reset_index()
    )


# Keep the thesis website focused on pretrained execution rows for compact comparisons.
def _dashboard_execution_view_df(execution_df: pd.DataFrame) -> pd.DataFrame:
    """Return one execution DataFrame filtered to pretrained thesis-facing modes."""
    if execution_df.empty:
        return execution_df.copy()
    pretrained_df = execution_df[execution_df["policy_variant"].astype(str) == "pretrained"].copy()
    return pretrained_df[
        pretrained_df["mode"].astype(str).isin(["predefined", "llm_only", "llm_lie"])
    ].copy()


# Keep the success/progress figure on all pretrained execution modes, including LLM Direct.
def _execution_success_progress_view_df(execution_df: pd.DataFrame) -> pd.DataFrame:
    """Return one execution DataFrame filtered to pretrained rows for success/progress."""
    return _dashboard_execution_view_df(execution_df)


# Build one compact execution summary grouped by mode for the website.
def _build_execution_cell_summary_df(
    execution_df: pd.DataFrame,
    *,
    mode_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return one thesis execution summary table grouped by mode."""
    if execution_df.empty:
        return pd.DataFrame(
            columns=[
                "mode",
                "num_trials",
                "strict_success_count",
                "strict_success_rate",
                "completed_goal_count",
                "total_goal_count",
                "mean_goal_completion_pct",
                "trajectory_rmse_m",
            ]
        )
    working_df = execution_df.copy()
    if "execution_goal_count" not in working_df.columns:
        working_df["execution_goal_count"] = 0.0
    execution_goal_counts = pd.to_numeric(
        working_df["execution_goal_count"], errors="coerce"
    ).fillna(0.0)
    goal_completion_fracs = (
        pd.to_numeric(working_df["goal_completion_pct"], errors="coerce").fillna(0.0) / 100.0
    )
    working_df["_completed_goal_count"] = execution_goal_counts * goal_completion_fracs
    working_df["_total_goal_count"] = execution_goal_counts
    summary_df = (
        working_df.groupby(["benchmark_cell"], dropna=False)
        .agg(
            mode=("mode", "first"),
            num_trials=("trial_id", "count"),
            strict_success_count=("strict_success", "sum"),
            strict_success_rate=("strict_success", "mean"),
            completed_goal_count=("_completed_goal_count", "sum"),
            total_goal_count=("_total_goal_count", "sum"),
            mean_goal_completion_pct=("goal_completion_pct", "mean"),
            trajectory_rmse_m=("translation_rmse_m", "mean"),
        )
        .reset_index()
    )
    if mode_order is None:
        observed_modes = set(summary_df["mode"].astype(str))
        mode_order = [mode_name for mode_name in THESIS_MODE_ORDER if mode_name in observed_modes]
    summary_df["mode"] = pd.Categorical(summary_df["mode"], categories=mode_order, ordered=True)
    summary_df = summary_df.sort_values("mode").reset_index(drop=True)
    return summary_df[
        [
            "mode",
            "num_trials",
            "strict_success_count",
            "strict_success_rate",
            "completed_goal_count",
            "total_goal_count",
            "mean_goal_completion_pct",
            "trajectory_rmse_m",
        ]
    ]


# Build one compact execution summary grouped by tool family and mode.
def _build_execution_family_cell_summary_df(execution_df: pd.DataFrame) -> pd.DataFrame:
    """Return one thesis execution summary table grouped by tool family and mode."""
    columns = [
        "tool_family",
        "mode",
        "num_trials",
        "strict_success_count",
        "strict_success_rate",
        "completed_goal_count",
        "total_goal_count",
        "mean_goal_completion_pct",
    ]
    if execution_df.empty:
        return pd.DataFrame(columns=columns)
    family_df = execution_df.copy()
    family_df["tool_family"] = family_df["object_name"].astype(str).map(_execution_object_family)
    family_df = family_df[family_df["tool_family"].isin(["hammer", "screwdriver"])].copy()
    if family_df.empty:
        return pd.DataFrame(columns=columns)
    working_df = family_df.copy()
    if "strict_success" not in working_df.columns:
        working_df["strict_success"] = False
    if "goal_completion_pct" not in working_df.columns:
        working_df["goal_completion_pct"] = 0.0
    if "execution_goal_count" not in working_df.columns:
        working_df["execution_goal_count"] = 0.0
    execution_goal_counts = pd.to_numeric(
        working_df["execution_goal_count"], errors="coerce"
    ).fillna(0.0)
    goal_completion_fracs = (
        pd.to_numeric(working_df["goal_completion_pct"], errors="coerce").fillna(0.0) / 100.0
    )
    working_df["_completed_goal_count"] = execution_goal_counts * goal_completion_fracs
    working_df["_total_goal_count"] = execution_goal_counts
    summary_df = (
        working_df.groupby(["tool_family", "benchmark_cell"], dropna=False)
        .agg(
            mode=("mode", "first"),
            num_trials=("trial_id", "count"),
            strict_success_count=("strict_success", "sum"),
            strict_success_rate=("strict_success", "mean"),
            completed_goal_count=("_completed_goal_count", "sum"),
            total_goal_count=("_total_goal_count", "sum"),
            mean_goal_completion_pct=("goal_completion_pct", "mean"),
        )
        .reset_index()
    )
    summary_df["tool_family"] = pd.Categorical(
        summary_df["tool_family"], categories=["hammer", "screwdriver"], ordered=True
    )
    summary_df["mode"] = pd.Categorical(
        summary_df["mode"], categories=THESIS_MODE_ORDER, ordered=True
    )
    return summary_df.sort_values(["tool_family", "mode"]).reset_index(drop=True)[columns].copy()


# Build one compact failure-attribution table grouped by thesis-facing mode.
def _build_execution_failure_attribution_df(execution_df: pd.DataFrame) -> pd.DataFrame:
    """Return mode-level success and failure attribution counts for pretrained execution rows."""
    if execution_df.empty:
        return pd.DataFrame(
            columns=[
                "mode",
                "num_trials",
                "successful_trials",
                "geometry_originated_failures",
                "execution_runtime_failures",
            ]
        )
    grouped_df = (
        execution_df.groupby(["mode"], dropna=False)
        .agg(
            num_trials=("trial_id", "count"),
            successful_trials=("execution_success", "sum"),
            geometry_originated_failures=("is_geometry_failure", "sum"),
            execution_runtime_failures=(
                "failure_group",
                lambda values: int(
                    pd.Series(values)
                    .astype(str)
                    .isin(["infrastructure", "policy_or_outcome"])
                    .sum()
                ),
            ),
        )
        .reset_index()
    )
    grouped_df["mode"] = pd.Categorical(
        grouped_df["mode"], categories=THESIS_MODE_ORDER, ordered=True
    )
    grouped_df = grouped_df.sort_values("mode").reset_index(drop=True)
    return grouped_df


# Return one human-readable label for a geometry mode used in figure legends.
def _pretty_mode_label(mode: str) -> str:
    """Return a compact thesis label for one geometry mode string."""
    return MODE_LABELS.get(str(mode), str(mode))


# Return one thesis-facing label for generated trajectory geometry modes.
def _pretty_geometry_mode_label(mode: str) -> str:
    """Return the Stage 2 display label for one trajectory-generation mode."""
    return _pretty_mode_label(mode)


# Return one human-readable object label for thesis website presentation.
def _pretty_object_label(object_name: str) -> str:
    """Return the thesis-facing display label for one object identifier."""
    object_name = str(object_name)
    return OBJECT_DISPLAY_LABELS.get(object_name, object_name.replace("_", " "))


# Return one stable family-first sort key for object-level website figures.
def _object_family_sort_key(object_name: str) -> tuple[int, int, str]:
    """Return a hammer-then-screwdriver ordering key for one object identifier."""
    object_name = str(object_name)
    family_name, within_family_order = OBJECT_FAMILY_ORDER.get(
        object_name,
        (
            (
                "hammer"
                if "hammer" in object_name
                else "screwdriver" if "screwdriver" in object_name else "other"
            ),
            99,
        ),
    )
    family_rank = {"hammer": 0, "screwdriver": 1}.get(family_name, 2)
    return family_rank, int(within_family_order), _pretty_object_label(object_name)


# Return one experiment-local directory that stores generated thesis figure assets.
def _results_dashboard_asset_dir(results_dir: Path) -> Path:
    """Return the per-experiment asset directory used by the thesis website."""
    return results_dir.resolve() / "website_assets"


# Return one shared matplotlib style context for all thesis figures.
def _figure_style_context() -> dict:
    """Return rcParams overrides for the canonical thesis figure style."""
    return {
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "legend.frameon": False,
        "svg.fonttype": "none",
    }


# Apply the shared thesis visual style to one matplotlib axis.
def _style_axis(axis: plt.Axes, *, x_grid: bool = True, y_grid: bool = False) -> None:
    """Apply the shared thesis figure styling to one axis."""
    axis.spines["left"].set_color("#cfd8dc")
    axis.spines["bottom"].set_color("#cfd8dc")
    axis.tick_params(axis="both", colors="#334e68", length=0)
    axis.set_axisbelow(True)
    if x_grid:
        axis.grid(axis="x", color="#e6dfd5", linewidth=0.8, alpha=0.9)
    if y_grid:
        axis.grid(axis="y", color="#e6dfd5", linewidth=0.8, alpha=0.9)


# Return one canonical mode color from either the raw key or the display label.
def _mode_color(mode_name: str) -> str:
    """Return the canonical thesis color for one mode or display label."""
    normalized_name = str(mode_name)
    if normalized_name in MODE_COLORS:
        return MODE_COLORS[normalized_name]
    for mode_key, mode_label in MODE_LABELS.items():
        if normalized_name == mode_label:
            return MODE_COLORS[mode_key]
    return "#6c757d"


# Save one matplotlib figure to disk as the canonical thesis website asset.
def _save_figure(figure: Figure, output_path: Path) -> None:
    """Persist one matplotlib figure to the requested SVG path and close it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="svg")
    plt.close(figure)


# Return one fallback figure with a centered message for empty-data cases.
def _empty_figure(title: str, message: str) -> Figure:
    """Return one minimal fallback figure when no data is available."""
    with plt.rc_context(_figure_style_context()):
        figure, axis = plt.subplots(figsize=(10, 3))
        axis.axis("off")
        axis.set_title(title, loc="left", color="#243b53", pad=10)
        axis.text(0.5, 0.5, message, ha="center", va="center", color="#52606d", fontsize=12)
        return figure


# Draw one shared lollipop-style percentage axis for thesis dashboard figures.
def _plot_lollipop_percentage_axis(
    axis: plt.Axes,
    *,
    category_labels: List[str],
    value_rows: List[np.ndarray],
    color_rows: List[List[str]],
    annotation_rows: Optional[List[List[str]]] = None,
    row_labels: Optional[List[str]] = None,
    x_label: str = "Rate",
    x_max: float = 1.0,
) -> None:
    """Render one shared lollipop-style percentage axis for one or more series."""
    positions = np.arange(len(category_labels))
    offsets = np.array([0.0]) if len(value_rows) <= 1 else np.linspace(-0.12, 0.12, len(value_rows))
    for row_index, (values, colors, offset) in enumerate(zip(value_rows, color_rows, offsets)):
        y_positions = positions + float(offset)
        axis.scatter(
            values,
            y_positions,
            s=85,
            color=colors,
            label=None if row_labels is None else row_labels[row_index],
            zorder=3,
        )
        annotations = (
            [f"{value:.0%}" for value in values]
            if annotation_rows is None
            else annotation_rows[row_index]
        )
        for value, y_position, color, annotation in zip(values, y_positions, colors, annotations):
            axis.hlines(y_position, 0.0, 1.0, color="#d9e2ec", linewidth=2, zorder=1)
            axis.hlines(y_position, 0.0, value, color=color, alpha=0.3, linewidth=3, zorder=2)
            ha = "left" if value < 0.92 else "right"
            text_x = value + 0.015 if ha == "left" else value - 0.015
            axis.text(
                text_x,
                y_position,
                annotation,
                va="center",
                ha=ha,
                color="#243b53",
                fontweight="600",
            )
    axis.set_yticks(positions)
    axis.set_yticklabels(category_labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, max(1.0, float(x_max)))
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel(x_label)


# Build one matplotlib figure for language grounding accuracy by prompt family and component.
def _build_language_accuracy_figure(
    family_summary_df: pd.DataFrame, component_summary_df: pd.DataFrame
) -> Figure:
    """Return one two-panel language figure for prompt families and component accuracies."""
    if family_summary_df.empty:
        return _empty_figure(
            "Language Grounding by Prompt Family", "No language benchmark results available."
        )
    family_plot_df = family_summary_df.copy().sort_values("prompt_family").reset_index(drop=True)
    metric_order = [
        "intent_accuracy",
        "object_accuracy",
        "target_accuracy",
        "clarification_accuracy",
    ]
    component_labels = [LANGUAGE_METRIC_LABELS[key] for key in metric_order]
    component_values = component_summary_df.iloc[0][metric_order].astype(float).to_numpy()
    with plt.rc_context(_figure_style_context()):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(12.4, 4.8),
            gridspec_kw={"width_ratios": [1.35, 0.9]},
            sharey=False,
        )
        _plot_lollipop_percentage_axis(
            axes[0],
            category_labels=[
                LANGUAGE_FAMILY_LABELS.get(
                    str(family_name), str(family_name).replace("_", " ").title()
                )
                for family_name in family_plot_df["prompt_family"].tolist()
            ],
            value_rows=[family_plot_df["exact_match_accuracy"].astype(float).to_numpy()],
            color_rows=[[SECTION_COLORS["language"]] * len(family_plot_df)],
            annotation_rows=[
                [
                    f"{float(row['exact_match_accuracy']):.0%}, n={int(row['num_trials'])}"
                    for _, row in family_plot_df.iterrows()
                ]
            ],
            x_label="Exact-Match Accuracy",
            x_max=1.16,
        )
        axes[0].set_ylabel("Prompt Family")
        axes[0].set_title("Exact Match by Prompt Family", loc="left", color="#243b53", pad=8)
        _style_axis(axes[0])
        _plot_lollipop_percentage_axis(
            axes[1],
            category_labels=component_labels,
            value_rows=[component_values],
            color_rows=[[SECTION_COLORS["language"]] * len(component_labels)],
            x_label="Accuracy",
            x_max=1.12,
        )
        axes[1].set_ylabel("Component")
        axes[1].set_title("Overall Component Accuracy", loc="left", color="#243b53", pad=8)
        _style_axis(axes[1])
        figure.suptitle(
            "Language Grounding by Prompt Family",
            x=0.06,
            y=0.99,
            ha="left",
            color="#243b53",
            fontsize=16,
        )
        figure.tight_layout()
        return figure


# Build one matplotlib figure for exact-match pass/fail counts by prompt family.
def _build_language_pass_fail_figure(family_summary_df: pd.DataFrame) -> Figure:
    """Return one stacked count figure showing exact-match passes and failures."""
    if family_summary_df.empty:
        return _empty_figure(
            "Language Prompt-Family Pass/Fail Counts", "No language benchmark results available."
        )
    family_plot_df = family_summary_df.copy().sort_values("prompt_family").reset_index(drop=True)
    if "num_exact_match" in family_plot_df:
        pass_counts = family_plot_df["num_exact_match"].astype(int).to_numpy()
    else:
        pass_counts = np.rint(
            family_plot_df["exact_match_accuracy"].astype(float)
            * family_plot_df["num_trials"].astype(int)
        ).astype(int)
    total_counts = family_plot_df["num_trials"].astype(int).to_numpy()
    fail_counts = np.maximum(total_counts - pass_counts, 0)
    family_labels = [
        LANGUAGE_FAMILY_LABELS.get(str(family_name), str(family_name).replace("_", " ").title())
        for family_name in family_plot_df["prompt_family"].tolist()
    ]
    positions = np.arange(len(family_plot_df))
    with plt.rc_context(_figure_style_context()):
        figure, axis = plt.subplots(figsize=(7.4, 4.8))
        axis.barh(
            positions,
            pass_counts,
            color=SECTION_COLORS["language"],
            label="Pass",
            height=0.58,
        )
        axis.barh(
            positions,
            fail_counts,
            left=pass_counts,
            color="#d95f5f",
            label="Fail",
            height=0.58,
        )
        for y_position, passed, failed, total in zip(
            positions, pass_counts, fail_counts, total_counts
        ):
            axis.text(
                total + 0.12,
                y_position,
                f"{passed}/{total}",
                va="center",
                ha="left",
                color="#243b53",
                fontweight="600",
            )
            if failed > 0:
                axis.text(
                    passed + (failed / 2.0),
                    y_position,
                    str(failed),
                    va="center",
                    ha="center",
                    color="white",
                    fontweight="700",
                )
        axis.set_yticks(positions)
        axis.set_yticklabels(family_labels)
        axis.invert_yaxis()
        axis.set_xlim(0, max(1.0, float(total_counts.max()) + 1.0))
        axis.set_xlabel("Exact-match trial count")
        axis.set_ylabel("Prompt Family")
        axis.legend(loc="lower right", frameon=False)
        _style_axis(axis)
        figure.tight_layout()
        return figure


# Build one family/mode geometry summary used by the presentation website.
def _build_geometry_mode_counts_df(geometry_df: pd.DataFrame) -> pd.DataFrame:
    """Return family-split geometry acceptance and target-error aggregates."""
    mode_order = THESIS_MODE_ORDER
    family_order = ["hammer", "screwdriver"]
    if geometry_df.empty:
        return pd.DataFrame(
            columns=[
                "tool_family",
                "mode",
                "total_trials",
                "valid_trials",
                "invalid_trials",
                "accepted_share",
                "mean_semantic_contact_point_xy_error_m",
            ]
        )
    summary_df = geometry_df.copy()
    if "tool_family" not in summary_df.columns:
        summary_df["tool_family"] = (
            summary_df["object_name"]
            .astype(str)
            .map(
                lambda object_name: (
                    "hammer"
                    if "hammer" in object_name
                    else "screwdriver" if "screwdriver" in object_name else "other"
                )
            )
        )
    summary_df["is_valid"] = summary_df["compile_success"].astype(bool) & summary_df[
        "validation_success"
    ].astype(bool)
    if "semantic_contact_point_xy_error_m" not in summary_df.columns:
        summary_df["semantic_contact_point_xy_error_m"] = np.nan
    if "trial_id" not in summary_df.columns:
        summary_df["trial_id"] = [f"geometry_row_{index}" for index in range(len(summary_df))]
    grouped_df = (
        summary_df.groupby(["tool_family", "mode"], dropna=False)
        .agg(
            total_trials=("trial_id", "count"),
            valid_trials=("is_valid", "sum"),
            mean_semantic_contact_point_xy_error_m=(
                "semantic_contact_point_xy_error_m",
                "mean",
            ),
        )
        .reset_index()
    )
    grouped_df["invalid_trials"] = grouped_df["total_trials"] - grouped_df["valid_trials"]
    grouped_df["accepted_share"] = (
        grouped_df["valid_trials"].astype(float) / grouped_df["total_trials"].replace(0, np.nan)
    ).fillna(0.0)
    grouped_df = grouped_df[
        grouped_df["mode"].isin(mode_order) & grouped_df["tool_family"].isin(family_order)
    ].copy()
    grouped_df["tool_family"] = pd.Categorical(
        grouped_df["tool_family"], categories=family_order, ordered=True
    )
    grouped_df["mode"] = pd.Categorical(grouped_df["mode"], categories=mode_order, ordered=True)
    grouped_df = grouped_df.sort_values(["tool_family", "mode"]).reset_index(drop=True)
    missing_pairs = [
        (family_name, mode_name)
        for family_name in family_order
        for mode_name in mode_order
        if (family_name, mode_name)
        not in {(str(row["tool_family"]), str(row["mode"])) for _, row in grouped_df.iterrows()}
    ]
    if missing_pairs:
        grouped_df = pd.concat(
            [
                grouped_df,
                pd.DataFrame(
                    {
                        "tool_family": [family_name for family_name, _ in missing_pairs],
                        "mode": [mode_name for _, mode_name in missing_pairs],
                        "total_trials": [0] * len(missing_pairs),
                        "valid_trials": [0] * len(missing_pairs),
                        "invalid_trials": [0] * len(missing_pairs),
                        "accepted_share": [0.0] * len(missing_pairs),
                        "mean_semantic_contact_point_xy_error_m": [np.nan] * len(missing_pairs),
                    }
                ),
            ],
            ignore_index=True,
        )
        grouped_df["tool_family"] = pd.Categorical(
            grouped_df["tool_family"], categories=family_order, ordered=True
        )
        grouped_df["mode"] = pd.Categorical(grouped_df["mode"], categories=mode_order, ordered=True)
        grouped_df = grouped_df.sort_values(["tool_family", "mode"]).reset_index(drop=True)
    return grouped_df


# Return one generated-mode subset so semantic figures compare trajectory generators directly.
def _geometry_generated_mode_subset(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows for generated geometry modes, excluding predefined references."""
    generated_df = summary_df[summary_df["mode"].astype(str).isin(GEOMETRY_GENERATED_MODE_ORDER)]
    generated_df = generated_df.copy()
    generated_df["mode"] = pd.Categorical(
        generated_df["mode"],
        categories=GEOMETRY_GENERATED_MODE_ORDER,
        ordered=True,
    )
    return generated_df.sort_values(["mode"]).reset_index(drop=True)


# Format one semantic target-error value in centimeters for direct figure labels.
def _format_semantic_error_cm(value_cm: float) -> str:
    """Return a compact centimeter label for one semantic target-error value."""
    value_cm = float(value_cm)
    if not np.isfinite(value_cm):
        return "n/a"
    if abs(value_cm) < 0.005:
        return "<0.01 cm"
    return f"{value_cm:.1f} cm"


# Build one family-split semantic target-error figure for generated trajectories.
def _build_geometry_validity_figure(geometry_df: pd.DataFrame) -> Figure:
    """Return one figure showing generated-mode semantic target error by family."""
    summary_df = _build_geometry_mode_counts_df(geometry_df)
    if int(summary_df["total_trials"].sum()) == 0:
        return _empty_figure(
            "Geometry Aggregate Summary",
            "No geometry benchmark results available.",
        )
    summary_df = _geometry_generated_mode_subset(summary_df)
    if summary_df.empty:
        return _empty_figure(
            "Geometry Aggregate Summary",
            "No generated geometry benchmark results available.",
        )
    families = ["hammer", "screwdriver"]
    mode_order = GEOMETRY_GENERATED_MODE_ORDER
    with plt.rc_context(_figure_style_context()):
        figure, axis = plt.subplots(1, 1, figsize=(8.2, 3.7))
        labels = ["Hammer", "Screwdriver"]
        error_rows: List[np.ndarray] = []
        for mode_name in mode_order:
            mode_subset = summary_df[summary_df["mode"].astype(str) == mode_name].copy()
            mode_subset["tool_family"] = pd.Categorical(
                mode_subset["tool_family"], categories=families, ordered=True
            )
            mode_subset = mode_subset.sort_values("tool_family")
            error_values = (
                mode_subset["mean_semantic_contact_point_xy_error_m"].astype(float).fillna(0.0)
            )
            error_rows.append((error_values * 100.0).to_numpy())
        positions = np.arange(len(labels))
        offsets = np.linspace(-0.14, 0.14, len(mode_order))
        all_error_values = np.asarray(error_rows, dtype=float)
        finite_error_values = all_error_values[np.isfinite(all_error_values)]
        max_error_value = float(np.max(finite_error_values)) if finite_error_values.size else 0.0
        x_max = max(1.2, max_error_value * 1.23)
        for offset, mode_name, values in zip(offsets, mode_order, error_rows):
            y_positions = positions + float(offset)
            axis.scatter(
                values,
                y_positions,
                s=92,
                color=_mode_color(mode_name),
                edgecolors="white",
                linewidths=0.8,
                label=_pretty_geometry_mode_label(mode_name),
                zorder=3,
            )
            for value, y_position in zip(values, y_positions):
                label_x = max(float(value), x_max * 0.035) + x_max * 0.025
                axis.hlines(y_position, 0.0, float(value), color="#d9e2ec", linewidth=1.8)
                axis.text(
                    label_x,
                    y_position,
                    _format_semantic_error_cm(float(value)),
                    va="center",
                    ha="left",
                    color="#243b53",
                    fontsize=10,
                )
        axis.set_yticks(positions)
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        axis.set_xlim(0.0, x_max)
        axis.set_xlabel("Semantic Target Error (cm)")
        axis.set_ylabel("Tool Family")
        _style_axis(axis)
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, -0.24),
            ncol=2,
            title="Mode",
            borderaxespad=0.0,
        )
        figure.tight_layout()
        return figure


# Build one object-split geometry summary used by the presentation website.
def _build_geometry_object_semantic_df(geometry_df: pd.DataFrame) -> pd.DataFrame:
    """Return object-split semantic target-error aggregates for generated modes."""
    if geometry_df.empty:
        return pd.DataFrame(
            columns=[
                "object_name",
                "tool_family",
                "mode",
                "num_trials",
                "mean_semantic_contact_point_xy_error_m",
            ]
        )
    summary_df = geometry_df.copy()
    if "tool_family" not in summary_df.columns:
        summary_df["tool_family"] = (
            summary_df["object_name"]
            .astype(str)
            .map(
                lambda object_name: (
                    "hammer"
                    if "hammer" in object_name
                    else "screwdriver" if "screwdriver" in object_name else "other"
                )
            )
        )
    if "semantic_contact_point_xy_error_m" not in summary_df.columns:
        summary_df["semantic_contact_point_xy_error_m"] = np.nan
    grouped_df = (
        summary_df[summary_df["mode"].astype(str).isin(GEOMETRY_GENERATED_MODE_ORDER)]
        .groupby(["object_name", "tool_family", "mode"], dropna=False)
        .agg(
            num_trials=("mode", "count"),
            mean_semantic_contact_point_xy_error_m=(
                "semantic_contact_point_xy_error_m",
                "mean",
            ),
        )
        .reset_index()
    )
    grouped_df["object_sort_key"] = grouped_df["object_name"].map(_object_family_sort_key)
    grouped_df["mode"] = pd.Categorical(
        grouped_df["mode"],
        categories=GEOMETRY_GENERATED_MODE_ORDER,
        ordered=True,
    )
    grouped_df = grouped_df.sort_values(["object_sort_key", "mode"]).reset_index(drop=True)
    return grouped_df.drop(columns=["object_sort_key"])


# Build one object-level semantic target-error figure for generated trajectories.
def _build_geometry_object_semantic_figure(geometry_df: pd.DataFrame) -> Figure:
    """Return one figure showing generated-mode semantic target error by object."""
    summary_df = _build_geometry_object_semantic_df(geometry_df)
    if summary_df.empty:
        return _empty_figure(
            "Semantic Target Error by Object",
            "No geometry benchmark results available.",
        )
    object_names = list(dict.fromkeys(summary_df["object_name"].astype(str).tolist()))
    labels = [_pretty_object_label(object_name).title() for object_name in object_names]
    with plt.rc_context(_figure_style_context()):
        figure, axis = plt.subplots(1, 1, figsize=(8.6, 4.8))
        positions = np.arange(len(object_names))
        offsets = np.linspace(-0.14, 0.14, len(GEOMETRY_GENERATED_MODE_ORDER))
        error_rows: List[np.ndarray] = []
        for mode_name in GEOMETRY_GENERATED_MODE_ORDER:
            values = []
            for object_name in object_names:
                value_series = summary_df[
                    (summary_df["mode"].astype(str) == mode_name)
                    & (summary_df["object_name"].astype(str) == object_name)
                ]["mean_semantic_contact_point_xy_error_m"].astype(float)
                values.append(
                    float(value_series.iloc[0]) * 100.0 if not value_series.empty else 0.0
                )
            error_rows.append(np.asarray(values, dtype=float))
        all_error_values = np.asarray(error_rows, dtype=float)
        finite_error_values = all_error_values[np.isfinite(all_error_values)]
        max_error_value = float(np.max(finite_error_values)) if finite_error_values.size else 0.0
        x_max = max(1.2, max_error_value * 1.23)
        family_names = [
            str(
                summary_df[summary_df["object_name"].astype(str) == object_name][
                    "tool_family"
                ].iloc[0]
            )
            for object_name in object_names
        ]
        for offset, mode_name, values in zip(
            offsets,
            GEOMETRY_GENERATED_MODE_ORDER,
            error_rows,
        ):
            y_positions = positions + float(offset)
            axis.scatter(
                values,
                y_positions,
                s=82,
                color=_mode_color(mode_name),
                edgecolors="white",
                linewidths=0.8,
                label=_pretty_geometry_mode_label(mode_name),
                zorder=3,
            )
            for value, y_position in zip(values, y_positions):
                label_x = max(float(value), x_max * 0.035) + x_max * 0.02
                axis.hlines(y_position, 0.0, float(value), color="#d9e2ec", linewidth=1.6)
                axis.text(
                    label_x,
                    y_position,
                    _format_semantic_error_cm(float(value)),
                    va="center",
                    ha="left",
                    color="#243b53",
                    fontsize=9,
                )
        for index, (current_family, next_family) in enumerate(zip(family_names, family_names[1:])):
            if current_family != next_family:
                axis.axhline(
                    index + 0.5,
                    color="#cfd8dc",
                    linewidth=1.0,
                    linestyle="--",
                    zorder=1,
                )
        axis.set_yticks(positions)
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        axis.set_xlim(0.0, x_max)
        axis.set_xlabel("Semantic Target Error (cm)")
        axis.set_ylabel("Object")
        _style_axis(axis)
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, -0.17),
            ncol=2,
            title="Mode",
            borderaxespad=0.0,
        )
        figure.tight_layout()
        return figure


# Return one family label for geometry artifacts that may predate flattened family columns.
def _geometry_tool_family(payload: Dict[str, object]) -> str:
    """Return the tool family from metrics or infer it from the object name."""
    metrics = payload.get("metrics", {})
    if isinstance(metrics, dict) and metrics.get("tool_family") is not None:
        return str(metrics["tool_family"])
    object_name = str(payload.get("object_name", ""))
    if "hammer" in object_name:
        return "hammer"
    if "screwdriver" in object_name:
        return "screwdriver"
    return "other"


# Build trial-level requested/implied target rows from saved raw geometry artifacts.
def _build_geometry_implied_target_topdown_df(results_dir: Path) -> pd.DataFrame:
    """Return generated trial rows with requested and implied tabletop target XY."""
    rows: List[Dict[str, object]] = []
    for payload in _load_geometry_raw_payloads(results_dir).values():
        mode = str(payload.get("mode", ""))
        if mode not in GEOMETRY_GENERATED_MODE_ORDER:
            continue
        if not bool(payload.get("compile_success")) or not bool(payload.get("validation_success")):
            continue
        target_xy = payload.get("target_xy")
        goals = payload.get("goals")
        object_name = str(payload.get("object_name", ""))
        if not isinstance(target_xy, list) or len(target_xy) < 2:
            continue
        if not isinstance(goals, list) or not goals:
            continue
        semantic_frame_index = _semantic_target_frame_index(object_name, goals)
        if semantic_frame_index is None:
            continue
        implied_target = semantic_strike_axis_table_intersection(
            goals[semantic_frame_index],
            object_name,
        )
        if implied_target is None:
            continue
        rows.append(
            {
                "trial_id": str(payload.get("trial_id", "")),
                "object_name": object_name,
                "tool_family": _geometry_tool_family(payload),
                "mode": mode,
                "requested_x": float(target_xy[0]),
                "requested_y": float(target_xy[1]),
                "implied_x": float(implied_target[0]),
                "implied_y": float(implied_target[1]),
                "semantic_target_frame_index": int(semantic_frame_index),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "trial_id",
                "object_name",
                "tool_family",
                "mode",
                "requested_x",
                "requested_y",
                "implied_x",
                "implied_y",
                "semantic_target_frame_index",
            ]
        )
    topdown_df = pd.DataFrame(rows)
    topdown_df["object_sort_key"] = topdown_df["object_name"].map(_object_family_sort_key)
    topdown_df["mode"] = pd.Categorical(
        topdown_df["mode"],
        categories=GEOMETRY_GENERATED_MODE_ORDER,
        ordered=True,
    )
    topdown_df = topdown_df.sort_values(
        ["tool_family", "mode", "object_sort_key", "requested_x", "requested_y"]
    ).reset_index(drop=True)
    return topdown_df.drop(columns=["object_sort_key"])


# Build one family-specific top-down requested-vs-implied target distribution figure.
def _build_geometry_implied_target_family_figure(results_dir: Path, family_name: str) -> Figure:
    """Return one top-down requested-vs-implied target plot for a tool family."""
    topdown_df = _build_geometry_implied_target_topdown_df(results_dir)
    if topdown_df.empty:
        return _empty_figure(
            "Requested vs Implied Target",
            "No generated geometry target rows available.",
        )
    family_labels = {"hammer": "Hammer", "screwdriver": "Screwdriver"}
    family_name = str(family_name)
    family_df = topdown_df[topdown_df["tool_family"].astype(str) == family_name].copy()
    if family_df.empty:
        return _empty_figure(
            f"{family_labels.get(family_name, family_name.title())} Requested vs Implied Target",
            "No generated geometry target rows available.",
        )
    with plt.rc_context(_figure_style_context()):
        figure, axis = plt.subplots(1, 1, figsize=(6.8, 5.2))
        requested_label = (
            "Requested hammer target"
            if family_name == "hammer"
            else (
                "Requested screwdriver target"
                if family_name == "screwdriver"
                else "Requested target"
            )
        )
        requested_df = family_df[["requested_x", "requested_y"]].drop_duplicates()
        family_x = pd.concat([family_df["requested_x"], family_df["implied_x"]]).astype(float)
        family_y = pd.concat([family_df["requested_y"], family_df["implied_y"]]).astype(float)
        x_padding = max(0.03, float(family_x.max() - family_x.min()) * 0.12)
        y_padding = max(0.03, float(family_y.max() - family_y.min()) * 0.12)
        requested_x_cm = requested_df["requested_x"].astype(float) * 100.0
        requested_y_cm = requested_df["requested_y"].astype(float) * 100.0
        axis.scatter(
            requested_x_cm,
            requested_y_cm,
            marker="x",
            s=64,
            color="#52606d",
            alpha=0.76,
            linewidths=1.8,
            label=requested_label,
            zorder=2,
        )
        for mode_name in GEOMETRY_GENERATED_MODE_ORDER:
            mode_df = family_df[family_df["mode"].astype(str) == mode_name]
            for _, row in mode_df.iterrows():
                axis.plot(
                    [float(row["requested_x"]) * 100.0, float(row["implied_x"]) * 100.0],
                    [float(row["requested_y"]) * 100.0, float(row["implied_y"]) * 100.0],
                    color=_mode_color(mode_name),
                    alpha=0.18,
                    linewidth=0.9,
                    zorder=1,
                )
            axis.scatter(
                mode_df["implied_x"].astype(float) * 100.0,
                mode_df["implied_y"].astype(float) * 100.0,
                s=42,
                color=_mode_color(mode_name),
                edgecolors="white",
                linewidths=0.6,
                alpha=0.88,
                label=_pretty_geometry_mode_label(mode_name),
                zorder=4,
            )
        axis.set_xlabel("Table X (cm)")
        axis.set_ylabel("Table Y (cm)")
        axis.set_xlim(
            (float(family_x.min()) - x_padding) * 100.0,
            (float(family_x.max()) + x_padding) * 100.0,
        )
        axis.set_ylim(
            (float(family_y.min()) - y_padding) * 100.0,
            (float(family_y.max()) + y_padding) * 100.0,
        )
        axis.set_aspect("equal", adjustable="box")
        _style_axis(axis)
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="x",
                color="#52606d",
                linestyle="None",
                markersize=8,
                markeredgewidth=1.7,
                label=requested_label,
            ),
            *[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=_mode_color(mode_name),
                    linestyle="None",
                    markersize=7,
                    label=_pretty_geometry_mode_label(mode_name),
                )
                for mode_name in GEOMETRY_GENERATED_MODE_ORDER
            ],
        ]
        axis.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.24 if family_name == "hammer" else -0.16),
            ncol=3,
            title="Marker",
            frameon=False,
        )
        figure.tight_layout(rect=(0, 0.14, 1, 1))
        return figure


# Load all saved raw geometry payloads for one experiment directory.
def _load_geometry_raw_payloads(results_dir: Path) -> Dict[str, dict]:
    """Return raw geometry payloads keyed by trial id when saved artifacts are available."""
    raw_dir = results_dir / "geometry" / "raw"
    if not raw_dir.exists():
        return {}
    payloads: Dict[str, dict] = {}
    for path in sorted(raw_dir.glob("*.json")):
        payload = read_json(path)
        trial_id = str(payload.get("trial_id", path.stem))
        payloads[trial_id] = payload
    return payloads


# Return one deterministic object exemplar for the requested family from saved geometry rows.
def _select_geometry_family_object(geometry_df: pd.DataFrame, family_name: str) -> Optional[str]:
    """Return the preferred saved geometry object for one family or None when unavailable."""
    family_order = {
        "hammer": ["claw_hammer", "mallet_hammer", "cuboid_hammer_v014"],
        "screwdriver": ["long_screwdriver", "short_screwdriver", "cylinder_screwdriver_v3009"],
    }
    available_objects = (
        set(geometry_df["object_name"].astype(str)) if not geometry_df.empty else set()
    )
    for object_name in family_order.get(family_name, []):
        if object_name in available_objects:
            return object_name
    return None


# Return the saved predefined payload for one object when present.
def _select_geometry_raw_payload(
    payloads: Dict[str, dict], *, object_name: str, mode: str
) -> Optional[dict]:
    """Return the first saved raw geometry payload matching one object and mode."""
    for payload in payloads.values():
        if str(payload.get("object_name")) != str(object_name):
            continue
        if str(payload.get("mode")) != str(mode):
            continue
        return payload
    return None


# Return one target-a XY location for the requested object/task pair.
def _target_a_xy(object_name: str, task_name: str) -> Optional[List[float]]:
    """Return target-a XY coordinates for one object/task pair when available."""
    named_points = get_named_strike_points(object_name=object_name, task_name=task_name)
    target_a = named_points.get("target_a")
    if not isinstance(target_a, list) or len(target_a) != 2:
        return None
    return [float(target_a[0]), float(target_a[1])]


# Load all saved supplemental target-a exemplar payloads for one experiment directory.
def _load_geometry_exemplar_payloads(results_dir: Path) -> Dict[tuple[str, str, str], dict]:
    """Return saved thesis exemplar payloads keyed by object, mode, and target name."""
    exemplar_dir = results_dir / "geometry" / "exemplars"
    if not exemplar_dir.exists():
        return {}
    payloads: Dict[tuple[str, str, str], dict] = {}
    for path in sorted(exemplar_dir.glob("*.json")):
        payload = read_json(path)
        key = (
            str(payload.get("object_name")),
            str(payload.get("mode")),
            str(payload.get("target_name")),
        )
        payloads[key] = payload
    return payloads


# Return one saved target-a exemplar payload when present.
def _select_geometry_exemplar_payload(
    payloads: Dict[tuple[str, str, str], dict], *, object_name: str, mode: str
) -> Optional[dict]:
    """Return one saved target-a exemplar payload for an object/mode pair."""
    return payloads.get((str(object_name), str(mode), "target_a"))


# Build one deterministic exemplar payload bundle for the requested tool family.
def _build_geometry_target_a_bundle(
    geometry_df: pd.DataFrame,
    payloads: Dict[str, dict],
    exemplar_payloads: Dict[tuple[str, str, str], dict],
    *,
    family_name: str,
) -> Optional[dict]:
    """Return one target-a comparison bundle for the requested family or None when unavailable."""
    object_name = _select_geometry_family_object(geometry_df, family_name)
    if object_name is None:
        return None
    predefined_payload = _select_geometry_raw_payload(
        payloads, object_name=object_name, mode="predefined"
    )
    if predefined_payload is None:
        return None
    task_name = str(predefined_payload.get("task_name"))
    target_a_xy = _target_a_xy(object_name, task_name)
    if target_a_xy is None:
        return None
    llm_lie_payload = _select_geometry_exemplar_payload(
        exemplar_payloads, object_name=object_name, mode="llm_lie"
    )
    llm_direct_payload = _select_geometry_exemplar_payload(
        exemplar_payloads, object_name=object_name, mode="llm_only"
    )
    missing_exemplar = llm_lie_payload is None or llm_direct_payload is None
    used_target_xy = (
        llm_lie_payload.get("target_xy")
        if llm_lie_payload is not None and isinstance(llm_lie_payload.get("target_xy"), list)
        else target_a_xy
    )
    requested_target_xy = (
        llm_lie_payload.get("metrics", {}).get("requested_target_xy")
        if llm_lie_payload is not None
        and isinstance(llm_lie_payload.get("metrics"), dict)
        and isinstance(llm_lie_payload.get("metrics", {}).get("requested_target_xy"), list)
        else target_a_xy
    )
    return {
        "family_name": family_name,
        "object_name": object_name,
        "task_name": task_name,
        "target_a_xy": target_a_xy,
        "used_target_xy": [float(value) for value in used_target_xy],
        "requested_target_xy": [float(value) for value in requested_target_xy],
        "predefined_goals": [list(goal) for goal in predefined_payload.get("goals", [])],
        "llm_lie_goals": (
            None if llm_lie_payload is None else [list(goal) for goal in llm_lie_payload["goals"]]
        ),
        "llm_direct_goals": (
            None
            if llm_direct_payload is None
            else [list(goal) for goal in llm_direct_payload["goals"]]
        ),
        "missing_exemplar": missing_exemplar,
    }


# Return world-space semantic contact points for one saved pose trajectory.
def _trajectory_semantic_contact_points(object_name: str, goals: List[List[float]]) -> np.ndarray:
    """Return world-space semantic contact points for each saved waypoint."""
    semantics = get_object_pose_semantics(object_name)
    contact_points: List[List[float]] = []
    for goal in goals:
        position = np.asarray(goal[:3], dtype=float)
        rotated_offset = np.asarray(
            quat_rotate_xyzw(goal[3:7], semantics.strike_point_local), dtype=float
        )
        contact_points.append((position + rotated_offset).tolist())
    return np.asarray(contact_points, dtype=float)


# Plot one trajectory overlay panel in either top view or side view.
def _plot_geometry_overlay_panel(
    axis: plt.Axes,
    *,
    title: str,
    object_name: str,
    trajectories: list[tuple[str, Optional[List[List[float]]], str]],
    used_target_xy: Optional[List[float]],
    requested_target_xy: Optional[List[float]],
    view: str,
) -> None:
    """Render one target-a trajectory comparison panel for one requested 2D projection."""
    plotted_any = False
    for label, goals, color in trajectories:
        if not goals:
            continue
        positions = np.asarray([goal[:3] for goal in goals], dtype=float)
        contact_points = _trajectory_semantic_contact_points(object_name, goals)
        line_alpha = 0.58 if label == "Predefined" else 0.9
        line_width = 1.8 if label == "Predefined" else 2.5
        if view == "top":
            axis.plot(
                positions[:, 0],
                positions[:, 1],
                color=color,
                linewidth=1.2,
                alpha=0.32,
                linestyle="--",
            )
            axis.plot(
                contact_points[:, 0],
                contact_points[:, 1],
                color=color,
                linewidth=line_width,
                alpha=line_alpha,
                label=label,
            )
            start_y = contact_points[0, 1]
            end_y = contact_points[-1, 1]
        else:
            axis.plot(
                positions[:, 0],
                positions[:, 2],
                color=color,
                linewidth=1.2,
                alpha=0.32,
                linestyle="--",
            )
            axis.plot(
                contact_points[:, 0],
                contact_points[:, 2],
                color=color,
                linewidth=line_width,
                alpha=line_alpha,
                label=label,
            )
            start_y = contact_points[0, 2]
            end_y = contact_points[-1, 2]
        axis.scatter(
            contact_points[0, 0],
            start_y,
            color=color,
            marker="o",
            s=42,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.85,
            zorder=4,
        )
        axis.scatter(
            contact_points[-1, 0],
            end_y,
            color=color,
            marker="s",
            s=52,
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
        plotted_any = True
    if view == "top" and used_target_xy is not None:
        if requested_target_xy is not None:
            requested_delta = np.linalg.norm(
                np.asarray(used_target_xy, dtype=float)
                - np.asarray(requested_target_xy, dtype=float)
            )
            if requested_delta > 1e-6:
                axis.scatter(
                    float(requested_target_xy[0]),
                    float(requested_target_xy[1]),
                    marker="X",
                    s=75,
                    color="#f4a261",
                    alpha=0.28,
                    edgecolors="none",
                    zorder=3,
                )
        axis.scatter(
            float(used_target_xy[0]),
            float(used_target_xy[1]),
            marker="X",
            s=90,
            color="#f4a261",
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
        axis.annotate(
            "used target",
            xy=(float(used_target_xy[0]), float(used_target_xy[1])),
            xytext=(8, 8),
            textcoords="offset points",
            color="#8a4f00",
            fontsize=9,
        )
    elif view == "side" and used_target_xy is not None:
        axis.axvline(
            float(used_target_xy[0]),
            color="#f4a261",
            linewidth=1.4,
            linestyle=":",
            alpha=0.9,
            zorder=1,
        )
    if not plotted_any:
        axis.text(
            0.5, 0.5, "Trajectory unavailable", ha="center", va="center", transform=axis.transAxes
        )
    axis.set_title(title, loc="left", color="#243b53", pad=8, fontsize=12)
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)" if view == "top" else "Z (m)")
    _style_axis(axis)


# Return normalized progress samples for one plotted trajectory.
def _trajectory_progress(goals: List[List[float]]) -> np.ndarray:
    """Return normalized progress coordinates for one pose list."""
    return np.linspace(0.0, 1.0, num=max(len(goals), 1))


# Return the primary-axis tilt from vertical for one trajectory.
def _trajectory_axis_tilt_deg(object_name: str, goals: List[List[float]]) -> np.ndarray:
    """Return per-waypoint tilt angles showing how upright the tool remains."""
    semantics = get_object_pose_semantics(object_name)
    tilts_deg: List[float] = []
    for goal in goals:
        primary_axis_world = quat_rotate_xyzw(goal[3:], semantics.primary_axis_local)
        vertical_alignment = max(-1.0, min(1.0, abs(float(primary_axis_world[2]))))
        tilts_deg.append(float(np.degrees(np.arccos(vertical_alignment))))
    return np.asarray(tilts_deg, dtype=float)


# Return the cumulative twist angle for one screwdriver trajectory.
def _trajectory_twist_angle_deg(object_name: str, goals: List[List[float]]) -> np.ndarray:
    """Return unwrapped face-normal azimuth angles relative to the first waypoint."""
    semantics = get_object_pose_semantics(object_name)
    azimuths_rad: List[float] = []
    for goal in goals:
        face_normal_world = quat_rotate_xyzw(goal[3:], semantics.face_normal_local)
        azimuths_rad.append(float(np.arctan2(face_normal_world[1], face_normal_world[0])))
    unwrapped_deg = np.degrees(np.unwrap(np.asarray(azimuths_rad, dtype=float)))
    return unwrapped_deg - float(unwrapped_deg[0])


# Plot one progress-aligned comparison panel for scalar geometric diagnostics.
def _plot_geometry_progress_panel(
    axis: plt.Axes,
    *,
    title: str,
    object_name: str,
    trajectories: list[tuple[str, Optional[List[List[float]]], str]],
    value_kind: str,
) -> None:
    """Render one scalar-vs-progress geometry panel for screwdriver exemplar diagnostics."""
    plotted_any = False
    ylabel = "Value"
    for label, goals, color in trajectories:
        if not goals:
            continue
        progress = _trajectory_progress(goals)
        line_alpha = 0.58 if label == "Predefined" else 0.9
        line_width = 1.8 if label == "Predefined" else 2.5
        if value_kind == "axis_tilt":
            values = _trajectory_axis_tilt_deg(object_name, goals)
            ylabel = "Primary-axis Tilt (deg), lower is better"
        else:
            values = _trajectory_twist_angle_deg(object_name, goals)
            ylabel = "Accumulated Twist (deg)"
        axis.plot(
            progress,
            values,
            color=color,
            linewidth=line_width,
            alpha=line_alpha,
            label=label,
        )
        axis.scatter(
            progress[0],
            values[0],
            color=color,
            marker="o",
            s=42,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.85,
        )
        axis.scatter(
            progress[-1],
            values[-1],
            color=color,
            marker="s",
            s=52,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        plotted_any = True
    if not plotted_any:
        axis.text(
            0.5, 0.5, "Trajectory unavailable", ha="center", va="center", transform=axis.transAxes
        )
    axis.axhline(0.0, color="#d9e2ec", linewidth=1.2, zorder=0)
    axis.set_title(title, loc="left", color="#243b53", pad=8, fontsize=12)
    axis.set_xlabel("Normalized Progress")
    axis.set_ylabel(ylabel)
    axis.set_xlim(0.0, 1.0)
    _style_axis(axis)


# Build one visual target-a comparison figure for one hammer and one screwdriver exemplar.
def _build_geometry_target_a_comparison_figure(
    results_dir: Path, geometry_df: pd.DataFrame
) -> Figure:
    """Return one target-a visual comparison figure for hammer and screwdriver exemplars."""
    if geometry_df.empty:
        return _empty_figure(
            "Target-A Trajectory Comparison",
            "No geometry benchmark results available.",
        )
    payloads = _load_geometry_raw_payloads(results_dir)
    exemplar_payloads = _load_geometry_exemplar_payloads(results_dir)
    hammer_bundle = _build_geometry_target_a_bundle(
        geometry_df,
        payloads,
        exemplar_payloads,
        family_name="hammer",
    )
    screwdriver_bundle = _build_geometry_target_a_bundle(
        geometry_df,
        payloads,
        exemplar_payloads,
        family_name="screwdriver",
    )
    bundles = [hammer_bundle, screwdriver_bundle]
    with plt.rc_context(_figure_style_context()):
        figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.8))
        for row_index, bundle in enumerate(bundles):
            family_label = "Hammer" if row_index == 0 else "Screwdriver"
            if bundle is None or bool(bundle.get("missing_exemplar", False)):
                for column_index in range(2):
                    axis = axes[row_index, column_index]
                    axis.axis("off")
                    axis.set_title(
                        f"{family_label} Exemplar",
                        loc="left",
                        color="#243b53",
                        pad=8,
                        fontsize=12,
                    )
                    axis.text(
                        0.5,
                        0.5,
                        "Target-a exemplar unavailable for this experiment.",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                        color="#52606d",
                    )
                continue
            trajectories = [
                ("Predefined", bundle["predefined_goals"], _mode_color("predefined")),
                (_pretty_mode_label("llm_lie"), bundle["llm_lie_goals"], _mode_color("llm_lie")),
                (
                    _pretty_mode_label("llm_only"),
                    bundle["llm_direct_goals"],
                    _mode_color("llm_only"),
                ),
            ]
            object_title = _pretty_object_label(str(bundle["object_name"]))
            if bundle["family_name"] == "hammer":
                _plot_geometry_overlay_panel(
                    axes[row_index, 0],
                    title=f"{family_label}: {object_title} Top View",
                    object_name=str(bundle["object_name"]),
                    trajectories=trajectories,
                    used_target_xy=bundle["used_target_xy"],
                    requested_target_xy=bundle["requested_target_xy"],
                    view="top",
                )
                _plot_geometry_overlay_panel(
                    axes[row_index, 1],
                    title=f"{family_label}: {object_title} Side View",
                    object_name=str(bundle["object_name"]),
                    trajectories=trajectories,
                    used_target_xy=bundle["used_target_xy"],
                    requested_target_xy=bundle["requested_target_xy"],
                    view="side",
                )
            else:
                _plot_geometry_overlay_panel(
                    axes[row_index, 0],
                    title=f"{family_label}: {object_title} Top View",
                    object_name=str(bundle["object_name"]),
                    trajectories=trajectories,
                    used_target_xy=bundle["used_target_xy"],
                    requested_target_xy=bundle["requested_target_xy"],
                    view="top",
                )
                _plot_geometry_overlay_panel(
                    axes[row_index, 1],
                    title=f"{family_label}: {object_title} Side View",
                    object_name=str(bundle["object_name"]),
                    trajectories=trajectories,
                    used_target_xy=bundle["used_target_xy"],
                    requested_target_xy=bundle["requested_target_xy"],
                    view="side",
                )
        legend_handles = [
            Line2D([0], [0], color=_mode_color("predefined"), linewidth=2.0, label="Predefined"),
            Line2D(
                [0],
                [0],
                color=_mode_color("llm_lie"),
                linewidth=2.4,
                label=_pretty_mode_label("llm_lie"),
            ),
            Line2D(
                [0],
                [0],
                color=_mode_color("llm_only"),
                linewidth=2.4,
                label=_pretty_mode_label("llm_only"),
            ),
            Line2D(
                [0],
                [0],
                color="#52606d",
                linewidth=2.2,
                label="Semantic contact",
            ),
            Line2D(
                [0],
                [0],
                color="#52606d",
                linewidth=1.2,
                linestyle="--",
                alpha=0.45,
                label="Pose origin",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#52606d",
                markeredgecolor="white",
                markersize=7,
                label="Start",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="#52606d",
                markeredgecolor="white",
                markersize=7,
                label="End",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="none",
                markerfacecolor="#f4a261",
                markeredgecolor="white",
                markersize=8,
                label="Used Target",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="none",
                markerfacecolor="#f4a261",
                markeredgecolor="none",
                alpha=0.35,
                markersize=8,
                label="Requested Target",
            ),
        ]
        figure.legend(
            legend_handles,
            [handle.get_label() for handle in legend_handles],
            loc="lower center",
            ncol=5,
            bbox_to_anchor=(0.5, 0.005),
        )
        figure.suptitle(
            "Target-A Trajectory Comparison",
            x=0.06,
            y=0.99,
            ha="left",
            color="#243b53",
            fontsize=16,
        )
        figure.tight_layout(rect=(0.0, 0.1, 1.0, 0.96))
        return figure


# Build one method-level execution rate figure.
def _build_execution_rate_figure(
    summary_df: pd.DataFrame,
    *,
    value_column: str,
    x_label: str,
    empty_title: str,
) -> Figure:
    """Return one lollipop figure for a method-level execution rate."""
    if summary_df.empty:
        return _empty_figure(
            empty_title,
            "No pretrained execution results available.",
        )
    plotting_df = summary_df.copy()
    plotting_df["mode"] = pd.Categorical(
        plotting_df["mode"], categories=THESIS_MODE_ORDER, ordered=True
    )
    plotting_df = plotting_df.sort_values("mode").reset_index(drop=True)
    labels = [_pretty_mode_label(str(mode_name)) for mode_name in plotting_df["mode"].astype(str)]
    if value_column == "mean_goal_completion_pct":
        values = plotting_df[value_column].astype(float).to_numpy() / 100.0
    else:
        values = plotting_df[value_column].astype(float).to_numpy()
    colors = [_mode_color(str(mode_name)) for mode_name in plotting_df["mode"].astype(str)]
    with plt.rc_context(_figure_style_context()):
        figure, axis = plt.subplots(figsize=(6.8, 3.3))
        trial_counts = plotting_df["num_trials"].astype(int).to_numpy()
        annotations = [
            f"{value:.0%}, n={trial_count}" for value, trial_count in zip(values, trial_counts)
        ]
        _plot_lollipop_percentage_axis(
            axis,
            category_labels=labels,
            value_rows=[values],
            color_rows=[colors],
            annotation_rows=[annotations],
            x_label=x_label,
            x_max=1.15,
        )
        _style_axis(axis)
        figure.tight_layout()
        return figure


# Build one method-level strict success figure for execution.
def _build_execution_strict_success_figure(summary_df: pd.DataFrame) -> Figure:
    """Return one method-level strict success rate figure."""
    return _build_execution_rate_figure(
        summary_df,
        value_column="strict_success_rate",
        x_label="Strict success rate (%)",
        empty_title="Strict Success",
    )


# Build one method-level goal completion figure for execution.
def _build_execution_goal_completion_figure(summary_df: pd.DataFrame) -> Figure:
    """Return one method-level goal completion rate figure."""
    return _build_execution_rate_figure(
        summary_df,
        value_column="mean_goal_completion_pct",
        x_label="Goal completion rate (%)",
        empty_title="Goal Completion",
    )


# Build one RMSE-style family marker figure for an execution percentage metric.
def _build_execution_family_percentage_figure(
    summary_df: pd.DataFrame,
    *,
    title: str,
    value_column: str,
    x_label: str,
    empty_message: str,
) -> Figure:
    """Return one grouped-marker family figure for an execution percentage metric."""
    if summary_df.empty:
        return _empty_figure(title, empty_message)
    plotting_df = summary_df.copy()
    plotting_df["tool_family"] = pd.Categorical(
        plotting_df["tool_family"], categories=["hammer", "screwdriver"], ordered=True
    )
    plotting_df["mode"] = pd.Categorical(
        plotting_df["mode"], categories=THESIS_MODE_ORDER, ordered=True
    )
    plotting_df = plotting_df.sort_values(["tool_family", "mode"]).reset_index(drop=True)
    family_labels = {
        "hammer": "Hammer tools",
        "screwdriver": "Screwdriver tools",
    }
    families = ["hammer", "screwdriver"]
    labels = [family_labels[family_name] for family_name in families]
    y_positions = np.arange(len(labels))
    mode_columns = [
        ("predefined", _pretty_mode_label("predefined")),
        ("llm_lie", _pretty_mode_label("llm_lie")),
        ("llm_only", _pretty_mode_label("llm_only")),
    ]
    row_lookup = {
        (str(row["tool_family"]), str(row["mode"])): row for _, row in plotting_df.iterrows()
    }
    with plt.rc_context(_figure_style_context()):
        figure, axis = plt.subplots(figsize=(9.6, 3.2))
        for y_position in y_positions:
            axis.hlines(y_position, 0.0, 1.08, color="#d9e2ec", linewidth=1.2, zorder=0)
        offsets = np.linspace(-0.14, 0.14, len(mode_columns))
        for offset, (mode_name, mode_label) in zip(offsets, mode_columns):
            values = []
            annotations = []
            for family_name in families:
                row = row_lookup.get((family_name, mode_name))
                if row is None:
                    values.append(np.nan)
                    annotations.append("")
                    continue
                raw_value = float(row[value_column])
                value = raw_value / 100.0 if raw_value > 1.0 else raw_value
                trial_count = int(round(float(row.get("num_trials", 0.0))))
                values.append(value)
                annotations.append(f"{value:.0%}, n={trial_count}")
            values_array = np.asarray(values, dtype=float)
            valid_mask = np.isfinite(values_array)
            axis.scatter(
                values_array[valid_mask],
                (y_positions + float(offset))[valid_mask],
                color=_mode_color(mode_name),
                s=82,
                label=mode_label,
                zorder=3,
            )
            for value, y_position, annotation in zip(
                values_array[valid_mask],
                y_positions[valid_mask],
                np.asarray(annotations, dtype=object)[valid_mask],
            ):
                axis.text(
                    float(value) + 0.026,
                    y_position + float(offset),
                    str(annotation),
                    va="center",
                    ha="left",
                    color="#243b53",
                    fontsize=10,
                )
        axis.set_yticks(y_positions)
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        axis.set_xlim(0.0, 1.13)
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axis.set_xlabel(x_label)
        axis.set_ylabel("Tool Family")
        axis.legend(loc="upper left", bbox_to_anchor=(1.14, 1.0), title="Mode", borderaxespad=0.0)
        _style_axis(axis)
        figure.tight_layout(rect=(0.0, 0.0, 0.76, 1.0))
        return figure


# Build one family-level grouped-marker figure for strict execution success.
def _build_execution_strict_success_by_family_figure(summary_df: pd.DataFrame) -> Figure:
    """Return one family-level strict-success figure for all execution modes."""
    return _build_execution_family_percentage_figure(
        summary_df,
        title="Strict Success by Tool Family",
        value_column="strict_success_rate",
        x_label="Strict success rate",
        empty_message="No pretrained execution results available.",
    )


# Build one family-level grouped-marker figure for goal-completion rate.
def _build_execution_goal_completion_by_family_figure(summary_df: pd.DataFrame) -> Figure:
    """Return one family-level goal-completion figure for all execution modes."""
    return _build_execution_family_percentage_figure(
        summary_df,
        title="Goal Completion Rate by Tool Family",
        value_column="mean_goal_completion_pct",
        x_label="Goal Completion Rate",
        empty_message="No pretrained execution results available.",
    )


# Build one object-level RMSE summary for the execution dumbbell figure.
def _build_execution_rmse_by_object_df(execution_df: pd.DataFrame) -> pd.DataFrame:
    """Return object-level pretrained translation RMSE aggregates for all three modes."""
    columns = [
        "object_name",
        "predefined_rmse_m",
        "llm_only_rmse_m",
        "llm_lie_rmse_m",
    ]
    if execution_df.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for object_name, object_df in execution_df.groupby("object_name", dropna=False):
        predefined_values = object_df[object_df["mode"].astype(str) == "predefined"][
            "translation_rmse_m"
        ].astype(float)
        llm_only_values = object_df[object_df["mode"].astype(str) == "llm_only"][
            "translation_rmse_m"
        ].astype(float)
        lie_values = object_df[object_df["mode"].astype(str) == "llm_lie"][
            "translation_rmse_m"
        ].astype(float)
        if predefined_values.empty and llm_only_values.empty and lie_values.empty:
            continue
        rows.append(
            {
                "object_name": str(object_name),
                "predefined_rmse_m": (
                    float(predefined_values.mean()) if not predefined_values.empty else np.nan
                ),
                "llm_only_rmse_m": (
                    float(llm_only_values.mean()) if not llm_only_values.empty else np.nan
                ),
                "llm_lie_rmse_m": float(lie_values.mean()) if not lie_values.empty else np.nan,
            }
        )
    summary_df = pd.DataFrame(rows, columns=columns)
    if summary_df.empty:
        return summary_df
    summary_df["_sort_key"] = summary_df["object_name"].map(_object_family_sort_key)
    return summary_df.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)


# Return the family label used to group execution RMSE rows.
def _execution_object_family(object_name: str) -> str:
    """Return the thesis-facing object family for one execution object name."""
    family_name, _ = OBJECT_FAMILY_ORDER.get(
        str(object_name),
        (
            (
                "hammer"
                if "hammer" in str(object_name)
                else "screwdriver" if "screwdriver" in str(object_name) else "other"
            ),
            99,
        ),
    )
    return str(family_name)


# Build one family-level RMSE summary for the execution overview figure.
def _build_execution_rmse_by_family_df(execution_df: pd.DataFrame) -> pd.DataFrame:
    """Return family-level pretrained translation RMSE aggregates for all three modes."""
    columns = [
        "tool_family",
        "predefined_rmse_m",
        "llm_lie_rmse_m",
        "llm_only_rmse_m",
    ]
    if execution_df.empty:
        return pd.DataFrame(columns=columns)
    family_df = execution_df.copy()
    family_df["tool_family"] = family_df["object_name"].astype(str).map(_execution_object_family)
    family_df = family_df[family_df["tool_family"].isin(["hammer", "screwdriver"])].copy()
    if family_df.empty:
        return pd.DataFrame(columns=columns)
    grouped_df = (
        family_df.groupby(["tool_family", "mode"], dropna=False)["translation_rmse_m"]
        .mean()
        .unstack("mode")
        .reset_index()
    )
    for mode_name in THESIS_MODE_ORDER:
        if mode_name not in grouped_df.columns:
            grouped_df[mode_name] = np.nan
    grouped_df = grouped_df.rename(
        columns={
            "predefined": "predefined_rmse_m",
            "llm_lie": "llm_lie_rmse_m",
            "llm_only": "llm_only_rmse_m",
        }
    )
    grouped_df["tool_family"] = pd.Categorical(
        grouped_df["tool_family"], categories=["hammer", "screwdriver"], ordered=True
    )
    return grouped_df.sort_values("tool_family").reset_index(drop=True)[columns]


# Build one family-level grouped-marker figure for translation RMSE.
def _build_execution_translation_rmse_by_family_figure(execution_df: pd.DataFrame) -> Figure:
    """Return one family-level translation RMSE figure for all execution modes."""
    title = "Translation RMSE by Tool Family"
    rmse_df = _build_execution_rmse_by_family_df(execution_df)
    if rmse_df.empty:
        return _empty_figure(title, "No pretrained execution rows available.")
    family_labels = {
        "hammer": "Hammer tools",
        "screwdriver": "Screwdriver tools",
    }
    labels = [family_labels.get(str(value), str(value).title()) for value in rmse_df["tool_family"]]
    y_positions = np.arange(len(labels))
    mode_columns = [
        ("predefined_rmse_m", "predefined"),
        ("llm_lie_rmse_m", "llm_lie"),
        ("llm_only_rmse_m", "llm_only"),
    ]
    finite_values_cm = [
        float(value) * 100.0
        for column_name, _ in mode_columns
        for value in rmse_df[column_name].astype(float).tolist()
        if np.isfinite(float(value))
    ]
    x_max = max(finite_values_cm, default=1.0) * 1.22
    with plt.rc_context(_figure_style_context()):
        figure, axis = plt.subplots(figsize=(9.6, 3.2))
        for y_position in y_positions:
            axis.hlines(y_position, 0.0, max(x_max, 0.01), color="#d9e2ec", linewidth=1.2, zorder=0)
        offsets = np.linspace(-0.14, 0.14, len(mode_columns))
        for offset, (column_name, mode_name) in zip(offsets, mode_columns):
            values_cm = rmse_df[column_name].astype(float).to_numpy() * 100.0
            valid_mask = np.isfinite(values_cm)
            axis.scatter(
                values_cm[valid_mask],
                (y_positions + float(offset))[valid_mask],
                color=_mode_color(mode_name),
                s=82,
                label=_pretty_mode_label(mode_name),
                zorder=3,
            )
            for value_cm, y_position in zip(values_cm[valid_mask], y_positions[valid_mask]):
                axis.text(
                    float(value_cm) + max(x_max, 1.0) * 0.035,
                    y_position + float(offset),
                    f"{float(value_cm):.1f} cm",
                    va="center",
                    ha="left",
                    color="#243b53",
                    fontsize=10,
                )
        axis.set_yticks(y_positions)
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        axis.set_xlim(0.0, max(x_max, 1.0))
        axis.set_xlabel("Translation RMSE (cm)")
        axis.set_ylabel("Tool Family")
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Mode", borderaxespad=0.0)
        _style_axis(axis)
        figure.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
        return figure


# Build one object-level dumbbell figure for translation RMSE.
def _build_execution_translation_rmse_figure(execution_df: pd.DataFrame) -> Figure:
    """Return one object-level grouped-marker translation RMSE figure for all modes."""
    title = "Translation RMSE by Object"
    rmse_df = _build_execution_rmse_by_object_df(execution_df)
    if rmse_df.empty:
        return _empty_figure(title, "No pretrained execution rows available.")
    labels = [_pretty_object_label(str(value)) for value in rmse_df["object_name"]]
    family_ranks = [
        _object_family_sort_key(str(value))[0] for value in rmse_df["object_name"].astype(str)
    ]
    y_positions = np.arange(len(labels))
    mode_columns = [
        ("predefined_rmse_m", "predefined"),
        ("llm_lie_rmse_m", "llm_lie"),
        ("llm_only_rmse_m", "llm_only"),
    ]
    finite_values = [
        float(value)
        for column_name, _ in mode_columns
        for value in rmse_df[column_name].astype(float).tolist()
        if np.isfinite(float(value))
    ]
    x_max = max(finite_values, default=0.01) * 1.25
    with plt.rc_context(_figure_style_context()):
        figure, axis = plt.subplots(figsize=(11.2, 5.2))
        for family_rank, family_label in ((0, "Hammer tools"), (1, "Screwdriver tools")):
            family_indices = [
                index for index, rank in enumerate(family_ranks) if int(rank) == family_rank
            ]
            if not family_indices:
                continue
            axis.axhspan(
                min(family_indices) - 0.48,
                max(family_indices) + 0.48,
                color="#f7f5f2" if family_rank == 0 else "#ffffff",
                zorder=-2,
            )
            axis.text(
                max(x_max, 0.01) * 0.985,
                float(np.mean(family_indices)),
                family_label,
                va="center",
                ha="right",
                color="#52606d",
                fontsize=10,
                fontweight="600",
            )
        for y_position in y_positions:
            axis.hlines(y_position, 0.0, max(x_max, 0.01), color="#d9e2ec", linewidth=1.2, zorder=0)
        offsets = np.linspace(-0.16, 0.16, len(mode_columns))
        for offset, (column_name, mode_name) in zip(offsets, mode_columns):
            values = rmse_df[column_name].astype(float).to_numpy()
            valid_mask = np.isfinite(values)
            axis.scatter(
                values[valid_mask],
                (y_positions + float(offset))[valid_mask],
                color=_mode_color(mode_name),
                s=72,
                label=_pretty_mode_label(mode_name),
                zorder=3,
            )
        axis.set_yticks(y_positions)
        axis.set_yticklabels(labels)
        for index in range(1, len(family_ranks)):
            if family_ranks[index] != family_ranks[index - 1]:
                axis.axhline(index - 0.5, color="#d9e2ec", linewidth=1.2, linestyle="--")
        axis.invert_yaxis()
        axis.set_xlim(0.0, max(x_max, 0.01))
        axis.set_xlabel("Translation RMSE (m), lower is better")
        axis.set_ylabel("Object")
        axis.set_title(title, loc="left", color="#243b53", pad=10)
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Mode", borderaxespad=0.0)
        _style_axis(axis)
        figure.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
        return figure


# Return one short methodology note block shown near the top of the website.
def _methodology_note(
    metadata: dict,
    language_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    execution_df: pd.DataFrame,
) -> str:
    """Return one short methodology summary string for thesis-facing context."""
    benchmark_version = str(metadata.get("benchmark_version", "v1"))
    if benchmark_version == "v3":
        legacy_note = (
            " This is a v3 thesis-facing benchmark run: prompt-taxonomy language scoring, "
            "a canonical 3x3 target grid, matched-input llm_only geometry, and pretrained "
            "execution comparisons across predefined, llm_only, and llm_lie."
        )
    else:
        legacy_note = (
            f" This is a pre-v3 bundle ({benchmark_version}) and should be treated as non-canonical "
            "for Chapter 5 figures."
        )
    return (
        f"Language trials: {len(language_df)}, geometry trials: {len(geometry_df)}, "
        f"execution trials: {len(execution_df)}, benchmark version {benchmark_version}. "
        "The site is organized as language grounding, trajectory generation, and trajectory execution."
        f"{legacy_note}"
    )


# Return one short benchmark-overview paragraph shown near the top of the website.
def _experiment_overview_note() -> str:
    """Return a compact summary of how one benchmark run flows through the three stages."""
    return (
        "Each benchmark trial follows the same staged protocol. A language prompt is first mapped "
        "to a structured intent, that intent is compiled into predefined, llm_only, or analytic "
        "Lie-based SE(3) tool trajectories, and a frozen RL policy then executes the saved trajectory "
        "in simulation. Because the stages are saved independently, the website can separate language "
        "failures, geometry failures, and policy-execution failures instead of collapsing them into one score."
    )


# Return one explanatory paragraph for the language section metrics.
def _language_section_note() -> str:
    """Return a concise explanation of the language-stage benchmark metrics."""
    return (
        "H1 asks whether language is grounded into the correct structured command. "
        "The pass/fail count figure shows exact-match outcomes by prompt family so isolated failures "
        "remain visible without hiding the trial counts behind aggregate percentages."
    )


# Return one explanatory paragraph for the geometry section metrics.
def _geometry_section_note() -> str:
    """Return a concise explanation of the geometry-stage benchmark metrics."""
    return (
        "H2 asks whether grounded commands become valid, semantically aligned tool trajectories. "
        "The object-level figure compares semantic target error across tools, while the top-down view "
        "shows where generated trajectories imply contact relative to the requested target grid."
    )


# Return one explanatory paragraph for the execution section metrics.
def _execution_section_note() -> str:
    """Return a concise explanation of the execution-stage benchmark metrics."""
    return (
        "H3 asks whether the frozen pretrained policy can execute those saved trajectories. "
        "The main execution story keeps all three geometry modes visible, pairing strict success with "
        "mean goal completion, then showing family-level translation RMSE and mode-level failure attribution."
    )


# Return one rounded copy of a summary table for static website rendering.
def _rounded_table_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return one static presentation table with rounded numeric values."""
    rounded_df = dataframe.copy()
    if rounded_df.empty:
        return rounded_df
    if "object_name" in rounded_df.columns:
        rounded_df["object_name"] = rounded_df["object_name"].map(
            lambda value: value if pd.isna(value) else _pretty_object_label(str(value))
        )
    for column_name in rounded_df.columns:
        if pd.api.types.is_float_dtype(rounded_df[column_name]):
            rounded_df[column_name] = rounded_df[column_name].round(3)
    return rounded_df


# Return one human-readable table column label for static thesis tables.
def _pretty_column_name(column_name: str) -> str:
    """Return a compact display label for one table column name."""
    overrides = {
        "object_name": "Object",
        "prompt_family": "Prompt Family",
        "prompt_variant": "Prompt Variant",
        "tool_family": "Tool Family",
        "mode": "Mode",
        "backend": "Backend",
        "benchmark_cell": "Benchmark Cell",
        "num_trials": "Trials",
        "valid_trajectory_rate": "Valid Trajectory Rate",
        "compile_success_rate": "Compile Success Rate",
        "validation_success_rate": "Validation Success Rate",
        "mean_num_waypoints": "Mean Waypoints",
        "mean_num_clamped_waypoints": "Mean Clamped Waypoints",
        "mean_semantic_contact_point_xy_error_m": "Mean Semantic Target Error (m)",
        "mean_screwdriver_max_primary_axis_tilt_deg": "Mean Screwdriver Tilt (deg)",
        "mean_screwdriver_twist_angle_span_deg": "Mean Screwdriver Twist Span (deg)",
        "mean_translation_error_m": "Mean Translation Error (m)",
        "mean_rotation_error_deg": "Mean Rotation Error (deg)",
        "mean_path_length_ratio": "Mean Path Length Ratio",
        "trajectory_rmse_m": "Translation RMSE (m)",
        "tracking_fidelity": "Tracking Fidelity",
        "strict_success_rate": "Strict Success Rate",
        "partial_success_rate": "Partial Success Rate",
        "mean_goal_completion_pct": "Goal Completion Rate (%)",
        "clean_trial_rate": "Clean Trial Rate",
        "strict_success_count": "Strict Successes",
        "successful_trials": "Successful Trials",
        "geometry_originated_failures": "Geometry-Originated Failures",
        "execution_runtime_failures": "Execution or Runtime Failures",
    }
    return overrides.get(column_name, column_name.replace("_", " ").title())


# Return one static Dash table component for one summary DataFrame.
def _summary_table(table_id: str, dataframe: pd.DataFrame) -> dash_table.DataTable:
    """Return one non-editable presentation table for one summary DataFrame."""
    rounded_df = _rounded_table_df(dataframe)
    return dash_table.DataTable(
        id=table_id,
        data=rounded_df.to_dict("records"),
        columns=[{"name": _pretty_column_name(name), "id": name} for name in rounded_df.columns],
        page_size=8,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={
            "textAlign": "left",
            "padding": "8px",
            "fontFamily": '"Source Sans 3", sans-serif',
        },
        style_header={"fontWeight": "700", "backgroundColor": "#f7f5f2"},
    )


# Build one small metric card with title, large numeric value, and optional detail text.
def _metric_card(
    title: str,
    value: str,
    *,
    element_id: str,
    color: str,
    detail: str | None = None,
    secondary_value: str | None = None,
) -> html.Div:
    """Return one styled overview card used in the thesis website."""
    children = [
        html.Div(title, className="metric-card-title"),
        html.Div(value, className="metric-card-value"),
    ]
    if secondary_value:
        children.append(
            html.Div(secondary_value, className="metric-card-value metric-card-value-secondary")
        )
    if detail:
        children.append(html.Div(detail, className="metric-card-detail"))
    return html.Div(
        children,
        id=element_id,
        className="metric-card",
        style={"borderTop": f"4px solid {color}"},
    )


# Format one meter-scale error as thesis-facing centimeters.
def _format_error_cm(error_m: float) -> str:
    """Return one compact centimeter error string from a meter value."""
    error_cm = max(0.0, float(error_m) * 100.0)
    if error_cm < 0.01:
        return "< 0.01 cm"
    return f"{error_cm:.1f} cm"


# Return one filtered mean semantic target error in meters.
def _mean_geometry_semantic_error_m(
    geometry_df: pd.DataFrame, *, mode: str, tool_family: str | None = None
) -> float:
    """Return mean semantic contact-point error for one geometry mode/family slice."""
    if geometry_df.empty or "semantic_contact_point_xy_error_m" not in geometry_df:
        return 0.0
    slice_df = geometry_df[geometry_df["mode"].astype(str) == mode].copy()
    if tool_family is not None and "tool_family" in slice_df:
        slice_df = slice_df[slice_df["tool_family"].astype(str) == tool_family].copy()
    if slice_df.empty:
        return 0.0
    return float(
        pd.to_numeric(slice_df["semantic_contact_point_xy_error_m"], errors="coerce")
        .fillna(0.0)
        .mean()
    )


# Return strict-success count and total for one execution mode/family slice.
def _strict_success_count_and_total(
    execution_df: pd.DataFrame, *, mode: str, tool_family: str | None = None
) -> tuple[int, int]:
    """Return strict-success numerator and total trial count for one execution slice."""
    if execution_df.empty or "strict_success" not in execution_df:
        return (0, 0)
    slice_df = execution_df[execution_df["mode"].astype(str) == mode].copy()
    if tool_family is not None:
        slice_df = slice_df[
            slice_df["object_name"].astype(str).map(_execution_object_family) == tool_family
        ].copy()
    return (int(slice_df["strict_success"].astype(bool).sum()), int(len(slice_df)))


# Build the screenshot-friendly main-results card panel.
def _main_results_cards(
    language_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    execution_df: pd.DataFrame,
) -> list[html.Div]:
    """Return thesis-style cards summarizing the main benchmark outcomes."""
    language_total = int(len(language_df))
    language_exact = (
        int(language_df["exact_match"].astype(bool).sum())
        if not language_df.empty and "exact_match" in language_df
        else 0
    )
    se3_error = _mean_geometry_semantic_error_m(geometry_df, mode="llm_lie")
    direct_error = _mean_geometry_semantic_error_m(geometry_df, mode="llm_only")
    direct_hammer_error = _mean_geometry_semantic_error_m(
        geometry_df, mode="llm_only", tool_family="hammer"
    )
    direct_screwdriver_error = _mean_geometry_semantic_error_m(
        geometry_df, mode="llm_only", tool_family="screwdriver"
    )
    se3_success, se3_total = _strict_success_count_and_total(execution_df, mode="llm_lie")
    direct_success, direct_total = _strict_success_count_and_total(execution_df, mode="llm_only")
    return [
        _metric_card(
            "Language Grounding",
            f"{language_exact}/{language_total}",
            detail="Exact-match structured tool calls",
            element_id="main-language-exact-card",
            color=SECTION_COLORS["language"],
        ),
        _metric_card(
            "SE(3) Tool Trajectory Method",
            _format_error_cm(se3_error),
            detail="Semantic target error",
            element_id="main-se3-target-error-card",
            color=SECTION_COLORS["geometry"],
        ),
        _metric_card(
            "Direct LLM Tool Trajectory Method",
            _format_error_cm(direct_error),
            detail=(
                f"Semantic target error; Hammer {_format_error_cm(direct_hammer_error)}; "
                f"Screwdriver {_format_error_cm(direct_screwdriver_error)}"
            ),
            element_id="main-direct-target-error-card",
            color=MODE_COLORS["llm_only"],
        ),
        _metric_card(
            "Execution Strict Success",
            f"SE(3) {se3_success}/{se3_total}",
            secondary_value=f"Direct LLM {direct_success}/{direct_total}",
            element_id="main-execution-strict-card",
            color=SECTION_COLORS["execution"],
        ),
    ]


# Return one website figure block that embeds one generated SVG asset.
def _figure_block(*, title: str, note: str | None, image_id: str, image_src: str) -> html.Div:
    """Return one titled figure block for one generated thesis asset."""
    children = [html.H3(title, className="figure-title")]
    if note:
        children.append(html.P(note, className="section-note"))
    children.append(html.Img(id=image_id, src=image_src, className="figure-image"))
    return html.Div(children, className="figure-block")


# Return one link to the generated static language trial viewer when available.
def _language_viewer_link(results_dir: Path) -> html.A | None:
    """Return a dashboard link to the generated static language viewer, if present."""
    if not default_language_viewer_path(results_dir).exists():
        return None
    return html.A(
        "Open Language Trial Viewer",
        id="language-trial-viewer-link",
        href="/language/viewer/index.html",
        className="section-link",
    )


# Generate the canonical thesis SVG assets for one saved experiment directory.
def _generate_website_assets(
    results_dir: Path,
    *,
    language_summary_df: pd.DataFrame,
    language_component_summary_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    geometry_summary_df: pd.DataFrame,
    execution_df: pd.DataFrame,
    execution_summary_df: pd.DataFrame,
    execution_cell_summary_df: pd.DataFrame,
) -> Dict[str, str]:
    """Generate all canonical website figures and return their filenames by asset key."""
    asset_dir = _results_dashboard_asset_dir(results_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    for filename in STALE_ASSET_FILENAMES:
        (asset_dir / filename).unlink(missing_ok=True)
    figure_builders = {
        "language_prompt_family_accuracy": _build_language_accuracy_figure(
            language_summary_df, language_component_summary_df
        ),
        "language_prompt_family_pass_fail": _build_language_pass_fail_figure(language_summary_df),
        "geometry_semantic_by_object": _build_geometry_object_semantic_figure(geometry_df),
        "geometry_implied_target_hammer": _build_geometry_implied_target_family_figure(
            results_dir, "hammer"
        ),
        "geometry_implied_target_screwdriver": _build_geometry_implied_target_family_figure(
            results_dir, "screwdriver"
        ),
        "execution_strict_success": _build_execution_strict_success_figure(
            execution_cell_summary_df
        ),
        "execution_goal_completion": _build_execution_goal_completion_figure(
            execution_cell_summary_df
        ),
        "execution_strict_success_by_family": _build_execution_strict_success_by_family_figure(
            _build_execution_family_cell_summary_df(execution_df)
        ),
        "execution_goal_completion_by_family": _build_execution_goal_completion_by_family_figure(
            _build_execution_family_cell_summary_df(execution_df)
        ),
        "execution_translation_rmse_by_family": _build_execution_translation_rmse_by_family_figure(
            execution_df
        ),
    }
    for asset_key, figure in figure_builders.items():
        _save_figure(figure, asset_dir / ASSET_FILENAMES[asset_key])
    return {asset_key: ASSET_FILENAMES[asset_key] for asset_key in ASSET_FILENAMES}


# Build the static Dash thesis website for one saved experiment root.
def create_app(
    results_dir: Path,
    *,
    strict_success_threshold_pct: float = 100.0,
    partial_success_threshold_pct: float = 50.0,
) -> Dash:
    """Return one static Dash app that embeds canonical matplotlib thesis figures."""
    metadata, language_df, geometry_df, execution_df = load_dashboard_data(results_dir)
    augmented_execution_df = _augment_execution_metrics(
        execution_df,
        strict_success_threshold_pct=float(strict_success_threshold_pct),
        partial_success_threshold_pct=float(partial_success_threshold_pct),
    )
    dashboard_execution_df = _dashboard_execution_view_df(augmented_execution_df)
    language_summary_df = _build_language_summary_df(language_df)
    language_component_summary_df = _build_language_component_summary_df(language_df)
    geometry_summary_df = _build_geometry_summary_df(geometry_df)
    execution_summary_df = _build_execution_summary_df(dashboard_execution_df)
    execution_cell_summary_df = _build_execution_cell_summary_df(dashboard_execution_df)
    execution_failure_attribution_df = _build_execution_failure_attribution_df(
        dashboard_execution_df
    )
    asset_manifest = _generate_website_assets(
        results_dir,
        language_summary_df=language_summary_df,
        language_component_summary_df=language_component_summary_df,
        geometry_df=geometry_df,
        geometry_summary_df=geometry_summary_df,
        execution_df=dashboard_execution_df,
        execution_summary_df=execution_summary_df,
        execution_cell_summary_df=execution_cell_summary_df,
    )
    app = Dash(__name__)

    # Serve one generated thesis asset file from the experiment-local website asset directory.
    @app.server.route("/_thesis_assets/<path:filename>")
    def _serve_generated_asset(filename: str):
        """Return one generated SVG asset from the experiment-local website asset directory."""
        return send_from_directory(_results_dashboard_asset_dir(results_dir), filename)

    # Serve the generated static language trial viewer from the experiment directory.
    @app.server.route("/language/viewer/index.html")
    @app.server.route("/language/viewer/")
    def _serve_language_viewer():
        """Return the generated static language trial viewer HTML."""
        return send_from_directory(
            default_language_viewer_path(results_dir).parent.resolve(), "index.html"
        )

    language_exact_match = (
        float(language_df["exact_match"].mean()) if not language_df.empty else 0.0
    )
    geometry_mode_counts_df = _build_geometry_mode_counts_df(geometry_df)
    geometry_valid_trials = int(geometry_mode_counts_df["valid_trials"].sum())
    geometry_total_trials = int(geometry_mode_counts_df["total_trials"].sum())
    execution_tracking = (
        float(dashboard_execution_df["translation_rmse_m"].mean())
        if not dashboard_execution_df.empty and "translation_rmse_m" in dashboard_execution_df
        else 0.0
    )
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1("Three-Stage Thesis Evaluation", id="dashboard-title"),
                    html.P(
                        f"Experiment: {metadata.get('experiment_name', results_dir.name)}",
                        id="experiment-name",
                    ),
                    html.P(
                        _methodology_note(
                            metadata, language_df, geometry_df, dashboard_execution_df
                        ),
                        id="benchmark-methodology-note",
                        className="section-note",
                    ),
                    html.P(
                        _experiment_overview_note(),
                        id="benchmark-overview-note",
                        className="section-note",
                    ),
                ],
                className="hero-section",
            ),
            html.Div(
                [
                    _metric_card(
                        "Language: Exact-Match Intent Accuracy",
                        f"{language_exact_match:.1%}",
                        element_id="language-exact-match-card",
                        color=SECTION_COLORS["language"],
                    ),
                    _metric_card(
                        "Geometry: Accepted Trajectories",
                        f"{geometry_valid_trials}/{geometry_total_trials}",
                        element_id="geometry-validity-card",
                        color=SECTION_COLORS["geometry"],
                    ),
                    _metric_card(
                        "Execution: Trajectory RMSE",
                        f"{execution_tracking:.3f} m",
                        element_id="execution-tracking-card",
                        color=SECTION_COLORS["execution"],
                    ),
                ],
                className="metric-card-grid",
            ),
            html.H2("Main Results", id="main-results-heading"),
            html.P(
                "Compact thesis-facing summary of the benchmark outcomes across language, geometry, and execution.",
                className="section-note",
            ),
            html.Div(
                _main_results_cards(language_df, geometry_df, dashboard_execution_df),
                id="main-results-card-grid",
                className="metric-card-grid main-results-card-grid",
            ),
            html.H2("1. Language Grounding"),
            html.P(
                "Scientific question: does the system infer the intended structured command from the user instruction?",
                className="section-note",
            ),
            html.P(_language_section_note(), id="language-section-note", className="section-note"),
            _language_viewer_link(results_dir),
            _figure_block(
                title="Prompt-Family Pass/Fail Counts",
                note=None,
                image_id="language-pass-fail-image",
                image_src=f"/_thesis_assets/{asset_manifest['language_prompt_family_pass_fail']}",
            ),
            _summary_table("language-table", _rounded_table_df(language_df)),
            html.H2("2. Trajectory Generation"),
            html.P(
                "Scientific question: can the system produce valid, spatially robust tool trajectories from that intent?",
                className="section-note",
            ),
            html.P(_geometry_section_note(), id="geometry-section-note", className="section-note"),
            _figure_block(
                title="Object-Level Geometry Summary",
                note=None,
                image_id="geometry-object-semantic-image",
                image_src=f"/_thesis_assets/{asset_manifest['geometry_semantic_by_object']}",
            ),
            _figure_block(
                title="Hammer Requested vs Implied Target",
                note=None,
                image_id="geometry-implied-target-hammer-image",
                image_src=f"/_thesis_assets/{asset_manifest['geometry_implied_target_hammer']}",
            ),
            _figure_block(
                title="Screwdriver Requested vs Implied Target",
                note=None,
                image_id="geometry-implied-target-screwdriver-image",
                image_src=(
                    f"/_thesis_assets/{asset_manifest['geometry_implied_target_screwdriver']}"
                ),
            ),
            _summary_table("geometry-table", _rounded_table_df(geometry_summary_df)),
            html.H2("3. Trajectory Execution"),
            html.P(
                "Scientific question: how faithfully can the frozen RL policy realize the commanded tool trajectory?",
                className="section-note",
            ),
            html.P(
                _execution_section_note(), id="execution-section-note", className="section-note"
            ),
            _figure_block(
                title="Strict Success",
                note=None,
                image_id="execution-strict-success-image",
                image_src=f"/_thesis_assets/{asset_manifest['execution_strict_success']}",
            ),
            _figure_block(
                title="Goal Completion",
                note=None,
                image_id="execution-goal-completion-image",
                image_src=f"/_thesis_assets/{asset_manifest['execution_goal_completion']}",
            ),
            _figure_block(
                title="Strict Success by Tool Family",
                note=None,
                image_id="execution-strict-success-family-image",
                image_src=(
                    f"/_thesis_assets/" f"{asset_manifest['execution_strict_success_by_family']}"
                ),
            ),
            _figure_block(
                title="Goal Completion Rate by Tool Family",
                note=None,
                image_id="execution-goal-completion-family-image",
                image_src=(
                    f"/_thesis_assets/" f"{asset_manifest['execution_goal_completion_by_family']}"
                ),
            ),
            _figure_block(
                title="Translation RMSE by Tool Family",
                note=None,
                image_id="execution-translation-rmse-family-image",
                image_src=(
                    f"/_thesis_assets/" f"{asset_manifest['execution_translation_rmse_by_family']}"
                ),
            ),
            _summary_table("execution-table", _rounded_table_df(execution_cell_summary_df)),
            _summary_table(
                "execution-failure-attribution-table",
                _rounded_table_df(execution_failure_attribution_df),
            ),
        ],
        className="dashboard-shell",
    )

    app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                margin: 0;
                background: #f7f5f2;
                color: #1f2933;
                font-family: "Source Sans 3", "Helvetica Neue", sans-serif;
            }
            .dashboard-shell {
                max-width: 1240px;
                margin: 0 auto;
                padding: 32px 24px 48px 24px;
            }
            .hero-section h1 {
                margin-bottom: 8px;
                font-size: 2.3rem;
            }
            .hero-section p {
                margin: 4px 0;
            }
            .metric-card-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 16px;
                margin: 24px 0 28px 0;
            }
            .main-results-card-grid {
                grid-template-columns: repeat(2, minmax(320px, 520px));
                justify-content: center;
                margin: 14px auto 28px auto;
                max-width: 1080px;
            }
            .metric-card {
                background: white;
                border: 1px solid #e6dfd5;
                border-radius: 14px;
                padding: 18px 20px;
                box-shadow: 0 2px 8px rgba(24, 39, 75, 0.06);
            }
            .metric-card-title {
                color: #52606d;
                font-size: 0.95rem;
                margin-bottom: 8px;
            }
            .metric-card-value {
                color: #102a43;
                font-size: 1.9rem;
                font-weight: 700;
            }
            .metric-card-value-secondary {
                margin-top: 6px;
            }
            .metric-card-detail {
                color: #52606d;
                font-size: 0.95rem;
                line-height: 1.35;
                margin-top: 6px;
            }
            .section-note {
                color: #52606d;
                max-width: 980px;
            }
            .figure-block {
                background: white;
                border: 1px solid #e6dfd5;
                border-radius: 14px;
                padding: 18px 20px 10px 20px;
                box-shadow: 0 2px 8px rgba(24, 39, 75, 0.06);
                margin-bottom: 24px;
            }
            .figure-title {
                margin: 0 0 8px 0;
                color: #102a43;
            }
            .figure-image {
                display: block;
                width: 100%;
                height: auto;
            }
            @media (max-width: 900px) {
                .metric-card-grid {
                    grid-template-columns: 1fr;
                }
                .main-results-card-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""
    return app


# Parse CLI arguments and serve the thesis website.
def main() -> None:
    """Entry point for the Dash thesis website CLI."""
    args = tyro.cli(ResultsDashboardArgs)
    app = create_app(
        args.results_dir,
        strict_success_threshold_pct=float(args.strict_success_threshold_pct),
        partial_success_threshold_pct=float(args.partial_success_threshold_pct),
    )
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
