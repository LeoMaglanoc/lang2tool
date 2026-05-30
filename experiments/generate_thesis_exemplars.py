"""Generate supplemental thesis target-a geometry exemplar artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import tyro

from dextoolbench.eval_config import (
    DEFAULT_CONTROL_HZ,
    DEFAULT_LLM_LIE_HORIZONTAL_STRIKE_CLEARANCE_M,
    DEFAULT_LLM_LIE_SCREWDRIVER_TWIST_EXTRA_HOVER_M,
    DEFAULT_LLM_LIE_TRAINING_RESAMPLING_ENABLED,
    DEFAULT_LLM_LIE_TRAINING_RESAMPLING_MIN_WAYPOINTS,
    DEFAULT_LLM_LIE_TRAINING_RESAMPLING_POS_SCALE_M,
    DEFAULT_LLM_LIE_TRAINING_RESAMPLING_ROT_SCALE_DEG,
    DEFAULT_LLM_LIE_TRAINING_RESAMPLING_TARGET_COST,
    DEFAULT_LLM_LIE_TRAINING_TARGET_VOLUME_MAXS,
    DEFAULT_LLM_LIE_TRAINING_TARGET_VOLUME_MINS,
    DEFAULT_LLM_LIE_TRAINING_VOLUME_CLAMP_ENABLED,
    DEFAULT_LLM_LIE_WAYPOINT_TABLE_CLEARANCE_M,
    DEFAULT_Z_OFFSET_M,
    TABLE_Z,
)
from dextoolbench.llm_lie_trajectory import compile_llm_lie_trajectory
from dextoolbench.object_start_poses import build_default_start_pose
from dextoolbench.predefined_baselines import resolve_predefined_baseline_object_name
from experiments.common import read_json, write_json
from geometric_tool_planning import (
    EvalGoalSourcesArgs,
    build_artifact_for_mode,
    build_reference_path,
    compute_path_metrics,
    get_named_strike_points,
    validate_pose_sequence,
)

HOST_REPO_ROOT = Path("/home/leo/code/simtoolreal")
CONTAINER_REPO_ROOT = Path("/workspace")

EXEMPLAR_SPECS = (
    ("hammer", "claw_hammer", "swing_down", "llm_only"),
    ("hammer", "claw_hammer", "swing_down", "llm_lie"),
    ("screwdriver", "long_screwdriver", "spin_vertical", "llm_only"),
    ("screwdriver", "long_screwdriver", "spin_vertical", "llm_lie"),
)


# Configure one supplemental thesis exemplar generation run from the CLI.
@dataclass
class ThesisExemplarArgs:
    """CLI arguments for generating saved thesis target-a exemplars."""

    experiment_dir: Path
    """Existing experiment directory that will receive geometry/exemplars/*.json."""

    backend: str = "openai"
    """LLM backend used for llm_only target-a exemplar generation."""

    seed: int | None = None
    """Optional seed override; defaults to the saved experiment config seed."""


# Load the saved geometry benchmark config from the experiment root.
def _load_experiment_config(experiment_dir: Path) -> Dict[str, Any]:
    """Return the saved experiment config or an empty mapping when absent."""
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        return {}
    return read_json(config_path)


# Resolve host absolute repo paths to the Docker workspace mount when needed.
def _resolve_experiment_dir(experiment_dir: Path) -> Path:
    """Return an experiment path that exists in the current runtime namespace."""
    if experiment_dir.exists():
        return experiment_dir
    try:
        relative_path = experiment_dir.relative_to(HOST_REPO_ROOT)
    except ValueError:
        return experiment_dir
    container_path = CONTAINER_REPO_ROOT / relative_path
    if container_path.exists():
        return container_path
    return experiment_dir


# Return one target-a instruction that names the selected object and task semantics.
def target_a_instruction(object_name: str, task_name: str) -> str:
    """Return the canonical natural-language prompt for one target-a exemplar."""
    pretty_object_name = str(object_name).replace("_", " ")
    if str(task_name) == "spin_vertical":
        return f"Use the {pretty_object_name} and twist upright on strike point a."
    return f"Use the {pretty_object_name} and swing down to strike point a."


# Resolve the named target-a coordinates for one exemplar object/task.
def _target_a_xy(object_name: str, task_name: str) -> List[float]:
    """Return target-a XY coordinates or raise if the task has no named point."""
    named_points = get_named_strike_points(object_name=object_name, task_name=task_name)
    target_a = named_points.get("target_a")
    if not isinstance(target_a, list) or len(target_a) != 2:
        raise ValueError(f"target_a is unavailable for {object_name}/{task_name}.")
    return [float(target_a[0]), float(target_a[1])]


# Return one config value with a deterministic fallback.
def _config_value(config: Dict[str, Any], key: str, default: Any) -> Any:
    """Return one saved config value while preserving explicit falsey values."""
    if key in config:
        return config[key]
    return default


# Build the raw-compatible payload for one direct LLM target-a exemplar.
def _build_llm_only_exemplar_payload(
    *,
    config: Dict[str, Any],
    family_name: str,
    object_name: str,
    task_name: str,
    backend: str,
    seed: int,
) -> Dict[str, Any]:
    """Return one saved llm_only target-a exemplar payload."""
    reference = build_reference_path(object_name=object_name, task_name=task_name)
    instruction = target_a_instruction(object_name, task_name)
    target_xy = _target_a_xy(object_name, task_name)
    pivot_point = build_default_start_pose(
        object_name,
        z_offset=float(_config_value(config, "z_offset", DEFAULT_Z_OFFSET_M)),
        table_z=float(TABLE_Z),
    )[:3]
    artifact = build_artifact_for_mode(
        "llm_only",
        EvalGoalSourcesArgs(
            goal_source="llm_only",
            object_name=object_name,
            task_name=task_name,
            instruction=instruction,
            target_xy=target_xy,
            resolved_baseline_object=resolve_predefined_baseline_object_name(object_name),
            pivot_point=[float(value) for value in pivot_point],
            llm_backend=str(backend),
            enable_viser=False,
            seed=int(seed),
            z_offset=float(_config_value(config, "z_offset", DEFAULT_Z_OFFSET_M)),
        ),
        reference,
    )
    saved_goals = validate_pose_sequence([list(goal) for goal in artifact.goals[1:]])
    return {
        "trial_id": f"{family_name}_llm_only_target_a",
        "object_name": object_name,
        "task_name": task_name,
        "mode": "llm_only",
        "seed": int(seed),
        "target_name": "target_a",
        "target_xy": target_xy,
        "instruction": instruction,
        "llm_backend": str(backend),
        "resolved_baseline_object": resolve_predefined_baseline_object_name(object_name),
        "pivot_point": [float(value) for value in pivot_point],
        "compile_success": True,
        "validation_success": True,
        "num_waypoints": len(saved_goals),
        "goals": [list(goal) for goal in saved_goals],
        "generation_context": dict(artifact.metadata.get("prompt_context", {})),
        "duration_sec": float(artifact.duration_sec),
        "sample_interval_sec": float(artifact.sample_interval_sec),
        "metrics": dict(artifact.metrics),
        "llm_raw": artifact.llm_raw,
    }


# Build the raw-compatible payload for one Lie target-a exemplar.
def _build_llm_lie_exemplar_payload(
    *,
    config: Dict[str, Any],
    family_name: str,
    object_name: str,
    task_name: str,
    backend: str,
    seed: int,
) -> Dict[str, Any]:
    """Return one saved llm_lie target-a exemplar payload."""
    target_xy = _target_a_xy(object_name, task_name)
    pivot_point = build_default_start_pose(
        object_name,
        z_offset=float(_config_value(config, "z_offset", DEFAULT_Z_OFFSET_M)),
        table_z=float(TABLE_Z),
    )[:3]
    clamped_target_xy, spec, compiled_goals, clamp_summary = compile_llm_lie_trajectory(
        object_name=object_name,
        task_name=task_name,
        pivot_point=pivot_point,
        strike_target_xy=target_xy,
        horizontal_strike_clearance_m=float(
            _config_value(
                config,
                "llm_lie_horizontal_strike_clearance_m",
                DEFAULT_LLM_LIE_HORIZONTAL_STRIKE_CLEARANCE_M,
            )
        ),
        waypoint_table_clearance_m=float(
            _config_value(
                config,
                "llm_lie_waypoint_table_clearance_m",
                DEFAULT_LLM_LIE_WAYPOINT_TABLE_CLEARANCE_M,
            )
        ),
        screwdriver_twist_extra_hover_m=float(
            _config_value(
                config,
                "llm_lie_screwdriver_twist_extra_hover_m",
                DEFAULT_LLM_LIE_SCREWDRIVER_TWIST_EXTRA_HOVER_M,
            )
        ),
        training_resampling_enabled=bool(
            _config_value(
                config,
                "llm_lie_training_resampling_enabled",
                DEFAULT_LLM_LIE_TRAINING_RESAMPLING_ENABLED,
            )
        ),
        training_resampling_pos_scale_m=float(
            _config_value(
                config,
                "llm_lie_training_resampling_pos_scale_m",
                DEFAULT_LLM_LIE_TRAINING_RESAMPLING_POS_SCALE_M,
            )
        ),
        training_resampling_rot_scale_deg=float(
            _config_value(
                config,
                "llm_lie_training_resampling_rot_scale_deg",
                DEFAULT_LLM_LIE_TRAINING_RESAMPLING_ROT_SCALE_DEG,
            )
        ),
        training_resampling_target_cost=float(
            _config_value(
                config,
                "llm_lie_training_resampling_target_cost",
                DEFAULT_LLM_LIE_TRAINING_RESAMPLING_TARGET_COST,
            )
        ),
        training_resampling_min_waypoints=int(
            _config_value(
                config,
                "llm_lie_training_resampling_min_waypoints",
                DEFAULT_LLM_LIE_TRAINING_RESAMPLING_MIN_WAYPOINTS,
            )
        ),
        training_volume_clamp_enabled=bool(
            _config_value(
                config,
                "llm_lie_training_volume_clamp_enabled",
                DEFAULT_LLM_LIE_TRAINING_VOLUME_CLAMP_ENABLED,
            )
        ),
        training_target_volume_mins=list(
            _config_value(
                config,
                "llm_lie_training_target_volume_mins",
                DEFAULT_LLM_LIE_TRAINING_TARGET_VOLUME_MINS,
            )
        ),
        training_target_volume_maxs=list(
            _config_value(
                config,
                "llm_lie_training_target_volume_maxs",
                DEFAULT_LLM_LIE_TRAINING_TARGET_VOLUME_MAXS,
            )
        ),
    )
    saved_goals = validate_pose_sequence(compiled_goals)
    reference = build_reference_path(object_name=object_name, task_name=task_name)
    duration_sec = len(saved_goals) / float(DEFAULT_CONTROL_HZ)
    return {
        "trial_id": f"{family_name}_llm_lie_target_a",
        "object_name": object_name,
        "task_name": task_name,
        "mode": "llm_lie",
        "seed": int(seed),
        "target_name": "target_a",
        "target_xy": [float(value) for value in clamped_target_xy],
        "instruction": target_a_instruction(object_name, task_name),
        "llm_backend": str(backend),
        "resolved_baseline_object": resolve_predefined_baseline_object_name(object_name),
        "pivot_point": [float(value) for value in pivot_point],
        "compile_success": True,
        "validation_success": True,
        "num_waypoints": len(saved_goals),
        "goals": [list(goal) for goal in saved_goals],
        "duration_sec": float(duration_sec),
        "sample_interval_sec": float(duration_sec / max(len(saved_goals) - 1, 1)),
        "clamp_summary": dict(clamp_summary),
        "resampling_summary": dict(spec.get("training_distribution_resampling", {})),
        "metrics": {
            **compute_path_metrics(reference.goals, saved_goals),
            "requested_target_xy": target_xy,
            "step_delta_summary": dict(spec.get("step_delta_summary", {})),
        },
    }


# Build one raw-compatible target-a exemplar payload for the requested mode.
def _build_exemplar_payload(
    *,
    config: Dict[str, Any],
    family_name: str,
    object_name: str,
    task_name: str,
    mode: str,
    backend: str,
    seed: int,
) -> Dict[str, Any]:
    """Return one target-a exemplar payload for one supported mode."""
    if mode == "llm_only":
        return _build_llm_only_exemplar_payload(
            config=config,
            family_name=family_name,
            object_name=object_name,
            task_name=task_name,
            backend=backend,
            seed=seed,
        )
    if mode == "llm_lie":
        return _build_llm_lie_exemplar_payload(
            config=config,
            family_name=family_name,
            object_name=object_name,
            task_name=task_name,
            backend=backend,
            seed=seed,
        )
    raise ValueError(f"Unsupported thesis exemplar mode: {mode}")


# Generate and save all canonical thesis target-a exemplar artifacts.
def generate_thesis_exemplars(args: ThesisExemplarArgs) -> List[Path]:
    """Write the canonical four thesis exemplar JSON files and return their paths."""
    experiment_dir = _resolve_experiment_dir(args.experiment_dir)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")
    config = _load_experiment_config(experiment_dir)
    seed = int(args.seed if args.seed is not None else _config_value(config, "seed", 0))
    output_dir = experiment_dir / "geometry" / "exemplars"
    written_paths: List[Path] = []
    for family_name, object_name, task_name, mode in EXEMPLAR_SPECS:
        payload = _build_exemplar_payload(
            config=config,
            family_name=family_name,
            object_name=object_name,
            task_name=task_name,
            mode=mode,
            backend=str(args.backend),
            seed=seed,
        )
        path = output_dir / f"{family_name}_{mode}_target_a.json"
        write_json(path, payload)
        written_paths.append(path)
    return written_paths


# Parse CLI arguments and generate the supplemental thesis exemplars.
def main() -> None:
    """Entry point for the thesis exemplar generation CLI."""
    written_paths = generate_thesis_exemplars(tyro.cli(ThesisExemplarArgs))
    for path in written_paths:
        print(f"Saved thesis exemplar: {path}")


if __name__ == "__main__":
    main()
