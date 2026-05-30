"""Isaac Lab interactive DexToolBench evaluation entrypoint.

This script keeps the legacy `eval_interactive.py` CLI name while delegating to the
current Isaac Lab evaluation stack used by `dextoolbench/eval.py`.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from dextoolbench.eval import EvalArgs, EvalRunner, _build_eval_env, log_warn
from dextoolbench.eval_config import (
    DEFAULT_EVAL_SUCCESS_TOLERANCE_M,
    DEFAULT_MAX_REALTIME_FACTOR,
    DEFAULT_OBJECT_CATEGORY,
    DEFAULT_OBJECT_NAME,
    DEFAULT_TASK_NAME,
    DEFAULT_Z_OFFSET_M,
)
from dextoolbench.metadata import DEXTOOLBENCH_DATA_STRUCTURE
from dextoolbench.shutdown_utils import close_simulation_app_with_timeout


# Build CLI arguments for interactive Isaac Lab evaluation.
def _build_parser() -> argparse.ArgumentParser:
    """Create the parser for eval_interactive compatibility flags."""
    parser = argparse.ArgumentParser(description="Run interactive DexToolBench evaluation.")
    parser.add_argument("--config-path", type=Path, required=True, help="Policy config YAML path")
    parser.add_argument(
        "--checkpoint-path", type=Path, required=True, help="Policy checkpoint path"
    )
    parser.add_argument(
        "--object-category", type=str, default=DEFAULT_OBJECT_CATEGORY, help="Object category"
    )
    parser.add_argument("--object-name", type=str, default=DEFAULT_OBJECT_NAME, help="Object name")
    parser.add_argument(
        "--task-name", type=str, default=DEFAULT_TASK_NAME, help="Task trajectory name"
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory")
    parser.add_argument(
        "--force-table-urdf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Always use default table URDF",
    )
    parser.add_argument(
        "--use-task-env-urdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use task-specific environment URDFs when available",
    )
    parser.add_argument(
        "--prompt-selection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prompt in terminal for category/object/task selection before startup",
    )
    parser.add_argument(
        "--z-offset", type=float, default=DEFAULT_Z_OFFSET_M, help="Start-pose Z safety offset"
    )
    parser.add_argument(
        "--interactive-autorun",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Automatically run one episode after startup",
    )
    parser.add_argument(
        "--exit-after-episodes",
        type=int,
        default=0,
        help="If >0, exit after this many completed interactive episodes",
    )
    parser.add_argument(
        "--telemetry-json-path", type=Path, default=None, help="Telemetry JSON output"
    )
    parser.add_argument(
        "--max-realtime-factor",
        type=float,
        default=DEFAULT_MAX_REALTIME_FACTOR,
        help="Fail when realtime factor repeatedly exceeds threshold",
    )
    parser.add_argument(
        "--eval-success-tolerance",
        type=float,
        default=DEFAULT_EVAL_SUCCESS_TOLERANCE_M,
        help="Evaluation success tolerance override",
    )
    parser.add_argument("--policy-name", type=str, default=None, help="Optional policy label in UI")
    return parser


# Build ordered choices and place preferred entry first when available.
def _ordered_choices(options: List[str], preferred: str) -> List[str]:
    """Return deterministic choices with preferred option first when present."""
    if preferred in options:
        return [preferred] + [option for option in options if option != preferred]
    return list(options)


# Prompt user for a single choice by index or exact name.
def _prompt_choice(label: str, options: List[str], preferred: str) -> str:
    """Prompt user to choose from options; accepts index or exact name."""
    if not sys.stdin.isatty():
        raise ValueError(
            "No TTY available. Disable --prompt-selection or pass explicit category/object/task."
        )
    choices = _ordered_choices(options, preferred)
    print(f"\nSelect {label}:")
    for idx, option in enumerate(choices, start=1):
        print(f"  {idx:2d}) {option}")

    while True:
        raw = input(f"\nChoose {label} by number or exact name (q to quit): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            raise SystemExit("Selection aborted by user.")
        if not raw:
            print("Please enter a selection.")
            continue
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(choices):
                return choices[index - 1]
            print(f"Invalid index: {index}. Choose between 1 and {len(choices)}.")
            continue
        if raw in choices:
            return raw
        print(f"Unknown {label} '{raw}'. Enter a listed index or exact name.")


# Resolve startup category/object/task from CLI defaults or terminal prompt mode.
def _resolve_startup_selection(args: argparse.Namespace) -> Tuple[str, str, str]:
    """Return startup (category, object, task), optionally from terminal prompts."""
    if not bool(args.prompt_selection):
        return str(args.object_category), str(args.object_name), str(args.task_name)

    categories = list(DEXTOOLBENCH_DATA_STRUCTURE.keys())
    category = _prompt_choice("category", categories, str(args.object_category))

    objects = list(DEXTOOLBENCH_DATA_STRUCTURE[category].keys())
    object_name = _prompt_choice("object", objects, str(args.object_name))

    tasks = list(DEXTOOLBENCH_DATA_STRUCTURE[category][object_name])
    task_name = _prompt_choice("task", tasks, str(args.task_name))
    return category, object_name, task_name


# Convert argparse namespace to the shared EvalArgs schema.
def _to_eval_args(args: argparse.Namespace) -> EvalArgs:
    """Translate compatibility CLI args into EvalArgs used by the Lab evaluator."""
    return EvalArgs(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        object_category=args.object_category,
        object_name=args.object_name,
        task_name=args.task_name,
        output_dir=args.output_dir,
        num_episodes=1,
        downsample_factor=1,
        policy_name=args.policy_name,
        interactive=True,
        enable_viser=True,
        force_table_urdf=bool(args.force_table_urdf),
        use_task_env_urdf=bool(args.use_task_env_urdf),
        z_offset=float(args.z_offset),
        custom_goals_json_path=None,
        interactive_autorun=bool(args.interactive_autorun),
        exit_after_episodes=int(args.exit_after_episodes),
        telemetry_json_path=args.telemetry_json_path,
        max_realtime_factor=float(args.max_realtime_factor),
        eval_success_tolerance=float(args.eval_success_tolerance),
    )


# Run interactive eval through the Isaac Lab app lifecycle.
def main() -> None:
    """Launch interactive evaluation with the same Isaac Lab backend as eval.py."""
    args = _build_parser().parse_args()
    selected_category, selected_object, selected_task = _resolve_startup_selection(args)
    args.object_category = selected_category
    args.object_name = selected_object
    args.task_name = selected_task
    eval_args = _to_eval_args(args)

    from isaaclab.app import AppLauncher

    # AppLauncher consumes process argv for Kit flags; isolate custom CLI flags first.
    argv_backup = list(sys.argv)
    sys.argv = [sys.argv[0]]
    try:
        app_launcher = AppLauncher(headless=True)
    finally:
        sys.argv = argv_backup
    simulation_app = app_launcher.app

    env, selected_table_urdf, _ = _build_eval_env(
        eval_args, eval_args.object_name, eval_args.task_name, app_launcher
    )

    runner = EvalRunner(
        env=env,
        config_path=eval_args.config_path,
        checkpoint_path=eval_args.checkpoint_path,
        object_name=eval_args.object_name,
        task_name=eval_args.task_name,
        table_urdf=selected_table_urdf,
        output_dir=eval_args.output_dir,
        policy_name=eval_args.policy_name,
        enable_viser=True,
        interactive_autorun=eval_args.interactive_autorun,
        exit_after_episodes=eval_args.exit_after_episodes,
        telemetry_json_path=eval_args.telemetry_json_path,
        max_realtime_factor=eval_args.max_realtime_factor,
        eval_args=eval_args,
        app_launcher=app_launcher,
        data_structure=DEXTOOLBENCH_DATA_STRUCTURE,
    )

    runner.run_interactive_eval()

    if eval_args.exit_after_episodes > 0:
        close_simulation_app_with_timeout(
            simulation_app,
            timeout_sec=15.0,
            log_warn_fn=log_warn,
        )
    else:
        simulation_app.close()


if __name__ == "__main__":
    main()
