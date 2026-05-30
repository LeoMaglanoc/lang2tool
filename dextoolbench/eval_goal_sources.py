"""CLI entrypoint for interactive goal-source comparison."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence

# Support script-style execution (`python3 dextoolbench/eval_goal_sources.py`) by ensuring
# repo root is importable before local package imports.
if importlib.util.find_spec("geometric_tool_planning") is None:
    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from dextoolbench.eval import (
    EvalArgs as PolicyEvalArgs,
)
from dextoolbench.eval import (
    EvalRunner as PolicyEvalRunner,
)
from dextoolbench.eval import _build_eval_env as policy_build_eval_env
from dextoolbench.metadata import DEXTOOLBENCH_DATA_STRUCTURE
from dextoolbench.shutdown_utils import close_simulation_app_with_timeout
from geometric_tool_planning import (
    EvalGoalSourcesArgs,
    build_goal_source_artifacts,
    build_reference_path,
    compute_path_metrics,
    goal_source_modes,
    load_predefined_goals,
    normalize_quaternion_xyzw,
    resample_goals,
    run_goal_source_comparison,
    sample_interval_sec,
    summary_markdown,
    validate_goal_source,
    validate_llm_lie_spec,
    validate_pose_sequence,
)
from geometric_tool_planning import (
    GoalSourceArtifact as _GoalSourceArtifact,
)
from geometric_tool_planning.viewer import infer_strike_target_xy

GoalSourceArtifact = _GoalSourceArtifact
_build_reference_path = build_reference_path
_build_goal_source_artifacts = build_goal_source_artifacts
_compute_path_metrics = compute_path_metrics
_goal_source_modes = goal_source_modes
_load_predefined_goals = load_predefined_goals
_normalize_quaternion_xyzw = normalize_quaternion_xyzw
_resample_goals = resample_goals
_sample_interval_sec = sample_interval_sec
_summary_markdown = summary_markdown
_validate_goal_source = validate_goal_source
_validate_llm_lie_spec = validate_llm_lie_spec
_validate_pose_sequence = validate_pose_sequence


# Build the fixed-goal list expected by the policy eval env for one comparison source.
def _policy_goals_for_artifact(artifact: GoalSourceArtifact) -> List[List[float]]:
    """Return comparison goals without the initial object spawn pose."""
    recorded_goals = artifact.metadata.get("recorded_goals")
    if isinstance(recorded_goals, list) and recorded_goals:
        return [list(goal) for goal in recorded_goals]
    if len(artifact.goals) <= 1:
        raise ValueError(f"Goal source '{artifact.mode}' must contain at least two poses.")
    return [list(goal) for goal in artifact.goals[1:]]


# Build the markdown summary shown in the policy viewer for the active goal source.
def _policy_summary_markdown(artifact: GoalSourceArtifact) -> str:
    """Return compact viewer markdown for one active comparison source."""
    lines = [
        f"**Active Source:** `{artifact.mode}`",
        f"- duration_sec: `{artifact.duration_sec:.3f}`",
        f"- num_samples: `{len(artifact.goals)}`",
    ]
    if artifact.metrics:
        lines.append(
            f"- mean_translation_error_m: `{artifact.metrics['mean_translation_error_m']:.4f}`"
        )
        lines.append(
            f"- mean_rotation_error_deg: `{artifact.metrics['mean_rotation_error_deg']:.2f}`"
        )
    if artifact.execution_metrics:
        lines.append(f"- episodes: `{int(artifact.execution_metrics.get('episodes', 0))}`")
        lines.append(
            f"- latest_goal_pct: `{float(artifact.execution_metrics.get('latest_goal_pct', 0.0)):.1f}`"
        )
        lines.append(
            f"- avg_goal_pct: `{float(artifact.execution_metrics.get('avg_goal_pct', 0.0)):.1f}`"
        )
    return "\n".join(lines)


# Mirror one one-shot goal list into a down-then-up cycle without duplicating turnaround endpoints.
def _build_mirrored_cycle(goals: Sequence[Sequence[float]]) -> List[List[float]]:
    """Return one mirrored cycle for a forward-only goal list."""
    forward_goals = [list(goal) for goal in goals]
    if len(forward_goals) <= 1:
        return forward_goals
    return forward_goals + [list(goal) for goal in reversed(forward_goals[1:-1])]


# Rebuild compared goal-source artifacts for a newly selected object/task pair.
def _rebuild_artifacts(
    args: EvalGoalSourcesArgs,
    *,
    object_name: str,
    task_name: str,
) -> List[GoalSourceArtifact]:
    """Return fresh compared artifacts after a GUI object/task selection change."""
    return build_goal_source_artifacts(
        replace(args, object_name=object_name, task_name=task_name, enable_viser=False)
    )


# Build one policy env using the selected comparison source's goals.
def _build_policy_env(
    args: EvalGoalSourcesArgs,
    *,
    object_name: str,
    task_name: str,
    artifact: GoalSourceArtifact,
    app_launcher,
):
    """Create a policy-eval env seeded with one comparison source's goals."""
    eval_args = PolicyEvalArgs(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        object_name=object_name,
        task_name=task_name,
        policy_name=args.policy_name,
        interactive=True,
        enable_viser=True,
        force_table_urdf=args.force_table_urdf,
        use_task_env_urdf=args.use_task_env_urdf,
        z_offset=args.z_offset,
        interactive_autorun=args.interactive_autorun,
        exit_after_episodes=args.exit_after_episodes,
        eval_success_tolerance=args.eval_success_tolerance,
        reset_time=args.reset_time,
    )
    return policy_build_eval_env(
        eval_args,
        object_name,
        task_name,
        app_launcher,
        custom_goals=_policy_goals_for_artifact(artifact),
    )


# Policy eval runner that adds goal-source switching to the standard interactive viewer.
class GoalSourcePolicyEvalRunner:
    """Wrap the standard policy EvalRunner with goal-source comparison controls."""

    # Create one comparison runner around the standard policy eval stack.
    def __init__(
        self,
        *,
        args: EvalGoalSourcesArgs,
        env,
        table_urdf: str,
        artifacts: Sequence[GoalSourceArtifact],
        app_launcher,
    ) -> None:
        self._args = args
        self._app_launcher = app_launcher
        self._artifacts_by_mode: Dict[str, GoalSourceArtifact] = {
            artifact.mode: artifact for artifact in artifacts
        }
        self._mode_order = [artifact.mode for artifact in artifacts]
        self._active_mode = self._mode_order[0]
        self._active_policy_goals: List[List[float]] = []
        self._cyclic_goal_chunk: List[List[float]] | None = None
        self._runner = PolicyEvalRunner(
            env=env,
            config_path=args.config_path,
            checkpoint_path=args.checkpoint_path,
            object_name=args.object_name,
            task_name=args.task_name,
            table_urdf=table_urdf,
            output_dir=None,
            policy_name=args.policy_name,
            enable_viser=args.enable_viser,
            interactive_autorun=False,
            exit_after_episodes=args.exit_after_episodes,
            telemetry_json_path=None,
            max_realtime_factor=1.10,
            eval_args=args,
            app_launcher=app_launcher,
            data_structure=DEXTOOLBENCH_DATA_STRUCTURE,
        )
        self._set_active_policy_goals(self._active_mode)
        self._install_active_goals_live()
        if self._runner.viser is not None:
            self._runner.viser.add_goal_source_selector(self._mode_order, self._active_mode)
            self._refresh_semantic_overlays()
            self._refresh_viewer_summary()

    # Match eval_llm cyclic execution semantics for predefined and llm_lie policy modes.
    def _mode_uses_cyclic_execution(self, mode: str) -> bool:
        """Return whether one goal-source mode should execute as a mirrored cycle."""
        return mode == "predefined" or mode == "llm_lie" or mode.startswith("llm_lie[")

    # Prepare the active policy goals and replenishment chunk for one selected source mode.
    def _set_active_policy_goals(self, mode: str) -> None:
        """Store the active live-goal sequence for the selected comparison source."""
        raw_goals = _policy_goals_for_artifact(self._artifacts_by_mode[mode])
        if self._mode_uses_cyclic_execution(mode):
            mirrored_cycle = _build_mirrored_cycle(raw_goals)
            self._active_policy_goals = [list(goal) for goal in mirrored_cycle]
            self._cyclic_goal_chunk = [list(goal) for goal in mirrored_cycle]
            return
        self._active_policy_goals = [list(goal) for goal in raw_goals]
        self._cyclic_goal_chunk = None

    # Keep the env success cap aligned with the active fixed-goal list after live updates and extensions.
    def _sync_live_goal_config(self) -> None:
        """Synchronize env success-target configuration with the stored active goal list."""
        self._runner.env.cfg.fixed_goal_states = [list(goal) for goal in self._active_policy_goals]
        if hasattr(self._runner.env.cfg, "max_consecutive_successes"):
            self._runner.env.cfg.max_consecutive_successes = len(self._active_policy_goals)
        if hasattr(self._runner.env, "max_consecutive_successes"):
            self._runner.env.max_consecutive_successes = len(self._active_policy_goals)

    # Apply the active policy goals to the live env and refresh the success-cap bookkeeping.
    def _install_active_goals_live(self) -> None:
        """Install the stored active goal list into the current env without recreating it."""
        self._runner._apply_goals_live(self._active_policy_goals)
        self._sync_live_goal_config()

    # Extend cyclic modes before the env reaches the current max-success terminal condition.
    def _maybe_extend_cyclic_goals(self) -> None:
        """Append another mirrored cycle when a cyclic run nears its current goal cap."""
        if self._cyclic_goal_chunk is None:
            return
        success_count = self._runner._success_count()
        if success_count + 2 < len(self._active_policy_goals):
            return
        self._active_policy_goals.extend([list(goal) for goal in self._cyclic_goal_chunk])
        self._sync_live_goal_config()

    # Keep cyclic episode stats meaningful by measuring completion against one mirrored cycle.
    def _episode_goal_pct(self, peak_success_count: int) -> float:
        """Return episode completion percentage for sidebar stats and summaries."""
        if self._cyclic_goal_chunk:
            return (
                100.0
                * min(peak_success_count, len(self._cyclic_goal_chunk))
                / max(1, len(self._cyclic_goal_chunk))
            )
        return 100.0 * peak_success_count / self._runner._success_target()

    # Render the active source's kinematics and rollout stats in the viewer sidebar.
    def _refresh_viewer_summary(self) -> None:
        """Update the viewer summary block for the active goal source."""
        if self._runner.viser is None:
            return
        self._runner.viser.sync_goal_source(self._active_mode)
        self._runner.viser.update_goal_source_summary(
            _policy_summary_markdown(self._artifacts_by_mode[self._active_mode])
        )

    # Refresh policy-view semantic overlays so they follow the active goal source.
    def _refresh_semantic_overlays(self) -> None:
        """Update the policy viewer semantic overlay context for the active source."""
        if self._runner.viser is None:
            return
        artifact = self._artifacts_by_mode[self._active_mode]
        try:
            self._runner.viser.set_semantic_goal_overlay_context(
                self._runner.object_name,
                infer_strike_target_xy(artifact, self._runner.object_name),
            )
        except Exception:
            self._runner.viser.clear_semantic_goal_overlay_context()

    # Clear rollout summary state after a source/object/task switch so per-mode stats restart cleanly.
    def _reset_rollout_stats(self) -> None:
        """Clear per-run episode stats after the active comparison source changes."""
        self._runner._reset_episode_tracking()

    # Swap only the active fixed-goal list when the object/task selection is unchanged.
    def _apply_goal_source_live(self, mode: str) -> None:
        """Apply a same-object/task goal-source change without recreating the env."""
        self._active_mode = mode
        self._set_active_policy_goals(mode)
        self._runner.policy.reset()
        self._install_active_goals_live()
        self._reset_rollout_stats()
        self._refresh_semantic_overlays()
        self._runner._refresh_viewer_state(reset_stats=True)
        self._refresh_viewer_summary()

    # Recreate the env only when object/task selections change in the viewer.
    def _apply_pending_selection(self) -> bool:
        """Apply buffered object/task/source selections before running or resetting."""
        if self._runner.viser is None:
            return False

        selected_object = self._runner.viser.pending_object_name
        selected_task = self._runner.viser.pending_task_name
        selected_mode = (
            self._runner.viser.pending_goal_source
            if self._runner.viser.pending_goal_source is not None
            else self._active_mode
        )
        object_or_task_changed = self._runner.viser.selection_changed
        mode_changed = self._runner.viser.goal_source_changed

        if not object_or_task_changed and not mode_changed:
            return False

        self._runner.viser.selection_changed = False
        self._runner.viser.goal_source_changed = False

        if object_or_task_changed:
            self._artifacts_by_mode = {
                artifact.mode: artifact
                for artifact in _rebuild_artifacts(
                    self._args,
                    object_name=selected_object,
                    task_name=selected_task,
                )
            }
            self._mode_order = list(self._artifacts_by_mode.keys())

        if selected_mode not in self._artifacts_by_mode:
            selected_mode = self._mode_order[0]
        self._active_mode = selected_mode
        self._set_active_policy_goals(selected_mode)

        if not object_or_task_changed:
            self._apply_goal_source_live(self._active_mode)
            return True

        if object_or_task_changed:
            self._args = replace(
                self._args,
                object_name=selected_object,
                task_name=selected_task,
            )

        self._runner._recreate_environment(
            self._args.object_name,
            self._args.task_name,
            custom_goals=self._active_policy_goals,
        )
        self._reset_rollout_stats()
        self._runner._reset_policy_and_env_state()
        self._sync_live_goal_config()
        self._refresh_semantic_overlays()
        self._runner._refresh_viewer_state(rebuild_scene=True, reset_stats=True)
        self._refresh_viewer_summary()
        return True

    # Reset the scene while honoring any pending object/task/source selections first.
    def _reset_scene(self) -> None:
        """Reset the current scene or apply a pending viewer selection."""
        if self._runner.viser is not None:
            self._runner.viser.reset_requested = True
            self._runner.viser._set_pause_state(False)
        self._runner._wait_for_episode_stop()
        if self._runner.viser is not None:
            self._runner.viser.reset_requested = False
        if self._apply_pending_selection():
            return
        self._set_active_policy_goals(self._active_mode)
        self._runner._reset_policy_and_env_state()
        self._install_active_goals_live()
        self._refresh_semantic_overlays()
        self._runner._refresh_viewer_state(reset_progress=True)
        self._refresh_viewer_summary()

    # Run one episode and record the resulting per-source execution stats.
    def _run_episode(self) -> None:
        """Run a policy episode for the currently active comparison source."""
        if not hasattr(self._runner, "_run_episode_in_progress"):
            self._runner._run_episode_in_progress = False
        if self._runner._run_episode_in_progress:
            return
        before_episode_count = self._runner.episode_count
        self._apply_pending_selection()
        self._set_active_policy_goals(self._active_mode)
        self._runner._run_episode_in_progress = True

        try:
            self._runner.policy.reset()
            self._install_active_goals_live()
            self._runner.obs = self._runner._reset()
            if self._runner.viser is not None:
                self._runner.viser.update(*self._runner._get_state())

            states, step, done = [], 0, False
            peak_success_count = self._runner._success_count()
            wall_start = time.time()
            realtime_violations = 0

            while not done:
                while self._runner.viser is not None and self._runner.viser.is_paused:
                    time.sleep(0.1)

                if self._runner.viser is not None and self._runner.viser.reset_requested:
                    return
                if self._runner.viser is not None and self._runner.viser.stop_requested:
                    self._runner.viser.stop_requested = False
                    wall_elapsed = time.time() - wall_start
                    sim_elapsed = step / self._runner.control_hz
                    self._runner._write_telemetry(
                        status="stopped",
                        step=step,
                        sim_time_sec=sim_elapsed,
                        wall_time_sec=wall_elapsed,
                        realtime_factor=sim_elapsed / max(wall_elapsed, 1e-6),
                    )
                    return

                if self._runner.record_video and step % self._runner.record_interval == 0:
                    states.append(tuple(x.copy() for x in self._runner._get_state()))
                self._maybe_extend_cyclic_goals()
                done, terminated, truncated, action = self._runner._sim_step(step)
                step += 1
                peak_success_count = max(peak_success_count, self._runner._success_count())
                wall_elapsed = time.time() - wall_start
                sim_elapsed = step / self._runner.control_hz
                realtime_factor = sim_elapsed / max(wall_elapsed, 1e-6)
                self._runner._write_telemetry(
                    status="running",
                    step=step,
                    sim_time_sec=sim_elapsed,
                    wall_time_sec=wall_elapsed,
                    realtime_factor=realtime_factor,
                    done=done,
                    terminated=terminated,
                    truncated=truncated,
                    action=action,
                    policy_obs=self._runner.obs,
                )
                if (
                    self._runner.max_realtime_factor > 0
                    and step > 5
                    and realtime_factor > self._runner.max_realtime_factor
                ):
                    realtime_violations += 1

            goal_pct = self._episode_goal_pct(peak_success_count)
            self._runner.episode_goal_pcts.append(goal_pct)
            self._runner.episode_lengths.append(step)
            self._runner.episode_count += 1
            avg_goal_pct = sum(self._runner.episode_goal_pcts) / len(self._runner.episode_goal_pcts)
            avg_time_sec = (
                sum(self._runner.episode_lengths)
                / len(self._runner.episode_lengths)
                / self._runner.control_hz
            )
            if self._runner.viser is not None:
                self._runner.viser.update_stats(
                    self._runner.episode_count, avg_goal_pct, avg_time_sec
                )
            if states and self._runner.record_video:
                self._runner._render_video(
                    states, self._runner.session_dir / f"{self._runner.episode_count}.mp4"
                )

            wall_elapsed = time.time() - wall_start
            sim_elapsed = step / self._runner.control_hz
            realtime_factor = sim_elapsed / max(wall_elapsed, 1e-6)
            self._runner._write_telemetry(
                status="done",
                step=step,
                sim_time_sec=sim_elapsed,
                wall_time_sec=wall_elapsed,
                realtime_factor=realtime_factor,
            )
            if self._runner.max_realtime_factor > 0 and realtime_violations > 3:
                raise RuntimeError(
                    f"Realtime factor exceeded threshold repeatedly: rtf={realtime_factor:.2f}, "
                    f"max={self._runner.max_realtime_factor:.2f}"
                )
        finally:
            self._runner._run_episode_in_progress = False

        if self._runner.episode_count <= before_episode_count:
            return
        artifact = self._artifacts_by_mode[self._active_mode]
        artifact.execution_metrics["episodes"] = int(self._runner.episode_count)
        artifact.execution_metrics["latest_goal_pct"] = float(self._runner.episode_goal_pcts[-1])
        artifact.execution_metrics["avg_goal_pct"] = float(
            sum(self._runner.episode_goal_pcts) / len(self._runner.episode_goal_pcts)
        )
        artifact.execution_metrics["avg_time_sec"] = float(
            sum(self._runner.episode_lengths)
            / len(self._runner.episode_lengths)
            / self._runner.control_hz
        )
        self._refresh_viewer_summary()
        print(_policy_summary_markdown(artifact))

    # Capture one replay frame from the current policy env state for offline export.
    def _capture_replay_frame(self, *, step: int, event: str) -> dict:
        """Return one cached replay frame from the current policy env state."""
        joint_pos, object_pose, goal_pose, *_ = self._runner._get_state()
        return {
            "tool_pose": [float(value) for value in object_pose.tolist()],
            "goal_pose": [float(value) for value in goal_pose.tolist()],
            "goal_index": int(self._runner._success_count()),
            "success_count": int(self._runner._success_count()),
            "sim_time_sec": float(step / max(self._runner.control_hz, 1e-6)),
            "event": str(event),
            "robot_joint_positions": [float(value) for value in joint_pos.tolist()],
        }

    # Run one policy episode silently and return cached replay frames for the selected source.
    def capture_policy_replay(
        self,
        mode: str,
        *,
        max_duration_sec: float | None = None,
    ) -> dict:
        """Return one offline replay payload for the selected policy goal-source mode."""
        if mode not in self._artifacts_by_mode:
            raise ValueError(f"Unknown goal-source mode: {mode}")
        if not hasattr(self._runner, "_run_episode_in_progress"):
            self._runner._run_episode_in_progress = False
        if self._runner._run_episode_in_progress:
            raise RuntimeError("Cannot capture replay while another episode is running.")
        max_steps = None
        if max_duration_sec is not None:
            if max_duration_sec <= 0.0:
                raise ValueError("max_duration_sec must be positive when provided.")
            max_steps = max(1, int(round(max_duration_sec * max(self._runner.control_hz, 1e-6))))

        self._active_mode = mode
        self._set_active_policy_goals(mode)
        self._runner._run_episode_in_progress = True
        try:
            self._runner.policy.reset()
            self._runner._reset_policy_and_env_state()
            self._install_active_goals_live()
            frames = [self._capture_replay_frame(step=0, event="startup")]
            step = 0
            done = False
            duration_cap_reached = False
            peak_success_count = self._runner._success_count()
            while not done:
                self._maybe_extend_cyclic_goals()
                done, terminated, truncated, action = self._runner._sim_step(step)
                del terminated, truncated, action
                step += 1
                duration_cap_reached = max_steps is not None and step >= max_steps and not done
                peak_success_count = max(peak_success_count, self._runner._success_count())
                frames.append(
                    self._capture_replay_frame(
                        step=step,
                        event=(
                            "done"
                            if done
                            else "duration_cap_reached" if duration_cap_reached else "running"
                        ),
                    )
                )
                if duration_cap_reached:
                    done = True
            goal_pct = self._episode_goal_pct(peak_success_count)
            artifact = self._artifacts_by_mode[mode]
            return {
                "mode": mode,
                "frames": frames,
                "summary": {
                    "episode_goal_pct": float(goal_pct),
                    "episode_length_steps": int(step),
                    "eval_success_tolerance": float(self._args.eval_success_tolerance),
                    "avg_time_sec": float(step / max(self._runner.control_hz, 1e-6)),
                    "duration_cap_reached": bool(duration_cap_reached),
                },
                "artifact": artifact,
            }
        finally:
            self._runner._run_episode_in_progress = False

    # Reuse the loaded policy runner for a new object/task/goal-source artifact without reloading weights.
    def reconfigure_for_artifact(
        self,
        *,
        args: EvalGoalSourcesArgs,
        artifact: GoalSourceArtifact,
    ) -> None:
        """Switch the wrapped policy runner to one new object/task/artifact bundle."""
        self._args = args
        self._artifacts_by_mode = {artifact.mode: artifact}
        self._mode_order = [artifact.mode]
        self._active_mode = artifact.mode
        self._set_active_policy_goals(artifact.mode)
        self._runner._eval_args = args
        self._runner._recreate_environment(
            args.object_name,
            args.task_name,
            custom_goals=self._active_policy_goals,
        )
        self._reset_rollout_stats()
        self._sync_live_goal_config()

    # Start the standard interactive viewer loop with comparison-aware callbacks.
    def run_interactive_eval(self) -> None:
        """Serve the interactive policy viewer with goal-source selection enabled."""
        if self._runner.viser is None:
            raise RuntimeError("Policy goal-source comparison requires the Viser viewer.")
        self._runner.viser.add_controls(
            self._run_episode,
            self._reset_scene,
            self._runner._stop_episode,
        )
        print(f"Open http://localhost:{self._runner.viser.port}")
        print("Click 'Run Episode' to start.")
        if self._args.interactive_autorun:
            self._run_episode()
        while True:
            if (
                self._runner.exit_after_episodes > 0
                and self._runner.episode_count >= self._runner.exit_after_episodes
            ):
                return
            time.sleep(1.0)


# Run the interactive policy comparison flow using the shared goal-source artifacts.
def _run_policy_goal_source_comparison(args: EvalGoalSourcesArgs) -> None:
    """Launch the standard policy eval stack with goal-source comparison controls."""
    if not args.enable_viser:
        raise ValueError("Policy backend requires enable_viser=True.")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    artifacts = build_goal_source_artifacts(replace(args, enable_viser=False))
    env, selected_table_urdf, _ = _build_policy_env(
        args,
        object_name=args.object_name,
        task_name=args.task_name,
        artifact=artifacts[0],
        app_launcher=app_launcher,
    )
    runner = GoalSourcePolicyEvalRunner(
        args=args,
        env=env,
        table_urdf=selected_table_urdf,
        artifacts=artifacts,
        app_launcher=app_launcher,
    )
    try:
        runner.run_interactive_eval()
    finally:
        close_simulation_app_with_timeout(
            simulation_app,
            timeout_sec=15.0,
            log_warn_fn=print,
        )


# Parse the CLI with tyro and dispatch to kinematics or policy comparison.
def main() -> None:
    """Run the interactive goal-source comparison CLI."""
    try:
        import tyro  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package/runtime
        raise RuntimeError("The eval_goal_sources CLI requires the tyro package.") from exc

    args = tyro.cli(EvalGoalSourcesArgs)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    if args.execution_backend == "kinematics":
        artifacts = run_goal_source_comparison(args)
        if not args.enable_viser:
            print(summary_markdown(artifacts))
        return
    if args.execution_backend == "policy":
        _run_policy_goal_source_comparison(args)
        return
    raise ValueError(
        f"Unsupported execution backend '{args.execution_backend}'. "
        "Supported: {'kinematics', 'policy'}."
    )


if __name__ == "__main__":
    main()
