"""Experiment-folder replay browser shared by geometry and execution entrypoints."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

if (
    importlib.util.find_spec("geometric_tool_planning") is None
    or importlib.util.find_spec("laptop") is None
):
    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from geometric_tool_planning.viewer import ToolTrajectoryViewer
from laptop.artifacts import load_offline_replay_artifact, replay_source_to_goal_source_artifact
from laptop.utils import log_info


@dataclass
class ExperimentReplayViewerArgs:
    """CLI args shared by experiment replay browser entrypoints."""

    experiment_dir: Path
    """Path to one experiment results directory."""

    port: int
    """Viser server port for the replay browser."""

    startup_only: bool = False
    """Initialize the replay browser once and exit without entering the serve loop."""

    show_goal: bool = False
    """Whether to draw the commanded goal mesh overlay."""

    show_target_error_jump: bool = False
    """Whether to expose a button that jumps to the semantic target-error pose."""


# Load the stage manifest from an experiment result directory.
def load_replay_manifest(experiment_dir: Path, *, stage: str) -> Dict[str, object]:
    """Return one replay manifest for the requested experiment stage."""
    manifest_path = experiment_dir / stage / "replay" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No {stage} replay manifest found at {manifest_path}. "
            "Regenerate this experiment with the replay-enabled benchmark pipeline."
        )
    with open(manifest_path) as file_obj:
        manifest = json.load(file_obj)
    entries = manifest.get("entries", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{stage} replay manifest has no replay entries.")
    return manifest


# Load the requested experiment target grid saved by the geometry benchmark.
def load_experiment_target_grid_xy(experiment_dir: Path) -> tuple[tuple[float, float], ...]:
    """Return requested target-grid XY coordinates for one experiment, if available."""
    aggregate_path = experiment_dir / "geometry" / "summaries" / "aggregate.json"
    if not aggregate_path.exists():
        return ()
    with open(aggregate_path) as file_obj:
        aggregate = json.load(file_obj)
    target_grid_xy = aggregate.get("target_grid_xy")
    if not isinstance(target_grid_xy, list):
        return ()
    normalized_grid: list[tuple[float, float]] = []
    for target_xy in target_grid_xy:
        if isinstance(target_xy, Sequence) and len(target_xy) >= 2:
            normalized_grid.append((float(target_xy[0]), float(target_xy[1])))
    return tuple(normalized_grid)


# Return a readable and unique dropdown label for one replay manifest entry.
def replay_entry_label(entry: Dict[str, object], *, stage: str) -> str:
    """Return one UI label for a replay manifest entry."""
    target_xy = entry.get("target_xy")
    target_label = ""
    if isinstance(target_xy, list) and len(target_xy) >= 2:
        target_label = f" x={float(target_xy[0]):+.3f} y={float(target_xy[1]):+.3f}"
    policy_label = ""
    if stage == "execution" and entry.get("policy_variant"):
        policy_label = f" {entry['policy_variant']}"
    status_label = ""
    if stage == "execution":
        status_label = " success" if entry.get("execution_success") else " failed"
    return (
        f"{entry.get('trial_id')} | {entry.get('object_name')} | "
        f"{entry.get('mode')}{policy_label}{target_label}{status_label}"
    )


# Resolve one manifest replay path relative to the experiment root.
def _replay_path(experiment_dir: Path, entry: Dict[str, object]) -> Path:
    """Return the absolute replay artifact path for one manifest entry."""
    raw_path = Path(str(entry["replay_artifact_path"]))
    return raw_path if raw_path.is_absolute() else experiment_dir / raw_path


class ExperimentReplayViewerApp:
    """Browse one experiment replay manifest through the shared tool trajectory viewer."""

    # Initialize the viewer from the first manifest entry.
    def __init__(
        self,
        args: ExperimentReplayViewerArgs,
        *,
        stage: str,
        show_robot: bool,
        markdown_title: str,
    ) -> None:
        """Create one experiment replay browser for a manifest-backed stage."""
        self._args = args
        self._stage = stage
        self._show_robot = show_robot
        self._manifest = load_replay_manifest(args.experiment_dir, stage=stage)
        self._target_grid_xy = load_experiment_target_grid_xy(args.experiment_dir)
        self._entries = [dict(entry) for entry in self._manifest["entries"]]
        self._labels = [replay_entry_label(entry, stage=stage) for entry in self._entries]
        self._entry_by_label = dict(zip(self._labels, self._entries))
        first_artifact, first_viewer_artifacts = self._load_entry(self._entries[0])
        self._artifact = first_artifact
        self._artifacts = first_viewer_artifacts
        initial_object_name = self._artifact.sources[self._artifact.mode_order[0]].object_name
        self.viewer = ToolTrajectoryViewer(
            object_name=initial_object_name,
            artifacts=self._artifacts,
            port=args.port,
            use_tabs=False,
            show_robot=show_robot,
            show_goal=args.show_goal,
            show_target_error_jump=args.show_target_error_jump,
            target_grid_xy=self._target_grid_xy,
            preloaded_object_names=tuple(
                dict.fromkeys(str(entry["object_name"]) for entry in self._entries)
            ),
        )
        self._suppress_trial_callback = False
        self.trial_dropdown = self.viewer.server.gui.add_dropdown(
            "Replay Trial",
            options=tuple(self._labels),
            initial_value=self._labels[0],
        )
        self.viewer.server.gui.add_markdown(
            f"**{markdown_title}**  \n"
            f"Loaded `{len(self._entries)}` replay artifacts from `{args.experiment_dir}`."
        )
        self.trial_dropdown.on_update(lambda _: self._on_trial_change())

    # Load one manifest entry into replay artifact and viewer-artifact values.
    def _load_entry(self, entry: Dict[str, object]):
        """Return parsed replay artifact plus viewer artifacts for one manifest entry."""
        artifact = load_offline_replay_artifact(_replay_path(self._args.experiment_dir, entry))
        viewer_artifacts = [
            replay_source_to_goal_source_artifact(artifact, mode=mode)
            for mode in artifact.mode_order
        ]
        return artifact, viewer_artifacts

    # Handle trial dropdown changes by swapping the active artifact and object mesh.
    def _on_trial_change(self) -> None:
        """Switch the active replay artifact from the selected manifest entry."""
        if self._suppress_trial_callback:
            return
        selected_label = str(self.trial_dropdown.value)
        entry = self._entry_by_label[selected_label]
        self._artifact, self._artifacts = self._load_entry(entry)
        object_name = self._artifact.sources[self._artifact.mode_order[0]].object_name
        self.viewer.switch_object(object_name, self._artifacts)
        self.viewer.playback.restart()
        self.viewer.playback.set_frame_index(0)
        self.viewer.playback.set_paused(True, now_sec=None)
        self.viewer._suppress_callbacks = True
        self.viewer.frame_slider.value = 0
        self.viewer._suppress_callbacks = False
        self.viewer.pause_button.name = "Play"

    # Advance cached playback one frame and keep the viewer synchronized.
    def tick(self, now_sec: float) -> int:
        """Advance the active replay and return the current frame index."""
        return self.viewer.tick(now_sec)

    # Run the replay browser indefinitely.
    def run_forever(self) -> None:
        """Serve the replay browser until interrupted."""
        log_info(f"Experiment replay viewer running at {getattr(self.viewer.server, 'url', '')}")
        while True:  # pragma: no cover - interactive runtime
            self.tick(time.monotonic())
            time.sleep(1.0 / 60.0)


# Run one stage-specific experiment replay browser from parsed args.
def run_replay_browser(
    args: ExperimentReplayViewerArgs,
    *,
    stage: str,
    show_robot: bool,
    markdown_title: str,
) -> ExperimentReplayViewerApp:
    """Create and optionally serve one experiment replay browser."""
    app = ExperimentReplayViewerApp(
        args,
        stage=stage,
        show_robot=show_robot,
        markdown_title=markdown_title,
    )
    if args.startup_only:
        log_info("Experiment replay viewer startup completed.")
        return app
    app.run_forever()
    return app
