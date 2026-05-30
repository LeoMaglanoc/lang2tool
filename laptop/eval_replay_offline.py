"""CPU-only replay viewer for the combined predefined hammer and screwdriver artifact."""

from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tyro

# Support script-style execution by ensuring the repo root is importable.
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

OFFLINE_REPLAY_SPEED_MULTIPLIER = 1.0
MAX_OFFLINE_REPLAY_SPEED_MULTIPLIER = 3.0


@dataclass
class OfflineReplayViewerArgs:
    """CLI args for laptop-friendly cached replay viewing."""

    artifact: Path
    """Path to one cached offline replay artifact json file."""

    port: int = 8081
    """Viser server port for the replay viewer."""

    startup_only: bool = False
    """Initialize the replay viewer once and exit without entering the serve loop."""


# Return the object/task identity encoded for one replay source.
def _source_object_task(artifact, mode: str) -> tuple[str, str]:
    """Return the resolved object/task pair for one replay source."""
    source = artifact.sources[mode]
    return (
        source.object_name or artifact.object_name,
        source.task_name or artifact.task_name,
    )


# Build the shared artifact list consumed by the multi-object tool viewer.
def _build_viewer_artifacts(artifact) -> list:
    """Return the ordered replay sources converted into GoalSourceArtifact values."""
    return [
        replay_source_to_goal_source_artifact(artifact, mode=mode) for mode in artifact.mode_order
    ]


class OfflineReplayViewerApp:
    """Drive the multi-object predefined replay viewer from a combined cached artifact."""

    # Initialize the predefined-only replay viewer around the shared tool trajectory viewer.
    def __init__(self, args: OfflineReplayViewerArgs) -> None:
        """Create one multi-object cached replay app for the combined predefined artifact."""
        self._args = args
        self._artifact = load_offline_replay_artifact(args.artifact)
        self._artifacts = _build_viewer_artifacts(self._artifact)
        if len(self._artifact.mode_order) != 2:
            raise ValueError("Offline replay viewer expects exactly two predefined replay sources.")
        initial_mode = self._artifact.mode_order[0]
        initial_object_name, _ = _source_object_task(self._artifact, initial_mode)
        self.viewer = ToolTrajectoryViewer(
            object_name=initial_object_name,
            artifacts=self._artifacts,
            port=args.port,
            use_tabs=False,
            show_robot=True,
            show_goal=True,
            initial_speed_multiplier=OFFLINE_REPLAY_SPEED_MULTIPLIER,
            max_speed_multiplier=MAX_OFFLINE_REPLAY_SPEED_MULTIPLIER,
            preloaded_object_names=tuple(
                dict.fromkeys(
                    _source_object_task(self._artifact, mode)[0]
                    for mode in self._artifact.mode_order
                )
            ),
        )
        self._suppress_source_callback = False
        self.viewer.server.gui.add_markdown(
            "**Offline Replay**  \n"
            "This viewer replays only the predefined hammer swing and predefined screwdriver twist, "
            "including cached robot joint playback and goal pose visualization."
        )
        self._bind_source_switching()
        self._activate_mode(initial_mode)

    # Wire the source dropdown to switch the active preloaded object when needed.
    def _bind_source_switching(self) -> None:
        """Connect the source dropdown to object-aware replay activation."""
        self.viewer.source_dropdown.on_update(lambda _: self._on_source_change())

    # Handle one source dropdown change from the viewer controls.
    def _on_source_change(self) -> None:
        """Switch the active object and source when the replay selection changes."""
        if self._suppress_source_callback:
            return
        self._activate_mode(str(self.viewer.source_dropdown.value))

    # Activate one replay source while keeping the preloaded object meshes in sync.
    def _activate_mode(self, mode: str) -> None:
        """Select one replay source and swap the visible tool mesh if required."""
        object_name, _ = _source_object_task(self._artifact, mode)
        self._suppress_source_callback = True
        try:
            self.viewer.switch_object(object_name, self._artifacts)
            self.viewer._activate_source(mode)
            self.viewer.playback.set_paused(True, now_sec=None)
            self.viewer.pause_button.name = "Play"
        finally:
            self._suppress_source_callback = False

    # Advance cached playback one step and keep the viewer synchronized.
    def tick(self, now_sec: float) -> int:
        """Advance the cached replay and return the active frame index."""
        if str(self.viewer.source_dropdown.value) != self.viewer.playback.mode:
            self._activate_mode(str(self.viewer.source_dropdown.value))
        return self.viewer.tick(now_sec)

    # Run the cached replay viewer indefinitely.
    def run_forever(self) -> None:
        """Serve the cached replay viewer until interrupted."""
        log_info(f"Offline replay viewer running at {getattr(self.viewer.server, 'url', '')}")
        while True:  # pragma: no cover - interactive runtime
            self.tick(time.monotonic())
            time.sleep(1.0 / 60.0)


# Parse CLI args and run one cached replay viewer instance.
def main() -> None:
    """Entry point for the laptop-friendly cached replay viewer."""
    args = tyro.cli(OfflineReplayViewerArgs)
    app = OfflineReplayViewerApp(args)
    if args.startup_only:
        log_info("Offline replay viewer startup completed.")
        return
    app.run_forever()


if __name__ == "__main__":
    main()
