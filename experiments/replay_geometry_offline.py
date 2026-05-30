"""Laptop-runnable browser for geometry benchmark replay artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from experiments.replay_browser import ExperimentReplayViewerArgs, run_replay_browser


@dataclass
class GeometryReplayViewerArgs(ExperimentReplayViewerArgs):
    """CLI args for geometry experiment replay browsing."""

    experiment_dir: Path
    """Path to one experiment results directory."""

    port: int = 8081
    """Viser server port for the geometry replay browser."""

    show_goal: bool = False
    """Whether to draw the commanded goal mesh overlay."""

    show_target_error_jump: bool = True
    """Whether to expose a button that jumps to the semantic target-error pose."""


# Parse CLI args and run the geometry replay browser.
def main() -> None:
    """Entry point for the laptop-runnable geometry replay browser."""
    run_replay_browser(
        tyro.cli(GeometryReplayViewerArgs),
        stage="geometry",
        show_robot=False,
        markdown_title="Geometry Replay",
    )


if __name__ == "__main__":
    main()
