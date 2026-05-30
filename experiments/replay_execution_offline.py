"""Laptop-runnable browser for execution benchmark replay artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from experiments.replay_browser import ExperimentReplayViewerArgs, run_replay_browser


@dataclass
class ExecutionReplayViewerArgs(ExperimentReplayViewerArgs):
    """CLI args for execution experiment replay browsing."""

    experiment_dir: Path
    """Path to one experiment results directory."""

    port: int = 8082
    """Viser server port for the execution replay browser."""

    show_goal: bool = True
    """Whether to draw the commanded goal mesh overlay."""


# Parse CLI args and run the execution replay browser.
def main() -> None:
    """Entry point for the laptop-runnable execution replay browser."""
    run_replay_browser(
        tyro.cli(ExecutionReplayViewerArgs),
        stage="execution",
        show_robot=True,
        markdown_title="Execution Replay",
    )


if __name__ == "__main__":
    main()
