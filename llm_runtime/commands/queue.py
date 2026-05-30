"""Thread-safe command queue for GUI thread -> sim loop handoff."""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Deque, List

from ..types import ToolCommand


class ToolCommandQueue:
    """Minimal locked FIFO for live tool commands."""

    # Initialize an empty command queue and synchronization lock.
    def __init__(self) -> None:
        self._queue: Deque[ToolCommand] = deque()
        self._lock = Lock()

    # Push one command from the chat callback thread.
    def push(self, command: ToolCommand) -> None:
        with self._lock:
            self._queue.append(command)

    # Drain all commands atomically for deterministic per-step processing.
    def pop_all(self) -> List[ToolCommand]:
        with self._lock:
            items = list(self._queue)
            self._queue.clear()
            return items
