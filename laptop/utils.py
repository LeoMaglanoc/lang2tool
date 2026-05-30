"""Lightweight shared helpers for laptop-friendly offline viewers."""

from __future__ import annotations

import html
from typing import Any

import numpy as np
from termcolor import colored


# Convert tensors / numpy values into plain JSON-safe Python primitives.
def to_json_compatible(value: Any) -> Any:
    """Return one nested value converted into JSON-safe primitives."""
    try:
        import torch
    except Exception:  # pragma: no cover - optional torch import in laptop-only helpers
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_json_compatible(nested_value) for key, nested_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(nested_value) for nested_value in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# Render one WhatsApp-style HTML chat transcript for the viewer chat panel.
def render_chat_html(chat_history: list[tuple[str, str]]) -> str:
    """Return compact HTML for one chat transcript."""
    if not chat_history:
        return "<div style='font-size:12px;color:#888;padding:4px;'><em>No messages yet.</em></div>"

    bubbles = []
    for role, text in chat_history:
        is_user = role == "user"
        bubble_color = "#d9fdd3" if is_user else "#ffffff"
        alignment = "flex-end" if is_user else "flex-start"
        label = "You" if is_user else "Assistant"
        safe_text = html.escape(text).replace("\n", "<br>")
        bubbles.append(
            "<div style='display:flex;justify-content:"
            f"{alignment};margin:6px 0;'>"
            "<div style='max-width:85%;background:"
            f"{bubble_color};border-radius:10px;padding:8px 10px;"
            "box-shadow:0 1px 2px rgba(0,0,0,0.08);font-size:13px;'>"
            f"<div style='font-weight:600;font-size:11px;color:#555;margin-bottom:4px;'>{label}</div>"
            f"<div>{safe_text}</div>"
            "</div></div>"
        )
    return "".join(bubbles)


# Print one informational log line in cyan for local offline scripts.
def log_info(text: str) -> None:
    """Print one informational log message."""
    print(colored(text, "cyan"))


# Print one warning log line in yellow for local offline scripts.
def log_warn(text: str) -> None:
    """Print one warning log message."""
    print(colored(text, "yellow"))
