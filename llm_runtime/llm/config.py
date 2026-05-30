"""Configuration for LLM-driven parametric goal generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GoalGeneratorConfig:
    """Store runtime settings for OpenAI-backed goal parameter generation."""

    model: str = "gpt-5.5"
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    schema_version: str = "v1"
    max_tool_round_trips: int = 5
