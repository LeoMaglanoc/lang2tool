"""High-level orchestration for LLM param generation and deterministic SE(3) conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol

from ..errors import ConverterError
from ..goals.schema import GeometricGoalV1, SE3PoseSequence, validate_geometric_goal_v1

# OpenAIParamClient is only needed for type-checking; openai is an optional runtime dep.
if TYPE_CHECKING:
    from .openai_client import OpenAIParamClient


class PoseSequenceConverter(Protocol):
    """Protocol for deterministic conversion from geometric params to SE(3) sequences."""

    # Convert validated geometric parameters into a smooth SE(3) sequence.
    def to_pose_sequence(self, params: GeometricGoalV1) -> SE3PoseSequence:
        """Return a smooth SE(3) pose sequence from validated geometric parameters."""


class LLMParametricGoalGenerator:
    """Orchestrate OpenAI param generation with deterministic sequence conversion."""

    # Wire the OpenAI param client and deterministic converter dependencies.
    def __init__(self, param_client: OpenAIParamClient, converter: PoseSequenceConverter) -> None:
        """Store dependencies required for end-to-end parametric goal generation."""
        self._param_client = param_client
        self._converter = converter

    # Generate and validate structured geometric parameters from instruction/context.
    def generate_params(
        self, user_instruction: str, scene_context: Optional[Dict[str, Any]] = None
    ) -> GeometricGoalV1:
        """Return validated geometric parameters in canonical v1 schema."""
        raw = self._param_client.generate_raw_params(
            user_instruction=user_instruction,
            scene_context=scene_context,
        )
        return validate_geometric_goal_v1(raw)

    # Generate SE(3) sequence by chaining LLM params and deterministic conversion.
    def generate_pose_sequence(
        self, user_instruction: str, scene_context: Optional[Dict[str, Any]] = None
    ) -> SE3PoseSequence:
        """Return a deterministic SE(3) sequence from an LLM-produced geometric goal."""
        params = self.generate_params(
            user_instruction=user_instruction, scene_context=scene_context
        )
        try:
            return self._converter.to_pose_sequence(params)
        except Exception as exc:
            raise ConverterError("Failed converting geometric params to SE(3) sequence.") from exc
