"""OpenAI Responses API wrapper for structured geometric parameter generation."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ..errors import LLMResponseFormatError, MissingEnvironmentError
from .config import GoalGeneratorConfig

SYSTEM_PROMPT_TEMPLATE = """You generate structured geometric goal parameters for a robotic policy.
Return JSON only, with no markdown and no extra prose.
The JSON must match schema_version \"{schema_version}\" exactly.
Required keys:
- schema_version
- task_label
- object_frame
- contact_point_object: [x, y, z]
- approach_direction_object: [x, y, z]
- tool_axis_object: [x, y, z]
- pregrasp_offset_m
- grasp_depth_m
- lift_height_m
- timing_s: {{\"approach\": float, \"close\": float, \"lift\": float}}
Use meters and seconds.
"""


# Read a required environment variable or raise a clear typed error.
def _get_env_var(name: str) -> str:
    """Return environment variable value or raise MissingEnvironmentError."""
    value = os.environ.get(name)
    if not value:
        raise MissingEnvironmentError(f"Missing required environment variable: {name}")
    return value


# Build the Responses API input payload for text-only generation.
def _build_response_input(system_prompt: str, user_payload: str) -> List[Dict[str, Any]]:
    """Construct the OpenAI Responses input message format."""
    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_payload}],
        },
    ]


# Extract output_text from a Responses result with a safe fallback.
def _extract_output_text(response: object) -> str:
    """Return response.output_text when present, else a string fallback."""
    output_text = getattr(response, "output_text", None)
    if output_text is not None:
        return output_text
    return str(response)


# Serialize user instruction and optional context into a single user payload string.
def _build_user_payload(user_instruction: str, scene_context: Optional[Dict[str, Any]]) -> str:
    """Return a compact JSON payload for the LLM user message."""
    payload: Dict[str, Any] = {"instruction": user_instruction}
    if scene_context is not None:
        payload["scene_context"] = scene_context
    return json.dumps(payload)


class OpenAIParamClient:
    """Generate raw geometric parameter dictionaries from an OpenAI model."""

    # Store OpenAI client dependency and generation config.
    def __init__(self, openai_client: OpenAI, config: GoalGeneratorConfig) -> None:
        """Initialize with an OpenAI client and a goal-generation config."""
        self._client = openai_client
        self._config = config

    # Call the Responses API and return parsed JSON as a dict.
    def generate_raw_params(
        self, user_instruction: str, scene_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a raw parameter dictionary from user text and optional scene context."""
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(schema_version=self._config.schema_version)
        user_payload = _build_user_payload(
            user_instruction=user_instruction, scene_context=scene_context
        )

        # Assemble optional generation kwargs only when configured.
        request_kwargs: Dict[str, Any] = {
            "model": self._config.model,
            "input": _build_response_input(system_prompt=system_prompt, user_payload=user_payload),
        }
        if self._config.temperature is not None:
            request_kwargs["temperature"] = self._config.temperature
        if self._config.max_output_tokens is not None:
            request_kwargs["max_output_tokens"] = self._config.max_output_tokens

        response = self._client.responses.create(**request_kwargs)
        output_text = _extract_output_text(response)

        # Parse and validate the top-level JSON container.
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise LLMResponseFormatError("LLM output is not valid JSON.") from exc

        if not isinstance(parsed, dict):
            raise LLMResponseFormatError("LLM output JSON must be an object.")
        return parsed


# Build the default OpenAI-backed param client using environment credentials.
def build_default_openai_param_client(
    config: Optional[GoalGeneratorConfig] = None,
) -> OpenAIParamClient:
    """Create OpenAIParamClient from OPENAI_API_KEY and optional OPENAI_MODEL override."""
    api_key = _get_env_var("OPENAI_API_KEY")
    configured = config or GoalGeneratorConfig()
    model = os.environ.get("OPENAI_MODEL", configured.model)
    resolved_config = GoalGeneratorConfig(
        model=model,
        temperature=configured.temperature,
        max_output_tokens=configured.max_output_tokens,
        schema_version=configured.schema_version,
    )
    client = OpenAI(api_key=api_key)
    return OpenAIParamClient(openai_client=client, config=resolved_config)
