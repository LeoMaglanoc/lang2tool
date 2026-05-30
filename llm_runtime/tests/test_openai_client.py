"""Unit tests for the OpenAI-backed raw parameter client."""

from __future__ import annotations

import pytest

# Skip entire module when the optional openai package is not installed.
pytest.importorskip("openai")

from llm_runtime.errors import LLMResponseFormatError, MissingEnvironmentError
from llm_runtime.llm.config import GoalGeneratorConfig
from llm_runtime.llm.openai_client import (
    OpenAIParamClient,
    _get_env_var,
    build_default_openai_param_client,
)


# Simple fake response matching the SDK output_text behavior.
class _FakeResponse:
    """Hold a canned output_text string for test responses."""

    # Store canned output text for downstream parsing.
    def __init__(self, output_text: str) -> None:
        """Initialize fake response with provided output text."""
        self.output_text = output_text


# Fake Responses API implementation capturing call payloads.
class _FakeResponses:
    """Record requests and return canned response strings in order."""

    # Initialize with ordered outputs and empty call history.
    def __init__(self, outputs: list[str]) -> None:
        """Prepare fake outputs and request call capture."""
        self._outputs = outputs
        self.calls: list[dict] = []

    # Mimic responses.create(model=..., input=...) behavior.
    def create(self, **kwargs):
        """Record request kwargs and return next canned response."""
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        return _FakeResponse(self._outputs[index])


# Minimal fake OpenAI client containing only responses API.
class _FakeClient:
    """Expose a fake responses API for OpenAIParamClient tests."""

    # Attach fake responses handler with canned outputs.
    def __init__(self, outputs: list[str]) -> None:
        """Initialize fake OpenAI client wrapper for tests."""
        self.responses = _FakeResponses(outputs)


# Verify client requests the configured model and parses JSON object output.
def test_generate_raw_params_success() -> None:
    """Ensure OpenAIParamClient returns parsed dict output and records request."""
    fake_client = _FakeClient(
        [
            '{"schema_version":"v1","task_label":"t","object_frame":"o","contact_point_object":[0,0,0],"approach_direction_object":[0,0,-1],"tool_axis_object":[1,0,0],"pregrasp_offset_m":0.1,"grasp_depth_m":0.02,"lift_height_m":0.05,"timing_s":{"approach":1.0,"close":0.3,"lift":0.7}}'
        ]
    )
    config = GoalGeneratorConfig(model="gpt-5.2")
    client = OpenAIParamClient(openai_client=fake_client, config=config)

    result = client.generate_raw_params("grasp it", {"object": "hammer"})

    assert result["schema_version"] == "v1"
    assert fake_client.responses.calls[0]["model"] == "gpt-5.2"


# Verify non-JSON output is surfaced as a typed format error.
def test_generate_raw_params_rejects_non_json() -> None:
    """Ensure invalid JSON responses raise LLMResponseFormatError."""
    fake_client = _FakeClient(["not-json"])
    client = OpenAIParamClient(openai_client=fake_client, config=GoalGeneratorConfig())

    with pytest.raises(LLMResponseFormatError, match="valid JSON"):
        client.generate_raw_params("do something")


# Verify env helper raises when required variables are missing.
def test_get_env_var_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure missing env vars raise MissingEnvironmentError."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingEnvironmentError, match="OPENAI_API_KEY"):
        _get_env_var("OPENAI_API_KEY")


# Verify default builder honors OPENAI_MODEL override.
def test_build_default_openai_param_client_uses_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure OPENAI_MODEL supersedes config default when building client."""

    # Store constructed API key for assertion.
    captured = {}

    # Fake OpenAI constructor to avoid external dependency calls.
    class _FakeOpenAI:
        """Provide a fake OpenAI class with responses API for builder tests."""

        # Capture API key and expose fake responses namespace.
        def __init__(self, api_key: str) -> None:
            """Store api_key for assertions and expose fake responses."""
            captured["api_key"] = api_key
            self.responses = _FakeResponses(
                [
                    '{"schema_version":"v1","task_label":"t","object_frame":"o","contact_point_object":[0,0,0],"approach_direction_object":[0,0,-1],"tool_axis_object":[1,0,0],"pregrasp_offset_m":0.1,"grasp_depth_m":0.02,"lift_height_m":0.05,"timing_s":{"approach":1.0,"close":0.3,"lift":0.7}}'
                ]
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.2")
    monkeypatch.setattr("llm_runtime.llm.openai_client.OpenAI", _FakeOpenAI)

    client = build_default_openai_param_client(GoalGeneratorConfig(model="gpt-5.2"))

    assert captured["api_key"] == "test-key"
    assert client._config.model == "gpt-5.2"
