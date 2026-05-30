"""Custom exception types for llm_runtime."""


class LLMIntegrationError(Exception):
    """Base exception for llm-runtime failures."""


class LLMResponseFormatError(LLMIntegrationError):
    """Raised when an LLM response cannot be parsed into the expected shape."""


class SchemaValidationError(LLMIntegrationError):
    """Raised when geometric goal parameters fail schema validation."""


class ConverterError(LLMIntegrationError):
    """Raised when conversion from parameters to an SE(3) sequence fails."""


class MissingEnvironmentError(LLMIntegrationError):
    """Raised when a required environment variable is missing."""
