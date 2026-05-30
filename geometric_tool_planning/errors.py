"""Errors for intent-contract validation and repair."""

from __future__ import annotations


# Signal schema-level validation failures for intent payloads.
class IntentSchemaError(ValueError):
    """Raised when an intent payload violates the declared schema."""


# Signal semantic-level validation failures after schema parsing succeeds.
class IntentSemanticError(ValueError):
    """Raised when an intent payload is structurally valid but semantically invalid."""
