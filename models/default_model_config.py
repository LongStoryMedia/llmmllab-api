"""
Default model configuration.

Holds the default_model field for UserConfig.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DefaultModelConfig(BaseModel):
    """Configuration for the user's default model fallback."""

    default_model: str | None = Field(
        default=None,
        description=(
            "Model ID to use when no model is specified in a request, "
            "or when the requested model is not available on any runner."
        ),
    )
