"""
ModelService — resolves the effective model for a request.

Rules:
1. If the caller specified a model and it is available on a runner, use it.
2. If the caller specified a model but it is *not* available, fall back to
   the user's ``default_model`` (from UserConfig).
3. If the caller did *not* specify a model at all, use the user's
   ``default_model``.
4. If no default_model is configured, return the original (possibly
   unavailable) model so the downstream error path can handle it.
"""

from __future__ import annotations

from typing import Optional

from utils.logging import llmmllogger

logger = llmmllogger.bind(component="model_service")


class ModelService:
    """Stateless helper that resolves model names against available runners."""

    @staticmethod
    async def resolve_default_model(
        requested_model: Optional[str],
        user_id: str,
    ) -> str:
        """Return the model ID that should be used for this request.

        Parameters
        ----------
        requested_model:
            The model name from the incoming API request (may be ``None``).
        user_id:
            The authenticated user ID (needed to look up ``default_model``).

        Returns
        -------
        str
            The resolved model ID.
        """
        from db import storage  # lazy import to avoid circular deps

        # Fast path: requested model is available on a runner
        if requested_model:
            available = await ModelService._available_model_ids()
            if requested_model in available:
                return requested_model

            # Requested model not found — try user's default_model
            fallback = await ModelService._user_default_model(user_id)
            if fallback:
                logger.info(
                    "Requested model not available, falling back to default_model",
                    extra={
                        "user_id": user_id,
                        "requested": requested_model,
                        "fallback": fallback,
                    },
                )
                return fallback

            # No fallback configured — return original so downstream can error
            logger.warning(
                "Requested model not available and no default_model configured",
                extra={"user_id": user_id, "requested": requested_model},
            )
            return requested_model

        # No model specified — use user's default_model
        fallback = await ModelService._user_default_model(user_id)
        if fallback:
            return fallback

        # Nothing to fall back to — return None so downstream errors
        logger.warning(
            "No model specified and no default_model configured",
            extra={"user_id": user_id},
        )
        return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _cached_model_ids: Optional[set[str]] = None

    @staticmethod
    async def _available_model_ids() -> set[str]:
        """Return the set of model IDs currently available on any runner."""
        if ModelService._cached_model_ids is not None:
            return ModelService._cached_model_ids

        from services.runner_client import runner_client

        try:
            models = await runner_client.list_models()
            ModelService._cached_model_ids = {m.id for m in models if m.id}
        except Exception as e:
            logger.warning(f"Failed to list models from runners: {e}")
            ModelService._cached_model_ids = set()

        return ModelService._cached_model_ids

    @staticmethod
    async def _user_default_model(user_id: str) -> Optional[str]:
        """Look up the user's configured default_model."""
        from db import storage

        try:
            config = await storage.user_config.get_user_config(user_id)
            if config and hasattr(config, "default_model") and config.default_model:
                return config.default_model
        except Exception as e:
            logger.warning(
                f"Failed to load user config for default_model lookup: {e}"
            )

        return None

    @staticmethod
    def invalidate_cache() -> None:
        """Clear the cached model list (call when models change)."""
        ModelService._cached_model_ids = None
