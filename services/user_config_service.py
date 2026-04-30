"""
UserConfigService — Business logic for user configuration management.

Provides a clean interface for user config operations without exposing
the underlying storage implementation.
"""

from typing import Optional

from models.user_config import UserConfig
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="user_config_service")


class UserConfigService:
    """Service layer for user configuration operations."""

    def __init__(self):
        self._storage = None

    def _get_storage(self):
        if self._storage is None:
            from db import storage  # pylint: disable=import-outside-toplevel

            if not storage.initialized or not storage.user_config:
                raise RuntimeError("Database not initialized")
            self._storage = storage.user_config
        return self._storage

    @property
    def available(self) -> bool:
        """Check if the service is available (DB initialized)."""
        from db import storage  # pylint: disable=import-outside-toplevel

        return storage.initialized and storage.user_config is not None

    async def get_user_config(self, user_id: str) -> UserConfig:
        """Get user configuration, creating with defaults if needed."""
        return await self._get_storage().get_user_config(user_id)

    async def update_user_config(self, user_id: str, config: UserConfig) -> None:
        """Update user configuration."""
        await self._get_storage().update_user_config(user_id, config)

    async def get_all_users(self) -> list[dict]:
        """Get all users (admin)."""
        return await self._get_storage().get_all_users()


user_config_service = UserConfigService()
