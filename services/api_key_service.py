"""
ApiKeyService — Business logic for API key management.

Provides a clean interface for API key CRUD and validation without
exposing the underlying storage implementation.
"""

from typing import List, Optional, Tuple

from models import ApiKey
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="api_key_service")


class ApiKeyService:
    """Service layer for API key operations."""

    def __init__(self):
        self._storage = None

    def _get_storage(self):
        if self._storage is None:
            from db import storage  # pylint: disable=import-outside-toplevel

            if not storage.initialized or not storage.api_key:
                raise RuntimeError("Database not initialized")
            self._storage = storage.api_key
        return self._storage

    @property
    def available(self) -> bool:
        """Check if the service is available (DB initialized)."""
        from db import storage  # pylint: disable=import-outside-toplevel

        return storage.initialized and storage.api_key is not None

    async def create_api_key(
        self,
        user_id: str,
        name: str,
        scopes: list[str],
        expires_in_days: Optional[int] = None,
    ) -> Tuple[str, ApiKey]:
        """Create a new API key. Returns (plaintext_key, api_key_object)."""
        return await self._get_storage().create_api_key(
            user_id=user_id,
            name=name,
            scopes=scopes,
            expires_in_days=expires_in_days,
        )

    async def list_api_keys_for_user(self, user_id: str) -> List[ApiKey]:
        """List all API keys for a user (without plaintext keys)."""
        return await self._get_storage().list_api_keys_for_user(user_id)

    async def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        """Revoke an API key. Returns True if successful."""
        return await self._get_storage().revoke_api_key(key_id, user_id)

    async def delete_api_key(self, key_id: str, user_id: str) -> bool:
        """Delete an API key. Returns True if successful."""
        return await self._get_storage().delete_api_key(key_id, user_id)

    async def validate_api_key(self, api_key: str):
        """Validate an API key and return the key object if valid."""
        return await self._get_storage().validate_api_key(api_key)

    async def update_last_used(self, key_id: str):
        """Update the last_used_at timestamp for an API key."""
        return await self._get_storage().update_last_used(key_id)


api_key_service = ApiKeyService()
