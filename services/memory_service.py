"""
MemoryService — Business logic for memory retrieval and storage.

Provides a clean interface for memory operations without exposing
the underlying storage implementation.
"""

from typing import List, Optional

from utils.logging import llmmllogger

logger = llmmllogger.bind(component="memory_service")


class MemoryService:
    """Service layer for memory operations."""

    def __init__(self):
        self._storage = None

    def _get_storage(self):
        if self._storage is None:
            from db import storage  # pylint: disable=import-outside-toplevel
            if not storage.initialized or not storage.memory:
                raise RuntimeError("Database not initialized")
            self._storage = storage.memory
        return self._storage

    @property
    def available(self) -> bool:
        """Check if the service is available (DB initialized)."""
        from db import storage  # pylint: disable=import-outside-toplevel
        return storage.initialized and storage.memory is not None

    async def search_similarity(
        self,
        embeddings: list,
        min_similarity: float = 0.7,
        limit: int = 10,
        user_id: Optional[str] = None,
        conversation_id: Optional[int] = None,
    ) -> list:
        """Search for memories similar to the given embeddings."""
        return await self._get_storage().search_similarity(
            embeddings=embeddings,
            min_similarity=min_similarity,
            limit=limit,
            user_id=user_id,
            conversation_id=conversation_id,
        )


memory_service = MemoryService()
