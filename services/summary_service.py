"""
SummaryService — Business logic for conversation summary management.

Provides a clean interface for summary operations without exposing
the underlying storage implementation.
"""

from typing import List, Optional

from utils.logging import llmmllogger

logger = llmmllogger.bind(component="summary_service")


class SummaryService:
    """Service layer for summary operations."""

    def __init__(self):
        self._storage = None

    def _get_storage(self):
        if self._storage is None:
            from db import storage  # pylint: disable=import-outside-toplevel
            if not storage.initialized or not storage.summary:
                raise RuntimeError("Database not initialized")
            self._storage = storage.summary
        return self._storage

    @property
    def available(self) -> bool:
        """Check if the service is available (DB initialized)."""
        from db import storage  # pylint: disable=import-outside-toplevel
        return storage.initialized and storage.summary is not None

    async def get_summaries_for_conversation(self, conversation_id: int) -> list:
        """Get all summaries for a conversation."""
        return await self._get_storage().get_summaries_for_conversation(conversation_id)


summary_service = SummaryService()
