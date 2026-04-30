"""
ConversationService — Business logic for conversation management.

Provides a clean interface for conversation CRUD without exposing
the underlying storage implementation.
"""

from typing import Optional

from models.conversation import Conversation
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="conversation_service")


class ConversationService:
    """Service layer for conversation operations."""

    def __init__(self):
        self._storage = None

    def _get_storage(self):
        if self._storage is None:
            from db import storage  # pylint: disable=import-outside-toplevel
            if not storage.initialized or not storage.conversation:
                raise RuntimeError("Database not initialized")
            self._storage = storage.conversation
        return self._storage

    @property
    def available(self) -> bool:
        """Check if the service is available (DB initialized)."""
        from db import storage  # pylint: disable=import-outside-toplevel
        return storage.initialized and storage.conversation is not None

    async def get_user_conversations(self, user_id: str) -> list[Conversation]:
        """Get all conversations for a user."""
        return await self._get_storage().get_user_conversations(user_id)

    async def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """Get a specific conversation by ID."""
        return await self._get_storage().get_conversation(conversation_id)

    async def create_conversation(self, conversation: Conversation) -> Optional[int]:
        """Create a new conversation. Returns the new ID."""
        return await self._get_storage().create_conversation(conversation)

    async def delete_conversation(self, conversation_id: int) -> None:
        """Delete a conversation and all its messages."""
        await self._get_storage().delete_conversation(conversation_id)


conversation_service = ConversationService()
