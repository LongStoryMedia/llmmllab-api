"""
MessageService — Business logic for message management.

Provides a clean interface for message CRUD without exposing
the underlying storage implementation.
"""

from typing import Optional

from models.message import Message
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="message_service")


class MessageService:
    """Service layer for message operations."""

    def __init__(self):
        self._storage = None

    def _get_storage(self):
        if self._storage is None:
            from db import storage  # pylint: disable=import-outside-toplevel
            if not storage.initialized or not storage.message:
                raise RuntimeError("Database not initialized")
            self._storage = storage.message
        return self._storage

    @property
    def available(self) -> bool:
        """Check if the service is available (DB initialized)."""
        from db import storage  # pylint: disable=import-outside-toplevel
        return storage.initialized and storage.message is not None

    async def get_messages_by_conversation_id(
        self, conversation_id: int, limit: int = 500, offset: int = 0
    ) -> list[Message]:
        """Get messages for a conversation."""
        return await self._get_storage().get_messages_by_conversation_id(
            conversation_id, limit, offset
        )

    async def get_conversation_history(self, conversation_id: int) -> list[Message]:
        """Get the full conversation history for a conversation."""
        return await self._get_storage().get_conversation_history(conversation_id)

    async def get_message(self, message_id: int) -> Optional[Message]:
        """Get a specific message by ID."""
        return await self._get_storage().get_message(message_id)

    async def add_message(self, message: Message) -> Optional[int]:
        """Store a message. Returns the new message ID."""
        return await self._get_storage().add_message(message)

    async def delete_message(self, message_id: int) -> None:
        """Delete a specific message."""
        await self._get_storage().delete_message(message_id)

    async def delete_all_from_message(self, message: Message) -> None:
        """Delete all messages from a given message onwards (for replay)."""
        await self._get_storage().delete_all_from_message(message)


message_service = MessageService()
