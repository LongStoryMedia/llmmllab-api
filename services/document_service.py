"""
DocumentService — Business logic for document management.

Provides a clean interface for document operations without exposing
the underlying storage implementation.
"""

from typing import List, Optional

from models.document import Document
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="document_service")


class DocumentService:
    """Service layer for document operations."""

    def __init__(self):
        self._storage = None

    def _get_storage(self):
        if self._storage is None:
            from db import storage  # pylint: disable=import-outside-toplevel

            if not storage.initialized or not storage.document:
                raise RuntimeError("Database not initialized")
            self._storage = storage.document
        return self._storage

    @property
    def available(self) -> bool:
        """Check if the service is available (DB initialized)."""
        from db import storage  # pylint: disable=import-outside-toplevel

        return storage.initialized and storage.document is not None

    async def store_document(
        self,
        message_id: int,
        user_id: str,
        filename: str,
        content_type: str,
        file_size: int,
        content: str,
        text_content: Optional[str] = None,
    ) -> Document:
        """Store a new document."""
        return await self._get_storage().store_document(
            message_id=message_id,
            user_id=user_id,
            filename=filename,
            content_type=content_type,
            file_size=file_size,
            content=content,
            text_content=text_content,
        )

    async def get_document(self, document_id: int) -> Optional[Document]:
        """Get a document by ID."""
        return await self._get_storage().get_document(document_id)

    async def get_documents_for_conversation(
        self, conversation_id: int
    ) -> List[Document]:
        """Get all documents for a conversation."""
        return await self._get_storage().get_documents_for_conversation(conversation_id)


document_service = DocumentService()
