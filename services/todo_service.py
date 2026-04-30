"""
TodoService — Business logic for todo item management.

Provides a clean interface for todo CRUD without exposing
the underlying storage implementation.
"""

from typing import List, Optional

from models.todo_item import TodoItem
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="todo_service")


class TodoService:
    """Service layer for todo operations."""

    def __init__(self):
        self._storage = None

    def _get_storage(self):
        if self._storage is None:
            from db import storage  # pylint: disable=import-outside-toplevel
            if not storage.initialized or not storage.todo:
                raise RuntimeError("Database not initialized")
            self._storage = storage.todo
        return self._storage

    @property
    def available(self) -> bool:
        """Check if the service is available (DB initialized)."""
        from db import storage  # pylint: disable=import-outside-toplevel
        return storage.initialized and storage.todo is not None

    async def add_todo(self, todo: TodoItem) -> TodoItem:
        """Create a new todo item."""
        return await self._get_storage().add_todo(todo)

    async def get_todos_by_user(self, user_id: str) -> List[TodoItem]:
        """Get all todos for a user."""
        return await self._get_storage().get_todos_by_user(user_id)

    async def get_todos_by_status(self, user_id: str, status: str) -> List[TodoItem]:
        """Get todos filtered by status."""
        return await self._get_storage().get_todos_by_status(user_id, status)

    async def get_todo_by_id(self, todo_id: int, user_id: str) -> Optional[TodoItem]:
        """Get a specific todo by ID."""
        return await self._get_storage().get_todo_by_id(todo_id, user_id)

    async def update_todo(self, todo: TodoItem) -> TodoItem:
        """Update a todo item."""
        return await self._get_storage().update_todo(todo)

    async def delete_todo(self, todo_id: int, user_id: str) -> bool:
        """Delete a todo item. Returns True if successful."""
        return await self._get_storage().delete_todo(todo_id, user_id)

    async def get_todos_by_conversation(
        self, user_id: str, conversation_id: int
    ) -> List[TodoItem]:
        """Get todos for a specific conversation."""
        return await self._get_storage().get_todos_by_conversation(
            user_id, conversation_id
        )


todo_service = TodoService()
