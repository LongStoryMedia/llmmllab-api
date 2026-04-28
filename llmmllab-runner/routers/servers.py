"""Server lifecycle router - create, status, delete, release servers."""

import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import RUNNER_PORT
from models import UserConfig
from server_manager import LlamaCppServerManager
from utils.model_loader import ModelLoader

router = APIRouter()
model_loader = ModelLoader()


class CreateServerRequest(BaseModel):
    model_id: str
    priority: int = 10
    config_override: Optional[Dict[str, Any]] = None


@router.post("/v1/server/create")
async def create_server(request: CreateServerRequest):
    """Start a new llama.cpp server for the given model.

    Returns server_id and base_url for proxying requests.
    """
    from app import server_cache

    model = model_loader.get_model_by_id(request.model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")

    user_config = None
    if request.config_override:
        try:
            user_config = UserConfig(**request.config_override)
        except Exception:
            user_config = None

    manager = LlamaCppServerManager(
        model=model,
        user_config=user_config,
    )

    # Run blocking start() in a thread pool to avoid blocking the event loop
    started = await asyncio.get_event_loop().run_in_executor(None, manager.start)
    if not started:
        raise HTTPException(status_code=500, detail="Failed to start llama.cpp server")

    server_id = server_cache.register(
        model_id=model.id,
        port=manager.port,
        manager=manager,
    )
    server_cache.increment_use(server_id)

    base_url = f"http://localhost:{RUNNER_PORT}/v1/server/{server_id}"

    return {
        "server_id": server_id,
        "base_url": base_url,
        "model": model.id,
        "port": manager.port,
    }


@router.get("/v1/server/{server_id}")
def get_server(server_id: str):
    """Get status of a running server."""
    from app import server_cache

    entry = server_cache.get(server_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

    is_running = entry.manager.is_running() if entry.manager else False

    return {
        "server_id": entry.server_id,
        "model_id": entry.model_id,
        "port": entry.port,
        "use_count": entry.use_count,
        "healthy": entry.healthy,
        "running": is_running,
        "created_at": entry.created_at,
    }


@router.delete("/v1/server/{server_id}")
def delete_server(server_id: str):
    """Stop and remove a server."""
    from app import server_cache

    entry = server_cache.get(server_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

    if entry.manager is not None:
        entry.manager.stop()

    server_cache.remove(server_id)

    return {"status": "deleted", "server_id": server_id}


@router.post("/v1/server/{server_id}/release")
def release_server(server_id: str):
    """Decrement use count for a server (signal that a client is done)."""
    from app import server_cache

    ok = server_cache.decrement_use(server_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

    entry = server_cache.get(server_id)
    return {
        "server_id": server_id,
        "use_count": entry.use_count if entry else 0,
    }
