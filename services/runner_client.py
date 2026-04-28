"""
RunnerClient — HTTP client for the llmmllab-runner service pool.

Routes requests among multiple runner instances based on health and
hardware capability (VRAM). Manages server lifecycle (acquire, release,
shutdown) and model discovery across all runners.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from config import RUNNER_ENDPOINTS
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="runner_client")


@dataclass
class ServerHandle:
    """Reference to an allocated llama.cpp server on a runner."""

    base_url: str
    server_id: str
    runner_host: str


class RunnerClient:
    """HTTP client that routes requests among multiple runner instances."""

    def __init__(self, endpoints: Optional[list[str]] = None):
        self._endpoints = endpoints if endpoints is not None else list(RUNNER_ENDPOINTS)
        self._healthy: list[str] = []

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def _health(self, endpoint: str) -> Optional[dict]:
        """Check health of a single runner. Returns health dict or None."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{endpoint}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    if endpoint not in self._healthy:
                        self._healthy.append(endpoint)
                    return data
                else:
                    if endpoint in self._healthy:
                        self._healthy.remove(endpoint)
                    return None
        except Exception as e:
            logger.warning(f"Runner {endpoint} health check failed: {e}")
            if endpoint in self._healthy:
                self._healthy.remove(endpoint)
            return None

    # ------------------------------------------------------------------
    # Runner selection
    # ------------------------------------------------------------------

    async def _select_runner(self, model_id: str) -> Optional[str]:
        """Iterate endpoints, pick highest VRAM runner with matching model."""
        best_url = None
        best_vram = -1
        for endpoint in self._endpoints:
            health = await self._health(endpoint)
            if not health:
                continue
            models = health.get("models", [])
            if model_id not in models:
                continue
            vram = health.get("gpu", {}).get("available_vram_bytes", 0)
            if vram > best_vram:
                best_vram = vram
                best_url = endpoint
        return best_url

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    async def acquire_server(
        self,
        model_id: str,
        task: str,
        config_override: Optional[dict] = None,
    ) -> ServerHandle:
        """Acquire a new llama.cpp server from a runner.

        Tries runners in order of VRAM capacity. Handles 507 (Insufficient
        Capacity) by falling through to the next runner.

        Returns:
            ServerHandle with connection details for the allocated server.

        Raises:
            RuntimeError: if no runner can satisfy the request.
        """
        config_override = config_override or {}
        payload: dict[str, Any] = {
            "model": model_id,
            "task": task,
            "config_override": config_override,
        }

        # Select the best runner by VRAM with matching model
        best = await self._select_runner(model_id)
        if best:
            # Put best runner first, then the rest
            ordered = [best]
            for ep in self._endpoints:
                if ep != best:
                    ordered.append(ep)
        else:
            ordered = list(self._endpoints)

        last_error = None
        for endpoint in ordered:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        f"{endpoint}/v1/server/create", json=payload
                    )

                    if resp.status_code == 507:
                        logger.warning(
                            f"Runner {endpoint} returned 507, trying next runner"
                        )
                        last_error = "Insufficient capacity"
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    handle = ServerHandle(
                        base_url=data["base_url"],
                        server_id=data["server_id"],
                        runner_host=endpoint,
                    )
                    logger.info(
                        f"Acquired server {handle.server_id} from {endpoint}"
                    )
                    return handle

            except Exception as e:
                logger.warning(f"Failed to acquire from {endpoint}: {e}")
                last_error = str(e)
                continue

        raise RuntimeError(
            f"No healthy runner available for model {model_id}. "
            f"Last error: {last_error}"
        )

    async def release_server(self, handle: ServerHandle) -> None:
        """Release an acquired server back to the runner."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{handle.runner_host}/v1/server/{handle.server_id}/release"
                )
                resp.raise_for_status()
            logger.info(f"Released server {handle.server_id}")
        except Exception as e:
            logger.error(f"Failed to release server {handle.server_id}: {e}")
            raise

    async def shutdown_server(self, handle: ServerHandle) -> None:
        """Permanently shut down a server on the runner."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.delete(
                    f"{handle.runner_host}/v1/server/{handle.server_id}"
                )
                resp.raise_for_status()
            logger.info(f"Shutdown server {handle.server_id}")
        except Exception as e:
            logger.error(f"Failed to shutdown server {handle.server_id}: {e}")
            raise

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    async def list_models(self) -> list[dict]:
        """List all available models across all runners, deduplicated by id."""
        seen_ids: set[str] = set()
        all_models: list[dict] = []

        tasks = []
        for endpoint in self._endpoints:

            async def fetch_models(ep=endpoint):
                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        resp = await client.get(f"{ep}/v1/models")
                        if resp.status_code == 200:
                            return resp.json()
                except Exception as e:
                    logger.warning(f"Failed to list models from {ep}: {e}")
                return []

            tasks.append(fetch_models())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                for model in result:
                    mid = model.get("id")
                    if mid and mid not in seen_ids:
                        seen_ids.add(mid)
                        all_models.append(model)

        return all_models

    async def model_by_task(self, task: str) -> Optional[dict]:
        """Find the first model matching the given task across all runners."""
        for endpoint in self._endpoints:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(
                        f"{endpoint}/v1/models", params={"task": task}
                    )
                    if resp.status_code == 200:
                        models = resp.json()
                        for model in models:
                            if model.get("task") == task:
                                return model
            except Exception as e:
                logger.warning(f"Failed to query models from {endpoint}: {e}")
                continue
        return None


# Module-level singleton
runner_client = RunnerClient()
