"""
Unit tests for services/runner_client.py.

Tests the RunnerClient HTTP client that routes requests among multiple
llmmllab-runner service instances.  The client now uses a persistent
``httpx.AsyncClient``, so tests mock ``_get_client()`` instead of patching
the ``httpx.AsyncClient`` constructor.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from services.runner_client import RunnerClient, ServerHandle
from models import ModelTask


class TestServerHandle:

    def test_construction(self):
        """Test ServerHandle dataclass fields."""
        handle = ServerHandle(
            base_url="http://runner:8000/v1/server/abc123",
            server_id="abc123",
            runner_host="http://runner:8000",
        )
        assert handle.base_url == "http://runner:8000/v1/server/abc123"
        assert handle.server_id == "abc123"
        assert handle.runner_host == "http://runner:8000"


def _mock_client(**overrides) -> AsyncMock:
    """Build an AsyncMock that behaves like an httpx.AsyncClient."""
    client = AsyncMock()
    client.is_closed = False
    for key, value in overrides.items():
        setattr(client, key, value)
    return client


class TestRunnerClientHealth:

    @pytest.mark.asyncio
    async def test_healthy_returns_data(self):
        """Mock 200 /health and verify _health returns dict."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ok",
            "gpu": {"available_vram_bytes": 12000000000},
            "active_servers": 0,
            "models": ["llama-3-8b"],
        }

        mock = _mock_client(get=AsyncMock(return_value=mock_response))

        client = RunnerClient(endpoints=["http://runner1:8000"])
        client._client = mock
        result = await client._health("http://runner1:8000")

        assert result is not None
        assert result["status"] == "ok"
        assert result["gpu"]["available_vram_bytes"] == 12000000000

    @pytest.mark.asyncio
    async def test_unhealthy_returns_none(self):
        """Mock httpx.RequestError and verify _health returns None."""
        mock = _mock_client(get=AsyncMock(side_effect=Exception("connection refused")))

        client = RunnerClient(endpoints=["http://runner1:8000"])
        client._client = mock
        result = await client._health("http://runner1:8000")

        assert result is None


class TestRunnerClientAcquire:

    @pytest.mark.asyncio
    async def test_acquire_returns_handle(self):
        """Mock health + create, verify acquire_server returns ServerHandle."""
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200
        mock_health_response.json.return_value = {
            "status": "ok",
            "gpu": {"available_vram_bytes": 12000000000},
            "active_servers": 0,
            "models": ["llama-3-8b"],
        }

        mock_create_response = MagicMock()
        mock_create_response.status_code = 201
        mock_create_response.json.return_value = {
            "server_id": "abc123",
            "base_url": "http://runner1:8000/v1/server/abc123",
            "model": "llama-3-8b",
        }
        mock_create_response.raise_for_status = MagicMock()

        mock = _mock_client(
            get=AsyncMock(return_value=mock_health_response),
            post=AsyncMock(return_value=mock_create_response),
        )

        client = RunnerClient(endpoints=["http://runner1:8000"])
        client._client = mock
        handle = await client.acquire_server("llama-3-8b", ModelTask.TEXTTOTEXT, {})

        assert isinstance(handle, ServerHandle)
        assert handle.server_id == "abc123"
        assert handle.base_url == "http://runner1:8000/v1/server/abc123"
        assert handle.runner_host == "http://runner1:8000"

    @pytest.mark.asyncio
    async def test_acquire_raises_if_none(self):
        """All runners unhealthy raises RuntimeError."""
        mock = _mock_client(
            get=AsyncMock(side_effect=Exception("connection refused")),
            post=AsyncMock(side_effect=Exception("connection refused")),
        )

        client = RunnerClient(
            endpoints=["http://runner1:8000", "http://runner2:8001"]
        )
        client._client = mock
        with pytest.raises(RuntimeError, match="No healthy runner"):
            await client.acquire_server("llama-3-8b", "TextGeneration", {})

    @pytest.mark.asyncio
    async def test_acquire_retries_on_507(self):
        """First runner returns 507, client tries next runner."""
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200
        mock_health_response.json.return_value = {
            "status": "ok",
            "gpu": {"available_vram_bytes": 12000000000},
            "active_servers": 0,
            "models": ["llama-3-8b"],
        }

        call_count = [0]

        async def mock_post(url, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                # First runner: 507
                resp = MagicMock()
                resp.status_code = 507
                resp.json.return_value = {"detail": "Insufficient capacity"}
                return resp
            else:
                # Second runner: success
                resp = MagicMock()
                resp.status_code = 201
                resp.json.return_value = {
                    "server_id": "ghi789",
                    "base_url": "http://runner2:8001/v1/server/ghi789",
                    "model": "llama-3-8b",
                }
                resp.raise_for_status = MagicMock()
                return resp

        mock = _mock_client(
            get=AsyncMock(return_value=mock_health_response),
            post=AsyncMock(side_effect=mock_post),
        )

        client = RunnerClient(
            endpoints=["http://runner1:8000", "http://runner2:8001"]
        )
        client._client = mock
        handle = await client.acquire_server("llama-3-8b", ModelTask.TEXTTOTEXT, {})

        assert handle is not None
        assert handle.server_id == "ghi789"


class TestRunnerClientRelease:

    @pytest.mark.asyncio
    async def test_release_calls_endpoint(self):
        """Verify release_server POSTs to /release endpoint."""
        mock_release_response = MagicMock()
        mock_release_response.status_code = 200
        mock_release_response.raise_for_status = MagicMock()

        mock = _mock_client(post=AsyncMock(return_value=mock_release_response))

        client = RunnerClient(endpoints=["http://runner1:8000"])
        client._client = mock
        handle = ServerHandle(
            base_url="http://runner1:8000/v1/server/abc123",
            server_id="abc123",
            runner_host="http://runner1:8000",
        )
        await client.release_server(handle)

        mock.post.assert_called_once()
        call_args = mock.post.call_args
        assert "/v1/server/abc123/release" in call_args[0][0]


class TestRunnerClientShutdown:

    @pytest.mark.asyncio
    async def test_shutdown_calls_endpoint(self):
        """Verify shutdown_server sends DELETE request."""
        mock_shutdown_response = MagicMock()
        mock_shutdown_response.status_code = 200
        mock_shutdown_response.raise_for_status = MagicMock()

        mock = _mock_client(delete=AsyncMock(return_value=mock_shutdown_response))

        client = RunnerClient(endpoints=["http://runner1:8000"])
        client._client = mock
        handle = ServerHandle(
            base_url="http://runner1:8000/v1/server/abc123",
            server_id="abc123",
            runner_host="http://runner1:8000",
        )
        await client.shutdown_server(handle)

        mock.delete.assert_called_once()
        call_args = mock.delete.call_args
        assert "/v1/server/abc123" in call_args[0][0]


class TestRunnerClientModels:

    @pytest.mark.asyncio
    async def test_list_models_aggregates(self):
        """Two runners return models, results are deduplicated by model id."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "llama-3-8b",
                "name": "Llama 3 8B",
                "model": "meta-llama/Llama-3-8B",
                "task": "TextToText",
                "modified_at": "2025-01-01",
                "digest": "abc123",
                "provider": "llama_cpp",
                "details": {"format": "gguf", "family": "llama", "families": ["llama"], "parameter_size": "8B", "size": 4000000000, "original_ctx": 8192},
            },
        ]

        mock = _mock_client(get=AsyncMock(return_value=mock_response))

        client = RunnerClient(
            endpoints=["http://runner1:8000", "http://runner2:8001"]
        )
        client._client = mock
        models = await client.list_models()

        # Deduplicated: should only have one entry
        model_ids = [m.id for m in models]
        assert model_ids.count("llama-3-8b") == 1

    @pytest.mark.asyncio
    async def test_model_by_task_filters(self):
        """GET /v1/models?task=TextToEmbeddings returns first match."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "nomic-embed",
                "name": "Nomic Embed",
                "model": "nomic-ai/nomic-embed",
                "task": "TextToEmbeddings",
                "modified_at": "2025-01-01",
                "digest": "def456",
                "provider": "llama_cpp",
                "details": {"format": "gguf", "family": "nomic", "families": ["nomic"], "parameter_size": "0.3B", "size": 200000000, "original_ctx": 8192},
            },
            {
                "id": "llama-3-8b",
                "name": "Llama 3 8B",
                "model": "meta-llama/Llama-3-8B",
                "task": "TextToText",
                "modified_at": "2025-01-01",
                "digest": "abc123",
                "provider": "llama_cpp",
                "details": {"format": "gguf", "family": "llama", "families": ["llama"], "parameter_size": "8B", "size": 4000000000, "original_ctx": 8192},
            },
        ]

        mock = _mock_client(get=AsyncMock(return_value=mock_response))

        client = RunnerClient(endpoints=["http://runner1:8000"])
        client._client = mock
        result = await client.model_by_task(ModelTask.TEXTTOEMBEDDINGS)

        assert result is not None
        assert result.id == "nomic-embed"
        assert result.task == ModelTask.TEXTTOEMBEDDINGS

        # Verify the task query param was included in the call
        call_args = mock.get.call_args
        assert call_args[1].get("params", {}).get("task") == "TextToEmbeddings"


class TestRunnerClientConfig:

    def test_default_refresh_interval(self):
        from config import MODEL_CACHE_REFRESH_SEC
        assert MODEL_CACHE_REFRESH_SEC == 60

    def test_refresh_interval_from_env(self, monkeypatch):
        """MODEL_CACHE_REFRESH_SEC reads from env var."""
        import importlib
        monkeypatch.setenv("MODEL_CACHE_REFRESH_SEC", "120")
        import config
        importlib.reload(config)
        assert config.MODEL_CACHE_REFRESH_SEC == 120


class TestRunnerClientConnectionPooling:
    """Tests for the persistent client / connection pooling behavior."""

    def test_get_client_creates_once(self):
        """Calling _get_client() twice returns the same instance."""
        client = RunnerClient(endpoints=["http://runner1:8000"])
        c1 = client._get_client()
        c2 = client._get_client()
        assert c1 is c2

    def test_get_client_recreates_after_close(self):
        """If the client is closed, _get_client() creates a new one."""
        client = RunnerClient(endpoints=["http://runner1:8000"])
        # Simulate a closed client
        old_mock = MagicMock()
        old_mock.is_closed = True
        client._client = old_mock

        c2 = client._get_client()
        # Should have created a fresh httpx.AsyncClient, not returned the mock
        assert c2 is not old_mock
        assert isinstance(c2, httpx.AsyncClient)

    @pytest.mark.asyncio
    async def test_aclose_closes_client(self):
        """aclose() closes the internal client."""
        client = RunnerClient(endpoints=["http://runner1:8000"])
        mock = AsyncMock()
        mock.is_closed = False
        client._client = mock

        await client.aclose()

        mock.aclose.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_aclose_noop_when_no_client(self):
        """aclose() is safe when no client has been created."""
        client = RunnerClient(endpoints=["http://runner1:8000"])
        await client.aclose()  # should not raise
