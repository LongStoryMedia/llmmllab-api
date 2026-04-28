"""
Unit tests for services/runner_client.py.

Tests the RunnerClient HTTP client that routes requests among multiple
llmmllab-runner service instances.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from services.runner_client import RunnerClient, ServerHandle


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

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("services.runner_client.httpx.AsyncClient", return_value=mock_client):
            client = RunnerClient(endpoints=["http://runner1:8000"])
            result = await client._health("http://runner1:8000")

        assert result is not None
        assert result["status"] == "ok"
        assert result["gpu"]["available_vram_bytes"] == 12000000000

    @pytest.mark.asyncio
    async def test_unhealthy_returns_none(self):
        """Mock httpx.RequestError and verify _health returns None."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("services.runner_client.httpx.AsyncClient", return_value=mock_client):
            client = RunnerClient(endpoints=["http://runner1:8000"])
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

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_health_response)
        mock_client.post = AsyncMock(return_value=mock_create_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("services.runner_client.httpx.AsyncClient", return_value=mock_client):
            client = RunnerClient(endpoints=["http://runner1:8000"])
            handle = await client.acquire_server("llama-3-8b", "TextGeneration", {})

        assert isinstance(handle, ServerHandle)
        assert handle.server_id == "abc123"
        assert handle.base_url == "http://runner1:8000/v1/server/abc123"
        assert handle.runner_host == "http://runner1:8000"

    @pytest.mark.asyncio
    async def test_acquire_skips_unhealthy(self):
        """First runner fails health, second succeeds."""
        client_index = [0]

        def fake_async_client(*args, **kwargs):
            idx = client_index[0]
            client_index[0] += 1
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)

            # First httpx.AsyncClient call is for health check on runner1
            # Second is for health check on runner2
            # Third is for POST /create on runner2 (since runner1 failed health)
            if idx <= 1:
                # Health check phase
                async def side_effect_get(url, **kw):
                    if "runner1" in url:
                        raise Exception("connection refused")
                    mock_resp = MagicMock()
                    mock_resp.status_code = 200
                    mock_resp.json.return_value = {
                        "status": "ok",
                        "gpu": {"available_vram_bytes": 8000000000},
                        "active_servers": 0,
                        "models": ["llama-3-8b"],
                    }
                    return mock_resp
                instance.get = AsyncMock(side_effect=side_effect_get)
            else:
                # POST /create phase — runner2 succeeds
                async def side_effect_get(url, **kw):
                    mock_resp = MagicMock()
                    mock_resp.status_code = 200
                    mock_resp.json.return_value = {
                        "status": "ok",
                        "gpu": {"available_vram_bytes": 8000000000},
                        "active_servers": 0,
                        "models": ["llama-3-8b"],
                    }
                    return mock_resp

                async def side_effect_post(url, **kw):
                    mock_resp = MagicMock()
                    mock_resp.status_code = 201
                    mock_resp.json.return_value = {
                        "server_id": "def456",
                        "base_url": "http://runner2:8001/v1/server/def456",
                        "model": "llama-3-8b",
                    }
                    mock_resp.raise_for_status = MagicMock()
                    return mock_resp

                instance.get = AsyncMock(side_effect=side_effect_get)
                instance.post = AsyncMock(side_effect=side_effect_post)

            return instance

        with patch("services.runner_client.httpx.AsyncClient", side_effect=fake_async_client):
            client = RunnerClient(
                endpoints=["http://runner1:8000", "http://runner2:8001"]
            )
            handle = await client.acquire_server("llama-3-8b", "TextGeneration", {})

        assert handle is not None
        assert handle.runner_host == "http://runner2:8001"

    @pytest.mark.asyncio
    async def test_acquire_raises_if_none(self):
        """All runners unhealthy raises RuntimeError."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
        mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("services.runner_client.httpx.AsyncClient", return_value=mock_client):
            client = RunnerClient(
                endpoints=["http://runner1:8000", "http://runner2:8001"]
            )
            with pytest.raises(RuntimeError, match="No healthy runner"):
                await client.acquire_server("llama-3-8b", "TextGeneration", {})

    @pytest.mark.asyncio
    async def test_acquire_retries_on_507(self):
        """First runner returns 507, client tries next runner."""
        client_index = [0]

        def fake_async_client(*args, **kwargs):
            idx = client_index[0]
            client_index[0] += 1
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)

            # Health checks always succeed
            async def side_effect_get(url, **kw):
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "status": "ok",
                    "gpu": {"available_vram_bytes": 12000000000},
                    "active_servers": 0,
                    "models": ["llama-3-8b"],
                }
                return mock_resp

            instance.get = AsyncMock(side_effect=side_effect_get)

            if idx <= 1:
                # Health check phase — no POST needed
                instance.post = AsyncMock()
            elif "runner1" in str(args) or idx == 2:
                # First POST attempt — runner1 returns 507
                async def side_effect_post(url, **kw):
                    mock_resp = MagicMock()
                    mock_resp.status_code = 507
                    mock_resp.json.return_value = {"detail": "Insufficient capacity"}
                    return mock_resp
                instance.post = AsyncMock(side_effect=side_effect_post)
            else:
                # Second POST attempt — runner2 succeeds
                async def side_effect_post(url, **kw):
                    mock_resp = MagicMock()
                    mock_resp.status_code = 201
                    mock_resp.json.return_value = {
                        "server_id": "ghi789",
                        "base_url": "http://runner2:8001/v1/server/ghi789",
                        "model": "llama-3-8b",
                    }
                    mock_resp.raise_for_status = MagicMock()
                    return mock_resp
                instance.post = AsyncMock(side_effect=side_effect_post)

            return instance

        with patch("services.runner_client.httpx.AsyncClient", side_effect=fake_async_client):
            client = RunnerClient(
                endpoints=["http://runner1:8000", "http://runner2:8001"]
            )
            handle = await client.acquire_server("llama-3-8b", "TextGeneration", {})

        assert handle is not None
        assert handle.runner_host == "http://runner2:8001"
        assert handle.server_id == "ghi789"


class TestRunnerClientRelease:

    @pytest.mark.asyncio
    async def test_release_calls_endpoint(self):
        """Verify release_server POSTs to /release endpoint."""
        mock_release_response = MagicMock()
        mock_release_response.status_code = 200
        mock_release_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_release_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("services.runner_client.httpx.AsyncClient", return_value=mock_client):
            client = RunnerClient(endpoints=["http://runner1:8000"])
            handle = ServerHandle(
                base_url="http://runner1:8000/v1/server/abc123",
                server_id="abc123",
                runner_host="http://runner1:8000",
            )
            await client.release_server(handle)

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/v1/server/abc123/release" in call_args[0][0]


class TestRunnerClientShutdown:

    @pytest.mark.asyncio
    async def test_shutdown_calls_endpoint(self):
        """Verify shutdown_server sends DELETE request."""
        mock_shutdown_response = MagicMock()
        mock_shutdown_response.status_code = 200
        mock_shutdown_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.delete = AsyncMock(return_value=mock_shutdown_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("services.runner_client.httpx.AsyncClient", return_value=mock_client):
            client = RunnerClient(endpoints=["http://runner1:8000"])
            handle = ServerHandle(
                base_url="http://runner1:8000/v1/server/abc123",
                server_id="abc123",
                runner_host="http://runner1:8000",
            )
            await client.shutdown_server(handle)

        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert "/v1/server/abc123" in call_args[0][0]


class TestRunnerClientModels:

    @pytest.mark.asyncio
    async def test_list_models_aggregates(self):
        """Two runners return models, results are deduplicated by model id."""

        def fake_async_client(*args, **kwargs):
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)

            async def side_effect_get(url, **kw):
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                # Both runners share llama-3-8b
                mock_resp.json.return_value = [
                    {
                        "id": "llama-3-8b",
                        "name": "Llama 3 8B",
                        "task": "TextGeneration",
                    },
                ]
                return mock_resp

            instance.get = AsyncMock(side_effect=side_effect_get)
            return instance

        with patch("services.runner_client.httpx.AsyncClient", side_effect=fake_async_client):
            client = RunnerClient(
                endpoints=["http://runner1:8000", "http://runner2:8001"]
            )
            models = await client.list_models()

        # Deduplicated: should only have one entry
        model_ids = [m["id"] for m in models]
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
                "task": "TextToEmbeddings",
            },
            {
                "id": "llama-3-8b",
                "name": "Llama 3 8B",
                "task": "TextGeneration",
            },
        ]

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("services.runner_client.httpx.AsyncClient", return_value=mock_client):
            client = RunnerClient(endpoints=["http://runner1:8000"])
            result = await client.model_by_task("TextToEmbeddings")

        assert result is not None
        assert result["id"] == "nomic-embed"
        assert result["task"] == "TextToEmbeddings"

        # Verify the task query param was included in the call
        call_args = mock_client.get.call_args
        # httpx accepts params as kwargs
        assert call_args[1].get("params", {}).get("task") == "TextToEmbeddings"
