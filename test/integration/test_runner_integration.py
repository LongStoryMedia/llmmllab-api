"""Integration tests for llmmllab-api -> llmmllab-runner HTTP communication.

Tests the API's RunnerClient, /models endpoint, and full chat completion
flow (runner → llama.cpp → LLM response) against a running llmmllab-runner
instance. Does NOT import runner source code.

Prerequisites:
1. llmmllab-runner running on a known port (default 8000)
   - Start manually: cd ../llmmllab-runner && make start
   - Or set RUNNER_ENDPOINTS env var to point to your runner
2. A llama.cpp binary and at least one small model configured in
   .models.local.yaml (the runner's .env sets LLAMA_SERVER_EXECUTABLE)

Run with:
    RUNNER_ENDPOINTS=http://localhost:8000 uv run pytest test/integration/test_runner_integration.py -v

Chat tests require the model to load and respond (can take 30-120s).
Use -k chat to run only the chat tests, or -k "not chat" to skip them:
    RUNNER_ENDPOINTS=http://localhost:8000 uv run pytest test/integration/test_runner_integration.py -v -k chat
"""

import asyncio
import os
import sys

import httpx
import pytest
import pytest_asyncio

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Integration tests require valid auth credentials.
# Read TEST_API_KEY from .env.local (seeded by app startup when TEST_USER_ID is set).
# Prevent huggingface_hub interactive login prompt
os.environ.setdefault("HF_TOKEN", "sk-dummy-testing-token")

RUNNER_ENDPOINT = os.environ.get("RUNNER_ENDPOINTS", "http://localhost:8000").split(",")[0]


def _check_runner_available():
    """Check if the runner is reachable."""
    try:
        r = httpx.get(f"{RUNNER_ENDPOINT}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# Skip entire module if runner is not available
pytestmark = pytest.mark.skipif(
    not _check_runner_available(),
    reason=f"llmmllab-runner not reachable at {RUNNER_ENDPOINT}",
)


# ---------------------------------------------------------------------------
# Runner endpoint tests (HTTP only, no imports from runner)
# ---------------------------------------------------------------------------


class TestRunnerHTTPEndpoints:
    """Test runner's HTTP endpoints via httpx (no cross imports)."""

    def test_health(self):
        """GET /health returns 200 with status ok."""
        resp = httpx.get(f"{RUNNER_ENDPOINT}/health", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "gpu" in data
        assert "active_servers" in data
        assert "models" in data

    def test_list_models(self):
        """GET /v1/models returns model list."""
        resp = httpx.get(f"{RUNNER_ENDPOINT}/v1/models", timeout=5)
        assert resp.status_code == 200
        models = resp.json()
        assert isinstance(models, list)
        assert len(models) > 0

        model = models[0]
        assert "id" in model
        assert "name" in model
        assert "task" in model
        assert "provider" in model
        assert "details" in model

    def test_list_models_by_task(self):
        """GET /v1/models?task= filters correctly."""
        resp = httpx.get(
            f"{RUNNER_ENDPOINT}/v1/models",
            params={"task": "TextToText"},
            timeout=5,
        )
        assert resp.status_code == 200
        models = resp.json()
        for m in models:
            assert m["task"] == "TextToText"

    def test_models_schema_details(self):
        """Models endpoint returns proper details sub-object."""
        resp = httpx.get(f"{RUNNER_ENDPOINT}/v1/models", timeout=5)
        models = resp.json()
        details = models[0]["details"]

        assert "format" in details
        assert "family" in details
        assert "size" in details
        assert "original_ctx" in details


# ---------------------------------------------------------------------------
# API RunnerClient tests (hits real runner via HTTP)
# ---------------------------------------------------------------------------


class TestRunnerClientIntegration:
    """Test RunnerClient against a live runner instance."""

    def test_list_models(self):
        """RunnerClient.list_models returns Model objects from runner."""
        from models import Model
        from services.runner_client import RunnerClient

        client = RunnerClient(endpoints=[RUNNER_ENDPOINT])

        async def _run():
            return await client.list_models()

        models = asyncio.new_event_loop().run_until_complete(_run())

        assert isinstance(models, list)
        assert len(models) > 0
        assert isinstance(models[0], Model)

        ids = [m.id for m in models]
        assert len(ids) == len(set(ids))

    def test_model_by_task(self):
        """RunnerClient.model_by_task returns correct model."""
        from models import ModelTask
        from services.runner_client import RunnerClient

        client = RunnerClient(endpoints=[RUNNER_ENDPOINT])

        async def _run():
            return await client.model_by_task(ModelTask.TEXTTOTEXT)

        model = asyncio.new_event_loop().run_until_complete(_run())

        assert model is not None
        assert model.task.value == "TextToText"
        assert model.id is not None

    def test_model_types_preserved(self):
        """Model objects from runner have correct types."""
        from models import Model, ModelDetails, ModelProvider
        from services.runner_client import RunnerClient

        client = RunnerClient(endpoints=[RUNNER_ENDPOINT])

        async def _run():
            return await client.list_models()

        models = asyncio.new_event_loop().run_until_complete(_run())
        model = models[0]

        assert isinstance(model, Model)
        assert isinstance(model.provider, ModelProvider)
        assert isinstance(model.details, ModelDetails)
        assert isinstance(model.details.size, int)
        assert isinstance(model.details.original_ctx, int)


# ---------------------------------------------------------------------------
# API /models endpoint tests (proxies to runner)
# ---------------------------------------------------------------------------


def _get_api_app():
    """Import app with HF login mocked to avoid network calls."""
    from unittest.mock import patch

    with patch("huggingface_hub.login"):
        from app import app

        return app


class TestAPIModelsEndpoint:
    """Test API's /models endpoint proxies to runner correctly."""

    def test_models_endpoint(self):
        """GET /models returns models from runner."""
        app = _get_api_app()
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            resp = client.get("/models/", headers=_auth_headers())
            assert resp.status_code == 200
            models = resp.json()
            assert isinstance(models, list)
            assert len(models) > 0

            model = models[0]
            assert "id" in model
            assert "name" in model
            assert "task" in model


def _read_test_api_key() -> str | None:
    """Read TEST_API_KEY from .env.local if it exists."""
    from pathlib import Path

    env_local = Path(_PROJECT_ROOT) / ".env.local"
    if not env_local.exists():
        return None
    for line in env_local.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == "TEST_API_KEY":
            return v.strip()
    return None


def _auth_headers() -> dict[str, str]:
    """Return auth headers using the test API key from .env.local."""
    api_key = _read_test_api_key()
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


# ---------------------------------------------------------------------------
# API /v1/models endpoint (OpenAI compat)
# ---------------------------------------------------------------------------


class TestAPIOpenAIModelsEndpoint:
    """Test API's /v1/models endpoint (OpenAI compatible)."""

    def test_openai_models_endpoint(self):
        """GET /v1/models returns OpenAI-compatible model list."""
        app = _get_api_app()
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            resp = client.get("/v1/models", headers=_auth_headers())
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "list"
            assert len(data["data"]) > 0

            m = data["data"][0]
            assert "id" in m
            assert "object" in m
            assert m["object"] == "model"


# ---------------------------------------------------------------------------
# Chat completion tests (runner → llama.cpp → LLM response)
# ---------------------------------------------------------------------------


class TestRunnerChatCompletion:
    """Test full chat completion through runner's llama.cpp proxy.

    These tests create a real llama.cpp server, send a chat completion
    request through the runner's proxy, and validate the LLM response.

    They are gated by the RUNNER_CHAT_TEST env var because they require
    a GPU, llama.cpp binary, and model on disk. Loading a model can take
    30-120 seconds.
    """

    @pytest.fixture(autouse=True)
    def _chat_enabled(self):
        """Skip all tests in this class unless RUNNER_CHAT_TEST=1."""
        if os.environ.get("RUNNER_CHAT_TEST") not in ("1", "true"):
            pytest.skip("Set RUNNER_CHAT_TEST=1 to run chat completion tests")

    @staticmethod
    def _cleanup(server_id: str):
        """Delete a server on the runner."""
        try:
            httpx.delete(
                f"{RUNNER_ENDPOINT}/v1/server/{server_id}",
                timeout=30,
            )
        except Exception:
            pass

    @staticmethod
    def _get_test_model_id() -> str:
        """Find a TextToText model from the runner for chat testing."""
        resp = httpx.get(
            f"{RUNNER_ENDPOINT}/v1/models",
            params={"task": "TextToText"},
            timeout=5,
        )
        resp.raise_for_status()
        models = resp.json()
        if not models:
            pytest.skip("No TextToText models available on runner")

        model_id = os.environ.get("RUNNER_CHAT_MODEL") or models[0]["id"]
        return model_id

    @staticmethod
    def _create_server(model_id: str) -> str:
        """Create a llama.cpp server on the runner and return server_id."""
        resp = httpx.post(
            f"{RUNNER_ENDPOINT}/v1/server/create",
            json={
                "model_id": model_id,
                "priority": 10,
            },
            timeout=180,  # Model loading can take a while
        )
        if resp.status_code == 507:
            pytest.skip(f"Runner returned 507 (insufficient capacity) for {model_id}")
        resp.raise_for_status()
        data = resp.json()
        return data["server_id"]

    def test_chat_completion_non_streaming(self):
        """Send a non-streaming chat completion and validate LLM response.

        Exercises: runner → llama.cpp subprocess → LLM → proxy response.
        """
        model_id = self._get_test_model_id()
        server_id = self._create_server(model_id)
        try:
            resp = httpx.post(
                f"{RUNNER_ENDPOINT}/v1/server/{server_id}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "user", "content": "What is 2+2? Answer with just the number."},
                    ],
                    "max_tokens": 32,
                    "temperature": 0.1,
                    "stream": False,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            assert "choices" in data, f"Missing 'choices' in response: {data}"
            assert len(data["choices"]) > 0, "No choices in response"

            choice = data["choices"][0]
            assert "message" in choice, f"Missing 'message' in choice: {choice}"
            assert choice["message"]["role"] == "assistant"

            content = choice["message"]["content"]
            assert content is not None, "Empty content from LLM"
            assert len(content) > 0, "Empty content from LLM"

            assert "4" in content or "four" in content.lower(), (
                f"Expected '4' or 'four' in response, got: {content}"
            )

            assert "usage" in data, "Missing 'usage' in response"
            usage = data["usage"]
            assert usage.get("prompt_tokens", 0) > 0, "Expected prompt_tokens > 0"
            assert usage.get("completion_tokens", 0) > 0, "Expected completion_tokens > 0"
        finally:
            self._cleanup(server_id)

    def test_chat_completion_streaming(self):
        """Send a streaming chat completion and validate SSE chunks.

        Exercises: runner → llama.cpp subprocess → SSE stream → proxy.
        """
        import json

        model_id = self._get_test_model_id()
        server_id = self._create_server(model_id)
        try:
            chunks = []
            with httpx.Client(timeout=60) as client:
                with client.stream(
                    "POST",
                    f"{RUNNER_ENDPOINT}/v1/server/{server_id}/v1/chat/completions",
                    json={
                        "model": model_id,
                        "messages": [
                            {"role": "user", "content": "What is 3+3? Answer with just the number."},
                        ],
                        "max_tokens": 16,
                        "temperature": 0.1,
                        "stream": True,
                    },
                ) as resp:
                    resp.raise_for_status()
                    assert "text/event-stream" in resp.headers.get(
                        "content-type", ""
                    )

                    # Collect SSE chunks (upstream may close after [DONE])
                    try:
                        for line in resp.iter_lines():
                            if line.startswith("data: "):
                                chunks.append(line[6:])
                    except httpx.RemoteProtocolError:
                        pass

            assert len(chunks) > 0, "No SSE chunks received"

            # Parse content from chunks
            content_parts = []
            finish_reason = None
            data_chunks = [c for c in chunks if c != "[DONE]"]
            for chunk_json in data_chunks:
                try:
                    chunk = json.loads(chunk_json)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    if chunk.get("choices", [{}])[0].get("finish_reason"):
                        finish_reason = chunk["choices"][0]["finish_reason"]
                except json.JSONDecodeError:
                    continue

            full_content = "".join(content_parts)
            assert len(full_content) > 0, "No content accumulated from stream"
            assert finish_reason == "stop", (
                f"Expected finish_reason 'stop', got: {finish_reason}"
            )
            assert "6" in full_content or "six" in full_content.lower(), (
                f"Expected '6' or 'six' in streamed response, got: {full_content}"
            )
        finally:
            self._cleanup(server_id)

    def test_chat_completion_with_system_prompt(self):
        """Chat completion with system + user messages."""
        model_id = self._get_test_model_id()
        server_id = self._create_server(model_id)
        try:
            resp = httpx.post(
                f"{RUNNER_ENDPOINT}/v1/server/{server_id}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that responds in JSON."},
                        {"role": "user", "content": 'Return {"answer": 42}'},
                    ],
                    "max_tokens": 64,
                    "temperature": 0.0,
                    "stream": False,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"]
            assert content is not None and len(content) > 0
            assert "42" in content, f"Expected '42' in response, got: {content}"
        finally:
            self._cleanup(server_id)


# ---------------------------------------------------------------------------
# Full completion flow: API CompletionService → Runner → llama.cpp
# Requires: TimescaleDB container (via testcontainers) + runner
# ---------------------------------------------------------------------------


class TestCompletionServiceChat:
    """Test the real CompletionService flow through to the runner.

    Uses the API's CompletionService.run_completion() which exercises:
    CompletionService → Composer → LangGraph → Agent → Pipeline → Runner

    Requires both a running llmmllab-runner AND a TimescaleDB container.
    """

    @pytest.fixture(autouse=True)
    def _chat_enabled(self):
        if os.environ.get("RUNNER_CHAT_TEST") not in ("1", "true"):
            pytest.skip("Set RUNNER_CHAT_TEST=1 to run chat completion tests")

    @pytest_asyncio.fixture
    async def _init_storage(self, engine):
        """Initialize the storage singleton with the test DB."""
        from db import storage
        from sqlalchemy.pool import NullPool  # noqa: F401

        conn_str = engine.url.render_as_string(hide_password=False).replace(
            "postgresql+asyncpg", "postgresql"
        )
        await storage.initialize(conn_str)
        yield
        try:
            await storage.close()
        except Exception:
            pass

    @pytest_asyncio.fixture
    async def _seed_user(self, session_factory):
        """Insert a test user required by FK constraints."""
        from sqlalchemy import text

        async with session_factory() as s:
            await s.execute(
                text("INSERT INTO users(id) VALUES (:uid) ON CONFLICT (id) DO NOTHING"),
                {"uid": "integration-test-user"},
            )
            await s.commit()

    async def _get_test_model_name(self) -> str:
        """Find a TextToText model name from the runner.

        Returns the model *name* (not ID), since CompletionService /
        IdeGraphBuilder match on `model.name`.
        """
        resp = httpx.get(
            f"{RUNNER_ENDPOINT}/v1/models",
            params={"task": "TextToText"},
            timeout=5,
        )
        resp.raise_for_status()
        models = resp.json()
        if not models:
            pytest.skip("No TextToText models available on runner")
        env_model = os.environ.get("RUNNER_CHAT_MODEL")
        if env_model:
            return env_model
        # runner's IdeGraphBuilder matches on model.name, so return the name
        return models[0]["name"]

    @pytest.mark.asyncio
    async def test_completion_service_run(
        self, _init_storage, _seed_user,  # noqa: ARG001
    ):
        """CompletionService.run_completion produces a valid LLM response.

        Exercises the full stack: CompletionService → Composer → LangGraph
        → Agent → RunnerClient → runner → llama.cpp → LLM response.
        """
        from unittest.mock import patch

        from models.message import (
            Message,
            MessageContent,
            MessageContentType,
            MessageRole,
        )
        from services import CompletionService

        model_name = await self._get_test_model_name()

        messages = [
            Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text="What is 5+5? Answer with just the number.",
                    )
                ],
            ),
        ]

        with patch("huggingface_hub.login"):
            result = await CompletionService.run_completion(
                user_id="integration-test-user",
                messages=messages,
                model_name=model_name,
            )

        assert result.chat_response is not None, (
            "CompletionService did not produce a response"
        )

        resp = result.chat_response
        assert resp.message is not None
        assert resp.message.content is not None

        text_parts = [
            c.text
            for c in resp.message.content
            if c.type == MessageContentType.TEXT and c.text
        ]
        content = "".join(text_parts)
        assert len(content) > 0, "Empty content from LLM"

        assert "10" in content or "ten" in content.lower(), (
            f"Expected '10' or 'ten' in response, got: {content}"
        )

    @pytest.mark.asyncio
    async def test_completion_service_stream(
        self, _init_storage, _seed_user,  # noqa: ARG001
    ):
        """CompletionService.stream_completion produces valid streamed events.

        Exercises streaming path: CompletionService.stream_completion →
        Composer → LangGraph → Agent → Runner → llama.cpp → SSE.
        """
        from unittest.mock import patch

        from models.message import (
            Message,
            MessageContent,
            MessageContentType,
            MessageRole,
        )
        from services import CompletionService

        model_name = await self._get_test_model_name()

        messages = [
            Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text="What is 7+7? Answer with just the number.",
                    )
                ],
            ),
        ]

        with patch("huggingface_hub.login"):
            events = []
            async for event, _ in CompletionService.stream_completion(
                user_id="integration-test-user",
                messages=messages,
                model_name=model_name,
            ):
                events.append(event)

        assert len(events) > 0, "No events from stream_completion"

        done_events = [e for e in events if e.done]
        assert len(done_events) > 0, "No done event in stream"

        final = done_events[-1]
        assert final.message is not None
        assert final.message.content is not None

        text_parts = [
            c.text
            for c in final.message.content
            if c.type == MessageContentType.TEXT and c.text
        ]
        content = "".join(text_parts)
        assert len(content) > 0, "Empty content from streamed LLM"

        assert "14" in content or "fourteen" in content.lower(), (
            f"Expected '14' or 'fourteen' in response, got: {content}"
        )
