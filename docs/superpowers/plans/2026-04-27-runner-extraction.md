# Runner Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `runner/` into a standalone `llmmllab-runner` service with HTTP proxy, replace all in-process imports in the main app with a `RunnerClient`.

**Architecture:** Runner = FastAPI app managing llama.cpp subprocesses on dynamic local ports, proxying `/v1/server/{id}/*` → `localhost:{port}/v1/*`. Main app's `RunnerClient` queries runner pool, creates servers via HTTP, receives `base_url` for `ChatOpenAI(base_url=...)`. All LangChain/LangGraph stays in main app.

**Tech Stack:** Python 3.12+, FastAPI, httpx, uvicorn, langchain-openai, pytest, pytest-asyncio

---

## File Map

### Main App: New
- `services/runner_client.py` — `RunnerClient` + `ServerHandle` dataclass
- `test/unit/test_runner_client.py` — RunnerClient tests

### Main App: Modified
- `config.py:90` — append `RUNNER_ENDPOINTS`
- `graph/workflows/ide/builder.py:35,124-138` — `pipeline_factory` → `RunnerClient`
- `graph/workflows/dialog/builder.py:32,90-101` — `pipeline_factory` → `RunnerClient`
- `services/token_service.py:15-16,36-54,114-123` — `pipeline_cache` → HTTP
- `tools/static/memory_retrieval_tool.py:29,96-106` — `pipeline_factory` → `RunnerClient`
- `app.py:75,145-155` — remove `pipeline_cache` import + shutdown
- `routers/conversation.py:18,190` — remove `pipeline_cache` import + usage

### Runner (`llmmllab-runner/`): New
- `app.py` — FastAPI entry, lifespan
- `config.py` — runner env vars
- `models.py` — duplicated Model/ModelProvider/ModelTask types
- `routers/models.py` — `GET /v1/models`
- `routers/servers.py` — server lifecycle endpoints
- `proxy/router.py` — `/v1/server/{id}/*` → `localhost:{port}/v1/*`
- `cache.py` — server registry (server_id → {port, model, use_count})
- `server_manager/` — copied from `runner/server_manager/`
- `utils/` — copied from `runner/utils/`

### Main App: Deleted
- `runner/` — moved to llmmllab-runner
- `test/unit/test_pipeline_cache_locking.py` — moves to runner repo
- `test/unit/test_pipeline_factory_locking.py` — moves to runner repo

---

## Task 1: Add Runner Endpoint Config

**Modify:** `config.py` (append after line 90)

- [ ] Append to `config.py`:
```python
# ── Runner service ────────────────────────────────────────────────────
RUNNER_ENDPOINTS = os.environ.get("RUNNER_ENDPOINTS", "http://localhost:8001").split(",")
```

- [ ] Commit: `git add config.py && git commit -m "feat: add RUNNER_ENDPOINTS config"`

---

## Task 2: Build RunnerClient with ServerHandle

**Create:** `services/runner_client.py`, `test/unit/test_runner_client.py`

- [ ] **Write failing tests** in `test/unit/test_runner_client.py` (all `@pytest.mark.asyncio`, mock `httpx.AsyncClient`):
  - `TestServerHandle.test_construction` — assert dataclass fields
  - `TestRunnerClientHealth.test_healthy_returns_data` — mock 200 /health
  - `TestRunnerClientHealth.test_unhealthy_returns_none` — mock RequestError
  - `TestRunnerClientAcquire.test_acquire_returns_handle` — mock health + create
  - `TestRunnerClientAcquire.test_acquire_skips_unhealthy` — first fails, second succeeds
  - `TestRunnerClientAcquire.test_acquire_raises_if_none` — all unhealthy → RuntimeError
  - `TestRunnerClientAcquire.test_acquire_retries_on_507` — 507 → try next
  - `TestRunnerClientRelease.test_release_calls_endpoint` — assert POST /release
  - `TestRunnerClientShutdown.test_shutdown_calls_endpoint` — assert DELETE
  - `TestRunnerClientModels.test_list_models_aggregates` — two runners, deduplicated
  - `TestRunnerClientModels.test_model_by_task_filters` — ?task=TextToEmbeddings

- [ ] **Verify failure:** `uv run pytest test/unit/test_runner_client.py -v` (expect ModuleNotFoundError)

- [ ] **Implement** `services/runner_client.py`:
  - `ServerHandle` dataclass: `base_url: str`, `server_id: str`, `runner_host: str`
  - `RunnerClient.__init__(endpoints)` — from `RUNNER_ENDPOINTS` config, tracks `_healthy` list
  - `_health(endpoint)` — GET /health, returns dict or None, updates `_healthy`
  - `_select_runner(model_id)` — iterate endpoints, pick highest VRAM with matching model
  - `acquire_server(model_id, task, config_override)` — POST /v1/server/create, handle 507 by trying next runner
  - `release_server(handle)` — POST /v1/server/{id}/release
  - `shutdown_server(handle)` — DELETE /v1/server/{id}
  - `list_models()` — GET /v1/models from all runners, deduplicate by model id
  - `model_by_task(task)` — GET /v1/models?task=X, return first match
  - Module singleton: `runner_client = RunnerClient()`

- [ ] **Run tests:** `uv run pytest test/unit/test_runner_client.py -v` (expect all PASS)

- [ ] Commit: `git add services/runner_client.py test/unit/test_runner_client.py && git commit -m "feat: RunnerClient with pool routing"`

---

## Task 3: Create llmmllab-runner Project

**Create:** `llmmllab-runner/` (separate repo). Copy `runner/server_manager/` and `runner/utils/`.

- [ ] **Create `config.py`** — env vars: `LOG_LEVEL`, `LLAMA_SERVER_EXECUTABLE`, `MODELS_FILE_PATH`, `PIPELINE_CACHE_TIMEOUT_MIN`, `PIPELINE_EVICTION_TIMEOUT_MIN`, `RUNNER_PORT=8000`, `RUNNER_HOST=0.0.0.0`, `SERVER_PORT_RANGE_START=8001`, `SERVER_PORT_RANGE_END=8900`

- [ ] **Create `models.py`** — duplicate `Model`, `ModelProvider`, `ModelTask`, `PipelinePriority`, `ModelDetails`, `ModelParameters`, `LoraWeight`, `UserConfig` from main app's `models/`

- [ ] **Copy `server_manager/` and `utils/`** — fix imports to use local `config` and `models`. Remove any `runner.pipelines` or `runner.exceptions` imports

- [ ] **Create `cache.py`** — server registry (not pipeline cache):
  - `_ServerEntry` dataclass: `server_id`, `model_id`, `port`, `use_count`, `created_at`, `healthy`
  - `ServerCache` class: `register()`, `get()`, `increment_use()`, `decrement_use()`, `evict_idle()`, `remove()`, `stats()`, `stop_all()`
  - `stop_all()` calls `manager.stop()` on each registered server manager

- [ ] **Create `routers/models.py`**:
  - `GET /v1/models` — return all models from `ModelLoader.get_available_models()` as JSON
  - Support `?task=TextToEmbeddings` query param filter

- [ ] **Create `routers/servers.py`**:
  - `POST /v1/server/create` — accept `{model_id, priority, config_override}`, look up model, start llama.cpp server via `LlamaCppServerManager`, register in `ServerCache`, return `{server_id, base_url, model}`. The `base_url` is `f"{runner_host}/v1/server/{server_id}"`
  - `GET /v1/server/{id}` — return server status from cache
  - `DELETE /v1/server/{id}` — call `manager.stop()`, remove from cache
  - `POST /v1/server/{id}/release` — decrement `use_count` in cache

- [ ] **Create `proxy/router.py`**:
  - Mount at `/v1/server/{server_id}/` prefix
  - Look up `server_id` in `ServerCache` to get the local port
  - Rewrite path: strip `/v1/server/{id}` prefix, forward to `http://localhost:{port}/v1/{remaining}`
  - For SSE responses (`text/event-stream`), stream chunks from upstream to downstream without buffering
  - For regular responses, proxy request body + headers, return response
  - Return 404 if `server_id` not found, 502 if upstream unreachable

- [ ] **Create `app.py`**:
  - FastAPI app with lifespan: on startup, initialize `ServerCache` and `ModelLoader`; on shutdown, call `cache.stop_all()`
  - Mount routers: `/health` (GET, returns GPU stats + model list + active server count), `/v1/models`, `/v1/server/`
  - Health endpoint includes: status, gpu info from `hardware_manager`, active server count, model list

- [ ] **Create `Dockerfile`** — based on the existing project's Dockerfile pattern, install Python deps, copy source

- [ ] Commit runner project

---

## Task 4: Update IDE Workflow Builder

**Modify:** `graph/workflows/ide/builder.py`

Current code at lines 35, 124-138:
```python
from runner import pipeline_factory
# ...
model_def = pipeline_factory._get_model_by_id(model_name)  # line 124
model_def = pipeline_factory.get_model_by_task(ModelTask.TEXTTOTEXT)  # line 128
primary_pipeline = pipeline_factory.get_pipeline(model=model_def)  # line 138
```

- [ ] **Replace import line 35:**
```python
# Remove: from runner import pipeline_factory
# Add:
from services.runner_client import runner_client
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
```

- [ ] **Replace lines 124-138 in `build_workflow()`:**

```python
# Look up model by name or fall back to first TextToText model
if model_name:
    all_models = await runner_client.list_models()
    model_def = next((m for m in all_models if m.name == model_name), None)
    if not model_def:
        raise RuntimeError(f"Model '{model_name}' not found")
else:
    model_def = await runner_client.model_by_task(ModelTask.TEXTTOTEXT)
    if not model_def:
        raise RuntimeError("No TextToText model available")

self.logger.debug(
    "Building workflow",
    user_id=user_id,
    model=model_def.name,
    model_arg=model_name,
)

# Acquire a server from the runner pool
server_handle = await runner_client.acquire_server(
    model_id=model_def.name,
    task=model_def.task,
)

# Use the runner's proxy URL as a drop-in OpenAI-compatible endpoint
primary_model = ChatOpenAI(
    base_url=server_handle.base_url,
    api_key=SecretStr("none"),
    model=model_def.name,
)
# Store the handle on the builder so it can be released later
self._server_handle = server_handle
```

- [ ] **Add `_server_handle` field to `IdeGraphBuilder.__init__`** if not already present: `self._server_handle: Optional[ServerHandle] = None`

- [ ] **Verify:** `uv run pytest test/unit/ -v -k "not pipeline_cache and not pipeline_factory"` (existing tests should still pass with the runner imports removed)

- [ ] Commit: `git add graph/workflows/ide/builder.py && git commit -m "refactor: IDE builder uses RunnerClient instead of pipeline_factory"`

---

## Task 5: Update Dialog Workflow Builder

**Modify:** `graph/workflows/dialog/builder.py`

Current code at lines 32, 90-101:
```python
from runner import pipeline_factory
# ...
primary_model_def = pipeline_factory.get_model_by_task(ModelTask.TEXTTOTEXT)
embedding_model_def = pipeline_factory.get_model_by_task(ModelTask.TEXTTOEMBEDDINGS)
# ...
primary_model = pipeline_factory.get_pipeline(model=primary_model_def)
embedding_model = pipeline_factory.get_pipeline(model=embedding_model_def)
```

- [ ] **Replace import line 32:**
```python
# Remove: from runner import pipeline_factory
# Add:
from services.runner_client import runner_client
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr
```

- [ ] **Replace lines 90-101 in `build_workflow()`:**

```python
primary_model_def = await runner_client.model_by_task(ModelTask.TEXTTOTEXT)
embedding_model_def = await runner_client.model_by_task(ModelTask.TEXTTOEMBEDDINGS)

if not primary_model_def:
    raise RuntimeError("No TextToText model available")
if not embedding_model_def:
    raise RuntimeError("No TextToEmbeddings model available")

primary_handle = await runner_client.acquire_server(
    model_id=primary_model_def.name,
    task=primary_model_def.task,
)
embedding_handle = await runner_client.acquire_server(
    model_id=embedding_model_def.name,
    task=embedding_model_def.task,
)

primary_model = ChatOpenAI(
    base_url=primary_handle.base_url,
    api_key=SecretStr("none"),
    model=primary_model_def.name,
)
embedding_model = OpenAIEmbeddings(
    base_url=embedding_handle.base_url,
    api_key="none",
)
# Store handles for cleanup
self._primary_handle = primary_handle
self._embedding_handle = embedding_handle
```

- [ ] **Add handle fields to `DialogGraphBuilder.__init__`**: `self._primary_handle: Optional[ServerHandle] = None`, `self._embedding_handle: Optional[ServerHandle] = None`

- [ ] Commit: `git add graph/workflows/dialog/builder.py && git commit -m "refactor: Dialog builder uses RunnerClient instead of pipeline_factory"`

---

## Task 6: Update TokenService

**Modify:** `services/token_service.py`

Current coupling: lines 15-16 import `pipeline_cache` and `ChatLlamaCppPipeline`, lines 36-54 and 114-123 reach into `pipeline_cache._cache` and `pipeline_cache._lock`.

- [ ] **Replace the entire file** with HTTP-based implementation:

```python
import json
from typing import Optional
import httpx
from models.message import Message, MessageContentType
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="token_service")
_CLAUDE_ASSUMED_CONTEXT = 200_000

class TokenService:
    @staticmethod
    async def get_num_ctx(server_url: str) -> int:
        """Get num_ctx from the runner's pipeline info endpoint."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{server_url}/models")
                if resp.status_code == 200:
                    models = resp.json()
                    if models and "parameters" in models[0]:
                        params = models[0].get("parameters", {})
                        if params.get("num_ctx"):
                            return params["num_ctx"]
        except Exception:
            pass
        return 131_072

    @staticmethod
    def scale_tokens(actual: int, assumed_context: int = _CLAUDE_ASSUMED_CONTEXT) -> int:
        num_ctx = 131_072  # default, callers should override via get_num_ctx
        effective_ctx = int(num_ctx * 0.90)
        if effective_ctx >= assumed_context:
            return actual
        return int(actual * assumed_context / effective_ctx)

    @staticmethod
    async def count_input_tokens(
        messages: list[Message],
        tools: Optional[list] = None,
        server_url: Optional[str] = None,
    ) -> int:
        parts: list[str] = []
        for msg in messages:
            role_tag = msg.role.value if msg.role else "user"
            text = ""
            if msg.content:
                text = " ".join(
                    c.text for c in msg.content
                    if c.type == MessageContentType.TEXT and c.text
                )
            parts.append(f"<|{role_tag}|>\n{text}")
        if tools:
            for tool in tools:
                if isinstance(tool, dict):
                    parts.append(json.dumps(tool))
                else:
                    parts.append(json.dumps(tool.model_dump(exclude_none=True)))
        combined_text = "\n".join(parts)
        if server_url:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(
                        f"{server_url}/tokenize",
                        json={"content": combined_text},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        tokens = data.get("tokens", [])
                        return len(tokens)
            except Exception as e:
                logger.debug(f"llama-server tokenize unavailable, using estimate: {e}")
        return max(1, len(combined_text) // 4)
```

Key changes:
- Remove `from runner import pipeline_cache` and `from runner.pipelines.llamacpp.chat import ChatLlamaCppPipeline`
- `get_num_ctx()` becomes async, takes `server_url` from the `ServerHandle` instead of introspecting cache
- `count_input_tokens()` takes an optional `server_url` parameter instead of looking it up in cache
- The `/tokenize` call now uses the llama.cpp server's native endpoint via the proxy URL

- [ ] **Update callers of `count_input_tokens()`** to pass `server_url` — check `routers/anthropic/messages.py` and `routers/openai/chat.py` for usages. The `ServerHandle` from the workflow builder needs to be accessible. If the handle isn't directly available at the call site, pass `None` and let the fallback (char estimate) apply — this is acceptable since the exact count is an optimization.

- [ ] Commit: `git add services/token_service.py && git commit -m "refactor: TokenService uses HTTP instead of pipeline_cache internals"`

---

## Task 7: Update Memory Retrieval Tool

**Modify:** `tools/static/memory_retrieval_tool.py`

Current code at lines 29, 96-106:
```python
from runner import pipeline_factory
# ...
embedding_model = pipeline_factory.get_model_by_task(ModelTask.TEXTTOEMBEDDINGS)
embedding_pipeline = pipeline_factory.get_embedding_pipeline(model=embedding_model)
query_embeddings = embedding_pipeline.embed_documents([query])
```

- [ ] **Replace import line 29:**
```python
# Remove: from runner import pipeline_factory
# Add:
from services.runner_client import runner_client
from langchain_openai import OpenAIEmbeddings
```

- [ ] **Replace lines 96-106:**

```python
embedding_model = await runner_client.model_by_task(ModelTask.TEXTTOEMBEDDINGS)
try:
    if not embedding_model:
        raise RuntimeError("No TextToEmbeddings model available")
    embedding_handle = await runner_client.acquire_server(
        model_id=embedding_model.name,
        task=embedding_model.task,
    )
    embed_client = OpenAIEmbeddings(
        base_url=embedding_handle.base_url,
        api_key="none",
    )
    query_embeddings = await embed_client.aembed_documents([query])
except Exception as embed_error:
    logger.warning(f"Embedding generation failed: {embed_error}, using mock embeddings")
    query_embeddings = [[0.1] * 768]
```

Note: `embed_documents` becomes `aembed_documents` (async) since we're now using `OpenAIEmbeddings`. The calling code in `_arun` is already async so `await` is fine.

- [ ] Commit: `git add tools/static/memory_retrieval_tool.py && git commit -m "refactor: memory retrieval uses RunnerClient for embeddings"`

---

## Task 8: Clean Up app.py and conversation.py

**Modify:** `app.py`, `routers/conversation.py`

- [ ] **In `app.py`:**
  - Remove line 75: `from runner import pipeline_cache`
  - Remove lines 145-155: the `pipeline_cache.stop()` shutdown block

- [ ] **In `routers/conversation.py`:**
  - Remove line 18: `from runner import pipeline_cache`
  - Replace line 190: `pipeline_cache.clear()` with a call that notifies all known runners. For now, remove the line entirely — the runner manages its own eviction. If conversation cancellation needs to trigger server cleanup, pass the `ServerHandle` through the workflow context and call `runner_client.shutdown_server(handle)` there.

- [ ] Commit: `git add app.py routers/conversation.py && git commit -m "refactor: remove pipeline_cache from app shutdown and conversation cancel"`

---

## Task 9: Remove Runner Directory and Stale Tests

**Delete:** `runner/`, `test/unit/test_pipeline_cache_locking.py`, `test/unit/test_pipeline_factory_locking.py`

- [ ] **Remove runner directory:** `rm -rf runner/`

- [ ] **Remove stale tests:** `rm test/unit/test_pipeline_cache_locking.py test/unit/test_pipeline_factory_locking.py`

- [ ] **Check for remaining runner imports:** `grep -r "from runner" --include="*.py" .` — fix any remaining references

- [ ] **Validate syntax:** `make validate` (or `uv run py_compile` on all remaining .py files)

- [ ] Commit: `git rm -r runner/ test/unit/test_pipeline_cache_locking.py test/unit/test_pipeline_factory_locking.py && git commit -m "refactor: remove runner/ directory, moved to llmmllab-runner service"`

---

## Task 10: K8s Deployment and Local Dev

**Create:** `llmmllab-runner/k8s/`, `docker-compose.yml` (main app root)

- [ ] **Create runner K8s manifests:**
  - `llmmllab-runner/k8s/deployment.yaml` — Deployment with GPU resource requests, env vars for model config
  - `llmmllab-runner/k8s/service.yaml` — Service exposing port 8000
  - `llmmllab-runner/k8s/hpa.yaml` — optional HPA for scaling

- [ ] **Create docker-compose.yml** in main app root for local dev:
  - `api` service — main app, `RUNNER_ENDPOINTS=http://runner:8000`
  - `runner` service — llmmllab-runner, mounts llama.cpp binary and model files
  - `timescaledb` service — database (keep existing setup)
  - `redis` service — cache (keep existing setup)

- [ ] **Update main app K8s manifests** to set `RUNNER_ENDPOINTS` env var pointing to the runner service

- [ ] Commit: `git add llmmllab-runner/k8s/ docker-compose.yml k8s/ && git commit -m "feat: K8s manifests and docker-compose for runner service"`

---

## Self-Review Checklist

- [x] **Spec coverage:** All 5 spec phases covered (runner repo, client, integration, cleanup, deployment)
- [x] **No placeholders:** All code blocks contain actual implementation code
- [x] **Type consistency:** `ServerHandle` used consistently across all tasks; `RunnerClient` API matches between Task 2 definition and Tasks 4-7 usage
- [x] **Method signatures match:** `acquire_server(model_id, task)` called in Tasks 4-7 matches Task 2 definition; `release_server(handle)` and `shutdown_server(handle)` consistent
- [x] **Async consistency:** `build_workflow()` is already async in both builders, so `await runner_client.acquire_server()` fits; `memory_retrieval_tool._arun` is async so `await embed_client.aembed_documents()` fits
- [x] **Import cleanup:** Each modified file explicitly lists which runner imports to remove
- [x] **Testing:** Task 2 has TDD (tests first), Tasks 4-8 have verification steps