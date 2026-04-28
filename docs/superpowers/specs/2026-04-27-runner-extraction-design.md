# Runner Extraction Design

**Date:** 2026-04-27
**Status:** Approved

Extract the `runner/` component into a standalone `llmmllab-runner` service. The runner manages llama.cpp server lifecycle (start, stop, evict, VRAM accounting). The main app talks to runners over HTTP, receiving OpenAI-compatible proxy endpoints that drop in as `base_url` for `ChatOpenAI`.

All LangChain/LangGraph orchestration stays in the main API app. The runner knows nothing about agents, workflows, or messages.

---

## Architecture

```
Main API App
┌──────────────────────────────────────────────────┐
│ Routers → CompletionService → Composer → LangGraph│
│                      │                           │
│              RunnerClient (HTTP + pool routing)   │
└──────────────────────┼───────────────────────────┘
                       │ HTTP
     ┌─────────────────┼─────────────────┐
     │     Runner Pool (service discovery)                      │
     └───────┬─────────┬─────────┬───────┘
             │         │         │
    ┌────────▼───┐ ┌───▼────┐ ┌─▼────────┐
    │ Runner #1  │ │Runner #2│ │Runner #3 │
    │ RTX 4090   │ │RTX 4060 │ │CPU only  │
    │            │ │        │ │          │
    │ proxy:8000 │ │:8000   │ │:8000     │
    │ ──►llama.cpp│ │─►llama │ │─►llama   │
    │ :8123      │ │:8124   │ │:8125     │
    └────────────┘ └────────┘ └──────────┘
```

Each runner is a standalone FastAPI app. It manages llama.cpp server subprocesses on dynamic local ports and proxies requests through its own stable address. The proxy rewrites `/v1/server/{id}/*` to `localhost:{port}/v1/*`.

---

## Runner Service API

Five endpoints. The runner knows only about server lifecycle and model availability.

### `GET /health`

GPU stats, VRAM, active server count. Used for health checking and routing decisions.

```json
{
  "status": "ok",
  "gpu": {"model": "RTX 4090", "total_vram_bytes": 25769803776, "available_vram_bytes": 12884901888},
  "active_servers": 1,
  "models": ["llama-3-8b", "nomic-embed"]
}
```

### `GET /v1/models`

List available models from the runner's model config file. Supports `?task=TextToEmbeddings` filter.

### `POST /v1/server/create`

Spin up a llama.cpp server with a given model. Returns a proxy URL that the API app uses as a `base_url`.

**Request:**
```json
{"model_id": "llama-3-8b", "priority": "normal", "config_override": {}}
```

**Response:**
```json
{
  "server_id": "abc123",
  "base_url": "http://runner-1:8000/v1/server/abc123",
  "model": "llama-3-8b"
}
```

### `GET /v1/server/{id}`

Server status: running, health, model, uptime.

### `DELETE /v1/server/{id}`

Hard shutdown — stop the llama.cpp subprocess immediately.

### `POST /v1/server/{id}/release`

Release the cache lock. Server stays running but becomes eligible for eviction.

---

## Proxy Layer

The runner proxies llama.cpp traffic through its own address. Internal mapping:

```
/v1/server/{id}/chat/completions  →  localhost:{port}/v1/chat/completions
/v1/server/{id}/embeddings        →  localhost:{port}/v1/embeddings
/v1/server/{id}/models            →  localhost:{port}/v1/models
/v1/server/{id}/tokenize          →  localhost:{port}/tokenize
```

The `{id}` segment tells the proxy which local llama.cpp server to forward to. SSE streams (chat completions) pass through transparently — the proxy streams chunks from llama.cpp to the client without buffering.

The API app receives `base_url: "http://runner-1:8000/v1/server/abc123"` and uses it as a drop-in replacement:

```python
ChatOpenAI(
    base_url="http://runner-1:8000/v1/server/abc123",
    api_key="none",
    model="llama-3-8b",
)
```

No translation layer needed. The llama.cpp server's native OpenAI-compatible protocol flows through unchanged.

---

## Runner Client (Main App)

A new `services/runner_client.py` replaces imports of `pipeline_factory` and `pipeline_cache`.

```python
class RunnerClient:
    async def acquire_server(self, model_id: str, task: ModelTask) -> ServerHandle:
        """Route to a runner with capacity, spin up a server, return its URL."""

    async def release_server(self, handle: ServerHandle) -> None:
        """Release the lock on a server."""

    async def shutdown_server(self, handle: ServerHandle) -> None:
        """Hard shutdown."""

    async def list_models(self) -> list[Model]:
        """Aggregate models across all runners."""

    async def model_by_task(self, task: ModelTask) -> Optional[Model]:
        """Find a model matching a task type."""
```

`ServerHandle` is a simple dataclass: `{base_url: str, server_id: str, runner_host: str}`.

---

## Multi-Instance Routing

The `RunnerClient` maintains a pool of runner endpoints. Configuration via `RUNNER_ENDPOINTS` env var (comma-separated URLs) or Kubernetes service discovery.

**Routing logic:**
1. Query each runner's `/health` for available models + VRAM
2. Filter to runners that have the requested model and sufficient resources
3. Pick the runner with the most available VRAM (least loaded)
4. `POST /v1/server/create` on the selected runner
5. Return the `ServerHandle` with the proxy URL

**Health checking:** Periodic `/health` pings. Unhealthy runners removed from pool until recovery.

---

## Integration Changes in Main App

| File | Change |
|------|--------|
| `graph/workflows/ide/builder.py` | Replace `pipeline_factory.get_pipeline(model)` with `runner_client.acquire_server()` → wrap in `ChatOpenAI` |
| `graph/workflows/dialog/builder.py` | Same |
| `services/token_service.py` | Replace `pipeline_cache._cache` access with HTTP call to `{base_url}/tokenize` |
| `tools/static/memory_retrieval_tool.py` | Replace `pipeline_factory.get_embedding_pipeline()` with `runner_client.acquire_server()` → wrap in `OpenAIEmbeddings` |
| `app.py` | Remove `pipeline_cache.stop()` on shutdown |
| `routers/conversation.py` | Replace `pipeline_cache` calls with `runner_client` calls |
| `runner/` | Removed (moved to `llmmllab-runner` repo) |

---

## Error Handling

- **Runner dies mid-stream:** `ChatOpenAI` raises connection error. LangGraph catches it. Client marks runner unhealthy, request fails. Retry spawns a server on a healthy runner.
- **VRAM exhaustion:** Runner returns 507. Client tries another runner. All fail → 503 to client.
- **Server health:** Runner polls each llama.cpp `/health`. Dead servers auto-cleaned.
- **Graceful shutdown:** Runner drains active servers on SIGTERM, then SIGTERM → SIGKILL after 10s.
- **Model not found:** Runner returns 404. Client tries other runners.

---

## Testing

- **Runner unit tests:** Server lifecycle (start/stop/evict), VRAM accounting, proxy correctness, model loading
- **Runner integration tests:** Real llama.cpp server, verify SSE proxy passthrough, verify eviction
- **Main app tests:** `RunnerClient` against mocked runner HTTP server, verify routing, verify `ChatOpenAI` gets correct `base_url`
- **Token service tests:** Verify `/tokenize` endpoint usage instead of cache internals

---

## Project Structure

```
llmmllab-runner/          # Separate repo
  app.py                  # FastAPI entry point
  config.py               # Runner-specific config
  models.py               # Model/ModelProvider/ModelTask (duplicated from API)
  server_manager/         # llama.cpp subprocess lifecycle
  proxy/                  # HTTP proxy: /v1/server/{id}/* → localhost:{port}/v1/*
  utils/                  # hardware_manager, model_loader
  Dockerfile
  k8s/

llmmllab-api/             # Current repo
  services/runner_client.py   # NEW - HTTP client + pool routing
  # runner/ removed
  # graph/workflows/*/builder.py updated
  # services/token_service.py updated
  # tools/static/memory_retrieval_tool.py updated
```

---

## Phases

| Phase | Description | Effort |
|---|---|---|
| 1 | Create `llmmllab-runner` repo: FastAPI app, server_manager, proxy layer, config | 4-6h |
| 2 | Build `RunnerClient` in main app with pool routing | 3-4h |
| 3 | Update main app integration points (builders, token_service, tools, app.py) | 3-4h |
| 4 | Remove `runner/` from main app, clean up imports | 1-2h |
| 5 | K8s deployment: runner manifests, service discovery, docker-compose for local dev | 2-3h |
| **Total** | | **~13-19h** |
