# Runner Extraction Plan

**Date:** 2026-04-27
**Goal:** Extract the `runner/` component into a standalone `llmmllab-runner` service, enabling multiple instances with different hardware and configuration.

## Assessment

**Feasibility:** High — the `runner/` directory is a coherent unit with clear internal structure.

**Complexity:** Medium-High — not because of code volume, but because:

1. The runner provides synchronous and streaming responses that the API layer depends on for SSE. Replacing in-process calls with HTTP while preserving stream fidelity is non-trivial.
2. `token_service` and `pipeline_cache` have deep coupling (reaching into private `_cache` and `_lock` internals).
3. There are broken imports (Flux, Qwen3 VL) that need fixing regardless.
4. Multi-instance routing adds a load-balancer/discovery layer that doesn't exist today.

---

## Phase 1: Clean Up Internal Runner Boundaries

**Goal:** Make the runner self-contained, remove broken code, and define a clear internal API.

1. **Fix broken imports** — `BasePipelineCore` (Flux pipelines) and `BaseLlamaCppPipeline` (Qwen3 VL) don't exist. Either restore the missing base classes or fix the imports to use what exists (`BasePipeline`).
2. **Decouple from shared config** — The runner reads `config.LOG_LEVEL`, `config.LLAMA_SERVER_EXECUTABLE`, `config.PIPELINE_CACHE_TIMEOUT_MIN`, `config.OPENAI_API_KEY`, etc. Extract these into a `runner/config.py` that the main app populates via env vars or a startup call. This creates a clean boundary.
3. **Decouple from shared models** — The runner imports `Model`, `ModelProvider`, `ModelTask`, etc. from the top-level `models/` package. These Pydantic models should either move into the runner or be defined in a shared types package. Recommend: keep them in `models/` but treat it as the contract — the runner owns the model definitions.
4. **Decouple from shared logging** — `utils.logging.llmmllogger` is used in every runner file. Either move the logger into runner or accept it as a dependency injected at startup.
5. **Remove singleton exports** — Replace module-level `pipeline_factory` and `pipeline_cache` singletons with instances that are created by the runner service on startup. The main app should import a `runner_client` instead.

**Estimated effort:** 2-3 hours

---

## Phase 2: Design the Runner Service API

**Goal:** Define the HTTP interface that replaces direct Python imports.

The runner service needs to expose:

| Current call | New API endpoint | Notes |
|---|---|---|
| `pipeline_factory.get_pipeline(model, priority, grammar, metadata)` | `POST /v1/pipelines/chat` | Returns a session ID, streams via SSE |
| `pipeline_factory.get_embedding_pipeline(model)` | `POST /v1/pipelines/embed` | Sync request/response |
| `pipeline_factory.unlock_pipeline(model)` | `POST /v1/pipelines/{id}/release` | Release cache lock |
| `pipeline_factory.clear_cache(model)` | `DELETE /v1/pipelines/{id}` | Evict from cache |
| `pipeline_factory.get_model_by_task(task)` | `GET /v1/models?task=TextToEmbeddings` | Model discovery |
| `pipeline_cache.stats()` | `GET /v1/health` | Include cache + GPU stats |
| `ModelLoader.get_available_models()` | `GET /v1/models` | List all models |
| (implicit stream) | SSE stream on chat session | The hard part — see below |

### Critical design decision: Streaming

Currently, the workflow calls `_stream()` on a pipeline, which yields LangChain events. These flow through LangGraph → Composer → CompletionService → SSE to the client. This is an in-process generator chain.

With a remote runner, we need:

- **Option A (SSE proxy):** The runner exposes an SSE endpoint. The API layer proxies the SSE chunks through transparently. Simplest, but the runner becomes a long-lived HTTP connection broker.
- **Option B (gRPC streaming):** Use gRPC bidirectional streaming for the inference path. Lower latency, stronger typing, but adds operational complexity.
- **Option C (Hybrid):** The runner exposes both a sync endpoint (for embeddings, status) and an SSE endpoint (for chat). The API layer acts as a transparent SSE proxy.

**Recommendation: Option A (SSE)** — it's the simplest and matches what the outer API already speaks. The runner becomes: accept a request, spin up/cache a pipeline, stream tokens back as SSE. The API layer proxies those SSE chunks through to the end client with minimal transformation.

**Estimated effort:** 1-2 hours (design only)

---

## Phase 3: Build the Runner Service

**Goal:** A standalone FastAPI app that implements the API from Phase 2.

1. Create `llmmllab-runner/` as a new project (can live as a subdirectory or separate repo).
2. Move `runner/pipelines/`, `runner/server_manager/`, `runner/utils/` into the new project.
3. Build a minimal FastAPI app with:
   - Model management endpoints (list, reload)
   - Chat endpoint (accepts messages, streams back SSE)
   - Embedding endpoint (sync)
   - Image generation endpoint (sync/async with polling)
   - Health endpoint (GPU stats, cache stats)
   - Pipeline lifecycle (release, evict)
4. Implement the SSE streaming layer. This wraps the existing LangChain `_stream()` calls and yields JSON-formatted chunks.
5. Handle graceful shutdown (stop all servers, clear cache, stop background threads).

**Estimated effort:** 4-6 hours

---

## Phase 4: Build the Runner Client in the Main App

**Goal:** Replace direct imports of `pipeline_factory` and `pipeline_cache` with an HTTP client.

1. Create `services/runner_client.py` — an async HTTP client that speaks to one or more runner instances.
2. Implement the chat path:
   - Send request to runner via HTTP POST
   - Proxy the SSE stream back through to the workflow/composer
   - Handle runner failures/timeouts with retry/failover
3. Implement the embedding path: simple HTTP POST/GET.
4. Replace usages:
   - `graph/workflows/dialog/builder.py` — use client instead of `pipeline_factory`
   - `graph/workflows/ide/builder.py` — same
   - `tools/static/memory_retrieval_tool.py` — use client for embedding requests
   - `services/token_service.py` — use client for cache introspection (this is the deepest coupling to untangle)
   - `app.py` — remove `pipeline_cache.stop()` on shutdown
   - `routers/conversation.py` — use client for cache operations
5. Remove the `runner/` directory from the main app (or keep a stub for local dev mode).

**Estimated effort:** 4-6 hours

---

## Phase 5: Multi-Instance Routing

**Goal:** Route requests across multiple runner instances based on hardware/capability.

1. **Runner registration:** Each runner instance registers with the main app (or a discovery service) with its capabilities:
   - Available GPUs (VRAM, model)
   - Loaded models
   - Current load (active sessions, cache usage)
2. **Routing logic:** The `runner_client` becomes a pool client:
   - Route by model capability (e.g., only runners with 24GB+ VRAM get Flux requests)
   - Route by task type (embedding runners vs chat runners)
   - Round-robin or least-connections within a capability tier
3. **Health checking:** Periodic health checks remove unhealthy instances from the pool.
4. **Failover:** If a runner mid-stream dies, the client detects the broken connection and can either fail the request or retry on another instance (tokens will be duplicated, which is acceptable for chat).

**Estimated effort:** 3-4 hours

---

## Summary

| Phase | Description | Effort | Risk |
|---|---|---|---|
| 1 | Clean internal boundaries, fix broken code | 2-3h | Low |
| 2 | Design runner service API | 1-2h | Low |
| 3 | Build standalone runner service | 4-6h | Medium (streaming) |
| 4 | Build client, replace imports | 4-6h | Medium (token_service coupling) |
| 5 | Multi-instance routing | 3-4h | Low-Medium |
| **Total** | | **~14-21h** | |

## Key Risks

1. **Streaming latency:** Adding a network hop between API and runner adds latency to each token. SSE over localhost/Docker network minimizes this (~0.5-2ms), but it's measurable.
2. **Token service coupling:** `services/token_service.py` reaches into `_cache` private dicts. This needs a proper endpoint on the runner or a redesign.
3. **State in the workflow:** The current workflow passes pipeline objects as LangChain nodes. With a remote runner, the "pipeline" becomes an HTTP session, which changes the LangGraph node signatures.
4. **Broken pipelines:** Flux and Qwen3 VL pipelines are currently broken. Fix or remove before extraction.

## Recommendation

Start with Phase 1 immediately (fix broken imports, decouple config). Then decide whether to go through all phases or take a hybrid approach: keep the runner as a local import for now but with clean boundaries, and extract to a separate process only when multi-instance routing is actually needed. The boundary work from Phase 1 makes the later extraction nearly free.
