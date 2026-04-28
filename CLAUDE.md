# CLAUDE.md

This file provides guidance for working with the llmmllab-api codebase.

## Commands

```bash
make start           # Start API server (uvicorn + reload)
make test            # Run all tests
make validate        # Python syntax check
make clean           # Remove build artifacts
make docker-build    # Build Docker image
make deploy          # Build, push, and apply k8s manifests
make sync-watch      # Watch mode: sync code to k8s on changes
```

## Architecture

### Structure

```
llmmllab-api/
  app.py             FastAPI entry point
  config.py          Environment-based configuration
  composer_init.py   Composer public API (workflow orchestration)
  routers/           API routes: openai/, anthropic/, common/
  middleware/        Auth, DB init, message validation
  services/          Business logic: completion, token, tool
  runner/            Model execution: pipeline_factory, pipeline_cache, pipelines/
  agents/            Agent implementations (chat, embed)
  core/              Core composer components (service, errors)
  graph/             LangGraph workflow builder, executor, state, nodes
  tools/             Tool registry and static tools
  db/                Multi-tier storage (PostgreSQL + Redis)
  models/            Pydantic data models (edit directly)
  utils/             Shared helpers
  k8s/               Kubernetes deployment manifests
  test/              Unit and integration tests
```

### Key Entry Points

| Component | Entry Point |
|-----------|-------------|
| FastAPI app | `app.py` |
| OpenAI chat | `routers/openai/chat.py` |
| Anthropic messages | `routers/anthropic/messages.py` |
| Pipeline creation | `runner/pipeline_factory.py` |
| Composer API | `composer_init.py` |
| Workflow builder | `graph/workflows/ide/builder.py` |

### Key Patterns

**Pipeline System**: `runner/pipeline_factory.py` creates pipelines (text, image, embeddings, multimodal). `runner/pipeline_cache.py` manages instances with memory-based eviction. All implementations in `runner/pipelines/`.

**Provider Compatibility**: OpenAI-compatible (`routers/openai/`) and Anthropic-compatible (`routers/anthropic/`) endpoints sharing the same runner/pipeline infrastructure.

**Streaming**: Chat and image responses stream token-by-token.

**Composer/LangGraph**: `composer_init.py` exposes the workflow orchestration API. `graph/workflows/` contains IDE and Dialog workflow builders. `graph/nodes/` contains LangGraph nodes for agent execution, tool calling, memory, and web search.

**Multi-Tier Caching**: User config flows memory → Redis → PostgreSQL. Configured via `db/multi_tier_cache.py`.

### Configuration

- Python: `pyproject.toml` (managed via uv)
- Pytest: `pytest.ini` (asyncio_mode: auto, testpaths: test/unit)
- Docker: `Dockerfile` (CUDA 12.8 runtime, llama.cpp compiled from source)
- Kubernetes: `k8s/deployment.yaml`, `k8s/service.yaml`
