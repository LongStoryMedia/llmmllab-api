# Multi-Runner Model Mapping Design

**Date**: 2026-05-01
**Status**: Approved

## Problem

The API currently interacts with multiple runner instances via `RUNNER_ENDPOINTS` (comma-separated env var). However, when acquiring a server for a specific model, the `_select_runner()` method picks the runner with the **most VRAM** and checks if the model appears in that runner's `/health` endpoint. This VRAM-first heuristic is suboptimal:

1. A model that exists **only** on `runner-small` will fail the VRAM-based selection, requiring a fallback loop through all endpoints before finding the right runner.
2. Every `acquire_server()` call triggers health checks on all runners, adding unnecessary latency.
3. The model-to-runner relationship is implicit (discovered per-request) rather than explicit and cached.

With `runner-small` now deployed alongside the main runner, models need to be routed to the correct instance efficiently.

## Current Flow

```
acquire_server(model_id)
  -> _select_runner(model_id)
    -> health check each endpoint
    -> pick runner with most VRAM that has model_id in /health["models"]
  -> POST {best_runner}/v1/server/create
  -> if 507, try next endpoint in list
```

## Design: Cached Model-to-Runner Map with Sliding Refresh

### Core Data Structure

A `Dict[str, List[str]]` mapping `model_id → [runner_endpoints]` is maintained in `RunnerClient`:

```python
self._model_map: Dict[str, List[str]] = {}
# Example:
# {
#   "Qwen3_6_27B": ["http://runner-main:8000"],
#   "nomic_xlm": ["http://runner-main:8000", "http://runner-small:8000"],
#   "Qwen3_5_0_8B": ["http://runner-small:8000"],
# }
```

### Building the Map

`refresh_model_map()` queries `GET /v1/models` on all endpoints concurrently (same pattern as `list_models()`). For each runner, it records which models are available. The endpoint order in each model's list reflects the order endpoints were processed, preserving a deterministic fallback order.

### Sliding Window Refresh

- Configured by `MODEL_CACHE_REFRESH_SEC` env var (default: `60`).
- On startup: initial refresh.
- Each successful `acquire_server()` call schedules a refresh `MODEL_CACHE_REFRESH_SEC` seconds later.
- Implementation: an `asyncio.Task` that sleeps for the configured interval, then calls `refresh_model_map()`. Each new acquire cancels the pending task and schedules a new one.
- Effect: active usage keeps the map fresh (refresh fires after each idle window). Idle periods still get periodic refreshes.

### Fast Path in `acquire_server()`

```
acquire_server(model_id)
  -> look up model_id in _model_map
  -> if found: try each endpoint in order (507 -> next)
  -> if not found: fall back to full health-check scan (_select_runner)
  -> on success: schedule sliding refresh
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_CACHE_REFRESH_SEC` | `60` | Sliding window interval for model map refresh (seconds) |

### Unchanged Components

- **`list_models()`**: Already queries all runners concurrently and deduplicates by ID. No changes needed.
- **`model_by_task()`**: Already queries all runners sequentially. No changes needed.
- **`release_server()` / `shutdown_server()`**: Use `ServerHandle.runner_host` to target the correct runner. No changes needed.
- **Workflow builders** (`ide/builder.py`, `dialog/builder.py`): Call `runner_client.acquire_server(model_id)`. No changes needed.
- **Runner side**: No changes needed. Each runner already exposes `/v1/models` and `/health` with model IDs.

### Error Handling

- If `refresh_model_map()` fails for a runner (network error, 5xx), that runner's models are simply omitted from the map. The next refresh will pick them up.
- If `acquire_server()` fails for all endpoints in the map, fall back to the current `_select_runner()` health-check scan as a safety net.
- If the map is empty for a model_id (model not yet discovered), fall back to `_select_runner()` immediately.

### Testing

- Unit test: `refresh_model_map()` builds correct map from mock runner responses
- Unit test: `acquire_server()` uses cached map, falls back on 507
- Unit test: sliding window refresh is scheduled on success, cancels previous
- Unit test: empty map falls back to health-check scan
- Integration test: with two runner instances, models route to correct runner

## Files Changed

| File | Change |
|------|--------|
| `config.py` | Add `MODEL_CACHE_REFRESH_SEC` env var |
| `services/runner_client.py` | Add `_model_map`, `refresh_model_map()`, sliding refresh task, update `acquire_server()` |
| `test/unit/test_runner_client.py` | Add tests for cached map, sliding refresh, fallback behavior |
