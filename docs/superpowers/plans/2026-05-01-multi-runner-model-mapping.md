# Multi-Runner Model Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cache a model_id → [runner_endpoints] map in RunnerClient so `acquire_server()` routes directly to the correct runner without per-request health checks.

**Architecture:** Add `_model_map: Dict[str, List[str]]` to `RunnerClient` populated by `refresh_model_map()` querying `GET /v1/models` concurrently on all endpoints. A sliding-window `asyncio.Task` refreshes after each idle period (configurable via `MODEL_CACHE_REFRESH_SEC`). The `acquire_server()` fast path uses the cached map, falling back to `_select_runner()` health-check scan if the model isn't found.

**Tech Stack:** Python 3.12, asyncio, httpx, pytest-asyncio

---

### Task 1: Add MODEL_CACHE_REFRESH_SEC config variable

**Files:**
- Modify: `config.py`
- Test: `test/unit/test_runner_client.py`

- [ ] **Step 1: Write the failing test**

Add to `test/unit/test_runner_client.py`:
```python
class TestRunnerClientConfig:
    def test_default_refresh_interval(self):
        from config import MODEL_CACHE_REFRESH_SEC
        assert MODEL_CACHE_REFRESH_SEC == 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/unit/test_runner_client.py::TestRunnerClientConfig::test_default_refresh_interval -v --noconftest`
Expected: FAIL with ImportError/AttributeError

- [ ] **Step 3: Add config variable**

Add to `config.py` after `RUNNER_ENDPOINTS` block (line 82):
```python
MODEL_CACHE_REFRESH_SEC = int(os.environ.get("MODEL_CACHE_REFRESH_SEC", "60"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/unit/test_runner_client.py::TestRunnerClientConfig::test_default_refresh_interval -v --noconftest`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config.py test/unit/test_runner_client.py
git commit -m "feat: add MODEL_CACHE_REFRESH_SEC config variable (default 60s)"
```

---

### Task 2: Add _model_map and refresh_model_map() to RunnerClient

**Files:**
- Modify: `services/runner_client.py`
- Test: `test/unit/test_runner_client.py`

- [ ] **Step 1: Write the failing test**

Add to `test/unit/test_runner_client.py`:
```python
class TestRunnerClientModelMap:
    @pytest.mark.asyncio
    async def test_refresh_builds_map(self):
        r1 = MagicMock()
        r1.status_code = 200
        r1.json.return_value = [
            {"id": "model-a", "name": "A", "model": "a", "task": "TextToText", "modified_at": "2025-01-01", "digest": "a", "provider": "llama_cpp", "details": {"format": "gguf", "family": "llama", "families": ["llama"], "parameter_size": "8B", "size": 4e9, "original_ctx": 8192}},
            {"id": "model-b", "name": "B", "model": "b", "task": "TextToText", "modified_at": "2025-01-01", "digest": "b", "provider": "llama_cpp", "details": {"format": "gguf", "family": "llama", "families": ["llama"], "parameter_size": "7B", "size": 3e9, "original_ctx": 4096}},
        ]
        r2 = MagicMock()
        r2.status_code = 200
        r2.json.return_value = [
            {"id": "model-b", "name": "B", "model": "b", "task": "TextToText", "modified_at": "2025-01-01", "digest": "b", "provider": "llama_cpp", "details": {"format": "gguf", "family": "llama", "families": ["llama"], "parameter_size": "7B", "size": 3e9, "original_ctx": 4096}},
            {"id": "model-c", "name": "C", "model": "c", "task": "TextToEmbeddings", "modified_at": "2025-01-01", "digest": "c", "provider": "llama_cpp", "details": {"format": "gguf", "family": "nomic", "families": ["nomic"], "parameter_size": "0.3B", "size": 2e8, "original_ctx": 8192}},
        ]
        idx = [0]
        async def mock_get(url, **kw):
            r = [r1, r2][idx[0]]; idx[0] += 1; return r
        mock = _mock_client(get=AsyncMock(side_effect=mock_get))
        client = RunnerClient(endpoints=["http://r1:8000", "http://r2:8001"])
        client._client = mock
        await client.refresh_model_map()
        assert client._model_map["model-a"] == ["http://r1:8000"]
        assert client._model_map["model-b"] == ["http://r1:8000", "http://r2:8001"]
        assert client._model_map["model-c"] == ["http://r2:8001"]

    @pytest.mark.asyncio
    async def test_refresh_skips_unhealthy_runner(self):
        r1 = MagicMock()
        r1.status_code = 200
        r1.json.return_value = [
            {"id": "model-a", "name": "A", "model": "a", "task": "TextToText", "modified_at": "2025-01-01", "digest": "a", "provider": "llama_cpp", "details": {"format": "gguf", "family": "llama", "families": ["llama"], "parameter_size": "8B", "size": 4e9, "original_ctx": 8192}},
        ]
        idx = [0]
        async def mock_get(url, **kw):
            v = [r1, Exception("conn refused")][idx[0]]; idx[0] += 1
            raise v if isinstance(v, Exception) else None or v
        mock = _mock_client(get=AsyncMock(side_effect=mock_get))
        client = RunnerClient(endpoints=["http://r1:8000", "http://r2:8001"])
        client._client = mock
        await client.refresh_model_map()
        assert client._model_map["model-a"] == ["http://r1:8000"]
        assert all("http://r2:8001" not in v for v in client._model_map.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/unit/test_runner_client.py::TestRunnerClientModelMap -v --noconftest`
Expected: FAIL with AttributeError (no refresh_model_map)

- [ ] **Step 3: Add _model_map, _refresh_task, and refresh_model_map()**

In `RunnerClient.__init__` after `self._healthy` line:
```python
        self._model_map: Dict[str, List[str]] = {}
        self._refresh_task: Optional[asyncio.Task] = None
```

After `shutdown_server` method, add:
```python
    async def refresh_model_map(self) -> None:
        """Query all runners and build a model_id -> [endpoints] map."""
        new_map: Dict[str, List[str]] = {}
        client = self._get_client()
        tasks = []
        for endpoint in self._endpoints:
            async def fetch_models(ep=endpoint):
                try:
                    resp = await client.get(f"{ep}/v1/models")
                    if resp.status_code == 200:
                        return [(m["id"], ep) for m in resp.json() if "id" in m]
                except Exception as e:
                    logger.warning(f"Failed to list models from {ep}: {e}")
                return []
            tasks.append(fetch_models())
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                for model_id, endpoint in result:
                    if model_id not in new_map:
                        new_map[model_id] = []
                    new_map[model_id].append(endpoint)
        self._model_map = new_map
        logger.info(f"Model map refreshed: {len(new_map)} models across {len(self._endpoints)} endpoints")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/unit/test_runner_client.py::TestRunnerClientModelMap -v --noconftest`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/runner_client.py test/unit/test_runner_client.py
git commit -m "feat: add cached model-to-runner map with refresh_model_map()"
```

---

### Task 3: Add sliding-window refresh scheduling

**Files:**
- Modify: `services/runner_client.py`
- Test: `test/unit/test_runner_client.py`

- [ ] **Step 1: Write the failing test**

Add to `test/unit/test_runner_client.py`:
```python
class TestRunnerClientSlidingRefresh:
    @pytest.mark.asyncio
    async def test_schedule_refresh_on_acquire(self):
        mock_health = MagicMock()
        mock_health.status_code = 200
        mock_health.json.return_value = {"status": "ok", "gpu": {"available_vram_bytes": 12e9}, "active_servers": 0, "models": ["model-a"]}
        mock_create = MagicMock()
        mock_create.status_code = 201
        mock_create.json.return_value = {"server_id": "abc", "base_url": "http://r1:8000/v1/server/abc", "model": "model-a"}
        mock_create.raise_for_status = MagicMock()
        mock = _mock_client(get=AsyncMock(return_value=mock_health), post=AsyncMock(return_value=mock_create))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000"]}
        handle = await client.acquire_server("model-a")
        assert handle is not None
        assert client._refresh_task is not None
        assert isinstance(client._refresh_task, asyncio.Task)

    @pytest.mark.asyncio
    async def test_new_schedule_cancels_pending(self):
        mock_health = MagicMock()
        mock_health.status_code = 200
        mock_health.json.return_value = {"status": "ok", "gpu": {"available_vram_bytes": 12e9}, "active_servers": 0, "models": ["model-a"]}
        mock_create = MagicMock()
        mock_create.status_code = 201
        mock_create.json.return_value = {"server_id": "abc", "base_url": "http://r1:8000/v1/server/abc", "model": "model-a"}
        mock_create.raise_for_status = MagicMock()
        mock = _mock_client(get=AsyncMock(return_value=mock_health), post=AsyncMock(return_value=mock_create))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000"]}
        await client.acquire_server("model-a")
        first_task = client._refresh_task
        await client.acquire_server("model-a")
        second_task = client._refresh_task
        assert first_task is not second_task
        assert first_task.cancelled()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/unit/test_runner_client.py::TestRunnerClientSlidingRefresh -v --noconftest`
Expected: FAIL (refresh task not yet scheduled)

- [ ] **Step 3: Add _schedule_refresh, update imports, integrate**

Update config import:
```python
from config import MODEL_CACHE_REFRESH_SEC, RUNNER_ENDPOINTS
```

Add method after `refresh_model_map`:
```python
    def _schedule_refresh(self) -> None:
        if self._refresh_task is not None:
            self._refresh_task.cancel()
        async def _do_refresh():
            try:
                await asyncio.sleep(MODEL_CACHE_REFRESH_SEC)
                await self.refresh_model_map()
            except asyncio.CancelledError:
                pass
            finally:
                self._refresh_task = None
        self._refresh_task = asyncio.create_task(_do_refresh())
```

In `acquire_server`, add before `return handle`:
```python
                self._schedule_refresh()
```

Update `aclose` to cancel pending refresh:
```python
    async def aclose(self) -> None:
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/unit/test_runner_client.py::TestRunnerClientSlidingRefresh -v --noconftest`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/runner_client.py test/unit/test_runner_client.py
git commit -m "feat: add sliding-window refresh scheduling to runner client"
```

---

### Task 4: Update acquire_server() to use cached model map

**Files:**
- Modify: `services/runner_client.py`
- Test: `test/unit/test_runner_client.py`

- [ ] **Step 1: Write the failing test**

Add to `test/unit/test_runner_client.py`:
```python
class TestRunnerClientAcquireWithMap:
    @pytest.mark.asyncio
    async def test_acquire_uses_cached_map(self):
        """acquire_server uses cached map, skips health checks."""
        mock_create = MagicMock()
        mock_create.status_code = 201
        mock_create.json.return_value = {"server_id": "abc", "base_url": "http://r2:8001/v1/server/abc", "model": "model-c"}
        mock_create.raise_for_status = MagicMock()
        mock = _mock_client(post=AsyncMock(return_value=mock_create))
        client = RunnerClient(endpoints=["http://r1:8000", "http://r2:8001"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000"], "model-c": ["http://r2:8001"]}
        handle = await client.acquire_server("model-c")
        assert handle.server_id == "abc"
        assert handle.runner_host == "http://r2:8001"
        # Should NOT have called get() for health check
        mock.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_acquire_fallback_on_missing_model(self):
        """Model not in map falls back to health-check scan."""
        mock_health = MagicMock()
        mock_health.status_code = 200
        mock_health.json.return_value = {"status": "ok", "gpu": {"available_vram_bytes": 12e9}, "active_servers": 0, "models": ["model-x"]}
        mock_create = MagicMock()
        mock_create.status_code = 201
        mock_create.json.return_value = {"server_id": "def", "base_url": "http://r1:8000/v1/server/def", "model": "model-x"}
        mock_create.raise_for_status = MagicMock()
        mock = _mock_client(get=AsyncMock(return_value=mock_health), post=AsyncMock(return_value=mock_create))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {}  # empty map
        handle = await client.acquire_server("model-x")
        assert handle.server_id == "def"
        # Should have called get() for health check fallback
        mock.get.assert_called()

    @pytest.mark.asyncio
    async def test_acquire_fallback_on_507(self):
        """507 on primary runner falls through to next in map."""
        calls = [0]
        async def mock_post(url, **kw):
            calls[0] += 1
            if calls[0] == 1:
                r = MagicMock(); r.status_code = 507; return r
            r = MagicMock()
            r.status_code = 201
            r.json.return_value = {"server_id": "ghi", "base_url": "http://r2:8001/v1/server/ghi", "model": "model-b"}
            r.raise_for_status = MagicMock()
            return r
        mock = _mock_client(post=AsyncMock(side_effect=mock_post))
        client = RunnerClient(endpoints=["http://r1:8000", "http://r2:8001"])
        client._client = mock
        client._model_map = {"model-b": ["http://r1:8000", "http://r2:8001"]}
        handle = await client.acquire_server("model-b")
        assert handle.server_id == "ghi"
        assert handle.runner_host == "http://r2:8001"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/unit/test_runner_client.py::TestRunnerClientAcquireWithMap -v --noconftest`
Expected: FAIL (acquire_server still uses _select_runner, would call get())

- [ ] **Step 3: Update acquire_server to use cached map**

Replace the `acquire_server` method's endpoint ordering logic (lines 147-156) to use the cached map first:

```python
    async def acquire_server(self, model_id: str, **kwargs) -> ServerHandle:
        """Acquire a new llama.cpp server from a runner.

        Uses the cached model map for fast routing. Falls back to
        health-check scan if the model isn't in the map.

        Extra kwargs are accepted for forward compatibility with callers
        that pass task/config_override (ignored, not used).
        """
        payload: dict[str, Any] = {"model_id": model_id}

        # Fast path: use cached model map
        mapped_endpoints = self._model_map.get(model_id)
        if mapped_endpoints:
            ordered = list(mapped_endpoints)
        else:
            # Fallback: health-check scan
            best = await self._select_runner(model_id)
            if best:
                ordered = [best]
                for ep in self._endpoints:
                    if ep != best:
                        ordered.append(ep)
            else:
                ordered = list(self._endpoints)

        last_error = None
        for endpoint in ordered:
            try:
                client = self._get_client()
                resp = await client.post(
                    f"{endpoint}/v1/server/create",
                    json=payload,
                    timeout=_ACQUIRE_TIMEOUT,
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
                    base_url=f"{endpoint}/v1/server/{data['server_id']}",
                    server_id=data["server_id"],
                    runner_host=endpoint,
                )
                logger.info(f"Acquired server {handle.server_id} from {endpoint}")
                self._schedule_refresh()
                return handle

            except Exception as e:
                logger.warning(f"Failed to acquire from {endpoint}: {e}")
                last_error = str(e)
                continue

        raise RuntimeError(
            f"No healthy runner available for model {model_id}. "
            f"Last error: {last_error}"
        )
```

Note: Added `**kwargs` to signature to accept `task` and `config_override` from callers (IDE builder, dialog builder) without raising TypeError.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/unit/test_runner_client.py::TestRunnerClientAcquireWithMap -v --noconftest`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/runner_client.py test/unit/test_runner_client.py
git commit -m "feat: use cached model map in acquire_server with fallback"
```

---

### Task 5: Fix existing tests and add startup refresh

**Files:**
- Modify: `test/unit/test_runner_client.py` — fix stale test signatures
- Modify: `services/runner_client.py` — no startup refresh needed (sliding window covers it)
- Modify: `test/unit/test_runner_client.py` — ensure all tests work with new signature

- [ ] **Step 1: Fix stale test calls**

The existing tests in `TestRunnerClientAcquire` call `acquire_server` with extra args (`task`, `config_override`) that the old signature didn't accept. With the new `**kwargs` signature, these will work, but verify:

Run: `uv run pytest test/unit/test_runner_client.py -v --noconftest`
Expected: All tests PASS

If any tests fail due to the new behavior (e.g., tests that mock `_select_runner` path), update the mocks to include `_model_map` or adjust assertions.

- [ ] **Step 2: Commit**

```bash
git add test/unit/test_runner_client.py
git commit -m "fix: update runner client tests for new acquire_server signature"
```

---

### Task 6: Run full test suite and verify

**Files:**
- Run: full test suite

- [ ] **Step 1: Run all runner client tests**

Run: `uv run pytest test/unit/test_runner_client.py -v`
Expected: All tests PASS

- [ ] **Step 2: Verify no regressions in broader test suite**

Run: `uv run pytest test/unit/ -v --ignore=test/unit/conftest.py 2>&1 | tail -40`

Note: If conftest.py requires torch (which isn't installed), run with `--noconftest` or install torch. The key verification is that the runner_client tests pass.

- [ ] **Step 3: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "test: ensure all runner client tests pass with new model map"
```

---

## Self-Review

1. **Spec coverage:**
   - Cached model map (`_model_map`) — Task 2
   - Sliding window refresh — Task 3
   - Fast path in acquire_server — Task 4
   - Fallback to health-check scan — Task 4
   - Configurable refresh interval — Task 1
   - Error handling (507 fallback, empty map fallback) — Task 4
   - Testing — Tasks 1-5

2. **Placeholder scan:** No TBDs, TODOs, or vague instructions. All code blocks are complete.

3. **Type consistency:** `acquire_server(model_id: str, **kwargs) -> ServerHandle` consistent across all tasks. `_model_map: Dict[str, List[str]]` used uniformly. `ServerHandle` fields unchanged.

4. **Bonus fix:** Added `**kwargs` to `acquire_server` to fix the preexisting TypeError when IDE/dialog builders pass `task=` kwarg.