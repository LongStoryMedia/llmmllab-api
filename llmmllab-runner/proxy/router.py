"""HTTP proxy router - forwards requests to llama.cpp servers.

Catch-all route for /v1/server/{server_id}/* that rewrites the path
and forwards to the appropriate local llama.cpp server instance.
"""

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from config import RUNNER_PORT

router = APIRouter()


@router.api_route("/v1/server/{server_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_request(request: Request, server_id: str, path: str):
    """Proxy a request to the target llama.cpp server.

    Path rewriting:
      /v1/server/{id}/v1/chat/completions  ->  http://127.0.0.1:{port}/v1/chat/completions
      /v1/server/{id}/health               ->  http://127.0.0.1:{port}/health

    SSE responses are streamed without buffering.
    """
    from app import server_cache

    entry = server_cache.get(server_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

    # Increment use count to keep server alive during request
    server_cache.increment_use(server_id)

    try:
        target_host = f"http://127.0.0.1:{entry.port}"

        # Rewrite path: strip the /v1/server/{id} prefix
        remaining = path
        upstream_url = f"{target_host}/{remaining}"

        # Read request body
        body = await request.body()

        # Build headers (exclude hop-by-hop)
        hop_by_hop = {
            "host", "connection", "keep-alive", "transfer-encoding",
            "upgrade", "proxy-authenticate", "proxy-authorization",
            "te", "trailers", "proxy-connection",
        }
        headers = dict(request.headers)
        for h in hop_by_hop:
            headers.pop(h, None)
        headers["host"] = f"127.0.0.1:{entry.port}"

        method = request.method
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                method=method,
                url=upstream_url,
                headers=headers,
                content=body if body else None,
            ) as response:
                response_headers = dict(response.headers)
                is_sse = "text/event-stream" in response_headers.get("content-type", "")

                if is_sse:
                    return StreamingResponse(
                        content=response.aiter_bytes(),
                        status_code=response.status_code,
                        headers={k: v for k, v in response_headers.items()
                                 if k.lower() not in ("transfer-encoding", "content-length")},
                    )
                else:
                    content = b""
                    async for chunk in response.aiter_bytes():
                        content += chunk

                    from fastapi.responses import Response
                    return Response(
                        content=content,
                        status_code=response.status_code,
                        headers={k: v for k, v in response_headers.items()
                                 if k.lower() not in ("transfer-encoding", "content-length")},
                    )

    except httpx.ConnectError:
        raise HTTPException(
            status_code=502,
            detail=f"Upstream server {server_id} on port {entry.port} is unreachable",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"Upstream server {server_id} timed out",
        )
    finally:
        # Decrement use count after request completes
        server_cache.decrement_use(server_id)
