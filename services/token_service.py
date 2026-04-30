import json
from typing import Optional
import httpx
from models.message import Message, MessageContentType
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="token_service")


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
                    c.text
                    for c in msg.content
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
