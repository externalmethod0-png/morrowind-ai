"""
llamacpp_provider.py — llama.cpp server provider via OpenAI-compatible HTTP API.

Uses the /v1/chat/completions endpoint exposed by `llama-server` (llama.cpp).
No API key required by default; an optional bearer token can be configured.

Vision is NOT supported. If image_bytes is passed, a warning is logged and
the image is ignored — the request proceeds as text-only.

Default base URL: http://localhost:8080
"""

import logging

import aiohttp

from .base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class LlamaCppProvider(LLMProvider):
    """
    LLM provider backed by a running llama.cpp server.

    Config keys:
        base_url    — Server URL (default: "http://localhost:8080")
        model       — Model name passed in the request (default: "local")
                      Most llama.cpp deployments ignore this field.
        bearer_token — Optional bearer token if the server requires auth.
    """

    def __init__(self, cfg: dict) -> None:
        self.base_url: str = cfg.get("base_url", "http://localhost:8080").rstrip("/")
        self.model_name: str = cfg.get("model", "local")
        self._bearer: str | None = cfg.get("bearer_token")
        logger.info(
            "LlamaCppProvider ready: base_url=%s model=%s", self.base_url, self.model_name
        )

    async def complete(
        self,
        system: str,
        messages: list[dict],
        image_bytes: bytes | None = None,
        **kwargs,
    ) -> LLMResponse:
        if image_bytes is not None:
            logger.warning(
                "LlamaCppProvider does not support vision — image_bytes ignored. "
                "Use gemini, openai, anthropic, or ollama (with a vision model) instead."
            )

        temperature: float = kwargs.get("temperature", 0.8)
        max_tokens: int = kwargs.get("max_tokens", 512)

        # Build OpenAI-compatible message list
        oai_messages: list[dict] = [{"role": "system", "content": system}]
        for msg in messages:
            oai_messages.append(
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            )

        payload = {
            "model": self.model_name,
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers: dict = {"Content-Type": "application/json"}
        if self._bearer:
            headers["Authorization"] = f"Bearer {self._bearer}"

        url = f"{self.base_url}/v1/chat/completions"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        raw_text: str = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        usage = data.get("usage", {})
        tokens_in: int = usage.get("prompt_tokens", 0)
        tokens_out: int = usage.get("completion_tokens", 0)

        return LLMResponse(
            text=raw_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=0.0,  # local inference — no API cost
            model=self.model_name,
            provider="llamacpp",
        )
