"""
ollama_provider.py — Local Ollama provider via HTTP /api/chat.

No API key required.
Vision: images are passed as base64 in the message content if the model
supports it (e.g. llava, bakllava, llama3.2-vision).

Default base URL: http://localhost:11434
"""

import base64
import json
import logging

import aiohttp

from .base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """
    LLM provider backed by a locally running Ollama instance.

    Config keys:
        model      — Ollama model tag (default: "llama3.2")
        base_url   — Ollama server URL (default: "http://localhost:11434")
    """

    def __init__(self, cfg: dict) -> None:
        self.model_name: str = cfg.get("model", "llama3.2")
        self.base_url: str = cfg.get("base_url", "http://localhost:11434").rstrip("/")
        logger.info(
            "OllamaProvider ready: model=%s base_url=%s", self.model_name, self.base_url
        )

    async def complete(
        self,
        system: str,
        messages: list[dict],
        image_bytes: bytes | None = None,
        **kwargs,
    ) -> LLMResponse:
        temperature: float = kwargs.get("temperature", 0.8)

        # Build Ollama message list
        ollama_messages: list[dict] = [{"role": "system", "content": system}]

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Ollama vision: add images list to the last user message
            if (
                image_bytes is not None
                and role == "user"
                and i == len(messages) - 1
            ):
                b64 = base64.b64encode(image_bytes).decode()
                ollama_messages.append(
                    {"role": role, "content": content, "images": [b64]}
                )
            else:
                ollama_messages.append({"role": role, "content": content})

        payload = {
            "model": self.model_name,
            "messages": ollama_messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        url = f"{self.base_url}/api/chat"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        raw_text: str = data.get("message", {}).get("content", "")

        # Ollama reports eval_count (output tokens) and prompt_eval_count (input tokens)
        tokens_in: int = data.get("prompt_eval_count", 0)
        tokens_out: int = data.get("eval_count", 0)

        return LLMResponse(
            text=raw_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=0.0,  # local inference — no API cost
            model=self.model_name,
            provider="ollama",
        )
