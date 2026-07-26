"""
openai_provider.py — OpenAI provider using the openai SDK (AsyncOpenAI).

Supports vision via base64 image_url content parts.
API key read from OPENAI_API_KEY in ~/.nemoclaw_env.

No cost constants are hardcoded here — GPT-4o pricing changes frequently.
cost_usd is set to 0.0; callers who need accurate billing should extend this
with per-model rate tables.
"""

import base64
import logging

from openai import AsyncOpenAI

from .base import LLMProvider, LLMResponse, read_nemoclaw_env

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    LLM provider backed by OpenAI chat completions.

    Config keys:
        model      — Model name (default: "gpt-4o")
        api_key    — Override; if absent, read OPENAI_API_KEY from ~/.nemoclaw_env
        base_url   — Optional custom base URL (e.g. Azure OpenAI endpoint)
    """

    def __init__(self, cfg: dict) -> None:
        self.model_name: str = cfg.get("model", "gpt-4o")
        api_key: str = cfg.get("api_key") or read_nemoclaw_env("OPENAI_API_KEY")
        base_url: str | None = cfg.get("base_url")

        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**kwargs)
        logger.info("OpenAIProvider ready: model=%s", self.model_name)

    async def complete(
        self,
        system: str,
        messages: list[dict],
        image_bytes: bytes | None = None,
        **kwargs,
    ) -> LLMResponse:
        temperature: float = kwargs.get("temperature", 0.8)
        max_tokens: int = kwargs.get("max_tokens", 512)

        # Build the openai-format message list
        oai_messages: list[dict] = [{"role": "system", "content": system}]

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Attach image to the LAST user turn only
            if (
                image_bytes is not None
                and role == "user"
                and i == len(messages) - 1
            ):
                b64 = base64.b64encode(image_bytes).decode()
                oai_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": content},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}",
                                    "detail": "auto",
                                },
                            },
                        ],
                    }
                )
            else:
                oai_messages.append({"role": role, "content": content})

        response = await self._client.chat.completions.create(
            model=self.model_name,
            messages=oai_messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )

        raw_text: str = response.choices[0].message.content or ""
        usage = response.usage
        tokens_in: int = usage.prompt_tokens if usage else 0
        tokens_out: int = usage.completion_tokens if usage else 0

        # Cost not computed here — rates vary by model and tier
        return LLMResponse(
            text=raw_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=0.0,
            model=self.model_name,
            provider="openai",
        )
