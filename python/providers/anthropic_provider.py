"""
anthropic_provider.py — Anthropic Claude provider using the anthropic SDK.

Supports vision via base64 image source.
API key read from ANTHROPIC_API_KEY in ~/.nemoclaw_env.
Default model: claude-sonnet-4-6

Pricing (USD per million tokens, claude-sonnet-4-6 as of 2025-Q2):
  Input:  $3.00 / 1M
  Output: $15.00 / 1M
"""

import base64
import logging

import anthropic

from .base import LLMProvider, LLMResponse, read_nemoclaw_env

logger = logging.getLogger(__name__)

# claude-sonnet-4-6 rates — update if model changes
_INPUT_COST_PER_M = 3.00
_OUTPUT_COST_PER_M = 15.00


class AnthropicProvider(LLMProvider):
    """
    LLM provider backed by Anthropic Claude.

    Config keys:
        model      — Claude model name (default: "claude-sonnet-4-6")
        api_key    — Override; if absent, read ANTHROPIC_API_KEY from ~/.nemoclaw_env
    """

    def __init__(self, cfg: dict) -> None:
        self.model_name: str = cfg.get("model", "claude-sonnet-4-6")
        api_key: str = cfg.get("api_key") or read_nemoclaw_env("ANTHROPIC_API_KEY")
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        logger.info("AnthropicProvider ready: model=%s", self.model_name)

    async def complete(
        self,
        system: str,
        messages: list[dict],
        image_bytes: bytes | None = None,
        **kwargs,
    ) -> LLMResponse:
        temperature: float = kwargs.get("temperature", 0.8)
        max_tokens: int = kwargs.get("max_tokens", 512)

        # Build Anthropic message list
        anthropic_messages: list[dict] = []

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Attach image to the LAST user turn only
            if (
                image_bytes is not None
                and role == "user"
                and i == len(messages) - 1
            ):
                b64 = base64.standard_b64encode(image_bytes).decode()
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": content},
                        ],
                    }
                )
            else:
                anthropic_messages.append({"role": role, "content": content})

        response = await self._client.messages.create(
            model=self.model_name,
            system=system,
            messages=anthropic_messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )

        raw_text: str = (
            response.content[0].text
            if response.content and hasattr(response.content[0], "text")
            else ""
        )

        tokens_in: int = response.usage.input_tokens if response.usage else 0
        tokens_out: int = response.usage.output_tokens if response.usage else 0

        cost = (tokens_in / 1_000_000) * _INPUT_COST_PER_M + (
            tokens_out / 1_000_000
        ) * _OUTPUT_COST_PER_M

        return LLMResponse(
            text=raw_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            model=self.model_name,
            provider="anthropic",
        )
