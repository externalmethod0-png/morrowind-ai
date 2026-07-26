"""
gemini_provider.py — Google Gemini provider using the new google.genai SDK.

Supports vision (image_bytes → inline Part).
Default model: gemini-2.5-flash

Pricing (USD per million tokens, as of 2025-Q2):
  Input:  $0.075 / 1M
  Output: $0.30  / 1M
"""

import logging

from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors

from .base import LLMProvider, LLMResponse, read_nemoclaw_env

logger = logging.getLogger(__name__)

_INPUT_COST_PER_M = 0.075
_OUTPUT_COST_PER_M = 0.30


class GeminiProvider(LLMProvider):
    """
    LLM provider backed by Google Gemini (new google-genai SDK).

    Config keys:
        model      — Gemini model name (default: "gemini-2.5-flash")
        api_key    — Override; if absent, read GOOGLE_API_KEY from ~/.nemoclaw_env
    """

    def __init__(self, cfg: dict) -> None:
        self.model_name: str = cfg.get("model", "gemini-2.5-flash")
        # Multiple free-tier keys rotate on quota (429): each Google account has
        # its own daily pool. Reads GOOGLE_API_KEY, GOOGLE_API_KEY2..N, and any
        # comma-separated list inside those. Falls back to a single key.
        self._keys: list[str] = self._collect_keys(cfg)
        self._key_idx: int = 0
        self._clients: list = [genai.Client(api_key=k) for k in self._keys]
        # Whether this model accepts thinking_budget=0 (disable reasoning). Some
        # models (e.g. gemini-3.x flash) reject it; we detect once and remember.
        self._thinking_ok: bool = True
        logger.info("GeminiProvider ready: model=%s, keys=%d", self.model_name, len(self._keys))

    @staticmethod
    def _collect_keys(cfg: dict) -> list[str]:
        keys: list[str] = []
        if cfg.get("api_key"):
            keys.append(str(cfg["api_key"]))
        for name in ("GOOGLE_API_KEY", "GOOGLE_API_KEY2", "GOOGLE_API_KEY3",
                     "GOOGLE_API_KEY4", "GOOGLE_API_KEY5"):
            try:
                raw = read_nemoclaw_env(name)
            except ValueError:
                continue
            for part in str(raw).split(","):
                k = part.strip()
                if k and k not in keys:
                    keys.append(k)
        if not keys:
            keys.append(read_nemoclaw_env("GOOGLE_API_KEY"))  # raises a clear error
        return keys

    @staticmethod
    def _is_quota(exc) -> bool:
        """Is this the daily free-tier limit, i.e. should we rotate key/model?

        Matching only "RESOURCE_EXHAUSTED" was too narrow: a quota error worded
        any other way propagated as a hard failure and the NPC went mute for the
        rest of the day instead of moving to the next key.
        """
        if getattr(exc, "code", None) == 429:
            return True
        text = str(exc).lower()
        return any(s in text for s in
                   ("resource_exhausted", "quota", "rate limit", "429"))

    @property
    def _client(self):
        return self._clients[self._key_idx]

    async def complete(
        self,
        system: str,
        messages: list[dict],
        image_bytes: bytes | None = None,
        **kwargs,
    ) -> LLMResponse:
        temperature: float = kwargs.get("temperature", 0.8)
        max_output_tokens: int = kwargs.get("max_tokens", 512)

        # Flatten system + prior history into a single prompt string.
        history_lines: list[str] = [system, ""]
        for msg in messages[:-1]:
            role = msg.get("role", "user").upper()
            history_lines.append(f"{role}: {msg.get('content', '')}")
        if len(messages) > 1:
            history_lines.append("")

        last_msg = messages[-1] if messages else {"role": "user", "content": ""}
        history_lines.append(f"USER: {last_msg.get('content', '')}")
        prompt_text = "\n".join(history_lines)

        # Build contents. For text-only, a string is fine; for vision, assemble Parts.
        if image_bytes is not None:
            contents = [
                genai_types.Part.from_text(text=prompt_text),
                genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ]
        else:
            contents = prompt_text

        base_kwargs = dict(temperature=temperature, max_output_tokens=max_output_tokens)

        # Two axes of quota escape on a 429:
        #  1. sibling MODELS (each model has its own per-key daily pool);
        #  2. the next KEY (each Google account has its own pool entirely).
        model_chain = [self.model_name] + [
            m for m in ("gemini-3-flash-preview", "gemini-3.1-flash-lite")
            if m != self.model_name
        ]

        _is_quota = self._is_quota

        async def _call(cfg):
            last_exc = None
            tried_keys = 0
            n_keys = len(self._clients)
            while tried_keys < n_keys:
                client = self._clients[self._key_idx]
                for m in model_chain:
                    try:
                        return await client.aio.models.generate_content(
                            model=m, contents=contents, config=cfg,
                        )
                    except genai_errors.ClientError as exc:
                        if _is_quota(exc):
                            last_exc = exc
                            continue          # next model on this key
                        raise
                # every model on this key is quota-exhausted -> rotate key
                tried_keys += 1
                self._key_idx = (self._key_idx + 1) % n_keys
                if tried_keys < n_keys:
                    logger.warning("all models quota-hit on key #%d — switching key", self._key_idx)
            raise last_exc

        # Disable "thinking" so reasoning tokens don't eat the output budget and
        # truncate short NPC replies. If the model rejects thinking_budget=0
        # (some 3.x flash models do), fall back once and stop trying.
        # No retry here — callers (agents) already wrap .complete() in call_with_retry.
        if self._thinking_ok:
            try:
                response = await _call(genai_types.GenerateContentConfig(
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                    **base_kwargs,
                ))
            except genai_errors.ClientError as exc:
                # 400 = model won't accept thinking_budget=0. Anything else (e.g.
                # 429 quota) is a real error — re-raise it.
                if getattr(exc, "code", None) == 400 or " 400 " in f" {exc} ":
                    self._thinking_ok = False
                    logger.info("model %s rejects thinking_budget=0; disabling that path", self.model_name)
                    response = await _call(genai_types.GenerateContentConfig(**base_kwargs))
                else:
                    raise
        else:
            response = await _call(genai_types.GenerateContentConfig(**base_kwargs))

        raw_text: str = (getattr(response, "text", None) or "") or ""

        usage = getattr(response, "usage_metadata", None)
        tokens_in: int = getattr(usage, "prompt_token_count", 0) or 0
        tokens_out: int = getattr(usage, "candidates_token_count", 0) or 0

        cost = (tokens_in / 1_000_000) * _INPUT_COST_PER_M + (
            tokens_out / 1_000_000
        ) * _OUTPUT_COST_PER_M

        return LLMResponse(
            text=raw_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            model=self.model_name,
            provider="gemini",
        )
