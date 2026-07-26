"""
base.py — Abstract base classes and shared utilities for all LLM providers.

All providers return LLMResponse so callers can log cost/tokens uniformly
regardless of which backend is active.
"""

import pathlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    model: str
    provider: str


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    # Умеет ли провайдер отдавать ответ по частям. Переопределяют те, кто умеет:
    # тогда мост показывает реплику по мере набора, а не после полного ответа.
    supports_stream: bool = False

    async def complete_stream(
        self,
        system: str,
        messages: list[dict],
        on_text=None,
        image_bytes: bytes | None = None,
        **kwargs,
    ) -> "LLMResponse":
        """Как complete(), но зовёт on_text(текст_целиком_на_текущий_момент)
        по мере набора.

        Базовая реализация ждёт ответ целиком и отдаёт его одним куском —
        провайдеру без поддержки потока переопределять ничего не нужно, а
        вызывающий код одинаков для всех.
        """
        resp = await self.complete(system=system, messages=messages,
                                   image_bytes=image_bytes, **kwargs)
        if on_text is not None and resp.text:
            on_text(resp.text)
        return resp

    @abstractmethod
    async def complete(
        self,
        system: str,
        messages: list[dict],
        image_bytes: bytes | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Run a chat completion.

        Args:
            system:      System prompt string.
            messages:    Conversation turns: [{"role": "user"|"assistant", "content": str}]
            image_bytes: Raw image bytes for vision requests, or None.
            **kwargs:    Provider-specific overrides (e.g. temperature, max_tokens).

        Returns:
            LLMResponse with text, token counts, cost, model name, and provider name.
        """


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def read_nemoclaw_env(key: str) -> str:
    """
    Read a value from ~/.nemoclaw_env.

    Raises ValueError if the key is not found.
    Never logs the value — it may be a secret.
    """
    env_file = pathlib.Path.home() / ".nemoclaw_env"
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    raise ValueError(f"{key} not found in ~/.nemoclaw_env")


def log_llm_response(agent_name: str, resp: LLMResponse) -> None:
    """
    Append a cost/token line to ~/morrowind-ai/logs/costs.log.

    Format is compatible with the existing log_cost() output in base_agent.py:
        <timestamp> | <agent> [<provider>/<model>] | in=N out=N | cost=$X.XXXXXX
    """
    logs_dir = pathlib.Path.home() / "morrowind-ai" / "logs"
    costs_log = logs_dir / "costs.log"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = (
        f"{timestamp} | {agent_name} [{resp.provider}/{resp.model}] | "
        f"in={resp.tokens_in} out={resp.tokens_out} | "
        f"cost=${resp.cost_usd:.6f}\n"
    )
    try:
        with costs_log.open("a") as fh:
            fh.write(line)
    except OSError:
        pass  # non-fatal — caller already has the data in LLMResponse
