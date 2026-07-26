"""
base_agent.py — Shared utilities for all Morrowind AI agents.

Provides:
  - load_api_key(): reads GOOGLE_API_KEY from ~/.nemoclaw_env
  - log_cost(): appends token/cost records to ~/morrowind-ai/logs/costs.log
  - call_with_retry(): wraps a coroutine with exponential backoff (max 3 retries)
"""

import asyncio
import logging
import os
import pathlib
import time

logger = logging.getLogger(__name__)

# Gemini 2.5 Flash pricing (USD per million tokens)
GEMINI_INPUT_COST_PER_M = 0.075
GEMINI_OUTPUT_COST_PER_M = 0.30

LOGS_DIR = pathlib.Path.home() / "morrowind-ai" / "logs"
COSTS_LOG = LOGS_DIR / "costs.log"


def load_api_key() -> str:
    """Read GOOGLE_API_KEY from ~/.nemoclaw_env."""
    env_file = pathlib.Path.home() / ".nemoclaw_env"
    for line in env_file.read_text().splitlines():
        if line.startswith("GOOGLE_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise ValueError("GOOGLE_API_KEY not found in ~/.nemoclaw_env")


def log_cost(agent_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Compute and log the USD cost of a Gemini call.

    Appends a line to ~/morrowind-ai/logs/costs.log and returns the total cost.
    Creates the log directory if it doesn't exist.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    input_cost = (input_tokens / 1_000_000) * GEMINI_INPUT_COST_PER_M
    output_cost = (output_tokens / 1_000_000) * GEMINI_OUTPUT_COST_PER_M
    total_cost = input_cost + output_cost

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log_line = (
        f"{timestamp} | {agent_name} | "
        f"in={input_tokens} out={output_tokens} | "
        f"cost=${total_cost:.6f}\n"
    )

    try:
        with COSTS_LOG.open("a") as f:
            f.write(log_line)
    except OSError as exc:
        logger.warning("Could not write to costs.log: %s", exc)

    return total_cost


async def call_with_retry(coro_factory, max_retries: int = 3, base_delay: float = 1.0):
    """
    Call an async coroutine factory with exponential backoff on failure.

    Args:
        coro_factory: A zero-argument callable that returns a coroutine.
                      Called fresh on each retry so the coroutine is not reused.
        max_retries: Maximum number of attempts (default 3).
        base_delay: Initial delay in seconds; doubles each retry.

    Returns:
        The result of the coroutine on success.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exc = None
    delay = base_delay

    for attempt in range(1, max_retries + 1):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == max_retries:
                logger.error(
                    "All %d retries exhausted. Last error: %s", max_retries, exc
                )
                raise
            logger.warning(
                "Attempt %d/%d failed (%s). Retrying in %.1fs…",
                attempt,
                max_retries,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
            delay *= 2

    raise last_exc  # unreachable but satisfies type checkers
