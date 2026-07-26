"""
pixel_agent.py — Vision agent for the Morrowind AI system.

Continuously captures the OpenMW game window and analyses each frame using
the configured vision-capable LLM provider. Results are cached and consumed
by the LoreAgent and OBSDirector to add scene context to NPC responses
and camera direction.

Provider is configured in config.yaml under models.pixel_agent:
    provider: gemini | openai | anthropic | ollama  (NOT llamacpp — no vision)
    model:    <model name>

IMPORTANT: This agent requires a vision-capable provider. Using llamacpp will
log a warning and the analysis will be empty/meaningless.

Usage:
    agent = PixelAgent(config)
    asyncio.create_task(agent.capture_and_analyze())  # start background loop
    ...
    ctx = agent.get_latest_context()  # thread-safe read of cached result
"""

import asyncio
import io
import logging
import subprocess
import time
from typing import Any

try:
    import mss
    import mss.tools
    _MSS_AVAILABLE = True
except ImportError:
    _MSS_AVAILABLE = False

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from providers.factory import get_provider
from providers.base import log_llm_response

from .base_agent import call_with_retry

logger = logging.getLogger(__name__)

# Providers that actually support vision
_VISION_PROVIDERS = {"gemini", "openai", "anthropic", "ollama"}

# Target resolution for cost-controlled vision calls
TARGET_WIDTH = 1024
TARGET_HEIGHT = 768

# How often to refresh the scene analysis (seconds)
DEFAULT_CAPTURE_INTERVAL = 3.0

ANALYSIS_PROMPT = """\
You are analysing a screenshot from the game The Elder Scrolls III: Morrowind (OpenMW engine).
Describe what you see concisely for use by an NPC dialogue system.

Respond ONLY with valid JSON in this exact structure (no markdown, no code fences):
{
  "scene_description": "<1-2 sentence summary of the scene visible>",
  "player_state": "<one of: exploring, in_dialogue, in_combat, in_inventory, resting, loading>",
  "threats": ["<threat1>", "<threat2>"],
  "notable_items": ["<item1>", "<item2>"]
}

Focus on:
- The environment (interior/exterior, location name if visible in HUD)
- Player health/magicka bars if visible (describe as 'full', 'low', 'critical')
- Any enemies or hostile creatures visible
- Any important items, NPCs, or signs in the scene
- The current UI state (dialogue box open? inventory open? map open?)

Keep every string under 80 characters. Return ONLY the JSON object.
"""


def _find_openmw_window_region() -> dict[str, int] | None:
    """
    Use xdotool to find the OpenMW window position and size.

    Returns a dict with keys 'left', 'top', 'width', 'height' suitable
    for mss.grab(), or None if the window cannot be found.
    """
    try:
        search_result = subprocess.run(
            ["xdotool", "search", "--name", "OpenMW"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if search_result.returncode != 0 or not search_result.stdout.strip():
            logger.debug("xdotool: no OpenMW window found")
            return None

        window_ids = search_result.stdout.strip().splitlines()
        wid = window_ids[0]

        geom_result = subprocess.run(
            ["xdotool", "getwindowgeometry", "--shell", wid],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if geom_result.returncode != 0:
            logger.debug("xdotool getwindowgeometry failed")
            return None

        env: dict[str, int] = {}
        for line in geom_result.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                try:
                    env[k.strip()] = int(v.strip())
                except ValueError:
                    pass

        if all(k in env for k in ("X", "Y", "WIDTH", "HEIGHT")):
            return {
                "left": env["X"],
                "top": env["Y"],
                "width": env["WIDTH"],
                "height": env["HEIGHT"],
            }

    except FileNotFoundError:
        logger.warning("xdotool not found — cannot locate OpenMW window by title")
    except subprocess.TimeoutExpired:
        logger.warning("xdotool timed out looking for OpenMW window")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error in _find_openmw_window_region: %s", exc)

    return None


def _capture_screen(region: dict[str, int] | None) -> bytes | None:
    """
    Capture the screen (or a region) using mss and return JPEG bytes.

    Resizes to TARGET_WIDTH x TARGET_HEIGHT for cost control.
    Returns None if capture or PIL processing fails.
    """
    if not _MSS_AVAILABLE:
        logger.error("mss is not installed — cannot capture screen")
        return None
    if not _PIL_AVAILABLE:
        logger.error("Pillow is not installed — cannot resize screenshot")
        return None

    try:
        with mss.mss() as sct:
            monitor = region if region is not None else sct.monitors[0]
            sct_img = sct.grab(monitor)

        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    except Exception as exc:  # noqa: BLE001
        logger.error("Screen capture failed: %s", exc)
        return None


def _parse_analysis(raw_text: str) -> dict[str, Any]:
    """Parse the JSON response from the vision model."""
    import json

    fallback = {
        "scene_description": "Unable to analyse scene.",
        "player_state": "exploring",
        "threats": [],
        "notable_items": [],
    }

    try:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            inner = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(inner)

        data = json.loads(cleaned)

        return {
            "scene_description": str(data.get("scene_description", fallback["scene_description"])),
            "player_state": str(data.get("player_state", "exploring")),
            "threats": [str(t) for t in data.get("threats", [])],
            "notable_items": [str(i) for i in data.get("notable_items", [])],
        }
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        logger.warning("Failed to parse vision JSON: %s | raw=%r", exc, raw_text[:200])
        return fallback


class PixelAgent:
    """
    Vision agent that continuously analyses the OpenMW game window.

    Runs a background asyncio loop that captures frames every
    `capture_interval` seconds, sends them to the configured vision LLM,
    and caches the result for consumption by other agents.

    Provider and model are read from config['models']['pixel_agent'].
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialise the PixelAgent.

        Args:
            config: Full config dict (as loaded from config.yaml). Must contain
                    a 'models.pixel_agent' key with 'provider' and 'model'.
                    Also reads config['pixel'] for capture settings.
        """
        provider_cfg: dict = config.get("models", {}).get(
            "pixel_agent", {"provider": "gemini", "model": "gemini-2.5-flash"}
        )

        provider_name = provider_cfg.get("provider", "gemini").lower()
        if provider_name not in _VISION_PROVIDERS:
            logger.warning(
                "PixelAgent: provider '%s' does not support vision. "
                "Frame analysis will produce empty/incorrect results. "
                "Switch to gemini, openai, anthropic, or a vision-capable ollama model.",
                provider_name,
            )

        self.llm = get_provider(provider_cfg)
        self._temperature: float = provider_cfg.get("temperature", 0.2)
        self._max_tokens: int = 512

        pixel_cfg: dict = config.get("pixel", {})
        self.capture_interval: float = float(
            pixel_cfg.get("capture_interval_sec", DEFAULT_CAPTURE_INTERVAL)
        )
        self.window_title: str = pixel_cfg.get("window_title", "OpenMW")

        self._latest_context: dict[str, Any] = {
            "scene_description": "Not yet analysed.",
            "player_state": "exploring",
            "threats": [],
            "notable_items": [],
            "timestamp": 0.0,
        }
        self._lock = asyncio.Lock()
        self._running = False

        logger.info(
            "PixelAgent initialised | provider=%s model=%s | interval=%.1fs | mss=%s | PIL=%s",
            provider_name,
            provider_cfg.get("model"),
            self.capture_interval,
            _MSS_AVAILABLE,
            _PIL_AVAILABLE,
        )

    def get_latest_context(self) -> dict[str, Any]:
        """Return the most recent scene analysis (non-blocking)."""
        return dict(self._latest_context)

    async def analyze_frame(self, frame_bytes: bytes) -> dict[str, Any]:
        """
        Send a JPEG frame to the vision LLM and return structured analysis.

        Args:
            frame_bytes: Raw JPEG bytes of the game screenshot.

        Returns:
            Dict with scene_description, player_state, threats, notable_items.
        """
        messages = [
            {
                "role": "user",
                "content": "Analyze this Morrowind screenshot and return the structured game state JSON.",
            }
        ]

        resp = await call_with_retry(
            lambda: self.llm.complete(
                system=ANALYSIS_PROMPT,
                messages=messages,
                image_bytes=frame_bytes,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        )

        log_llm_response("PixelAgent", resp)

        result = _parse_analysis(resp.text)
        logger.debug(
            "PixelAgent | state=%s | threats=%d | tokens=%d | cost=$%.5f",
            result["player_state"],
            len(result["threats"]),
            resp.tokens_in + resp.tokens_out,
            resp.cost_usd,
        )
        return result

    async def capture_and_analyze(self) -> None:
        """
        Continuous background loop: capture → analyse → cache → sleep.

        Designed to run as an asyncio Task. Exits cleanly when cancelled.
        If mss or PIL are unavailable, logs a warning and returns immediately.
        """
        if not _MSS_AVAILABLE or not _PIL_AVAILABLE:
            logger.warning(
                "PixelAgent disabled: mss=%s, PIL=%s. "
                "Install with: pip install mss Pillow",
                _MSS_AVAILABLE,
                _PIL_AVAILABLE,
            )
            return

        self._running = True
        logger.info("PixelAgent capture loop started")

        try:
            while self._running:
                loop_start = time.monotonic()

                region = _find_openmw_window_region()

                frame_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, _capture_screen, region
                )

                if frame_bytes is not None:
                    try:
                        analysis = await self.analyze_frame(frame_bytes)
                        analysis["timestamp"] = time.time()

                        async with self._lock:
                            self._latest_context = analysis

                    except Exception as exc:  # noqa: BLE001
                        logger.error("Frame analysis failed: %s", exc)

                elapsed = time.monotonic() - loop_start
                sleep_time = max(0.0, self.capture_interval - elapsed)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info("PixelAgent capture loop cancelled")
            self._running = False
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("PixelAgent capture loop crashed: %s", exc)
            self._running = False
            raise
