"""
obs_director.py — OBS automation agent for the Morrowind AI system.

Reacts to game state changes detected by PixelAgent and makes intelligent
camera/scene decisions using the configured LLM provider when the state
is ambiguous.

Provider is configured in config.yaml under models.obs_director:
    provider: gemini | openai | anthropic | ollama | llamacpp
    model:    <model name>

(Vision is NOT required here — all decisions are text-only.)

Usage:
    director = OBSDirector(config)
    await director.connect()
    ...
    await director.on_game_state(state_dict)  # called by main orchestrator
"""

import asyncio
import json
import logging
import time
from typing import Any

try:
    import obsws_python as obs
    _OBS_AVAILABLE = True
except ImportError:
    obs = None  # type: ignore[assignment]
    _OBS_AVAILABLE = False

from providers.factory import get_provider
from providers.base import log_llm_response

from .base_agent import call_with_retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic scene routing — used when player_state is unambiguous
# ---------------------------------------------------------------------------

# Maps player_state → preferred OBS scene name
DETERMINISTIC_ROUTES: dict[str, str] = {
    "in_combat": "Combat Cam",
    "in_dialogue": "Dialogue Cam",
    "exploring": "Wide Cam",
    "resting": "Wide Cam",
}

# Sources that should be toggled by state
STATE_OVERLAYS: dict[str, dict[str, bool]] = {
    "in_inventory": {"Inventory Overlay": True},
    "in_combat": {"Inventory Overlay": False},
    "in_dialogue": {"Inventory Overlay": False},
    "exploring": {"Inventory Overlay": False},
}

# If health description contains these strings, trigger alert overlay
LOW_HEALTH_KEYWORDS = {"critical", "low", "red", "danger", "near death"}

# Minimum confidence from LLM to act on an ambiguous switch
AMBIGUOUS_SWITCH_THRESHOLD = 0.8

DIRECTOR_SYSTEM_PROMPT = """\
You are an OBS director for a Morrowind livestream.
Your task is to decide whether to switch to a different OBS camera scene
based on the current game state.

Consider:
- "Combat Cam" for fights, danger, action sequences
- "Dialogue Cam" when NPCs are being spoken to
- "Wide Cam" for exploration, calm moments, travel
- Stay on the current scene if the switch would be jarring or unnecessary

Respond ONLY with valid JSON (no markdown, no code fences):
{
  "should_switch": true|false,
  "target_scene": "<exact scene name or empty string>",
  "confidence": <float 0.0-1.0>,
  "reason": "<one sentence>"
}
"""


def _build_director_user_message(
    current_scene: str,
    available_scenes: list[str],
    state: dict[str, Any],
) -> str:
    return (
        f"Current OBS scene: {current_scene}\n"
        f"Available scenes: {json.dumps(available_scenes)}\n"
        f"\n"
        f"Game state from vision analysis:\n"
        f"  player_state: {state.get('player_state', 'exploring')}\n"
        f"  scene_description: {state.get('scene_description', '')}\n"
        f"  threats: {json.dumps(state.get('threats', []))}\n"
        f"  notable_items: {json.dumps(state.get('notable_items', []))}\n"
        f"\n"
        f"Should the stream switch to a different OBS scene?"
    )


class OBSDirector:
    """
    OBS automation agent.

    Connects to OBS via WebSocket and reacts to game state changes.
    Uses the configured LLM for ambiguous director decisions (text-only).
    Falls back gracefully if OBS is not connected.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialise the OBSDirector.

        Args:
            config: Full config dict (as loaded from config.yaml). Must contain
                    a 'models.obs_director' key with 'provider' and 'model'.
                    Also reads config['obs'] for WebSocket connection settings.
        """
        obs_cfg: dict = config.get("obs", {})
        self.host: str = obs_cfg.get("host", "127.0.0.1")
        self.port: int = int(obs_cfg.get("port", 4455))
        self.password: str = obs_cfg.get("password", "")
        self.switch_threshold: float = float(
            config.get("switch_threshold", AMBIGUOUS_SWITCH_THRESHOLD)
        )

        provider_cfg: dict = config.get("models", {}).get(
            "obs_director", {"provider": "gemini", "model": "gemini-2.5-flash"}
        )
        self.llm = get_provider(provider_cfg)
        self._temperature: float = provider_cfg.get("temperature", 0.2)
        self._max_tokens: int = 256

        self._client: Any = None
        self._connected: bool = False
        self._current_scene: str = ""
        self._available_scenes: list[str] = []

        self._last_switch_time: float = 0.0
        self._min_switch_interval: float = float(config.get("min_switch_interval", 5.0))

        logger.info(
            "OBSDirector initialised | obs=%s:%d | provider=%s model=%s | obs_lib=%s",
            self.host,
            self.port,
            provider_cfg.get("provider"),
            provider_cfg.get("model"),
            _OBS_AVAILABLE,
        )

    async def connect(self) -> bool:
        """
        Connect to OBS WebSocket.

        Returns True on success, False if OBS is unavailable.
        Never raises — failures are logged and the director runs in no-op mode.
        """
        if not _OBS_AVAILABLE:
            logger.warning(
                "obsws_python not installed — OBS automation disabled. "
                "Install with: pip install obsws-python"
            )
            return False

        try:
            def _connect_sync() -> Any:
                return obs.ReqClient(
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    timeout=5,
                )

            self._client = await asyncio.get_event_loop().run_in_executor(
                None, _connect_sync
            )
            self._connected = True

            await self._refresh_scene_list()
            logger.info(
                "OBSDirector connected to OBS | current_scene=%s | scenes=%s",
                self._current_scene,
                self._available_scenes,
            )
            return True

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "OBSDirector: could not connect to OBS at %s:%d — %s. "
                "Running in no-op mode.",
                self.host,
                self.port,
                exc,
            )
            self._connected = False
            return False

    async def _refresh_scene_list(self) -> None:
        """Fetch and cache the current scene name and scene list from OBS."""
        if not self._connected or self._client is None:
            return

        def _fetch_sync():
            scene_list = self._client.get_scene_list()
            current = self._client.get_current_program_scene()
            return scene_list, current

        try:
            scene_list_resp, current_resp = await asyncio.get_event_loop().run_in_executor(
                None, _fetch_sync
            )
            self._available_scenes = [
                s.get("sceneName", "") for s in (scene_list_resp.scenes or [])
            ]
            self._current_scene = current_resp.current_program_scene_name or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("OBSDirector: failed to refresh scene list: %s", exc)

    async def switch_scene(self, scene_name: str) -> bool:
        """
        Switch OBS to the named scene.

        Returns True if the switch was performed, False otherwise.
        """
        if not self._connected or self._client is None:
            logger.debug("OBSDirector (no-op): would switch to scene '%s'", scene_name)
            return False

        if scene_name not in self._available_scenes:
            logger.debug(
                "OBSDirector: scene '%s' not in available scenes %s — skipping",
                scene_name,
                self._available_scenes,
            )
            return False

        if scene_name == self._current_scene:
            logger.debug("OBSDirector: already on scene '%s'", scene_name)
            return False

        now = time.monotonic()
        if now - self._last_switch_time < self._min_switch_interval:
            logger.debug(
                "OBSDirector: switch rate-limited (%.1fs since last switch)",
                now - self._last_switch_time,
            )
            return False

        def _switch_sync():
            self._client.set_current_program_scene(name=scene_name)

        try:
            await asyncio.get_event_loop().run_in_executor(None, _switch_sync)
            logger.info(
                "OBSDirector: switched scene '%s' → '%s'",
                self._current_scene,
                scene_name,
            )
            self._current_scene = scene_name
            self._last_switch_time = time.monotonic()
            return True

        except Exception as exc:  # noqa: BLE001
            logger.error("OBSDirector: scene switch failed: %s", exc)
            return False

    async def set_source_visible(self, source: str, visible: bool) -> bool:
        """
        Toggle a scene item (source) visible/hidden in the current scene.

        Returns True if the operation succeeded.
        """
        if not self._connected or self._client is None:
            logger.debug(
                "OBSDirector (no-op): would set source '%s' visible=%s", source, visible
            )
            return False

        def _toggle_sync():
            item_list = self._client.get_scene_item_list(scene_name=self._current_scene)
            for item in item_list.scene_items:
                if item.get("sourceName") == source:
                    item_id = item.get("sceneItemId")
                    self._client.set_scene_item_enabled(
                        scene_name=self._current_scene,
                        scene_item_id=item_id,
                        scene_item_enabled=visible,
                    )
                    return True
            return False

        try:
            found = await asyncio.get_event_loop().run_in_executor(None, _toggle_sync)
            if found:
                logger.info(
                    "OBSDirector: source '%s' set visible=%s in scene '%s'",
                    source,
                    visible,
                    self._current_scene,
                )
            else:
                logger.debug(
                    "OBSDirector: source '%s' not found in scene '%s'",
                    source,
                    self._current_scene,
                )
            return found

        except Exception as exc:  # noqa: BLE001
            logger.error("OBSDirector: set_source_visible failed: %s", exc)
            return False

    async def _llm_director_decision(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Ask the LLM whether to switch scenes given the current ambiguous state.

        Returns a dict: {should_switch, target_scene, confidence, reason}
        """
        user_message = _build_director_user_message(
            self._current_scene, self._available_scenes, state
        )
        messages = [{"role": "user", "content": user_message}]

        resp = await call_with_retry(
            lambda: self.llm.complete(
                system=DIRECTOR_SYSTEM_PROMPT,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        )

        log_llm_response("OBSDirector", resp)

        fallback: dict[str, Any] = {
            "should_switch": False,
            "target_scene": "",
            "confidence": 0.0,
            "reason": "Parse error",
        }

        try:
            cleaned = resp.text.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(
                    l for l in cleaned.splitlines() if not l.strip().startswith("```")
                )
            data = json.loads(cleaned)
            return {
                "should_switch": bool(data.get("should_switch", False)),
                "target_scene": str(data.get("target_scene", "")),
                "confidence": float(data.get("confidence", 0.0)),
                "reason": str(data.get("reason", "")),
            }
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning(
                "OBSDirector: failed to parse LLM JSON: %s | raw=%r",
                exc,
                resp.text[:200],
            )
            return fallback

    async def on_game_state(self, state: dict[str, Any]) -> None:
        """
        React to a game state update from the orchestrator.

        Handles deterministic routing for known states, and falls back to
        LLM-powered director decisions for ambiguous states.

        Args:
            state: Dict from PixelAgent.get_latest_context() plus any extra
                   bridge data. Expected keys:
                   - player_state (str)
                   - scene_description (str)
                   - threats (list)
                   - notable_items (list)
        """
        await self._refresh_scene_list()

        player_state = state.get("player_state", "exploring")
        scene_desc = state.get("scene_description", "")

        # --- Low health alert (check first regardless of scene) ---
        scene_lower = scene_desc.lower()
        if any(kw in scene_lower for kw in LOW_HEALTH_KEYWORDS):
            await self.set_source_visible("Low Health Alert", True)
        else:
            await self.set_source_visible("Low Health Alert", False)

        # --- Source visibility for inventory ---
        if player_state in STATE_OVERLAYS:
            for source_name, visibility in STATE_OVERLAYS[player_state].items():
                await self.set_source_visible(source_name, visibility)

        # --- Deterministic scene routing ---
        if player_state in DETERMINISTIC_ROUTES:
            target = DETERMINISTIC_ROUTES[player_state]
            await self.switch_scene(target)
            return

        # --- Ambiguous state: ask LLM ---
        logger.debug(
            "OBSDirector: ambiguous state '%s', consulting LLM", player_state
        )
        try:
            decision = await self._llm_director_decision(state)
            logger.debug(
                "OBSDirector LLM decision: switch=%s target=%r confidence=%.2f reason=%s",
                decision["should_switch"],
                decision["target_scene"],
                decision["confidence"],
                decision["reason"],
            )

            if (
                decision["should_switch"]
                and decision["confidence"] >= self.switch_threshold
                and decision["target_scene"]
            ):
                await self.switch_scene(decision["target_scene"])
            else:
                logger.debug(
                    "OBSDirector: staying on current scene (confidence %.2f < threshold %.2f "
                    "or should_switch=False)",
                    decision["confidence"],
                    self.switch_threshold,
                )

        except Exception as exc:  # noqa: BLE001
            logger.error("OBSDirector: LLM director decision failed: %s", exc)
            # Fail safely — don't touch OBS on error
