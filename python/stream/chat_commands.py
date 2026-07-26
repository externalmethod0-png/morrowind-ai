"""
chat_commands.py — Parse and route viewer chat commands to game world IPC events.

Commands write JSON files to ~/morrowind-ai/ipc/events/{uuid}.json which the
OpenMW Lua mod polls and consumes.
"""

import asyncio
import html
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_EVENTS_DIR = '/home/nemoclaw/morrowind-ai/ipc/events'
_LOGS_DIR = '/home/nemoclaw/morrowind-ai/logs'

# ---------------------------------------------------------------------------
# Lore item mapping: common viewer words -> Morrowind item IDs
# ---------------------------------------------------------------------------
ITEM_MAP: dict[str, str] = {
    "gold":            "gold_001",
    "coin":            "gold_001",
    "coins":           "gold_001",
    "potion":          "p_restore_health_s",
    "health potion":   "p_restore_health_s",
    "restore potion":  "p_restore_health_s",
    "stamina potion":  "p_restore_fatigue_s",
    "mana potion":     "p_restore_magicka_s",
    "scroll":          "sc_almsivi_intervention",
    "intervention":    "sc_almsivi_intervention",
    "arrow":           "w_iron_arrow",
    "arrows":          "w_iron_arrow",
    "silver arrow":    "w_silver_arrow",
    "bread":           "food_bread_01",
    "apple":           "ingred_apple_01",
    "torch":           "torch_01",
    "candle":          "candle_01",
    "kwama egg":       "ingred_kwama_egg_01",
    "saltrice":        "ingred_saltrice_01",
    "comberry":        "ingred_comberry_01",
    "wickwheat":       "ingred_wickwheat_01",
    "moon sugar":      "ingred_moon_sugar_01",
    "skooma":          "potion_skooma_01",
    "lockpick":        "misc_de_goblet_01_lockpick",  # vanilla lockpick ID
    "probe":           "misc_probe_steel_01",
    "map":             "misc_map_01",
    "lantern":         "misc_lantern_01",
}

# ---------------------------------------------------------------------------
# Creature mapping: viewer names -> Morrowind creature IDs
# ---------------------------------------------------------------------------
CREATURE_MAP: dict[str, str] = {
    "scamp":        "scamp",
    "rat":          "rat",
    "mudcrab":      "mudcrab",
    "kwama":        "kwama_worker",
    "kwama worker": "kwama_worker",
    "cliff racer":  "cliff_racer",
    "cliftracer":   "cliff_racer",
    "cliff_racer":  "cliff_racer",
    "netch":        "betty_netch",
    "betty netch":  "betty_netch",
    "guar":         "guar",
    "pack guar":    "pack_guar",
    "alit":         "alit",
    "shalk":        "shalk",
    "dreugh":       "dreugh",
    "slaughterfish":"slaughterfish",
}

# Safe creature whitelist (must be a key in CREATURE_MAP)
CREATURE_WHITELIST = frozenset(CREATURE_MAP.keys())

# ---------------------------------------------------------------------------
# Simple profanity blocklist — add more as needed
# ---------------------------------------------------------------------------
_PROFANITY = frozenset({
    "fuck", "shit", "ass", "bitch", "cunt", "dick", "cock", "pussy",
    "nigger", "nigga", "faggot", "retard", "whore", "slut",
})

# ---------------------------------------------------------------------------
# Gemini import (optional dependency)
# ---------------------------------------------------------------------------
try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    _GEMINI_AVAILABLE = False


def _sanitize_text(text: str, max_len: int = 100) -> str:
    """Strip HTML, truncate, and block obvious profanity."""
    # Unescape then re-escape to normalize any HTML entities
    text = html.unescape(text)
    # Strip any residual HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Collapse whitespace
    text = ' '.join(text.split())
    # Truncate
    if len(text) > max_len:
        text = text[:max_len].rstrip() + '...'
    # Profanity filter — replace whole words only
    words = text.split()
    cleaned = [
        '***' if w.lower().strip('.,!?;:"\'') in _PROFANITY else w
        for w in words
    ]
    return ' '.join(cleaned)


def _write_event(event: dict) -> str:
    """Write an IPC event JSON to ipc/events/{uuid}.json. Returns the path."""
    os.makedirs(_EVENTS_DIR, exist_ok=True)
    event_id = str(uuid.uuid4())
    path = os.path.join(_EVENTS_DIR, f"{event_id}.json")
    # Stamp the event with an id and wall time
    event.setdefault('event_id', event_id)
    event.setdefault('written_at', datetime.now(timezone.utc).isoformat())
    try:
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(event, fh, indent=2)
        logger.debug("event written: %s", path)
    except OSError as exc:
        logger.error("_write_event: failed to write %s: %s", path, exc)
    return path


class ChatCommandHandler:
    """
    Parse !command messages from YouTube chat and dispatch IPC events.

    Cooldowns are tracked in-memory (reset on restart).
    Session bounty total is capped at BOUNTY_SESSION_CAP gold.
    """

    BOUNTY_MAX_PER_CMD   = 1000   # gold
    BOUNTY_SESSION_CAP   = 5000   # gold total across session
    SAY_MAX_LEN          = 100    # characters
    QUEST_MAX_LEN        = 200    # characters for raw fallback

    # Per-user cooldowns in seconds
    COOLDOWN = {
        'bounty':   10,
        'generate': 15,
        'spawn':    30,
        'quest':    60,
        'say':       5,
    }

    def __init__(self, config: dict):
        self.config = config
        self._bounty_session_total: int = 0
        # {cmd_name: {username: last_used_timestamp}}
        self._cooldowns: dict[str, dict[str, float]] = {k: {} for k in self.COOLDOWN}

        # Gemini setup (optional)
        self._gemini_model = None
        if _GEMINI_AVAILABLE:
            api_key = config.get('gemini_api_key') or os.environ.get('GEMINI_API_KEY')
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    self._gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                    logger.info("Gemini model loaded for !quest sanitization")
                except Exception as exc:
                    logger.warning("Gemini init failed: %s — falling back to raw text", exc)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def handle_message(self, author: str, message: str, timestamp: str) -> None:
        """Entry point called by YouTubeChatListener for every chat message."""
        try:
            stripped = message.strip()
            if not stripped.startswith('!'):
                return
            parts = stripped[1:].split(None, 1)  # ['command', 'rest...']
            if not parts:
                return
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ''

            dispatch = {
                'bounty':   self._cmd_bounty,
                'generate': self._cmd_generate,
                'spawn':    self._cmd_spawn,
                'quest':    self._cmd_quest,
                'say':      self._cmd_say,
            }
            handler = dispatch.get(cmd)
            if handler is None:
                return  # Unknown command — silently ignore

            # Cooldown check
            if not self._check_cooldown(cmd, author):
                logger.debug("!%s from %s blocked by cooldown", cmd, author)
                return

            await handler(author=author, arg=arg, timestamp=timestamp)
            self._mark_cooldown(cmd, author)

        except Exception as exc:
            logger.error("handle_message error (author=%s msg=%r): %s", author, message, exc)

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _cmd_bounty(self, author: str, arg: str, timestamp: str) -> None:
        """!bounty <amount> — add bounty to the player character."""
        try:
            amount = int(re.sub(r'[^\d]', '', arg.split()[0]) if arg.split() else '0')
        except (ValueError, IndexError):
            logger.debug("!bounty: bad amount from %s: %r", author, arg)
            return

        if amount <= 0:
            return

        amount = min(amount, self.BOUNTY_MAX_PER_CMD)

        remaining = self.BOUNTY_SESSION_CAP - self._bounty_session_total
        if remaining <= 0:
            logger.info("!bounty: session cap reached, ignoring from %s", author)
            return
        amount = min(amount, remaining)

        self._bounty_session_total += amount
        event = {
            'type':      'bounty',
            'amount':    amount,
            'author':    author,
            'timestamp': timestamp,
            'session_total': self._bounty_session_total,
        }
        _write_event(event)
        logger.info("!bounty: %d gold from %s (session total: %d)", amount, author, self._bounty_session_total)

    async def _cmd_generate(self, author: str, arg: str, timestamp: str) -> None:
        """!generate <item_name> — drop a lore-appropriate item near player."""
        if not arg.strip():
            return
        item_key = arg.strip().lower()
        item_id = ITEM_MAP.get(item_key)

        if item_id is None:
            # Try partial match on any key that starts with the input
            for map_key, map_id in ITEM_MAP.items():
                if item_key in map_key or map_key in item_key:
                    item_id = map_id
                    break

        if item_id is None:
            logger.debug("!generate: unknown item %r from %s", item_key, author)
            return

        event = {
            'type':      'drop_item',
            'item':      item_id,
            'quantity':  1,
            'author':    author,
            'timestamp': timestamp,
        }
        _write_event(event)
        logger.info("!generate: %s -> %s for %s", item_key, item_id, author)

    async def _cmd_spawn(self, author: str, arg: str, timestamp: str) -> None:
        """!spawn <creature> — spawn an enemy near the player."""
        if not arg.strip():
            return
        creature_key = arg.strip().lower()
        creature_id = CREATURE_MAP.get(creature_key)

        if creature_id is None:
            # Partial match
            for map_key, map_id in CREATURE_MAP.items():
                if creature_key in map_key or map_key in creature_key:
                    creature_id = map_id
                    break

        if creature_id is None:
            logger.debug("!spawn: unknown creature %r from %s", creature_key, author)
            return

        event = {
            'type':      'spawn_enemy',
            'creature':  creature_id,
            'author':    author,
            'timestamp': timestamp,
        }
        _write_event(event)
        logger.info("!spawn: %s -> %s for %s", creature_key, creature_id, author)

    async def _cmd_quest(self, author: str, arg: str, timestamp: str) -> None:
        """!quest <description> — add a viewer-authored journal entry."""
        if not arg.strip():
            return

        raw_text = arg.strip()[:self.QUEST_MAX_LEN]
        lore_text = await self._loreify_quest(raw_text, author)

        event = {
            'type':      'journal_update',
            'text':      lore_text,
            'author':    author,
            'timestamp': timestamp,
        }
        _write_event(event)
        logger.info("!quest from %s: %r", author, lore_text[:80])

    async def _cmd_say(self, author: str, arg: str, timestamp: str) -> None:
        """!say <text> — display a message box in-game."""
        if not arg.strip():
            return
        sanitized = _sanitize_text(arg, max_len=self.SAY_MAX_LEN)
        if not sanitized:
            return

        display_text = f"Chat says: {sanitized}"
        event = {
            'type':      'message',
            'text':      display_text,
            'author':    author,
            'timestamp': timestamp,
        }
        _write_event(event)
        logger.info("!say from %s: %r", author, display_text)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_cooldown(self, cmd: str, author: str) -> bool:
        """Returns True if the user is allowed to run this command now."""
        now = time.monotonic()
        last = self._cooldowns[cmd].get(author, 0.0)
        return (now - last) >= self.COOLDOWN[cmd]

    def _mark_cooldown(self, cmd: str, author: str) -> None:
        self._cooldowns[cmd][author] = time.monotonic()

    async def _loreify_quest(self, raw_text: str, author: str) -> str:
        """
        Use Gemini to rewrite the text in Morrowind journal style.
        Falls back to cleaned raw text if Gemini is unavailable or fails.
        """
        if self._gemini_model is not None:
            prompt = (
                "You are a Morrowind journal writer. Rewrite the following viewer suggestion "
                "as a single short journal entry in the style of The Elder Scrolls III: Morrowind. "
                "Keep it under 150 words. Keep lore-appropriate: no modern references. "
                "Viewer name for attribution: " + author + "\n\n"
                "Suggestion: " + raw_text
            )
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self._gemini_model.generate_content(prompt)
                )
                lore_text = response.text.strip()
                if lore_text:
                    return lore_text[:300]
            except Exception as exc:
                logger.warning("Gemini quest loreify failed: %s — using raw text", exc)

        # Fallback: sanitize and use raw text
        return _sanitize_text(raw_text, max_len=self.QUEST_MAX_LEN)
