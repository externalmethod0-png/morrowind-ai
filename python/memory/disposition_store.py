"""
disposition_store.py — Per-NPC opinion vector, mood residue, life facts.
Extracted verbatim from chroma_memory.py so it can be used without importing
chromadb (which is not installed in this lightweight setup).
"""

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DispositionStore — per-NPC opinion vector, mood residue, life facts
# ---------------------------------------------------------------------------
#
# Stored as a single sidecar JSON file next to the Chroma directory. Chroma is
# built for searchable turn history; a single-row upsert per NPC is cheaper as
# plain JSON and keeps the feature auditable (tail the file to see state).
#
# Shape:
#   {
#     "npc_id_or_safe_name": {
#       "disposition": -100..100,   # cumulative opinion, 0 = neutral
#       "last_mood":  "happy",      # emotion tag from most recent turn
#       "last_seen":  "2026-04-17T18:30:00+00:00",
#       "life_facts": ["...", "...", "..."],   # 3 short non-plot beats
#     },
#     ...
#   }

# Heuristic deltas — derived from the emotion+action the LoreAgent already
# returns. Keeps the LLM schema unchanged and the rule auditable.
EMOTION_DELTA: dict[str, float] = {
    "happy":     +2.0,
    "surprised": +0.5,
    "neutral":    0.0,
    "fearful":   -0.5,
    "disgusted": -1.5,
    "angry":     -2.5,
}
ACTION_DELTA: dict[str, float] = {
    "none":    0.0,
    "follow": +3.0,
    "trade":  +1.0,
    "flee":   -1.0,
    "attack": -10.0,
}

# Decay toward zero, measured in points per real-time hour since last_seen.
# A quarrel three days ago fades but doesn't fully reset.
DECAY_PER_HOUR: float = 0.5


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class DispositionStore:
    """
    Lightweight sidecar store for opinion vectors, mood residue, and life facts.

    Thread-safe via a single coarse lock (writes are rare). Both lore_agent and
    d2d_agent may touch the same file, hence atomic replace + lock.
    """

    DISPOSITION_MIN: float = -100.0
    DISPOSITION_MAX: float = +100.0

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = self._load()
        logger.info("DispositionStore initialised at %s (%d npcs)",
                    self.path, len(self._cache))

    # ---------------------------------------------------------- persistence

    def _load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("DispositionStore: could not load %s (%s) — starting empty",
                           self.path, exc)
        return {}

    def _save_locked(self) -> None:
        try:
            _atomic_write_json(self.path, self._cache)
        except OSError as exc:
            logger.error("DispositionStore: atomic write failed: %s", exc)

    # -------------------------------------------------------------- access

    def _entry_locked(self, npc_id: str) -> dict:
        if npc_id not in self._cache:
            self._cache[npc_id] = {
                "disposition": 0.0,
                "last_mood": "neutral",
                "last_seen": None,
                "life_facts": [],
            }
        return self._cache[npc_id]

    def _apply_decay_locked(self, entry: dict) -> None:
        """Move disposition toward zero based on wall-clock since last_seen."""
        last_seen = entry.get("last_seen")
        if not last_seen:
            return
        try:
            then = datetime.fromisoformat(last_seen)
        except ValueError:
            return
        now = datetime.now(timezone.utc)
        hours = max(0.0, (now - then).total_seconds() / 3600.0)
        if hours <= 0:
            return
        decay = DECAY_PER_HOUR * hours
        d = entry.get("disposition", 0.0)
        if d > 0:
            entry["disposition"] = max(0.0, d - decay)
        elif d < 0:
            entry["disposition"] = min(0.0, d + decay)

    def get(self, npc_id: str) -> dict:
        """Return a snapshot of the NPC's disposition state (with passive decay)."""
        with self._lock:
            entry = self._entry_locked(npc_id)
            self._apply_decay_locked(entry)
            return {
                "disposition": float(entry.get("disposition", 0.0)),
                "last_mood":   entry.get("last_mood", "neutral"),
                "last_seen":   entry.get("last_seen"),
                "life_facts":  list(entry.get("life_facts", [])),
            }

    def apply_turn(self, npc_id: str, emotion: str, action: str) -> dict:
        """
        Update disposition + mood after a player↔NPC turn.

        Returns the post-update snapshot.
        """
        delta = EMOTION_DELTA.get(emotion, 0.0) + ACTION_DELTA.get(action, 0.0)
        with self._lock:
            entry = self._entry_locked(npc_id)
            self._apply_decay_locked(entry)
            entry["disposition"] = _clamp(
                entry.get("disposition", 0.0) + delta,
                self.DISPOSITION_MIN, self.DISPOSITION_MAX,
            )
            entry["last_mood"] = emotion or "neutral"
            entry["last_seen"] = datetime.now(timezone.utc).isoformat()
            self._save_locked()
            snap = dict(entry)
        logger.debug("DispositionStore: %s += %.1f → %.1f (mood=%s)",
                     npc_id, delta, snap["disposition"], snap["last_mood"])
        return snap

    def set_life_facts(self, npc_id: str, facts: list[str]) -> None:
        """Persist 3 one-line life facts for an NPC. Idempotent."""
        facts = [f.strip() for f in facts if f and f.strip()]
        if not facts:
            return
        with self._lock:
            entry = self._entry_locked(npc_id)
            entry["life_facts"] = facts[:5]
            self._save_locked()
        logger.info("DispositionStore: stored %d life facts for %s",
                    len(facts), npc_id)

    @staticmethod
    def disposition_band(value: float) -> str:
        """Convert a disposition number into a short prompt-ready description."""
        if value <= -60:  return "This NPC despises the player. Cold hostility colours every word."
        if value <= -20:  return "This NPC distrusts the player. Guarded, terse, unfriendly."
        if value <   20:  return "This NPC is neutral toward the player."
        if value <   60:  return "This NPC is warmly disposed toward the player."
        return "This NPC is devoted to the player, within reason for their character."
