"""
json_memory.py — Lightweight JSON-backed NPC memory (drop-in for NPCMemory).

Why this exists:
  The upstream chroma_memory.NPCMemory pulls ChromaDB + onnxruntime and, on the
  first store_exchange(), downloads an ~80MB embedding model. For this install we
  only need recency-based conversation history (which is exactly what LoreAgent
  consumes), so we back it with a single JSON file. No vector search, no model
  download, no native deps — but full functional parity for the dialogue loop.

Interface matches chroma_memory.NPCMemory:
    store_exchange(npc_id, player_text, npc_response, location, emotion, action)
    get_history(npc_id, limit)  -> list[turn dicts newest-last]
    get_npc_summary(npc_id)     -> str
    clear_npc(npc_id)

get_history returns turn-level dicts so LoreAgent's RECENT CONVERSATION block
(which reads turn['role'] / turn['content']) renders correctly:
    [{"role": "player", "content": "..."}, {"role": "npc", "content": "..."}, ...]
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class NPCMemory:
    """Persistent per-NPC conversation history backed by one JSON file."""

    def __init__(self, persist_dir: str) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.persist_dir / "npc_history.json"
        self._lock = threading.Lock()
        self._data: dict[str, list[dict]] = self._load()
        logger.info("NPCMemory (JSON) initialised at %s (%d npcs)",
                    self.path, len(self._data))

    def _load(self) -> dict[str, list[dict]]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("NPCMemory: could not load %s (%s) — starting empty",
                           self.path, exc)
        return {}

    def _save_locked(self) -> None:
        try:
            _atomic_write_json(self.path, self._data)
        except OSError as exc:
            logger.error("NPCMemory: atomic write failed: %s", exc)

    def store_exchange(
        self,
        npc_id: str,
        player_text: str,
        npc_response: str,
        location: str,
        emotion: str = "neutral",
        action: str = "none",
    ) -> None:
        record = {
            "player_text": player_text,
            "npc_response": npc_response,
            "location": location,
            "emotion": emotion,
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._data.setdefault(npc_id, []).append(record)
            # Keep the tail bounded — we only ever read the last handful.
            if len(self._data[npc_id]) > 200:
                self._data[npc_id] = self._data[npc_id][-200:]
            self._save_locked()
        logger.debug("Stored exchange for NPC '%s' at %s", npc_id, location)

    def get_history(self, npc_id: str, limit: int = 10) -> list[dict]:
        """
        Return the most recent conversation turns for this NPC, newest-last.

        Each stored exchange expands to up to two turns (player, then npc);
        empty player turns (auto-greetings) are skipped.
        """
        with self._lock:
            records = list(self._data.get(npc_id, []))
        turns: list[dict] = []
        for rec in records:
            pt = (rec.get("player_text") or "").strip()
            nr = (rec.get("npc_response") or "").strip()
            ts = rec.get("timestamp", "")
            loc = rec.get("location", "")
            if pt:
                turns.append({"role": "player", "content": pt,
                              "location": loc, "timestamp": ts})
            if nr:
                turns.append({"role": "npc", "content": nr,
                              "location": loc, "timestamp": ts})
        return turns[-limit:]

    def get_npc_summary(self, npc_id: str) -> str:
        with self._lock:
            records = list(self._data.get(npc_id, []))
        if not records:
            return f"No prior interactions recorded for NPC '{npc_id}'."
        lines = [
            f"[{r.get('timestamp','?')} @ {r.get('location','?')}] "
            f"Player: {r.get('player_text','')} | NPC: {r.get('npc_response','')}"
            for r in records[-5:]
        ]
        return (f"Recent exchanges with NPC '{npc_id}' "
                f"({len(records)} total):\n" + "\n".join(lines))

    def clear_npc(self, npc_id: str) -> None:
        with self._lock:
            if npc_id in self._data:
                del self._data[npc_id]
                self._save_locked()
        logger.info("Cleared memory for NPC '%s'", npc_id)
