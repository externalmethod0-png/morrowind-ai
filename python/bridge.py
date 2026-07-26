"""
bridge.py — IPC file-watcher bridge between OpenMW Lua scripts and Python agents.

Polls ipc/request.json every 100 ms. On receipt:
  1. Reads + immediately clears the request file (prevents double-processing).
  2. Routes by request type:
       "dialogue"  → lore_agent generates NPC reply, stored in memory
       "npc_npc"   → generates inter-NPC dialogue, written to events/ dir
  3. Writes response to ipc/response.json.
  4. Logs each exchange to logs/dialogue.log.
"""

import asyncio
import json
import logging
import os
import pathlib
import tempfile
import time
import uuid
from typing import Optional

import aiofiles

logger = logging.getLogger(__name__)

_BASE = pathlib.Path("/home/nemoclaw/morrowind-ai")
_IPC_DIR = pathlib.Path("/home/nemoclaw/morrowind-ai/ipc")
_REQUEST_FILE = _IPC_DIR / "request.json"
_RESPONSE_FILE = _IPC_DIR / "response.json"
_EVENTS_DIR = _IPC_DIR / "events"
_DIALOGUE_LOG = _BASE / "logs" / "dialogue.log"


def _atomic_write(path: pathlib.Path, data: dict) -> None:
    """Write JSON atomically via tmp+replace so Lua readers see complete files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class IPCBridge:
    """
    Polls ipc/request.json and dispatches to the appropriate agent.

    Args:
        config:      Parsed config.yaml dict.
        lore_agent:  Agent with an async generate(npc_id, history, player_text, location) method.
        memory:      NPCMemory instance for storing and retrieving exchanges.
        pixel_agent: Agent with context the bridge may query (currently unused in routing).
    """

    def __init__(self, config: dict, lore_agent, memory, pixel_agent) -> None:
        self.config = config
        self.lore_agent = lore_agent
        self.memory = memory
        self.pixel_agent = pixel_agent

        ipc_cfg = config.get("ipc", {})
        self._poll_interval: float = ipc_cfg.get("poll_interval_ms", 100) / 1000.0

        _IPC_DIR.mkdir(parents=True, exist_ok=True)
        _EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        _DIALOGUE_LOG.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "IPCBridge ready (poll=%.3fs, request=%s)", self._poll_interval, _REQUEST_FILE
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Poll request.json continuously until cancelled."""
        logger.info("IPCBridge polling started")
        while True:
            try:
                request = self._read_request()
                if request is not None:
                    await self._handle_request(request)
            except asyncio.CancelledError:
                logger.info("IPCBridge shutting down")
                return
            except Exception as exc:  # noqa: BLE001
                logger.error("IPCBridge loop error: %s", exc, exc_info=True)
            await asyncio.sleep(self._poll_interval)

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    def _read_request(self) -> Optional[dict]:
        """
        Read request.json if it exists and is non-empty.
        Immediately truncates the file after reading to prevent double-processing.
        Returns None if the file is absent, empty, or unreadable.
        """
        if not _REQUEST_FILE.exists():
            return None

        try:
            raw = _REQUEST_FILE.read_text(encoding="utf-8").strip()
            if not raw:
                return None

            data = json.loads(raw)

            # Truncate immediately — do this before any async work so a
            # concurrent poll can't pick up the same request.
            _REQUEST_FILE.write_text("", encoding="utf-8")
            return data

        except json.JSONDecodeError as exc:
            logger.warning("IPCBridge: malformed request.json: %s", exc)
            # Clear corrupted file
            try:
                _REQUEST_FILE.write_text("", encoding="utf-8")
            except OSError:
                pass
            return None
        except OSError as exc:
            logger.warning("IPCBridge: could not read request.json: %s", exc)
            return None

    async def _handle_request(self, request: dict) -> None:
        """Route an incoming request to the correct handler."""
        req_type = request.get("type", "dialogue")
        logger.info("IPCBridge: handling request type='%s'", req_type)

        if req_type == "dialogue":
            await self._handle_dialogue(request)
        elif req_type == "npc_npc":
            await self._handle_npc_npc(request)
        else:
            logger.warning("IPCBridge: unknown request type '%s', ignoring", req_type)
            await self._write_response(
                {"error": f"Unknown request type '{req_type}'", "type": req_type}
            )

    async def _handle_dialogue(self, request: dict) -> None:
        """
        Player↔NPC dialogue:
          1. Retrieve conversation history from memory.
          2. Ask lore_agent to generate a response.
          3. Store the exchange.
          4. Write response.json.
          5. Append to dialogue.log.
        """
        npc_id = request.get("npc_id", "unknown_npc")
        player_text = request.get("player_text", "")
        location = request.get("location", "unknown")

        history = self.memory.get_history(
            npc_id, limit=self.config.get("memory", {}).get("history_limit", 10)
        )

        try:
            agent_request = {
                "npc_id": npc_id,
                "npc_name": request.get("npc_name", npc_id),
                "npc_race": request.get("npc_race", "Dunmer"),
                "npc_class": request.get("npc_class", "Commoner"),
                "npc_faction": request.get("npc_faction", ""),
                "player_input": player_text,
                "location": location,
                "conversation_history": history,
            }
            result = await self.lore_agent.generate_response(
                agent_request, memory_context=history
            )
            npc_response = result.get("response", "...") if isinstance(result, dict) else str(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("lore_agent.generate_response failed: %s", exc, exc_info=True)
            npc_response = "..."  # Fallback — NPC stays silent rather than crashing

        # Persist the exchange
        self.memory.store_exchange(npc_id, player_text, npc_response, location)

        response = {
            "type": "dialogue",
            "npc_id": npc_id,
            "npc_response": npc_response,
            "location": location,
        }
        await self._write_response(response)
        await self._log_dialogue(npc_id, player_text, npc_response, location)

    async def _handle_npc_npc(self, request: dict) -> None:
        """
        Inter-NPC dialogue:
          Generate a conversation snippet between two NPCs using contrasting
          system prompts, then write it to ipc/events/npc_dialogue_{uuid}.json
          for the Lua mod to pick up and display.
        """
        npc_a_id = request.get("npc_a_id", "npc_a")
        npc_b_id = request.get("npc_b_id", "npc_b")
        topic = request.get("topic", "the weather")
        location = request.get("location", "unknown")

        try:
            # Two contrasting personas: one formal, one blunt
            prompt_a = (
                f"You are {npc_a_id}, a formal and learned scholar of Morrowind. "
                f"Speak with gravitas about: {topic}."
            )
            prompt_b = (
                f"You are {npc_b_id}, a blunt, sceptical merchant of Morrowind. "
                f"React tersely to your companion's remark about: {topic}."
            )

            line_a = await self.lore_agent.generate_with_system(
                system_prompt=prompt_a,
                user_text=f"Say something to {npc_b_id} about {topic}.",
            )
            line_b = await self.lore_agent.generate_with_system(
                system_prompt=prompt_b,
                user_text=line_a,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("npc_npc generation failed: %s", exc, exc_info=True)
            line_a = f"{npc_a_id} mutters something unintelligible."
            line_b = f"{npc_b_id} nods absently."

        event = {
            "type": "npc_npc",
            "npc_a_id": npc_a_id,
            "npc_b_id": npc_b_id,
            "npc_a_line": line_a,
            "npc_b_line": line_b,
            "location": location,
            "timestamp": _now_iso(),
        }

        event_filename = _EVENTS_DIR / f"npc_dialogue_{uuid.uuid4().hex}.json"
        try:
            _atomic_write(event_filename, event)
            logger.info("npc_npc event written: %s", event_filename.name)
        except OSError as exc:
            logger.error("Could not write npc_npc event file: %s", exc)

        # Also acknowledge via response.json
        await self._write_response(
            {"type": "npc_npc", "status": "ok", "event_file": event_filename.name}
        )

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    async def _write_response(self, response: dict) -> None:
        """Atomically write a response dict to ipc/response.json."""
        response.setdefault("timestamp", _now_iso())
        try:
            _atomic_write(_RESPONSE_FILE, response)
            logger.debug("IPCBridge: response written -> %s", _RESPONSE_FILE)
        except OSError as exc:
            logger.error("IPCBridge: could not write response.json: %s", exc)

    async def _log_dialogue(
        self,
        npc_id: str,
        player_text: str,
        npc_response: str,
        location: str,
    ) -> None:
        """Append a dialogue record to logs/dialogue.log."""
        ts = _now_iso()
        line = (
            f"[{ts}] NPC={npc_id} LOC={location}\n"
            f"  PLAYER: {player_text}\n"
            f"  NPC:    {npc_response}\n"
            "---\n"
        )
        try:
            async with aiofiles.open(_DIALOGUE_LOG, "a", encoding="utf-8") as fh:
                await fh.write(line)
        except OSError as exc:
            logger.warning("Could not write dialogue.log: %s", exc)


def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
