"""
openmw_log_bridge.py — Windows-compatible IPC bridge for OpenMW 0.49.

Why this exists:
  The OpenMW 0.49 Lua sandbox on Windows does NOT expose `io` or a writeable
  `os`. Global scripts can't touch arbitrary files. So we use:

    Lua  -> Python : print('[MWAI_REQ] <json>')   (tagged line in openmw.log)
    Python -> Lua  : write C:\\morrowind-ai-mod\\ai_inbox\\response.txt, which
                     the Lua VFS can read because C:\\morrowind-ai-mod is a
                     data= path in openmw.cfg.

This script tails openmw.log, dispatches tagged lines to the existing
lore_agent + memory, and atomically overwrites response.txt with a new
`req_id` so the Lua side can dedup.

Run as a standalone process (do NOT restart mw-bridge; the WSL IPC path is
owned by another agent). This is an additive alternative path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pathlib
import queue
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# WSL-visible paths for Windows OpenMW install
OPENMW_LOG = pathlib.Path(
    "/mnt/c/Users/rneeb/Documents/My Games/OpenMW/openmw.log"
)
MOD_ROOT        = pathlib.Path("/mnt/c/morrowind-ai-mod")
INBOX_DIR       = MOD_ROOT / "ai_inbox"
RESPONSE_FILE   = INBOX_DIR / "response.txt"
JOURNAL_FILE    = INBOX_DIR / "responses.ndjson"   # append-only reply journal
NPC_SPEECH_FILE = INBOX_DIR / "npc_speech.txt"
PLAYER_TEXT_FILE = INBOX_DIR / "player_text.txt"  # written by chat_window_vfs.py

# Lua tag prefixes emitted via print()
REQ_RE = re.compile(r"\[MWAI_REQ\]\s+(\{.*\})\s*$")


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    """Write then rename so the Lua VFS never sees a half-flushed file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Reply journal (NDJSON)
# ---------------------------------------------------------------------------
# Every reply channel (dialogue, narrate, companion, bystander, voice) used to
# overwrite ONE response.txt slot. Delayed lines (companion +2s, bystander
# +3.5s) raced with it: replies were lost, arrived out of order, or executed
# against whatever NPC happened to be locked later. Now each reply is APPENDED
# as one line with a monotonic `seq`; the Lua side keeps the last seq it has
# seen and drains everything newer in order. Rotation keeps the file small and
# is safe because consumers key off `seq`, not line position.

_seq_lock = threading.Lock()
_seq_counter = 0

JOURNAL_MAX_LINES = 400
JOURNAL_KEEP_LINES = 80


_reply_q: "queue.Queue[dict]" = queue.Queue()
REPLY_SPACING_S = 0.9   # long enough for the Lua side (polls at 0.2 s) to see each


# Сколько знаков реплики помещается в окно разговора, не уезжая за границу.
# Мерено по окну на 620 пикселей при кегле 17: четыре строки истории плюс
# текущая реплика. Это ПОСЛЕДНИЙ рубеж, а не средство укоротить речь: за
# краткость отвечают правила в промпте, здесь мы лишь не даём непослушной
# модели залить экран.
REPLY_CHARS = 400


def trim_reply(text: str, limit: int = REPLY_CHARS) -> str:
    """Обрезать длинную реплику ПО КОНЦУ ПРЕДЛОЖЕНИЯ, а не по счёту знаков.

    Обрыв на полуслове читается как поломка мода; законченная мысль, пусть и
    короче задуманной, — как немногословный характер.
    """
    s = " ".join(str(text or "").split())
    if len(s) <= limit:
        return s
    cut = s[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "),
              cut.rfind("… "), cut.rfind(" — "))
    if end >= limit // 3:                    # нашлось предложение приличной длины
        return cut[:end + 1].strip()
    space = cut.rfind(" ")
    return (cut[:space] if space > 0 else cut).rstrip(" ,;:—-") + "…"


# Позиция в openmw.log, с которой мост начнёт читать. Запоминается КАК МОЖНО
# РАНЬШЕ — в момент, когда мост объявляет себя готовым, а не когда доберётся до
# цикла чтения: между этими событиями поднимаются распознавание и озвучка, это
# добрый десяток секунд, и запросы игры в этом промежутке иначе теряются.
_log_start_pos: int | None = None


def mark_log_position() -> int:
    """Запомнить (один раз) конец лога и вернуть его."""
    global _log_start_pos
    if _log_start_pos is None:
        try:
            _log_start_pos = OPENMW_LOG.stat().st_size
        except OSError:
            _log_start_pos = 0
    return _log_start_pos


def publish_reply(reply: dict) -> None:
    """Queue a reply for delivery to the game.

    Delivery is ONE atomically-replaced slot file, not an appended journal:
    OpenMW's virtual file system serves a script the size the file had when the
    game started, so a growing file is read forever as its first line — replies
    existed on disk and never reached the player. A queue with spacing gives us
    what the journal was for (nothing lost, order kept) on a mechanism the
    engine actually re-reads.
    """
    global _seq_counter
    with _seq_lock:
        _seq_counter += 1
        reply["seq"] = _seq_counter
    _reply_q.put(reply)


SLOT_BYTES = 16384   # every reply occupies exactly this much — see _write_slot


def read_complete_lines(path: pathlib.Path, pos: int) -> tuple[list[str], int]:
    """Read whole lines only, and report where the last complete one ended.

    A dialogue request is one enormous line — the NPC's canon lines run to
    several thousand characters. Polling the log could easily catch such a line
    half-written: the fragment has no "[MWAI_REQ]" marker to match, the reader
    moved past it anyway, and the rest arrived without its beginning. The
    request vanished without a trace, and the player pressed H to silence.

    Reading in binary keeps the position exact; anything after the last
    newline stays unread until the game finishes writing it.
    """
    with path.open("rb") as fh:
        fh.seek(pos)
        data = fh.read()
    cut = data.rfind(b"\n")
    if cut == -1:
        return [], pos                      # ни одной дописанной строки
    text = data[:cut + 1].decode("utf-8", "replace")
    return text.splitlines(), pos + cut + 1


def _fit_slot(reply: dict) -> str:
    """Serialise a reply so it ALWAYS fits the slot exactly.

    Padding alone was not enough: a reply longer than the slot was written at
    its own length, the file grew, and we were back to the failure that kept
    NPCs silent for two days (the VFS serves the size the file had at game
    start). So an over-long reply is SHORTENED here — and the spoken line is
    what gives way, never the tags: losing a sentence is a blemish, losing
    ACTION or GOLD means the world stops matching what was said.
    """
    line = json.dumps(reply, ensure_ascii=False)
    if len(line.encode("utf-8")) <= SLOT_BYTES:
        return line

    trimmed = dict(reply)
    # Порядок жертв: сперва то, что можно прислать снова, потом сама реплика.
    for field in ("companion_arc", "life_facts", "rumor", "player_echo"):
        if len(json.dumps(trimmed, ensure_ascii=False).encode("utf-8")) <= SLOT_BYTES:
            break
        if trimmed.get(field):
            logger.warning("ответ не влезал в слот — убрано поле %s", field)
            trimmed[field] = [] if isinstance(trimmed[field], list) else ""

    text = str(trimmed.get("npc_response") or "")
    while len(json.dumps(trimmed, ensure_ascii=False).encode("utf-8")) > SLOT_BYTES:
        if len(text) <= 40:
            trimmed["npc_response"] = "(реплика не поместилась)"
            break
        text = text[: int(len(text) * 0.8)].rstrip() + "…"
        trimmed["npc_response"] = text
        logger.warning("реплика обрезана до %d символов — не влезала в слот", len(text))

    line = json.dumps(trimmed, ensure_ascii=False)
    if len(line.encode("utf-8")) > SLOT_BYTES:      # последний рубеж
        line = json.dumps({"req_id": reply.get("req_id"), "seq": reply.get("seq"),
                           "type": "dialogue", "npc_id": reply.get("npc_id", ""),
                           "npc_response": "(ответ слишком длинный)",
                           "emotion": "neutral", "action": "none", "target": "none",
                           "disp": 0, "gold": 0, "item": "none"},
                          ensure_ascii=False)
        logger.error("ответ пришлось заменить заглушкой — не поместился в слот")
    return line


def _write_slot(path: pathlib.Path, line: str) -> None:
    """Write text padded to a CONSTANT size.

    OpenMW's VFS hands a script the file size it saw when the game started, so
    a slot whose length changes per reply risks being served truncated. Padding
    removes that; the Lua side cuts at the closing brace before decoding.
    """
    blob = line.encode("utf-8")
    if len(blob) > SLOT_BYTES:
        # Сюда попадать нельзя: значит, кто-то обошёл _fit_slot.
        logger.error("слот переполнен на %d байт — обрезаю", len(blob) - SLOT_BYTES)
        line = blob[:SLOT_BYTES].decode("utf-8", "ignore")
        blob = line.encode("utf-8")
    if len(blob) < SLOT_BYTES:
        line += " " * (SLOT_BYTES - len(blob))
    _atomic_write_text(path, line)


def _reply_writer() -> None:
    while True:
        reply = _reply_q.get()
        line = _fit_slot(reply)
        try:
            _write_slot(RESPONSE_FILE, line)
        except OSError as exc:
            logger.error("could not write reply slot: %s", exc)
        # keep a human-readable trail for debugging (game never reads this)
        try:
            with JOURNAL_FILE.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass
        time.sleep(REPLY_SPACING_S)


threading.Thread(target=_reply_writer, daemon=True, name="reply-writer").start()


def _rotate_journal() -> None:
    try:
        if not JOURNAL_FILE.exists():
            return
        with JOURNAL_FILE.open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        if len(lines) <= JOURNAL_MAX_LINES:
            return
        _atomic_write_text(JOURNAL_FILE, "".join(lines[-JOURNAL_KEEP_LINES:]))
    except OSError as exc:
        logger.warning("journal rotation failed: %s", exc)


class OpenMWLogBridge:
    """Tail openmw.log, dispatch to lore_agent / d2d_agent, write inbox responses."""

    def __init__(
        self,
        config: dict,
        lore_agent,
        memory,
        d2d_agent=None,
        dispositions=None,
    ) -> None:
        self.config = config
        self.lore_agent = lore_agent
        self.memory = memory
        self.d2d_agent = d2d_agent
        self.dispositions = dispositions   # legacy, unused: social state lives in the savegame
        self.enable_life_facts = bool((config.get("features") or {}).get("disposition"))
        self._locked_npc: dict = {}  # most recent lock_npc context
        self._seen_req_ids: set[str] = set()
        self._counter: int = 0
        INBOX_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(
            "OpenMWLogBridge ready (log=%s inbox=%s disposition=%s)",
            OPENMW_LOG, RESPONSE_FILE, "on" if dispositions else "off",
        )

    # ------------------------------------------------------------------ tail

    async def _watch_game(self) -> None:
        """Silence the voices when the game is gone.

        Speech that is still queued belongs to a conversation that no longer
        exists: after closing OpenMW the player heard a batch of old replies
        talking to an empty desktop.
        """
        try:
            import psutil
        except ImportError:
            return
        was_running = False
        while True:
            await asyncio.sleep(3)
            try:
                # psutil reports name=None for processes it cannot read, and a
                # process can vanish mid-iteration. Neither may be allowed to
                # raise: this watcher shares a task group with the log tail,
                # and one exception here stopped the bridge from answering at
                # all — the game kept sending requests into the void.
                running = False
                for proc in psutil.process_iter(["name"]):
                    if (proc.info.get("name") or "").lower() == "openmw.exe":
                        running = True
                        break
            except Exception as exc:  # noqa: BLE001
                logger.debug("не смог перечислить процессы: %s", exc)
                continue
            if was_running and not running:
                tts = getattr(self, "tts", None)
                if tts is not None and hasattr(tts, "stop"):
                    tts.stop()
                logger.info("игра закрыта — недоговорённые реплики сброшены")
            was_running = running

    async def _supervised(self, name: str, factory, essential: bool) -> None:
        """Run a task so its failure cannot take the bridge down with it.

        asyncio.gather cancels every sibling when one task raises. That is how
        a single exception in the process watcher stopped the bridge from
        answering at all, while the game went on writing requests nobody read.
        Essential loops come back up; the rest just report and step aside.
        """
        while True:
            try:
                await factory()
                return
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("задача «%s» упала", name)
                if not essential:
                    logger.warning("«%s» отключена, мост продолжает работать", name)
                    return
                await asyncio.sleep(2)
                logger.info("перезапускаю «%s»", name)

    async def run(self) -> None:
        # Run log-tail, player-text watcher, and (optional) YouTube chat concurrently.
        tasks = [
            self._supervised("чтение лога игры", self._run_log_tail, True),
            self._supervised("текст игрока", self._run_player_text_watch, True),
            self._supervised("наблюдение за игрой", self._watch_game, False),
        ]

        stream_cfg = self.config.get("stream", {}) or {}
        if stream_cfg.get("enabled") and stream_cfg.get("youtube_video_id"):
            try:
                from stream.youtube_chat import YouTubeChatListener  # type: ignore
                from stream.chat_commands import ChatCommandHandler  # type: ignore
                handler = ChatCommandHandler(self.config)
                listener = YouTubeChatListener(
                    {"video_id": stream_cfg["youtube_video_id"]},
                    handler,
                )
                tasks.append(listener.start())
                logger.info("YouTube chat listener enabled (video_id=%s)",
                            stream_cfg["youtube_video_id"])
            except Exception as exc:  # noqa: BLE001
                logger.error("Could not start YouTube chat listener: %s", exc)
        else:
            logger.info("YouTube chat disabled (stream.enabled=false or no video_id)")

        await asyncio.gather(*tasks)

    async def _run_player_text_watch(self) -> None:
        """
        Watch ai_inbox/player_text.txt for text entered by the external chat
        window (chat_window_vfs.py). Each new file is treated as one dialogue
        request against the most recently locked NPC.
        """
        last_mtime = 0.0
        while True:
            try:
                if PLAYER_TEXT_FILE.exists():
                    mtime = PLAYER_TEXT_FILE.stat().st_mtime
                    if mtime != last_mtime:
                        last_mtime = mtime
                        text = PLAYER_TEXT_FILE.read_text(encoding="utf-8").strip()
                        if text:
                            self._counter += 1
                            req = {
                                "req_id": f"chat-{int(time.time())}-{self._counter}",
                                "type": "dialogue",
                                "player_text": text,
                            }
                            await self._handle_dialogue(req)
            except OSError as exc:
                logger.warning("player_text watch error: %s", exc)
            await asyncio.sleep(0.25)

    async def _run_log_tail(self) -> None:
        """Tail openmw.log by REOPENING it each poll.

        Holding one handle open across game restarts silently breaks the whole
        mod: OpenMW recreates openmw.log on launch, and the old handle keeps
        pointing at the replaced file — requests stream into the log while the
        bridge sees nothing at all. Reopening costs nothing at 4 Hz and is
        immune to truncation, recreation and rotation alike.
        """
        logger.info("OpenMWLogBridge tailing %s", OPENMW_LOG)
        while not OPENMW_LOG.exists():
            await asyncio.sleep(1.0)

        # Откуда читать. Позицию мог запомнить запуск — ЕЩЁ ДО того, как
        # поднялись озвучка и распознавание: между «мост готов» и этим циклом
        # проходит с десяток секунд, и всё сказанное игрой в этот промежуток
        # иначе пропадало молча (так терялся первый же запрос).
        pos = mark_log_position()
        while True:
            try:
                try:
                    size = OPENMW_LOG.stat().st_size
                except OSError:
                    await asyncio.sleep(0.5)
                    continue

                if size < pos:
                    logger.info("openmw.log restarted (%d < %d) — rewinding", size, pos)
                    pos = 0

                if size > pos:
                    lines, pos = read_complete_lines(OPENMW_LOG, pos)
                    for line in lines:
                        m = REQ_RE.search(line)
                        if m:
                            await self._handle_request_line(m.group(1))
                    if not lines:
                        await asyncio.sleep(0.1)   # строка ещё дописывается
                else:
                    await asyncio.sleep(0.25)
            except asyncio.CancelledError:
                logger.info("OpenMWLogBridge cancelled")
                return
            except Exception as exc:  # noqa: BLE001
                logger.error("tail error: %s", exc, exc_info=True)
                await asyncio.sleep(1.0)

    # ------------------------------------------------------------- dispatch

    async def _handle_request_line(self, payload: str) -> None:
        try:
            req = json.loads(payload)
        except json.JSONDecodeError as exc:
            logger.warning("bad MWAI_REQ json: %s (%s)", exc, payload[:120])
            return

        rid = str(req.get("req_id") or "")
        if not rid or rid in self._seen_req_ids:
            return
        self._seen_req_ids.add(rid)
        # cap memory
        if len(self._seen_req_ids) > 512:
            self._seen_req_ids = set(list(self._seen_req_ids)[-256:])

        rtype = req.get("type", "dialogue")
        if rtype == "lock_npc":
            self._locked_npc = req
            logger.info("lock_npc: %s (%s)", req.get("npc_name"), req.get("npc_id"))
            return
        if rtype == "stop_voice":
            tts = getattr(self, "tts", None)
            if tts is not None and hasattr(tts, "stop"):
                tts.stop()
                logger.info("voice stopped (player left the conversation)")
            return
        if rtype == "dialogue":
            # A new player line ends the previous exchange: whatever is still
            # queued or half-spoken from it is stale. Without this the queue
            # filled up with outdated lines and every new reply was dropped
            # unspoken ("очередь переполнена").
            if str(req.get("player_text") or "").strip():
                tts = getattr(self, "tts", None)
                if tts is not None and hasattr(tts, "new_turn"):
                    tts.new_turn()
            await self._handle_dialogue(req)
            return
        if rtype == "voice_start":
            stt = getattr(self, "stt", None)
            if stt is not None:
                asyncio.create_task(stt.ptt_start())
            return
        if rtype == "voice_stop":
            asyncio.create_task(self._handle_voice_stop(req))
            return
        if rtype == "voice":
            # Voice mode: record the mic, transcribe, then run the normal
            # dialogue pipeline with the recognized text.
            asyncio.create_task(self._handle_voice(req))
            return
        if rtype == "narrate":
            await self._handle_narrate(req)
            return
        if rtype == "npc_npc":
            await self._handle_d2d(req)
            return
        if rtype == "scene":
            # Сцена считается долго — не держим разбор входящих строк.
            asyncio.create_task(self._handle_scene(req))
            return
        if rtype == "scene_say":
            # Игра дошла до такта и просит озвучить реплику. Голос выбирается
            # так же, как в обычном разговоре: по имени, полу и расе.
            tts = getattr(self, "tts", None)
            if tts is not None:
                tts.speak_async(
                    str(req.get("text") or ""), str(req.get("npc_id") or "scene"),
                    bool(req.get("npc_is_male", True)),
                    distance=float(req.get("distance") or 0),
                    race=str(req.get("npc_race") or ""))
            return
        logger.warning("unknown MWAI type '%s'", rtype)

    async def _handle_scene(self, req: dict) -> None:
        """Поставить сцену и отдать её игре одним куском.

        Такты уезжают в обычный слот ответов: игра играет их по очереди сама,
        отмеряя паузы по длине реплики. Слот того же постоянного размера, что и
        всегда, поэтому длина сцены ограничена — лишние такты просто не влезут,
        и лучше их отрезать здесь, чем получить обрезанный JSON в игре.
        """
        agent = getattr(self, "scene_agent", None)
        if agent is None:
            logger.debug("сцена запрошена, но постановщик не поднят")
            return

        # Повод и жребий фарса решаются ЗДЕСЬ, а не в игре: ручки характера
        # мира читаются на лету, и держать их в одном месте проще, чем гонять
        # каждую правку через файлы VFS.
        try:
            import random as _rnd

            import world_tuning as wt
            from agents.scene_agent import (SCENE_KINDS, is_absurd_roll,
                                            kinds_allowed_for)
            # Заодно перекладываем ручки в игру: игрок мог поправить файл
            # только что, а частота событий считается на стороне Lua.
            dials = wt.publish()
            if not req.get("order"):
                pool = kinds_allowed_for(dials["опасность"], req.get("fit") or [])
                if not pool:
                    logger.info("сцена не состоялась: при опасности %d подходящих "
                                "поводов нет", dials["опасность"])
                    return
                req["kind"] = _rnd.choice(pool)
            req["absurd"] = is_absurd_roll(dials["нелепость"], _rnd.random())
            req["used_jokes"] = list(getattr(self, "_used_jokes", []))
        except Exception as exc:  # noqa: BLE001
            logger.warning("ручки характера мира не применились: %s", exc)

        try:
            result = await agent.stage(req)
        except Exception as exc:  # noqa: BLE001
            logger.error("постановщик сцены не справился: %s", exc, exc_info=True)
            return
        beats = result.get("beats") or []
        if not beats:
            logger.info("сцена не состоялась: тактов нет (повод %r)", req.get("kind"))
            return
        for b in beats:
            b["line"] = trim_reply(b.get("line", ""), 220)
        publish_reply({
            "req_id": str(req.get("req_id") or ""),
            "scene": beats,
            "kind": result.get("kind", ""),
            "timestamp": _now_iso(),
        })
        logger.info("сцена «%s»: %d тактов, состав %d",
                    result.get("kind") or "по указанию", len(beats),
                    len(req.get("cast") or []))

    async def _handle_d2d(self, req: dict) -> None:
        if not self.d2d_agent:
            logger.debug("D2D event received but no d2d_agent configured — skipping")
            return
        try:
            result = await self.d2d_agent.generate(req)
        except Exception as exc:  # noqa: BLE001
            logger.error("d2d_agent failed: %s", exc, exc_info=True)
            return

        payload = {
            "req_id": result.get("req_id", req.get("req_id")),
            "exchanges": result.get("exchanges", []),
            "timestamp": _now_iso(),
        }
        try:
            _write_slot(NPC_SPEECH_FILE, json.dumps(payload, ensure_ascii=False))
            logger.info(
                "D2D wrote %d exchanges for req_id=%s",
                len(payload["exchanges"]), payload["req_id"],
            )
        except OSError as exc:
            logger.error("could not write npc_speech.txt: %s", exc)

    # ----------------------------------------------------------- narrator

    async def _handle_narrate(self, req: dict) -> None:
        """BG3-style narrator: scene descriptions AND a running conversation —
        the player may ask the voice-over questions about what they perceive."""
        location = str(req.get("location") or "Вварденфелл")
        scene = str(req.get("player_context") or "")
        bystanders = str(req.get("bystanders") or "")
        question = str(req.get("player_text") or "").strip()
        system = (
            "Ты — закадровый РАССКАЗЧИК в духе Baldur's Gate 3, ведущий игрока по "
            "The Elder Scrolls III: Morrowind (3E 427, Вварденфелл). Пиши ТОЛЬКО "
            "по-русски, во втором лице, настоящем времени: бархатно, образно, с "
            "лёгкой иронией и интригой. Ты описываешь то, что игрок ВОСПРИНИМАЕТ: "
            "виды, звуки, запахи, взгляды NPC, деталь, которая «не так». На вопросы "
            "игрока отвечаешь как голос за кадром — направляешь внимание, дразнишь "
            "догадкой, но НИКОГДА не выдумываешь квесты и факты, которых нет, не "
            "решаешь за игрока и не раскрываешь того, чего он не мог бы заметить. "
            "3-5 предложений."
        )
        context_line = (
            f"Локация: {location}\nСцена: {scene}\n"
            f"Рядом: {bystanders or 'никого приметного'}"
        )
        messages = []
        hist = req.get("conversation_history")
        if isinstance(hist, list):
            for turn in hist[-10:]:
                role = "user" if turn.get("role") == "player" else "assistant"
                messages.append({"role": role, "content": str(turn.get("content") or "")})
        if question:
            messages.append({"role": "user",
                             "content": context_line + "\n\nИГРОК СПРАШИВАЕТ У РАССКАЗЧИКА: " + question})
        else:
            messages.append({"role": "user",
                             "content": context_line + "\n\nОпиши сцену, встречая игрока."})
        # The narrator LOOKS: one downscaled frame per request (only when the
        # player asks), so descriptions are of the REAL scene instead of being
        # invented from a text summary. Falls back to text-only on failure.
        shot = None
        if (self.config.get("narrator") or {}).get("vision", True):
            try:
                from screen_grab import grab_screen_png
                shot = await asyncio.to_thread(grab_screen_png)
            except Exception as exc:  # noqa: BLE001
                logger.warning("narrator vision unavailable: %s", exc)
        if shot:
            messages[-1]["content"] += (
                "\n\n(К этому сообщению приложен КАДР ИГРЫ — описывай то, что на "
                "нём действительно видно: помещение, предметы, свет, кто в кадре.)"
            )

        try:
            resp = await self.lore_agent.llm.complete(
                system=system, messages=messages,
                image_bytes=shot,
                temperature=0.9, max_tokens=400,
            )
            text = (resp.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("narrate failed: %s", exc)
            return
        if not text:
            return
        reply = {
            "req_id": req.get("req_id"), "type": "dialogue", "npc_id": "narrator",
            "speaker_id": "narrator", "speaker_name": "Рассказчик",
            "npc_response": text, "emotion": "neutral", "action": "none",
            "target": "none", "disp": 0, "gold": 0, "item": "none",
            "rumor": "", "life_facts": [],
            "location": location, "timestamp": _now_iso(),
        }
        try:
            publish_reply(reply)
        except OSError as exc:
            logger.error("could not write narrate response: %s", exc)
        tts = getattr(self, "tts", None)
        if tts is not None:
            tts.speak_async(text, "narrator", True, distance=0.0)

    # ---------------------------------------------------- baked character
    # Character is TIMELINE-INDEPENDENT (like the voice): the same NPC has the
    # same innate temperament in every save and every new game. Traits are
    # deterministic from the npc id; life facts are generated once EVER and
    # kept in a persistent bridge-side store.

    TRAITS = [
        "гордый (не терпит снисхождения)", "жадноватый", "добросердечный",
        "трусливый", "вспыльчивый", "расчётливо-холодный", "суеверный",
        "весельчак и балагур", "угрюмый молчун", "любопытный до чужих дел",
        "циничный", "набожный", "хвастливый", "меланхоличный",
        "упрямый как гуар", "льстивый",
    ]
    MONEY_ATTITUDES = [
        "к деньгам: гордец — подачек НЕ принимает, дары только как знак уважения",
        "к деньгам: практичен — монета есть монета, возьмёт с благодарностью",
        "к деньгам: жаден — возьмёт и ещё попросит",
        "к деньгам: щепетилен — возьмёт только заработанное или в долг с отдачей",
    ]

    # Чего человек хочет САМ, помимо игрока. Без этого NPC только отвечает на
    # вопросы: исчезни игрок — и жизнь персонажа замирает. Цель, страх и нужда
    # запекаются намертво по его id, поэтому характер не плавает от встречи к
    # встрече и не стоит ни одного лишнего запроса к модели.
    GOALS_BY_TRADE = {
        "merchant": [
            "скопить на вторую лавку и уйти от поставщика-кровопийцы",
            "выкупить долг у Дома Хлаалу, пока проценты не сожрали всё",
            "сбыть партию, которую взял не подумав, и не сесть за это",
            "переманить покупателей у соседа напротив",
        ],
        "guard": [
            "дослужиться до десятника и перевестись поближе к дому",
            "закрыть дело, которое начальство велело замять",
            "накопить на выкуп брата из долговой ямы",
            "дотянуть смену без происшествий — семья ждёт",
        ],
        "priest": [
            "отмолить грех, о котором не знает даже наставник",
            "собрать на восстановление придорожного святилища",
            "переубедить прихожан, тянущихся к Шестому Дому",
            "получить перевод в Вивек, ближе к Трибуналу",
        ],
        "warrior": [
            "вернуть родовой клинок, проигранный в кости",
            "найти того, кто вырезал его отряд, и рассчитаться",
            "наняться в приличную дружину, а не к кому попало",
            "заработать на выкуп своего имени после позорной отставки",
        ],
        "thief": [
            "сорвать одно дело и завязать навсегда",
            "рассчитаться с гильдией, пока не прислали спросить",
            "найти покупателя на вещь, которую боится держать дома",
            "уйти из города прежде, чем всплывёт прошлое дело",
        ],
        "commoner": [
            "скопить на переезд из этой дыры",
            "выдать дочь замуж и не опозориться приданым",
            "вернуть занятое у соседа, пока тот не начал болтать",
            "пережить сезон, не влезая в долги",
            "найти пропавшего родича — живым или мёртвым",
        ],
    }
    FEARS = [
        "боится стражи больше, чем даэдра", "боится нищеты и голода",
        "боится, что вскроется одна старая история", "боится болезни Шестого Дома",
        "боится потерять место и уважение соседей", "боится смерти в чужом краю",
        "боится собственной вспыльчивости", "боится, что его сочтут доносчиком",
    ]

    @staticmethod
    def _trade_of(npc_class: str) -> str:
        c = (npc_class or "").lower()
        for key, words in (
            ("merchant", ("trader", "merchant", "publican", "pawnbroker", "smith",
                          "apothecary", "bookseller", "creeper")),
            ("guard", ("guard", "ordinator", "legion", "soldier", "watchman")),
            ("priest", ("priest", "monk", "acolyte", "healer", "temple")),
            ("warrior", ("warrior", "knight", "crusader", "barbarian",
                         "mercenary", "archer", "scout")),
            ("thief", ("thief", "bandit", "rogue", "assassin", "smuggler",
                       "agent", "pilgrim")),
        ):
            if any(w in c for w in words):
                return key
        return "commoner"

    def _drives_for(self, npc_id: str, npc_class: str) -> str:
        h = int(hashlib.md5(("goal:" + npc_id).encode("utf-8", "ignore")).hexdigest(), 16)
        pool = self.GOALS_BY_TRADE[self._trade_of(npc_class)]
        goal = pool[h % len(pool)]
        fear = self.FEARS[(h // 137) % len(self.FEARS)]
        return f"главное желание: {goal}; {fear}"

    # Заминка перед ответом. Не «фразы-заглушки», а то, что человек правда
    # произносит, обдумывая: вздох, короткое согласие, что-то себе под нос.
    FILLERS = [
        "Хм…", "Так…", "Дай-ка подумать.", "Погоди.", "Ну-у…",
        "Это ты к чему?", "Тьфу…", "Ага…",
    ]

    # Как часто отдавать недописанную реплику в игру. Чаще смысла нет: игра
    # опрашивает файл раз в 0.2 с, а слот пишется атомарной заменой.
    PARTIAL_EVERY_S = 0.35

    def _partial_sender(self, req: dict, npc_id: str):
        """Возвращает функцию, которую агент зовёт по мере набора ответа."""
        state = {"at": 0.0, "last": ""}

        def send(text: str) -> None:
            text = (text or "").strip()
            if not text or text == state["last"]:
                return
            now = time.monotonic()
            if now - state["at"] < self.PARTIAL_EVERY_S:
                return
            state["at"], state["last"] = now, text
            publish_reply({
                "req_id": req.get("req_id"),
                "type": "dialogue",
                "npc_id": npc_id,
                "npc_response": text,
                # Пометка для игры: строку показать, но действий не исполнять,
                # в память не писать и голосом не озвучивать — ответ ещё пишется.
                "partial": True,
                "emotion": "neutral", "action": "none", "target": "none",
                "disp": 0, "gold": 0, "item": "none",
                "location": str(req.get("location") or ""),
                "timestamp": _now_iso(),
            })

        return send

    def _speak_filler(self, npc_id: str, req: dict, ctx: dict, player_text: str,
                      force: bool = False) -> None:
        bank   = getattr(self, "filler_bank", None)
        filler = getattr(self, "filler", None)
        if bank is None and filler is None:
            return
        if req.get("_filler_done"):
            return                      # уже мялся в этом же обмене
        # Только на настоящую реплику игрока: на приветствия и служебные
        # обращения заминка выглядит странно. force=True — голосовой режим:
        # там заминка идёт ДО распознавания, и текста ещё нет.
        text = (player_text or "").strip()
        if not force and (not text or text.startswith("__")):
            return
        req["_filler_done"] = True
        if force and not text:
            text = str(req.get("req_id") or "")
        try:
            is_male = req.get("npc_is_male")
            if is_male is None:
                is_male = ctx.get("npc_is_male", True)
            race = str(ctx.get("npc_race") or req.get("npc_race") or "")
            dist = float(req.get("distance") or 0)
            # Готовая заминка ГОЛОСОМ ЭТОГО ЖЕ персонажа — играется мгновенно,
            # без синтеза. Синтезированная piper'ом звучала чужим тембром:
            # NPC мялся одним голосом, а отвечал другим.
            if bank is not None and bank.available:
                # Флажок «ответ пошёл»: заминка перестаёт мяться, как только
                # реплика поехала в игру.
                done = threading.Event()
                req["_filler_done_evt"] = done
                if bank.play_async(npc_id, bool(is_male), race,
                                   distance=dist, salt=text, until=done):
                    return
            if filler is None:
                return
            h = int(hashlib.md5((npc_id + text).encode("utf-8", "ignore")).hexdigest(), 16)
            filler.speak_async(
                self.FILLERS[h % len(self.FILLERS)], npc_id, bool(is_male),
                distance=dist, race=race,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("заминка не прозвучала: %s", exc)

    def _characters_path(self) -> pathlib.Path:
        base = pathlib.Path(self.config.get("memory", {}).get("chroma_dir", "."))
        return base.parent / "characters.json"

    def _character_for(self, npc_id: str) -> tuple[str, list[str]]:
        """(baked trait line, stored life facts or [])."""
        h = int(hashlib.md5(("chr:" + npc_id).encode("utf-8", "ignore")).hexdigest(), 16)
        t1 = self.TRAITS[h % len(self.TRAITS)]
        t2 = self.TRAITS[(h // 31) % len(self.TRAITS)]
        if t2 == t1:
            t2 = self.TRAITS[(h // 31 + 1) % len(self.TRAITS)]
        money = self.MONEY_ATTITUDES[(h // 977) % len(self.MONEY_ATTITUDES)]
        traits = f"{t1}; {t2}; {money}"
        facts: list[str] = []
        try:
            data = json.loads(self._characters_path().read_text(encoding="utf-8"))
            facts = [str(x) for x in (data.get(npc_id) or [])]
        except (OSError, json.JSONDecodeError):
            pass
        return traits, facts

    def _store_facts(self, npc_id: str, facts: list[str]) -> None:
        try:
            try:
                data = json.loads(self._characters_path().read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            data[npc_id] = facts
            _atomic_write_text(self._characters_path(), json.dumps(data, ensure_ascii=False, indent=1))
        except OSError as exc:
            logger.warning("could not store character facts: %s", exc)

    # ------------------------------------------------------------- gossip

    @staticmethod
    def _make_rumor(req: dict, action: str, emotion: str, location: str) -> str:
        """Rumor text for a notable outcome. Storage is SAVE-SCOPED: the string
        is returned in the reply and the Lua side keeps it inside the savegame,
        so reloading an earlier save also unmakes the rumor."""
        npc_name = str(req.get("npc_name") or "кто-то")
        loc = location if location not in ("", "unknown") else "окрестностях"
        if action == "attack":
            return f"дело дошло до драки между чужаком и {npc_name} ({loc})"
        if action == "callguards":
            return f"{npc_name} звал стражу из-за чужака ({loc})"
        if action == "defend":
            return f"чужак жаловался страже, и та вступилась за него ({loc})"
        if action == "follow":
            return f"{npc_name} теперь путешествует вместе с чужаком"
        if emotion == "angry":
            return f"чужак крепко повздорил с {npc_name} ({loc})"
        return ""

    # ----------------------------------------------------- companion react

    async def _companion_react(
        self, req: dict, npc_response: str, npc_emotion: str,
        base_req_id: str, npc_id: str, location: str, player_text: str,
    ) -> None:
        """Second in-scene voice: the player's companion reacts to a heated
        exchange — scolds, calms, or physically intervenes (ACTION aimed at the
        interlocutor, handled by the Lua side via speaker_id)."""
        comp_id = str(req.get("companion_id"))
        comp_name = str(req.get("companion_name") or "Спутник")
        npc_name = str(req.get("npc_name") or npc_id)
        try:
            situation = (
                f"Ты — спутник и соратник игрока, вы путешествуете вместе. Вы стоите рядом, "
                f"игрок разговаривает с персонажем по имени {npc_name}. Разговор накалился.\n"
                f"Игрок сказал: «{player_text}»\n"
                f"{npc_name} ответил (настроение: {npc_emotion}): «{npc_response}»\n\n"
                f"Отреагируй как участник сцены, встав на сторону игрока так, как свойственно "
                f"твоему характеру: можешь рявкнуть на {npc_name}, попытаться разрядить обстановку, "
                f"съязвить, пригрозить — или, если {npc_name} перешёл черту, вмешаться делом "
                f"(ACTION:attack направит ТВОЮ атаку на {npc_name}, не на игрока). Одна-две фразы."
            )
            agent_request = {
                "npc_id": comp_id,
                "npc_name": comp_name,
                "npc_race": str(req.get("companion_race") or "Dunmer"),
                "npc_class": str(req.get("companion_class") or "Commoner"),
                "npc_faction": "",
                "player_input": situation,
                "location": location,
                "conversation_history": [],
                "is_greeting": False,
            }
            result = await self.lore_agent.generate_response(agent_request, memory_context=[])
            comp_text = trim_reply(result.get("response", ""))
            if not comp_text:
                return
            await asyncio.sleep(2.0)   # let the main reply display first
            reply = {
                "req_id": base_req_id + "-comp",
                "type": "dialogue",
                "npc_id": npc_id,                # history stays with the interlocutor
                "speaker_id": comp_id,           # but the SPEAKER is the companion
                "speaker_name": comp_name,
                "npc_response": comp_text,
                "emotion": result.get("emotion", "neutral"),
                "action": result.get("action", "none"),
                "target": result.get("target", "none"),
                "location": location,
                "timestamp": _now_iso(),
            }
            # Спутник влезает в разговор только когда предыдущий договорил.
            await self._await_quiet()
            publish_reply(reply)
            tts = getattr(self, "tts", None)
            if tts is not None:
                tts.speak_async(comp_text, comp_id, bool(req.get("companion_is_male", True)),
                                distance=float(req.get("distance") or 0),
                                race=str(req.get("companion_race") or ""))
            logger.info("companion '%s' reacted (action=%s)", comp_name, reply["action"])
        except Exception as exc:  # noqa: BLE001
            logger.error("companion react failed: %s", exc, exc_info=True)

    async def _bystander_react(
        self, req: dict, npc_response: str,
        base_req_id: str, npc_id: str, location: str, player_text: str,
    ) -> None:
        """A nearby NPC overheard a line that concerns them (their name or
        office came up). They may butt in — or stay silent (reply NONE)."""
        lid = str(req.get("listener_id"))
        lname = str(req.get("listener_name") or "Прохожий")
        npc_name = str(req.get("npc_name") or npc_id)
        try:
            why = ("прозвучало твоё имя или твоя служба"
                   if req.get("listener_reason") == "mentioned"
                   else "сказанное задевает закон, честь или порядок — и ты это слышал")
            situation = (
                f"Ты стоишь неподалёку и НЕВОЛЬНО СЛЫШИШЬ чужой разговор. Игрок говорит "
                f"с персонажем {npc_name} и произнёс: «{player_text}» — это касается ТЕБЯ "
                f"({why}).\n"
                f"{npc_name} ответил: «{npc_response}»\n\n"
                f"Решай по своему характеру и положению: вмешаться — осадить наглеца, "
                f"пригрозить (ACTION:threaten — движок исполнит), заявить о преступлении "
                f"(ACTION:callguards — штраф и арест), вспыхнуть (ACTION:attack), съязвить; "
                f"стражник не потерпит оскорбления власти в лицо. Но если тебя это не "
                f"задевает всерьёз — ответь ровно одним словом: NONE (ты промолчал)."
            )
            agent_request = {
                "npc_id": lid,
                "npc_name": lname,
                "npc_race": str(req.get("listener_race") or "Dunmer"),
                "npc_class": str(req.get("listener_class") or "Commoner"),
                "npc_faction": "",
                "player_input": situation,
                "location": location,
                "conversation_history": [],
                "is_greeting": False,
            }
            result = await self.lore_agent.generate_response(agent_request, memory_context=[])
            btext = trim_reply(result.get("response") or "")
            if not btext or btext.upper().startswith("NONE"):
                return
            await asyncio.sleep(3.5)   # let the main (and companion) lines land first
            reply = {
                "req_id": base_req_id + "-bys",
                "type": "dialogue",
                "npc_id": npc_id,
                "speaker_id": lid,
                "speaker_name": lname,
                "speaker_kind": "bystander",
                "npc_response": btext,
                "emotion": result.get("emotion", "neutral"),
                "action": result.get("action", "none"),
                "target": result.get("target", "none"),
                "disp": 0, "gold": 0, "item": "none", "rumor": "", "life_facts": [],
                "location": location, "timestamp": _now_iso(),
            }
            # Свидетель вступает ТОЛЬКО после того, как отзвучал предыдущий.
            # Раньше он влезал поверх и двое говорили разом, а при переполнении
            # очереди чья-то реплика просто пропадала.
            await self._await_quiet()
            publish_reply(reply)
            tts = getattr(self, "tts", None)
            if tts is not None:
                tts.speak_async(btext, lid, bool(req.get("listener_is_male", True)),
                                distance=float(req.get("distance") or 0) + 250,
                                race=str(req.get("listener_race") or ""))
            logger.info("bystander '%s' stepped in (action=%s)", lname, reply["action"])
        except Exception as exc:  # noqa: BLE001
            logger.error("bystander react failed: %s", exc, exc_info=True)

    async def _await_quiet(self, timeout: float = 12.0) -> None:
        """Ждём, пока договорит тот, кто говорит сейчас.

        Реплики озвучиваются по очереди, но МИР их порождал не считаясь с этим:
        свидетель влезал поверх собеседника, спутник поверх свидетеля. В звуке
        они выстраивались друг за другом, а в разговоре получалась каша, и при
        переполнении очереди чья-то реплика пропадала совсем.

        Ждём в отдельном потоке, чтобы не застопорить мост: пока идёт речь,
        он продолжает читать журнал игры.
        """
        tts = getattr(self, "tts", None)
        if tts is None or not hasattr(tts, "wait_quiet"):
            return
        if not await asyncio.to_thread(tts.wait_quiet, timeout):
            logger.warning("не дождался тишины за %.0f с — говорим поверх", timeout)

    async def _handle_voice_stop(self, req: dict) -> None:
        """Player released the talk key: transcribe what was recorded and run
        the normal dialogue pipeline with it."""
        # ЗАМИНКА ИДЁТ ПЕРВОЙ, до распознавания. Кто говорит — известно ещё с
        # нажатия клавиши, а Whisper думает над фразой почти три секунды.
        # Раньше эти секунды были мёртвой тишиной: игрок отпускал клавишу и не
        # получал ни звука, пока не отработает вся цепочка.
        self._speak_filler(str(req.get("npc_id") or ""), req, {}, "", force=True)

        stt = getattr(self, "stt", None)
        text = await stt.ptt_stop() if stt is not None else ""
        if not text:
            reply = {
                "req_id": req.get("req_id"), "type": "dialogue",
                "npc_id": str(req.get("npc_id") or ""), "voice": True,
                "player_echo": "", "npc_response": "",
                "emotion": "neutral", "action": "none", "target": "none",
                "disp": 0, "gold": 0, "item": "none", "loan": "no",
                "deal": "none", "cond": "none", "rumor": "", "life_facts": [],
                "location": str(req.get("location") or ""), "timestamp": _now_iso(),
            }
            publish_reply(reply)
            return
        req["player_text"] = text
        req["voice"] = True
        await self._handle_dialogue(req)

    async def _handle_voice(self, req: dict) -> None:
        """Voice mode: mic -> whisper -> the standard dialogue pipeline.
        The reply carries voice=true + player_echo so the Lua side shows
        subtitles instead of the chat window."""
        stt = getattr(self, "stt", None)
        if stt is None or not getattr(stt, "ready", False):
            reply = {
                "req_id": req.get("req_id"), "type": "dialogue",
                "npc_id": str(req.get("npc_id") or ""), "voice": True, "player_echo": "",
                "npc_response": "(голосовой режим не готов — распознавание ещё грузится)",
                "emotion": "neutral", "action": "none", "target": "none",
                "disp": 0, "gold": 0, "item": "none", "rumor": "", "life_facts": [],
                "location": str(req.get("location") or ""), "timestamp": _now_iso(),
            }
            publish_reply(reply)
            return
        text = await stt.listen(max_s=8.0)
        if not text:
            reply = {
                "req_id": req.get("req_id"), "type": "dialogue",
                "npc_id": str(req.get("npc_id") or ""), "voice": True, "player_echo": "",
                "npc_response": "", "emotion": "neutral", "action": "none",
                "target": "none", "disp": 0, "gold": 0, "item": "none",
                "rumor": "", "life_facts": [],
                "location": str(req.get("location") or ""), "timestamp": _now_iso(),
            }
            publish_reply(reply)
            return
        req["player_text"] = text
        req["voice"] = True
        await self._handle_dialogue(req)

    async def _handle_dialogue(self, req: dict) -> None:
        ctx = self._locked_npc or {}
        npc_id = req.get("npc_id") or ctx.get("npc_id") or "unknown_npc"
        location = req.get("location") or ctx.get("location") or "unknown"
        player_text = req.get("player_text", "")

        # Auto-greeting: player locked on NPC, no text yet (kenshi __greet__ equivalent)
        is_proactive = player_text == "__proactive__"
        is_surrender = player_text == "__surrender__"
        theft_item = ""
        if player_text.startswith("__theft__:"):
            theft_item = player_text[len("__theft__:"):].strip()
            player_text = ""
        death_react = ""
        if player_text.startswith("__death_react__:"):
            death_react = player_text[len("__death_react__:"):].strip()
            player_text = ""
        # Стражник пришёл на вызов и разбирается, кто виноват. Раньше вызов
        # стражи МГНОВЕННО вешал на игрока нападение — до всякого разбора.
        inquiry = ""
        if player_text.startswith("__inquiry__:"):
            inquiry = player_text[len("__inquiry__:"):].strip()
            player_text = ""
        if player_text in ("__greet__", "__proactive__", "__surrender__"):
            player_text = ""   # lore_agent will generate a greeting unprompted

        # Save-scoped history from the Lua side is the ONLY history source:
        # it lives inside the savegame, so reloading an earlier save rewinds NPC
        # memory with it. The local JSON store is a write-only archive — never
        # read it here, or NPCs would "remember" other timelines/new games.
        # (Note: Lua's json encodes an empty list as {}, hence the dict check.)
        lua_hist = req.get("conversation_history")
        history = lua_hist if isinstance(lua_hist, list) else []

        # EVENT memory (history, mood) is save-scoped and comes from Lua.
        # CHARACTER (traits, life facts) is timeline-independent and baked:
        # deterministic traits + facts generated once ever, stored bridge-side.
        disp_band: Optional[str] = None
        last_mood: Optional[str] = str(req.get("npc_last_mood") or "") or None
        baked_traits, life_facts = self._character_for(npc_id)
        new_life_facts: list[str] = []
        if not life_facts:
            # Adopt facts from older saves (they were stored savegame-side before).
            lf = req.get("npc_life_facts")
            if isinstance(lf, list) and lf:
                life_facts = [str(x) for x in lf]
                self._store_facts(npc_id, life_facts)
        if (
            not life_facts
            and self.enable_life_facts
            and hasattr(self.lore_agent, "generate_life_facts")
        ):
            # First-ever meeting anywhere: generate 3 life facts ONCE, forever.
            try:
                life_facts = await self.lore_agent.generate_life_facts(
                    npc_name=ctx.get("npc_name", npc_id),
                    npc_race=ctx.get("npc_race", "Dunmer"),
                    npc_class=ctx.get("npc_class", "Commoner"),
                    npc_faction=ctx.get("npc_faction", ""),
                )
                new_life_facts = life_facts
                if life_facts:
                    self._store_facts(npc_id, life_facts)
            except Exception as exc:  # noqa: BLE001
                logger.warning("life_facts gen failed for %s: %s", npc_id, exc)

        # A companion earns a hidden story of their own: generated once, kept in
        # the savegame, and opened one stage at a time as the player's own tale
        # moves forward. Only for people actually travelling with the player.
        new_arc: list[str] = []
        if (str(req.get("is_companion") or "") in ("1", "true", "True")
                and not (req.get("companion_arc") or [])):
            try:
                new_arc = await self.lore_agent.generate_companion_arc(
                    npc_name=ctx.get("npc_name", npc_id),
                    npc_race=ctx.get("npc_race", "Dunmer"),
                    npc_class=ctx.get("npc_class", "Commoner"),
                    npc_faction=ctx.get("npc_faction", ""),
                    npc_canon=str(req.get("npc_canon") or ""),
                    quests=str(req.get("active_quests") or ""),
                )
                if new_arc:
                    logger.info("арка спутника %s: %d ступени",
                                ctx.get("npc_name", npc_id), len(new_arc))
            except Exception as exc:  # noqa: BLE001
                logger.warning("companion arc failed for %s: %s", npc_id, exc)

        try:
            agent_request = {
                "npc_id": npc_id,
                "npc_name": ctx.get("npc_name", npc_id),
                "npc_race": ctx.get("npc_race", "Dunmer"),
                "npc_class": ctx.get("npc_class", "Commoner"),
                "npc_faction": ctx.get("npc_faction", ""),
                "player_input": player_text,
                "location": location,
                "conversation_history": history,
                "is_greeting": player_text == "",
                "disposition_band": disp_band,
                "last_mood": last_mood,
                "life_facts": life_facts,
                "player_context": str(req.get("player_context") or ""),
                "active_quests": str(req.get("active_quests") or ""),
                "rumors": (req.get("rumors") if isinstance(req.get("rumors"), list) else []),
                "talkativeness": ["terse", "normal", "chatty"][
                    int(hashlib.md5(npc_id.encode("utf-8", "ignore")).hexdigest(), 16) % 3
                ],
                "npc_disposition": req.get("npc_disposition"),
                # Идёт ли этот человек за игроком прямо сейчас. Флаг приходил
                # из игры и использовался только чтобы завести спутнику
                # предысторию — в промпт он не попадал вовсе. Из-за этого
                # Телери шла следом и в том же разговоре отрицала, что следует,
                # уверяя, что это игрок за ней увязался.
                "is_companion": str(req.get("is_companion") or "") in ("1", "true", "True"),
                "bystanders": str(req.get("bystanders") or ""),
                # Bodies in view and whose roof the NPC is under. Without these
                # a guard who had just killed a man in someone else's house
                # claimed to be at home and to have never seen a corpse.
                "corpses": str(req.get("corpses") or ""),
                "npc_place": str(req.get("npc_place") or ""),
                "npc_fate": str(req.get("npc_fate") or ""),
                # Своя цель и свой страх: с ними NPC перестаёт быть автоответчиком.
                "npc_drives": self._drives_for(npc_id, ctx.get("npc_class", "")),
                "risk_note": str(req.get("risk_note") or ""),
                # Скрытая арка спутника и то, сколько её ступеней уже открыто.
                "companion_arc": (req.get("companion_arc")
                                  if isinstance(req.get("companion_arc"), list)
                                  else new_arc),
                "arc_reveal": int(req.get("arc_reveal") or 1),
                "npc_condition": str(req.get("npc_condition") or ""),
                "is_proactive": is_proactive,
                "is_surrender": is_surrender,
                "theft_item": theft_item,
                # «кто звал | кто начал» — дело, с которым стражник пришёл.
                "inquiry": inquiry,
                "npc_canon": str(req.get("npc_canon") or ""),
                "npc_inventory": str(req.get("npc_inventory") or ""),
                "death_react": death_react,
                "baked_traits": baked_traits,
                "debt_note": str(req.get("debt_note") or ""),
                "deal_note": str(req.get("deal_note") or ""),
            }
            # Пока модель думает и голос синтезируется, NPC молчит секунд
            # пять-восемь и выглядит зависшим. Короткое «хм…» быстрым движком
            # закрывает эту паузу — человек так и делает, прежде чем ответить.
            self._speak_filler(npc_id, req, ctx, player_text)

            # Показ реплики по мере набора. Каждая порция везёт ВЕСЬ текст на
            # текущий момент, а не приращение: если игра пропустит одну из-за
            # опроса раз в 0.2 с, следующая всё равно покажет полную строку.
            agent_request["on_partial"] = self._partial_sender(req, npc_id)

            result = await self.lore_agent.generate_response(
                agent_request, memory_context=history
            )
            if isinstance(result, dict):
                npc_response = trim_reply(result.get("response", "..."))
                npc_emotion  = result.get("emotion",  "neutral")
                npc_action   = result.get("action",   "none")
                npc_target   = result.get("target",   "none")
                npc_disp     = int(result.get("disp", 0) or 0)
                npc_gold     = int(result.get("gold", 0) or 0)
                npc_item     = result.get("item", "none")
                npc_loan     = result.get("loan", "no")
                npc_deal     = result.get("deal", "none")
                npc_cond     = result.get("cond", "none")
                npc_fate     = result.get("fate", "none")
            else:
                npc_response = str(result)
                npc_emotion  = "neutral"
                npc_action   = "none"
                npc_target   = "none"
                npc_disp     = 0
                npc_gold     = 0
                npc_item     = "none"
                npc_loan     = "no"
                npc_deal     = "none"
                npc_cond     = "none"
                npc_fate     = "none"
        except Exception as exc:  # noqa: BLE001
            logger.error("lore_agent failed: %s", exc, exc_info=True)
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                npc_response = "(перевёл дух и молчит — лимит нейросети; попробуй через минуту)"
            else:
                npc_response = "(смотрит сквозь тебя — что-то сломалось, загляни в окно моста)"
            npc_emotion  = "neutral"
            npc_action   = "none"
            npc_target   = "none"
            npc_disp     = 0
            npc_gold     = 0
            npc_item     = "none"
            npc_loan     = "no"
            npc_deal     = "none"
            npc_cond     = "none"

        self.memory.store_exchange(
            npc_id, player_text, npc_response, location,
            emotion=npc_emotion, action=npc_action,
        )

        # ── NPC voice (TTS) — non-blocking, gender-matched, per-NPC stable ──
        # Реплика готова — заминке пора умолкать (доиграет начатое и ещё одну:
        # между готовым текстом и первым звуком лежит синтез).
        evt = req.get("_filler_done_evt")
        if evt is not None:
            evt.set()
        tts = getattr(self, "tts", None)
        if tts is not None:
            is_male = req.get("npc_is_male")
            if is_male is None:
                is_male = ctx.get("npc_is_male", True)
            tts.speak_async(npc_response, npc_id, bool(is_male),
                            distance=float(req.get("distance") or 0),
                            race=str(ctx.get("npc_race") or req.get("npc_race") or ""))

        # ── Gossip: notable outcomes become rumors (stored in the SAVE by Lua) ─
        rumor_text = ""
        try:
            rumor_text = self._make_rumor(req, npc_action, npc_emotion, location)
        except Exception as exc:  # noqa: BLE001
            logger.debug("rumor make failed: %s", exc)

        # ── Companion intervention: second in-scene voice on conflict ───────
        comp_id = str(req.get("companion_id") or "")
        if (
            comp_id
            and comp_id != npc_id
            and (npc_action not in ("none", "trade")
                 or npc_emotion in ("angry", "disgusted", "fearful"))
        ):
            asyncio.create_task(self._companion_react(
                req, npc_response=npc_response, npc_emotion=npc_emotion,
                base_req_id=str(req.get("req_id")), npc_id=npc_id,
                location=location, player_text=player_text,
            ))

        # ── Overheard: a bystander steps in when they were MENTIONED by name
        # or when the interlocutor judged the line alarming (HEARD:alarm) ──
        listener_id = str(req.get("listener_id") or "")
        heard_alarm = (
            isinstance(result, dict) and str(result.get("heard") or "") == "alarm"
        )
        if (
            listener_id
            and listener_id not in (npc_id, comp_id)
            and (req.get("listener_mentioned") or heard_alarm)
        ):
            req["listener_reason"] = ("mentioned" if req.get("listener_mentioned")
                                      else "overheard")
            asyncio.create_task(self._bystander_react(
                req, npc_response=npc_response,
                base_req_id=str(req.get("req_id")), npc_id=npc_id,
                location=location, player_text=player_text,
            ))

        # Reply req_id = echo of request so Lua dedup sees each new reply.
        reply = {
            "req_id": req.get("req_id"),
            "type": "dialogue",
            "npc_id": npc_id,
            "npc_response": npc_response,
            "emotion": npc_emotion,
            "action": npc_action,
            "target": npc_target,
            "disp": npc_disp,
            "gold": npc_gold,
            "item": npc_item,
            "loan": npc_loan,
            "deal": npc_deal,
            "cond": npc_cond,
            # Кем этот человек станет там, куда уходит: движок реально поселит
            # его в лавке, при таверне или на паперти.
            "fate": npc_fate,
            # Новая арка возвращается игре ОДИН раз — дальше она живёт в сейве.
            "companion_arc": new_arc,
            "voice": bool(req.get("voice")),
            "player_echo": (player_text if req.get("voice") else ""),
            "rumor": rumor_text,
            "life_facts": new_life_facts,
            "location": location,
            "timestamp": _now_iso(),
        }
        try:
            publish_reply(reply)
            logger.info("wrote inbox response for req_id=%s", reply["req_id"])
        except OSError as exc:
            logger.error("could not write inbox response: %s", exc)


# ----------------------------------------------------------- optional entry

async def _main() -> None:
    import yaml
    from agents.lore_agent import LoreAgent  # type: ignore
    from memory.chroma_memory import NPCMemory  # type: ignore

    logging.basicConfig(level=logging.INFO)
    cfg = yaml.safe_load(open("config.yaml"))
    mem = NPCMemory(cfg["memory"]["chroma_dir"])
    lore = LoreAgent(cfg)
    bridge = OpenMWLogBridge(cfg, lore, mem)
    await bridge.run()


if __name__ == "__main__":
    asyncio.run(_main())
