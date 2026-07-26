"""
voice_stt.py — bridge-side client for the STT daemon (stt_daemon.py running
under the Wisper venv, CUDA faster-whisper on the CMP 70HX).

Keeps the daemon warm; listen() records from the mic and returns the text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Распознавание переехало на Vosk и считает на процессоре, поэтому демон живёт
# в НАШЕМ окружении. Отдельный venv Whisper'а с одолженными у XTTS библиотеками
# CUDA больше не нужен — вместе с ним ушёл и целый класс поломок: чужой вывод в
# канал протокола, ненайденные cublas и драка за видеокарту.
WISPER_PYTHON = Path(__file__).resolve().parent.parent / "venv" / "Scripts" / "python.exe"
DAEMON = Path(__file__).resolve().parent / "stt_daemon.py"


class VoiceSTT:
    def __init__(self, device_hint: str = "", compute_device: str = "cpu") -> None:
        self.device_hint = device_hint
        self.compute_device = compute_device
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.ready = False
        if not WISPER_PYTHON.exists():
            logger.error("VoiceSTT: Wisper python not found at %s — voice mode off", WISPER_PYTHON)
            return
        threading.Thread(target=self._start, daemon=True, name="stt-start").start()

    def _start(self) -> None:
        try:
            # Keep the daemon's diagnostics (gate levels, recorded peak): with
            # stderr in DEVNULL a silent mic is indistinguishable from a broken
            # one, and that cost us an evening.
            err_path = Path(__file__).resolve().parent.parent / "data" / "stt_daemon.log"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            self._err = err_path.open("w", encoding="utf-8", errors="replace")
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                       MWAI_STT_DEVICE=self.compute_device)
            self._proc = subprocess.Popen(
                [str(WISPER_PYTHON), str(DAEMON)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._err, text=True, encoding="utf-8", errors="replace",
                env=env, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            info = self._read_reply()
            self.ready = bool(info.get("ready"))
            # Устройство спрашиваем у самого демона: строка была захардкожена
            # «CUDA», и лог уверял ровно в обратном тому, что считалось.
            logger.info("VoiceSTT: демон готов=%s, распознавание на %s",
                        self.ready, str(info.get("device") or "?"))
        except Exception as exc:  # noqa: BLE001
            logger.error("VoiceSTT start failed: %s", exc)

    def _listen_blocking(self, max_s: float) -> str:
        with self._lock:
            if not (self.ready and self._proc and self._proc.poll() is None):
                return ""
            try:
                self._proc.stdin.write(json.dumps(
                    {"cmd": "listen", "max_s": max_s, "device": self.device_hint}) + "\n")
                self._proc.stdin.flush()
                resp = self._read_reply()
                if resp.get("ok"):
                    txt = str(resp.get("text") or "").strip()
                    logger.info("VoiceSTT: heard %r (%.1fs)", txt[:60], resp.get("sec", 0))
                    return txt
                logger.warning("VoiceSTT error: %s", resp.get("err"))
            except Exception as exc:  # noqa: BLE001
                logger.error("VoiceSTT listen failed: %s", exc)
                self.ready = False
            return ""

    async def listen(self, max_s: float = 8.0) -> str:
        return await asyncio.to_thread(self._listen_blocking, max_s)

    # ── push-to-talk: record while the key is held ──────────────────────────

    def _read_reply(self) -> dict:
        """Read the daemon's answer, stepping over any stray output.

        One line printed into the pipe by a native library (PortAudio, CUDA)
        used to kill voice input for the rest of the session: the decode error
        marked the daemon dead. A line we cannot parse simply is not our reply.
        """
        for _ in range(50):
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("демон распознавания закрыл поток")
            line = line.strip()
            if not line.startswith("{"):
                logger.warning("STT: посторонний вывод в канале: %.80s", line)
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                logger.warning("STT: нечитаемая строка в канале: %.80s", line)
        raise RuntimeError("ответ демона распознавания не найден")

    def _cmd(self, payload: dict) -> dict:
        with self._lock:
            if not (self.ready and self._proc and self._proc.poll() is None):
                return {}
            try:
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
                return self._read_reply()
            except Exception as exc:  # noqa: BLE001
                logger.error("VoiceSTT command failed: %s", exc)
                self.ready = False
                return {}

    async def ptt_start(self) -> bool:
        resp = await asyncio.to_thread(
            self._cmd, {"cmd": "ptt_start", "device": self.device_hint})
        ok = bool(resp.get("ok"))
        logger.info("VoiceSTT: запись начата (%s)", "ok" if ok else resp.get("err"))
        return ok

    async def ptt_stop(self) -> str:
        resp = await asyncio.to_thread(self._cmd, {"cmd": "ptt_stop"})
        txt = str(resp.get("text") or "").strip()
        logger.info("VoiceSTT: услышано %r (%.1fs)", txt[:60], resp.get("sec", 0))
        return txt
