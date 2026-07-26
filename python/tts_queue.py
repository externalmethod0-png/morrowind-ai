"""
tts_queue.py — serial speech queue shared by every TTS backend.

Backends used to cancel any pending line the moment a new one arrived
("newest wins"). In a real scene two or three lines land within a second of
each other — the NPC's answer, a bystander reacting, a companion butting in —
so each line cancelled the previous and the player heard nothing at all.

Lines are now queued and spoken one after another. Only stop() (the player
closing the conversation) clears the queue.

A backend mixes this in and implements:
    _speak_blocking(text, npc_id, is_male, race, distance)
and calls _start_speech_queue() from its __init__.
"""

from __future__ import annotations

import logging
import queue
import threading

logger = logging.getLogger(__name__)

MAX_QUEUED = 6   # a scene is a few speakers; more than that means we lag badly

# Personal pitch per NPC, the same one every time. Any TTS backend has a small
# pool of base voices, so without this every guard sounds like every other one.
PITCH_RANGE = (0.88, 0.92, 0.96, 1.0, 1.04, 1.08, 1.12)


def pitch_for(npc_id: str) -> float:
    import hashlib
    h = int(hashlib.md5(("pitch:" + str(npc_id)).encode("utf-8", "ignore")).hexdigest(), 16)
    return round(PITCH_RANGE[h % len(PITCH_RANGE)], 3)


def shift_pitch_wav(path: str, pitch: float) -> None:
    """Shift a finished 16-bit wav in place, pitch and formants together.

    Resampling the samples while keeping the declared rate makes the file play
    back that much faster, which is what reads as a different person.
    """
    if abs(pitch - 1.0) < 0.005:
        return
    import wave
    import numpy as np
    with wave.open(path, "rb") as w:
        params = w.getparams()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if params.nchannels > 1:
        pcm = pcm.reshape(-1, params.nchannels).mean(axis=1).astype(np.int16)
    n_out = max(1, int(len(pcm) / pitch))
    idx = np.linspace(0, len(pcm) - 1, n_out)
    out = np.interp(idx, np.arange(len(pcm)), pcm.astype(np.float32))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(params.framerate)
        w.writeframes(np.clip(out, -32768, 32767).astype(np.int16).tobytes())


class SerialSpeaker:
    def _start_speech_queue(self, name: str = "tts") -> None:
        self._q: queue.Queue = queue.Queue()
        self._epoch = 0
        threading.Thread(target=self._speak_worker, daemon=True,
                         name=f"{name}-speak").start()

    def epoch(self) -> int:
        """Token identifying the current exchange.

        A backend that speaks a reply in several pieces compares this between
        pieces: if the exchange is over, the rest must not be played.
        """
        return getattr(self, "_epoch", 0)

    def new_turn(self) -> None:
        """The player said something new — the previous exchange is over.

        Lines of ONE exchange (the NPC, a bystander butting in, a companion)
        must all be heard, so they queue. But a line answering the player's
        previous message is stale the moment they type again: keeping it only
        filled the queue until new replies were dropped unspoken.
        """
        self.stop()

    def speak_async(self, text: str, npc_id: str, is_male: bool,
                    distance: float = 0.0, race: str = "") -> None:
        text = (text or "").strip()
        if not text:
            return
        q = getattr(self, "_q", None)
        if q is None:
            return
        if q.qsize() >= MAX_QUEUED:
            logger.warning("TTS: очередь переполнена, реплика '%s' пропущена", text[:40])
            return
        q.put((text, npc_id, is_male, race, distance))

    def stop(self) -> None:
        """Player closed the window: drop what is pending and cut the sound."""
        self._epoch = getattr(self, "_epoch", 0) + 1
        q = getattr(self, "_q", None)
        if q is not None:
            try:
                while True:
                    q.get_nowait()
                    q.task_done()
            except queue.Empty:
                pass
        try:
            from audio_out import stop as stop_audio
            stop_audio()
        except Exception:  # noqa: BLE001
            pass

    def _speak_worker(self) -> None:
        while True:
            text, npc_id, is_male, race, distance = self._q.get()
            try:
                self._speak_blocking(text, npc_id, is_male, race, distance)
            except Exception as exc:  # noqa: BLE001
                logger.error("TTS worker: %s", exc)
            finally:
                self._q.task_done()
