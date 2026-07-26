"""
tts.py — Fast Russian TTS for NPC replies via Silero (v4_ru), CPU.

Design goals:
  - FAST: model loads once in a background thread; synthesis of a 1-2 sentence
    NPC line takes ~0.3-0.8s on a Ryzen 7700 (well under the LLM latency).
  - Per-NPC stable voices: a deterministic hash of npc_id picks the speaker
    (male/female pools, matched to the NPC's gender) and a pitch level, so the
    same NPC always speaks with the same voice and no two neighbours sound alike.
  - Non-blocking: speak_async() queues the line; a serial worker speaks the
    queue in order so overlapping replies are heard one after another.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import wave
from pathlib import Path

from tts_queue import SerialSpeaker

logger = logging.getLogger(__name__)

MALE_SPEAKERS = ["aidar", "eugene"]
FEMALE_SPEAKERS = ["baya", "kseniya", "xenia"]
PITCHES = ["x-low", "low", "medium", "high", "x-high"]
SAMPLE_RATE = 48000


def _voice_for(npc_id: str, is_male: bool) -> tuple[str, str]:
    """Deterministic (speaker, pitch) for this NPC — stable across sessions."""
    h = int(hashlib.md5(npc_id.encode("utf-8", "ignore")).hexdigest(), 16)
    pool = MALE_SPEAKERS if is_male else FEMALE_SPEAKERS
    speaker = pool[h % len(pool)]
    pitch = PITCHES[(h // 7) % len(PITCHES)]
    return speaker, pitch


def _ssml_escape(text: str) -> str:
    return (text.replace("&", " и ").replace("<", " ").replace(">", " ")
                .replace('"', "'"))


class SileroTTS(SerialSpeaker):
    """Lazy-loading Silero TTS with per-NPC voices and queued playback."""

    def __init__(self, out_dir: str | Path, device: str = "cpu") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._model = None
        self._slot = 0
        self._ready = threading.Event()
        self._lock = threading.Lock()   # serialize synthesis calls
        threading.Thread(target=self._load, daemon=True, name="tts-load").start()
        self._start_speech_queue("silero")

    # ------------------------------------------------------------------ load

    def _load(self) -> None:
        try:
            import torch
            torch.set_num_threads(6)
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="ru",
                speaker="v4_ru",
                trust_repo=True,
            )
            model.to(self.device)
            self._model = model
            self._ready.set()
            logger.info("SileroTTS ready (v4_ru, %s)", self.device)
        except Exception as exc:  # noqa: BLE001
            logger.error("SileroTTS failed to load: %s — NPC voices disabled", exc)

    # ----------------------------------------------------------------- speak

    def _speak_blocking(self, text: str, npc_id: str, is_male: bool,
                        race: str = "", distance: float = 0.0) -> None:
        if not self._ready.wait(timeout=0.05) and self._model is None:
            # Model still loading (first minutes of first-ever run) — skip line.
            logger.info("TTS not ready yet; skipping voice for this line")
            return
        if npc_id == "narrator":
            # The Narrator: a voice no NPC has — unhurried, low, velvet (NPCs
            # never use rate variation, so this cadence is unique by design).
            speaker, pitch = "xenia", "low+slow"
            ssml = (f'<speak><prosody rate="slow" pitch="low">'
                    f'{_ssml_escape(text)}</prosody></speak>')
        else:
            speaker, pitch = _voice_for(npc_id, is_male)
            ssml = f'<speak><prosody pitch="{pitch}">{_ssml_escape(text)}</prosody></speak>'
        try:
            with self._lock:
                audio = self._model.apply_tts(
                    ssml_text=ssml, speaker=speaker, sample_rate=SAMPLE_RATE,
                )
            self._slot = (self._slot + 1) % 4
            path = self.out_dir / f"voice_{self._slot}.wav"
            self._write_wav(path, audio)
            # pygame, not winsound: winsound is silent when called off the main
            # thread, which is exactly where every reply is spoken from.
            from audio_out import play
            logger.info("TTS(silero): '%s' говорит (%s/%s, дист=%d)",
                        npc_id, speaker, pitch, int(distance or 0))
            play(str(path), distance, wait=True, npc_id=npc_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TTS synthesis failed: %s", exc)

    @staticmethod
    def _write_wav(path: Path, audio) -> None:
        import numpy as np
        pcm = (audio.numpy() * 32767.0).clip(-32768, 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
