"""
tts_edge.py — Natural Russian NPC voices via Microsoft Edge neural TTS.

Quality is dramatically better than Silero; needs internet (like Gemini).
Two base Russian neural voices (Dmitry / Svetlana) are spread into many
per-NPC variants via deterministic pitch/rate offsets — the same NPC always
sounds the same, gender-matched.

Playback goes through audio_out (pygame) so distance attenuation applies.
Interface matches the other backends:
    speak_async(text, npc_id, is_male, distance=0.0, race="")
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path

from tts_queue import SerialSpeaker

logger = logging.getLogger(__name__)

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

MALE_VOICE = "ru-RU-DmitryNeural"
FEMALE_VOICE = "ru-RU-SvetlanaNeural"
PITCHES = ["-15Hz", "-8Hz", "+0Hz", "+8Hz", "+15Hz"]
RATES = ["-8%", "+0%", "+8%"]


def _voice_for(npc_id: str, is_male: bool) -> tuple[str, str, str]:
    h = int(hashlib.md5(npc_id.encode("utf-8", "ignore")).hexdigest(), 16)
    voice = MALE_VOICE if is_male else FEMALE_VOICE
    return voice, PITCHES[h % len(PITCHES)], RATES[(h // 11) % len(RATES)]


class EdgeTTS(SerialSpeaker):
    """Async Edge-TTS with per-NPC stable voice variants; lines are queued and
    spoken in order, with volume falling off over in-game distance."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._slot = 0
        logger.info("EdgeTTS ready (%s / %s + pitch/rate variants)",
                    MALE_VOICE, FEMALE_VOICE)
        self._start_speech_queue("edge")

    def _speak_blocking(self, text: str, npc_id: str, is_male: bool,
                        race: str = "", distance: float = 0.0) -> None:
        if npc_id == "narrator":
            voice, pitch, rate = FEMALE_VOICE, "-12Hz", "-20%"   # unique cadence
        else:
            voice, pitch, rate = _voice_for(npc_id, is_male)
        self._slot = (self._slot + 1) % 6
        path = self.out_dir / f"voice_{self._slot}.mp3"
        try:
            import edge_tts

            async def synth() -> None:
                com = edge_tts.Communicate(text, voice=voice, pitch=pitch, rate=rate)
                await com.save(str(path))

            asyncio.run(synth())
            from audio_out import play, volume_for_distance
            logger.info("TTS(edge): '%s' говорит (%s %s %s, дист=%d, громк=%.2f)",
                        npc_id, voice, pitch, rate, int(distance or 0),
                        volume_for_distance(distance))
            play(str(path), distance, wait=True, npc_id=npc_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("EdgeTTS failed (%s) — line skipped", exc)
