"""
tts_piper.py — Piper TTS (local ONNX, CPU): fast and noticeably more natural
than Silero. ~0.5s per phrase including process startup.

Voices: ru_RU-dmitri-medium (male), ru_RU-irina-medium (female).
Per-NPC uniqueness: deterministic length_scale (tempo) + noise_w variation.
Narrator: irina, slow and velvet (tempo no NPC uses).

Interface matches tts.SileroTTS: speak_async(text, npc_id, is_male), stop().
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path

from tts_queue import SerialSpeaker, pitch_for, race_pitch, shift_pitch_wav

logger = logging.getLogger(__name__)

PIPER_DIR = Path(__file__).resolve().parent.parent / "piper"
PIPER_EXE = PIPER_DIR / "piper" / "piper.exe"
MALE_MODEL = PIPER_DIR / "ru_RU-dmitri-medium.onnx"
FEMALE_MODEL = PIPER_DIR / "ru_RU-irina-medium.onnx"

LENGTHS = ["0.88", "0.95", "1.0", "1.06", "1.14"]   # tempo variants
NOISES  = ["0.55", "0.667", "0.8"]                  # timbre variance

# Замеренная высота самих голосов: синтез трёх фраз, медиана по автокорреляции.
# Дмитрий заметно высок для мужчины, Ирина низка для женщины — оба сидят около
# 177 Гц, поэтому без поправки данмер и босмер звучали одинаково.
PIPER_HZ = {True: 178.0, False: 176.0}


def _voice_for(npc_id: str, is_male: bool) -> tuple[Path, str, str]:
    h = int(hashlib.md5(npc_id.encode("utf-8", "ignore")).hexdigest(), 16)
    model = MALE_MODEL if is_male else FEMALE_MODEL
    return model, LENGTHS[h % len(LENGTHS)], NOISES[(h // 13) % len(NOISES)]


class PiperTTS(SerialSpeaker):
    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._slot = 0
        self._ok = PIPER_EXE.exists() and MALE_MODEL.exists() and FEMALE_MODEL.exists()
        if not self._ok:
            logger.error("PiperTTS: files missing under %s — voices disabled", PIPER_DIR)
        else:
            logger.info("PiperTTS ready (dmitri/irina medium + tempo variants)")
        self._start_speech_queue("piper")

    def _speak_blocking(self, text: str, npc_id: str, is_male: bool,
                        race: str = "", distance: float = 0.0) -> None:
        if not self._ok:
            return
        pitch = 1.0
        if npc_id == "narrator":
            model, length, noise = FEMALE_MODEL, "1.28", "0.5"   # unhurried voice-over
        else:
            model, length, noise = _voice_for(npc_id, is_male)
            # Two base voices for a whole province is not enough: a personal
            # pitch keeps two guards from sounding like the same man.
            #
            # Раса важнее личного разброса: в самой игре данмер-мужчина звучит
            # на 80 Гц, а босмер на 171 — вдвое выше. Оба базовых голоса piper
            # стоят на ~177, так что без поправки все расы были на одно лицо.
            pitch = round(pitch_for(npc_id)
                          * race_pitch(race, is_male, PIPER_HZ[bool(is_male)]), 3)
            length = f"{float(length) * pitch:.3f}"   # компенсируем темп
        self._slot = (self._slot + 1) % 6
        path = self.out_dir / f"voice_{self._slot}.wav"
        try:
            proc = subprocess.run(
                [str(PIPER_EXE), "-m", str(model), "-f", str(path),
                 "--length_scale", length, "--noise_w", noise],
                input=text.encode("utf-8"),
                capture_output=True, timeout=20,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if proc.returncode != 0 or not path.exists():
                logger.warning("piper failed rc=%s: %s", proc.returncode,
                               proc.stderr.decode("utf-8", "ignore")[-200:])
                return
            shift_pitch_wav(str(path), pitch)
            from audio_out import play, volume_for_distance
            logger.info("TTS(piper): '%s' говорит (%s×%.2f, дист=%s, громкость=%.2f)",
                        npc_id, model.stem, pitch, int(distance or 0),
                        volume_for_distance(distance))
            play(str(path), distance, wait=True, npc_id=npc_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PiperTTS failed: %s", exc)
