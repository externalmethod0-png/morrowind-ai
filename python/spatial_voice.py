"""
spatial_voice.py — отдать реплику ДВИЖКУ игры, чтобы голос звучал из NPC.

Сейчас озвучка играется мимо игры (pygame), и громкость мы подгоняем сами по
расстоянию. Движок умеет лучше: звук идёт из точки, где стоит собеседник, —
слышно направление, его глушат стены, он замолкает вместе с паузой в игре.

Расплата — правила VFS. Игра составляет опись файлов при запуске и потом отдаёт
скриптам ТОТ размер, что был на старте. Значит:

  * файлы-слоты создаются ДО запуска игры;
  * каждый слот всегда одного и того же размера — синтез дописывается нулями;
  * в заголовке WAV длина настоящая, поэтому хвост из нулей не звучит.

Слотов несколько по кругу — пока играет один, следующая реплица пишется в
другой и не обрывает предыдущую.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

MOD_ROOT = Path(__file__).resolve().parent.parent
SOUND_DIR = MOD_ROOT / "openmw-mod" / "Sound" / "mwai"
CUE_FILE = MOD_ROOT / "openmw-mod" / "ai_inbox" / "voice_cue.txt"

SLOTS = 4
SLOT_SECONDS = 22          # самая длинная реплика, что влезет в слот
SLOT_BYTES = 22 * 48000 * 2 + 1024   # 22 с при 48 кГц моно 16 бит + запас
CUE_BYTES = 512            # у метки тоже постоянный размер — правило то же

# Тишина в слоте до первого синтеза: настоящий заголовок WAV с нулевой длиной
# данных. Игра такой файл открывает и молча закрывает, а не сыплет ошибками.
def _silent_wav(sample_rate: int = 22050) -> bytes:
    return (b"RIFF" + struct.pack("<I", 36) + b"WAVEfmt " + struct.pack("<I", 16)
            + struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16)
            + b"data" + struct.pack("<I", 0))


def _atomic_write(path: Path, blob: bytes) -> None:
    """Подменяем файл целиком: игра никогда не увидит его недописанным."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def slot_path(index: int) -> Path:
    return SOUND_DIR / f"voice_{index % SLOTS}.wav"


def vfs_name(index: int) -> str:
    """Путь так, как его видит игра (внутри каталога мода)."""
    return f"Sound/mwai/voice_{index % SLOTS}.wav"


def prepare_slots() -> list[Path]:
    """Создать слоты и метку ДО запуска игры. Идемпотентно."""
    made = []
    for i in range(SLOTS):
        p = slot_path(i)
        if not p.exists() or p.stat().st_size != SLOT_BYTES:
            blob = _silent_wav()
            _atomic_write(p, blob + b"\0" * (SLOT_BYTES - len(blob)))
        made.append(p)
    if not CUE_FILE.exists() or CUE_FILE.stat().st_size != CUE_BYTES:
        write_cue(seq=0, index=0, volume=0.0, npc_id="")
    return made


def fit_into_slot(wav_bytes: bytes) -> bytes | None:
    """Дополнить синтез нулями до размера слота.

    Возвращает None, если реплика в слот НЕ влезла: обрезать речь на полуслове
    хуже, чем сыграть её обычным путём — пусть вызывающий откатится на pygame.
    """
    if len(wav_bytes) > SLOT_BYTES:
        return None
    return wav_bytes + b"\0" * (SLOT_BYTES - len(wav_bytes))


def write_cue(seq: int, index: int, volume: float, npc_id: str) -> None:
    """Сказать игре, какой слот играть. Размер метки постоянный."""
    line = json.dumps({"seq": int(seq), "slot": int(index) % SLOTS,
                       "vol": round(float(volume), 3), "npc": str(npc_id)},
                      ensure_ascii=False)
    blob = line.encode("utf-8")
    if len(blob) > CUE_BYTES:
        blob = blob[:CUE_BYTES]
    _atomic_write(CUE_FILE, blob + b" " * (CUE_BYTES - len(blob)))


class SpatialSink:
    """Слоты по кругу плюс счётчик реплик."""

    def __init__(self) -> None:
        self._next = 0
        self._seq = 0
        self.available = False
        try:
            prepare_slots()
            self.available = True
        except Exception as exc:  # noqa: BLE001
            logger.error("слоты звука не готовы (%s) — играю мимо игры", exc)

    def play(self, wav_path: str | Path, volume: float, npc_id: str) -> bool:
        """Отдать файл движку. False — не вышло, играй обычным путём."""
        if not self.available:
            return False
        try:
            blob = Path(wav_path).read_bytes()
        except OSError as exc:
            logger.warning("не читается синтез %s: %s", wav_path, exc)
            return False
        padded = fit_into_slot(blob)
        if padded is None:
            logger.info("реплика длиннее слота (%d байт) — играю мимо игры", len(blob))
            return False
        idx = self._next % SLOTS
        self._next += 1
        self._seq += 1
        try:
            _atomic_write(slot_path(idx), padded)
            write_cue(self._seq, idx, volume, npc_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("слот звука не записался: %s", exc)
            return False
        logger.debug("звук отдан движку: слот %d, реплика %d", idx, self._seq)
        return True
