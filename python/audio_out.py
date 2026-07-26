"""
audio_out.py — playback with distance attenuation.

A voice from across the square must not hit the ear like a whisper next to it.
Volume follows an inverse-ish curve over the in-game distance the Lua side
reports (OpenMW units: ~64 = one metre, dialogue range ~500, earshot ~1500).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

NEAR_UNITS = 220.0    # closer than this = full volume
FAR_UNITS  = 1600.0   # beyond this = barely audible
MIN_VOL    = 0.12

_mixer_ready = False
_spatial = None       # SpatialSink, когда голос отдан движку игры


def enable_spatial(on: bool) -> bool:
    """Играть голоса ЧЕРЕЗ ДВИЖОК, из точки, где стоит NPC.

    Возвращает то, что получилось на самом деле: если слоты подготовить не
    вышло, остаёмся на обычном воспроизведении и говорим об этом честно.
    """
    global _spatial
    if not on:
        _spatial = None
        return False
    try:
        from spatial_voice import SpatialSink
        sink = SpatialSink()
    except Exception as exc:  # noqa: BLE001
        logger.error("пространственный звук не поднялся (%s) — играю мимо игры", exc)
        _spatial = None
        return False
    _spatial = sink if sink.available else None
    return _spatial is not None


def wav_seconds(path: str) -> float:
    """Длительность по заголовку WAV. 0.0, если файл непонятный."""
    try:
        import wave
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            return w.getnframes() / rate if rate else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def volume_for_distance(distance: float) -> float:
    """1.0 up close, fading to MIN_VOL at the edge of earshot."""
    try:
        d = float(distance or 0.0)
    except (TypeError, ValueError):
        d = 0.0
    if d <= NEAR_UNITS:
        return 1.0
    if d >= FAR_UNITS:
        return MIN_VOL
    t = (d - NEAR_UNITS) / (FAR_UNITS - NEAR_UNITS)
    # perceptual-ish falloff: quiet drops fast at first, then levels out
    return max(MIN_VOL, 1.0 - (t ** 0.65) * (1.0 - MIN_VOL))


def _ensure_mixer() -> bool:
    global _mixer_ready
    if _mixer_ready:
        return True
    try:
        import pygame
        pygame.mixer.init()
        _mixer_ready = True
    except Exception as exc:  # noqa: BLE001
        logger.error("pygame mixer init failed: %s — voices muted", exc)
    return _mixer_ready


def play(path: str, distance: float = 0.0, wait: bool = False,
         npc_id: str = "") -> None:
    """Play a file at the volume its distance deserves.

    With wait=True the call returns only when the line has finished, so a queue
    of speakers is heard one after another instead of each one cutting off the
    previous (an NPC answer plus a bystander butting in used to cancel each
    other and nothing was heard at all).
    """
    vol = volume_for_distance(distance)

    # Голос из самого NPC: слот отдаём движку, а очередь реплик всё равно
    # держим — ждём столько, сколько длится сама речь, иначе следующая реплика
    # начнёт говорить поверх этой.
    if _spatial is not None and _spatial.play(path, 1.0, npc_id):
        if wait:
            import time as _t
            _t.sleep(wav_seconds(path) + 0.15)
        return

    if not _ensure_mixer():
        return
    try:
        import pygame
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.set_volume(vol)
        pygame.mixer.music.play()
        logger.debug("playing %s at vol=%.2f (dist=%s)", path, vol, distance)
        if wait:
            import time as _t
            while pygame.mixer.music.get_busy():
                _t.sleep(0.05)
    except Exception as exc:  # noqa: BLE001
        logger.warning("playback failed: %s", exc)


def stop() -> None:
    if not _mixer_ready:
        return
    try:
        import pygame
        pygame.mixer.music.stop()
    except Exception:  # noqa: BLE001
        pass
