"""
voices_live.py — проверка, что разные NPC звучат по-разному.

В озвучке Morrowind на расу и пол приходится ОДИН актёр, поэтому клонирование
давало всем имперским стражникам один голос. Тест синтезирует одну фразу от
имени трёх NPC одной расы и пола и МЕРЯЕТ высоту основного тона: голоса должны
разойтись, а один и тот же NPC — звучать одинаково всегда.

Запуск:  venv\\Scripts\\python.exe tests\\voices_live.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

PHRASE = "Проходи, чужак, и не задерживайся здесь надолго."
NPCS = ["0x101f7c2", "0x101f7c4", "0x101f7c5"]   # трое из сессии игрока

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def pitch_hz(path: str) -> float:
    """Средняя высота основного тона (автокорреляция по звонким кадрам)."""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    x = pcm.astype(np.float64) / 32768.0
    frame, hop = int(sr * 0.04), int(sr * 0.02)
    lo, hi = int(sr / 350), int(sr / 70)          # 70…350 Гц — голосовой диапазон
    vals = []
    for i in range(0, len(x) - frame, hop):
        f = x[i:i + frame]
        if np.sqrt(np.mean(f ** 2)) < 0.02:
            continue
        f = f - f.mean()
        corr = np.correlate(f, f, mode="full")[len(f) - 1:]
        seg = corr[lo:hi]
        if len(seg) == 0 or corr[0] <= 0:
            continue
        peak = int(np.argmax(seg)) + lo
        if corr[peak] / corr[0] > 0.3:
            vals.append(sr / peak)
    return float(np.median(vals)) if vals else 0.0


def main() -> int:
    from tts_xtts import XttsTTS

    tts = XttsTTS(ROOT / "data" / "tts")
    t0 = time.time()
    while not tts.ready and time.time() - t0 < 300:
        time.sleep(0.5)
    if not tts.ready:
        print("!! демон XTTS не поднялся")
        return 1

    def synth(npc: str, tag: str) -> str:
        ref = tts._ref_for(npc, True, "Imperial")
        out = ROOT / "data" / "tts" / f"voice_check_{tag}.wav"
        with tts._lock:
            tts._proc.stdin.write(json.dumps(
                {"cmd": "say", "text": PHRASE, "ref": ref, "out": str(out),
                 "pitch": tts._pitch_for(npc)}, ensure_ascii=False) + "\n")
            tts._proc.stdin.flush()
            while True:
                msg = tts._read_reply()
                if "chunk" in msg:
                    continue
                if not msg.get("ok"):
                    raise RuntimeError(msg.get("err"))
                break
        return str(out)

    print(f"фраза: {PHRASE!r}\n")
    pitches = []
    for i, npc in enumerate(NPCS):
        path = synth(npc, str(i))
        hz = pitch_hz(path)
        pitches.append(hz)
        print(f"   NPC {npc}: сдвиг ×{tts._pitch_for(npc):.3f} -> тон {hz:.0f} Гц")

    again = pitch_hz(synth(NPCS[0], "repeat"))
    print(f"   тот же NPC повторно: {again:.0f} Гц")

    fails = []
    spread = (max(pitches) - min(pitches)) / min(pitches) * 100
    drift = abs(again - pitches[0]) / pitches[0] * 100
    print(f"\n   разброс между NPC: {spread:.1f}%   собственный разброс модели: {drift:.1f}%")
    # Разница между персонажами обязана быть заметно больше, чем колебания
    # одной и той же реплики от прогона к прогону — иначе её просто не слышно.
    if spread < 10:
        fails.append(f"голоса почти одинаковы (разброс {spread:.1f}%)")
    if spread < drift * 1.5:
        fails.append(f"разница между NPC ({spread:.1f}%) тонет в разбросе модели ({drift:.1f}%)")

    print("\n>>> ПРОИГРЫВАЮ ТРИ ГОЛОСА ПОДРЯД <<<")
    from audio_out import play
    for i in range(len(NPCS)):
        play(str(ROOT / "data" / "tts" / f"voice_check_{i}.wav"), 120, wait=True)

    print("\n" + "=" * 58)
    if fails:
        print(" ГОЛОСА НЕ РАЗЛИЧАЮТСЯ:")
        for f in fails:
            print(f"   - {f}")
        return 1
    print(" ГОЛОСА РАЗЛИЧАЮТСЯ, И КАЖДЫЙ NPC СТАБИЛЕН")
    return 0


if __name__ == "__main__":
    sys.exit(main())
