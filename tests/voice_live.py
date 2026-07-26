"""
voice_live.py — живая проверка голосового ввода (клавиша V) без микрофона.

Поднимает демон распознавания ровно так, как это делает мост, и прогоняет
через него заведомо известную фразу. Проверяет то, на чём голос ломался:
  - демон вообще поднимается (CUDA/CPU),
  - в канале протокола нет постороннего вывода,
  - распознавание возвращает текст, а не пустоту,
  - цикл записи (нажал/отпустил V) отрабатывает без ошибок.

Запуск:  venv\\Scripts\\python.exe tests\\voice_live.py
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

PHRASE = "Здравствуй. Я хочу купить меч и немного припасов."
WAV = ROOT / "data" / "voice_probe.wav"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
fails: list[str] = []


def make_probe() -> bool:
    """Записываем эталонную фразу голосом piper (локально, без сети)."""
    piper = ROOT / "piper" / "piper" / "piper.exe"
    model = ROOT / "piper" / "ru_RU-dmitri-medium.onnx"
    if not (piper.exists() and model.exists()):
        return False
    WAV.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(piper), "-m", str(model), "-f", str(WAV)],
                       input=PHRASE.encode("utf-8"), capture_output=True)
    return r.returncode == 0 and WAV.exists()


def main() -> int:
    from voice_stt import VoiceSTT

    if not make_probe():
        print("!! не удалось создать эталонную запись (нет piper)")
        return 1
    print(f"эталон: {WAV.name}")

    stt = VoiceSTT(device_hint="fifine")
    t0 = time.time()
    while not stt.ready and time.time() - t0 < 180:
        time.sleep(0.5)
    if not stt.ready:
        print("!! демон распознавания не поднялся")
        return 1
    print(f"   демон готов за {time.time() - t0:.1f}с")

    # 1) распознавание заведомо известной фразы через живой демон
    resp = stt._cmd({"cmd": "transcribe", "path": str(WAV)})
    text = str(resp.get("text") or "")
    if not text:
        print(f"!! фраза не распознана: {resp}")
        fails.append("распознавание возвращает пустоту")
    else:
        words = [w for w in ("купить", "меч", "припас") if w in text.lower()]
        print(f"   распознано: {text!r} ({resp.get('sec')}с)")
        if len(words) < 2:
            fails.append(f"текст не совпадает с эталоном: {text!r}")

    # 2) цикл нажал/отпустил V
    async def ptt_cycle() -> bool:
        ok = await stt.ptt_start()
        await asyncio.sleep(1.5)
        await stt.ptt_stop()
        return ok

    if not asyncio.run(ptt_cycle()):
        print("!! запись по нажатию V не стартует")
        fails.append("ptt_start не работает")
    else:
        print("   цикл записи V отработал")

    # 3) в канале протокола не должно быть постороннего вывода
    if not stt.ready:
        print("!! демон помечен мёртвым — в канал попал посторонний вывод")
        fails.append("протокол засорён")

    log = ROOT / "data" / "stt_daemon.log"
    if log.exists():
        tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-4:]
        print("\n--- лог демона ---")
        for line in tail:
            print("   " + line)

    print("\n" + "=" * 58)
    if fails:
        print(" ГОЛОСОВОЙ ВВОД НЕ РАБОТАЕТ:")
        for f in fails:
            print(f"   - {f}")
        return 1
    print(" ГОЛОСОВОЙ ВВОД РАБОТАЕТ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
