"""
mic_live.py — проверка «голос в текст» вживую, с твоим микрофоном.

Три попытки: скрипт говорит «ГОВОРИ», пишет 5 секунд и показывает, что
распознал. Это ровно тот путь, что работает по клавише V в игре.

Запуск:  venv\\Scripts\\python.exe tests\\mic_live.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

PHRASES = [
    "Скажи что-нибудь обычное, например: «Здравствуй, где найти таможню?»",
    "Теперь тише и быстрее — как в разговоре",
    "И последнее — любая фраза, какую скажешь NPC в игре",
]


def main() -> int:
    from voice_stt import VoiceSTT

    stt = VoiceSTT(device_hint="fifine")
    t0 = time.time()
    while not stt.ready and time.time() - t0 < 180:
        time.sleep(0.5)
    if not stt.ready:
        print("!! демон распознавания не поднялся — см. data/stt_daemon.log")
        return 1
    print(f"микрофон готов ({time.time() - t0:.1f}с)\n")

    heard: list[str] = []
    for i, hint in enumerate(PHRASES, 1):
        print(f"--- попытка {i} из {len(PHRASES)} ---")
        print(f"    {hint}")
        for n in (3, 2, 1):
            print(f"    начинаю через {n}...", flush=True)
            time.sleep(1)

        async def take() -> str:
            await stt.ptt_start()
            print("    >>> ГОВОРИ (5 секунд) <<<", flush=True)
            await asyncio.sleep(5)
            print("    ...распознаю", flush=True)
            return await stt.ptt_stop()

        text = asyncio.run(take())
        heard.append(text)
        print(f"    УСЛЫШАНО: {text!r}\n" if text
              else "    УСЛЫШАНО: (пусто)\n")

    log = ROOT / "data" / "stt_daemon.log"
    if log.exists():
        print("--- уровни записи (лог демона) ---")
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("ptt:"):
                print("   " + line)

    ok = sum(1 for t in heard if t.strip())
    print("\n" + "=" * 58)
    if ok == len(PHRASES):
        print(" ГОЛОС В ТЕКСТ РАБОТАЕТ — распознаны все попытки")
        return 0
    print(f" РАСПОЗНАНО {ok} из {len(PHRASES)} — покажи вывод целиком")
    return 1


if __name__ == "__main__":
    sys.exit(main())
