"""
split_check.py — проверка, что нагрузка действительно разведена по устройствам.

Запускать ПОСЛЕ того, как игра была запущена хотя бы раз: смотрит, на чём
рисовала игра, на чём считает озвучка, на чём распознавание, и какие задержки
получились по журналу моста.

Запуск:  venv\\Scripts\\python.exe tests\\split_check.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GAME_LOG = Path(r"D:\Morrowind (ReBuild)\OPENMW\openmw.log")
BRIDGE_LOG = ROOT / "data" / "bridge.log"

fails: list[str] = []


def check_game_gpu() -> None:
    if not GAME_LOG.exists():
        print("!! openmw.log не найден — игра ещё не запускалась")
        fails.append("нет лога игры")
        return
    head = GAME_LOG.read_text(encoding="utf-8", errors="replace")[:4000]
    m = re.search(r"OpenGL Renderer: (.+)", head)
    if not m:
        print("!! в логе игры нет строки OpenGL Renderer")
        fails.append("не удалось определить видеокарту игры")
        return
    renderer = m.group(1).strip()
    print(f"   игра рисует на : {renderer}")
    if "NVIDIA" in renderer.upper() or "CMP" in renderer.upper():
        fails.append(f"игра всё ещё на ускорителе ({renderer}) — переключение не сработало")


def check_tts_device() -> None:
    if not BRIDGE_LOG.exists():
        print("!! bridge.log не найден")
        fails.append("нет лога моста")
        return
    text = BRIDGE_LOG.read_text(encoding="utf-8", errors="replace")
    m = None
    for m in re.finditer(r"XTTS daemon ready=(\w+) device=(\w+)", text):
        pass
    if m:
        print(f"   озвучка считает на : {m.group(2)} (готов={m.group(1)})")
        if m.group(2) != "cuda":
            fails.append(f"озвучка не на ускорителе ({m.group(2)})")
    else:
        print("   озвучка: XTTS в журнале не найден (движок piper?)")

    stt = ROOT / "data" / "stt_daemon.log"
    if stt.exists():
        for line in stt.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("whisper:"):
                print(f"   распознавание на : {line.split(':', 1)[1].strip()}")
                if "cpu" not in line:
                    fails.append("распознавание не на процессоре")
                break


def check_latency() -> None:
    """Сколько прошло от готового ответа до первого звука."""
    if not BRIDGE_LOG.exists():
        return
    times: list[tuple[datetime, str]] = []
    for line in BRIDGE_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"(\d\d:\d\d:\d\d) ", line)
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%H:%M:%S")
        if "wrote inbox response" in line:
            times.append((t, "reply"))
        elif "TTS(" in line:
            times.append((t, "voice"))
    gaps = []
    pending = None
    for t, kind in times:
        if kind == "reply":
            pending = t
        elif pending is not None:
            gaps.append((t - pending).total_seconds())
            pending = None
    if gaps:
        recent = gaps[-6:]
        print(f"\n   задержка ответ->голос (последние {len(recent)}): "
              + ", ".join(f"{g:.0f}с" for g in recent))
        worst = max(recent)
        if worst > 12:
            fails.append(f"голос опаздывает на {worst:.0f}с")
    else:
        print("\n   реплик с озвучкой в журнале пока нет")

    if "очередь переполнена" in BRIDGE_LOG.read_text(encoding="utf-8", errors="replace")[-8000:]:
        fails.append("очередь озвучки переполняется")


def main() -> int:
    print("РАСКЛАДКА НАГРУЗКИ\n")
    check_game_gpu()
    check_tts_device()
    check_latency()

    smi = Path(r"C:\Windows\System32\nvidia-smi.exe")
    if smi.exists():
        r = subprocess.run([str(smi), "--query-gpu=name,utilization.gpu,memory.used",
                            "--format=csv,noheader"], capture_output=True, text=True)
        print(f"\n   ускоритель сейчас: {(r.stdout or '').strip()}")

    print("\n" + "=" * 58)
    if fails:
        print(" РАСКЛАДКА НЕ СОШЛАСЬ:")
        for f in fails:
            print(f"   - {f}")
        return 1
    print(" ВСЁ РАЗВЕДЕНО ПРАВИЛЬНО")
    return 0


if __name__ == "__main__":
    sys.exit(main())
