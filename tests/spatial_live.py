"""
spatial_live.py — живая проверка звука, отданного ДВИЖКУ игры.

Проверяем не «файл появился», а то, что в слоте лежит НАСТОЯЩАЯ речь: синтез
идёт обычным путём мода, попадает в слот постоянного размера, а потом слот
распознаётся Whisper'ом и сверяется с исходной репликой. Заодно смотрим метку
для Lua — тот ли слот и тот ли говорящий.

Чего этот тест НЕ проверяет: как поведёт себя сама игра, когда прочитает слот.
Это можно узнать только запуском игры — VFS отдаёт скриптам опись файлов,
снятую на старте.

Запуск:  venv\\Scripts\\python.exe tests\\spatial_live.py
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

# Распознавание теперь на Vosk и живёт в нашем окружении.
WISPER_PY = ROOT / "venv" / "Scripts" / "python.exe"
LINE = ("Стой где стоишь, чужак. Дальше по мосткам будет таможня, "
        "а мне недосуг с тобой болтать.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

CHECKER = r'''
import sys, json, wave
import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("sttd", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
model = m._load_model()
path = sys.argv[2]
with wave.open(path, "rb") as w:
    rate, frames = w.getframerate(), w.getnframes()
    pcm = np.frombuffer(w.readframes(frames), dtype=np.int16).astype(np.float32) / 32768.0
a = m._to_16k_mono(pcm, rate)
print("@@RESULT@@" + json.dumps(
    {"sec": round(frames / rate, 2), "rate": rate,
     "text": m.transcribe(model, a, use_vad=False)}, ensure_ascii=False))
'''


def words(text: str) -> set[str]:
    return {w.strip(".,!?—-«»\"'").lower() for w in text.split() if len(w) > 3}


def main() -> int:
    import audio_out
    import spatial_voice as sv

    if not audio_out.enable_spatial(True):
        print("!! слоты звука не подготовились")
        return 1
    print(f"   слотов: {sv.SLOTS} по {sv.SLOT_BYTES} байт "
          f"({sv.SLOT_BYTES / 1024 / 1024:.1f} МБ всего: "
          f"{sv.SLOTS * sv.SLOT_BYTES / 1024 / 1024:.1f})")
    for i in range(sv.SLOTS):
        p = sv.slot_path(i)
        assert p.stat().st_size == sv.SLOT_BYTES, f"{p.name}: {p.stat().st_size}"
    print("   пустые слоты созданы, размеры одинаковые")

    from tts_morrowind import MorrowindTTS
    tts = MorrowindTTS(ROOT / "data" / "tts")
    t0 = time.time()
    while not tts.ready and time.time() - t0 < 120:
        time.sleep(0.5)
    if not tts.ready:
        print("!! голоса не поднялись (см. data/piper_daemon.log)")
        return 1
    print(f"   голоса готовы за {time.time() - t0:.1f}с: {', '.join(tts.voices)}")

    t0 = time.time()
    tts.speak_async(LINE, "живой_проверяющий", True, distance=600.0, race="Dunmer")
    deadline = time.time() + 90
    while time.time() < deadline:
        if sv.CUE_FILE.exists():
            body = sv.CUE_FILE.read_text(encoding="utf-8")
            cut = body[:body.rfind("}") + 1]
            try:
                cue = json.loads(cut)
            except json.JSONDecodeError:
                cue = {}
            if int(cue.get("seq", 0)) > 0:
                break
        time.sleep(0.2)
    else:
        print("!! метка для игры так и не появилась — звук движку не отдан")
        return 1
    print(f"   реплика ушла в движок за {time.time() - t0:.1f}с: "
          f"слот {cue['slot']}, говорящий «{cue['npc']}», громкость {cue['vol']}")

    if cue["npc"] != "живой_проверяющий":
        print(f"!! в метке чужой говорящий: {cue['npc']}")
        return 1
    if sv.CUE_FILE.stat().st_size != sv.CUE_BYTES:
        print(f"!! размер метки уплыл: {sv.CUE_FILE.stat().st_size}")
        return 1

    slot = sv.slot_path(int(cue["slot"]))
    if slot.stat().st_size != sv.SLOT_BYTES:
        print(f"!! размер слота уплыл: {slot.stat().st_size} — игра его не прочтёт")
        return 1
    with wave.open(str(slot), "rb") as w:
        secs = w.getnframes() / w.getframerate()
    print(f"   в слоте {slot.name}: {secs:.1f}с звука при неизменных "
          f"{slot.stat().st_size} байтах файла")
    if secs < 1.0:
        print("!! в слоте тишина")
        return 1

    if not WISPER_PY.exists():
        print("!! нет venv Whisper — содержимое слота проверить нечем")
        return 1
    script = ROOT / "data" / "tts" / "_spatial_check.py"
    script.write_text(CHECKER, encoding="utf-8")
    res = subprocess.run(
        [str(WISPER_PY), str(script), str(ROOT / "python" / "stt_daemon.py"), str(slot)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    marker = [l for l in (res.stdout or "").splitlines() if l.startswith("@@RESULT@@")]
    if not marker:
        print("!! распознавание не отработало:", (res.stderr or "")[-400:])
        return 1
    got = json.loads(marker[0][len("@@RESULT@@"):])
    heard = got["text"]
    print(f"\n   сказано:   {LINE}")
    print(f"   услышано:  {heard}")
    common = words(LINE) & words(heard)
    share = len(common) / max(1, len(words(LINE)))
    print(f"   совпало слов: {len(common)} из {len(words(LINE))} ({share:.0%}), "
          f"{got['sec']}с при {got['rate']} Гц")
    if share < 0.5:
        print("!! в слоте не та речь")
        return 1
    print("\n   ГОТОВО: настоящая речь лежит в слоте постоянного размера, "
          "метка для игры на месте.")
    print("   Осталось непроверенным: прочтёт ли слот сама игра — это видно "
          "только при запуске (tts.spatial: true).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
