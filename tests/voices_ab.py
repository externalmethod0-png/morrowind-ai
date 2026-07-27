"""Сравнение двух наборов голосов на слух распознавателя.

Вопрос всегда один: стало ли РАЗБОРЧИВЕЕ. Мнение тут не годится — прошлый раз
я на глаз решил, что переобучение помогло, а замер показал 30/42 против 32/42,
то есть шум.

Обе версии синтезируют ОДНИ И ТЕ ЖЕ фразы, обе идут в Vosk, считаем долю
верно узнанных слов. Голоса переключаются переменной MWAI_VOICES_DIR, которую
читает демон.

Запуск:
    venv\\Scripts\\python.exe tests\\voices_ab.py <старый_каталог> <новый_каталог>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

# Фразы обычной игровой длины, с разными звуками и без редких слов: меряем
# голос, а не словарь распознавателя.
LINES = [
    "Здравствуй чужеземец что тебе нужно в наших краях",
    "Я живу здесь всю жизнь и ничего подобного не видел",
    "Ступай своей дорогой и да хранят тебя Три",
    "Говорят на болотах опять видели пепельных тварей",
    "Не советую ходить туда одному без хорошего клинка",
    "Купец обещал заплатить но снова тянет время",
    "Стража сегодня злая лучше не попадайся им под руку",
]


def _pcm16k(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        sr, raw = w.getframerate(), w.readframes(w.getnframes())
        ch = w.getnchannels()
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        n = int(len(x) * 16000 / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return x.astype(np.int16).tobytes()


def synth(voices_dir: Path, pool: str, out_dir: Path) -> list[Path]:
    """Синтезируем все фразы указанным набором голосов, в отдельном процессе.

    Отдельный процесс — не прихоть: демон держит модели в памяти и берёт
    каталог из окружения ОДИН раз, при старте.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    script = out_dir / "_synth.py"
    script.write_text(
        "import sys, time, json\n"
        f"sys.path.insert(0, r'{ROOT / 'python'}')\n"
        "import audio_out\n"
        "audio_out.play = lambda *a, **k: None\n"
        "import tts_morrowind as t\n"
        "t.pitch_for = lambda npc_id: 1.0\n"
        "t.race_pitch = lambda race, male, base: 1.0\n"
        f"mw = t.MorrowindTTS(r'{out_dir}')\n"
        "time.sleep(3.0)\n"
        f"lines = json.loads(r'''{json.dumps(LINES, ensure_ascii=False)}''')\n"
        "import pathlib, shutil\n"
        f"outd = pathlib.Path(r'{out_dir}')\n"
        "for i, ln in enumerate(lines):\n"
        f"    mw._speak_blocking(ln, 'ab_%d' % i, {pool.endswith('m')!r},"
        f" {'\"dark elf\"' if pool.startswith('d') else '\"imperial\"'}, 0.0)\n"
        "    mw.wait_quiet(60)\n"
        "    time.sleep(0.3)\n"
        # Демон пишет в ШЕСТЬ слотов по кругу (mw_0…mw_5): седьмая фраза
        # затирала первую, и замер выходил короче на одну. Забираем копию
        # сразу, пока слот не переиспользован.
        "    got = sorted(outd.glob('mw_*.wav'), key=lambda p: p.stat().st_mtime)\n"
        "    if got: shutil.copy2(got[-1], outd / ('keep_%02d.wav' % i))\n",
        encoding="utf-8")

    env = dict(os.environ, MWAI_VOICES_DIR=str(voices_dir), PYTHONIOENCODING="utf-8")
    subprocess.run([str(ROOT / "venv" / "Scripts" / "python.exe"), str(script)],
                   env=env, capture_output=True, timeout=600)
    # Читаем СНЯТЫЕ копии, а не сами слоты: слотов шесть и они цикличны.
    return sorted(out_dir.glob("keep_*.wav"))


def score(files: list[Path], model) -> tuple[float, list[str]]:
    from vosk import KaldiRecognizer
    hits = total = 0
    heard = []
    for line, path in zip(LINES, files):
        rec = KaldiRecognizer(model, 16000)
        rec.AcceptWaveform(_pcm16k(path))
        said = json.loads(rec.FinalResult()).get("text", "")
        heard.append(said)
        got = said.lower().split()
        want = [w.lower() for w in line.split()]
        hits += sum(1 for w in want if w in got)
        total += len(want)
    return (hits / total if total else 0.0), heard


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    old_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
    pool = sys.argv[3] if len(sys.argv) > 3 else "dm"
    for d in (old_dir, new_dir):
        if not d.exists():
            print(f"нет каталога: {d}")
            return 1

    sys.stderr.write("грузим Vosk...\n")
    from vosk import Model, SetLogLevel
    SetLogLevel(-1)
    vosk = Model(str(ROOT / "data" / "vosk" / "vosk-ru-0.42-fast"))

    tmp = Path(tempfile.mkdtemp(prefix="mwai_ab_"))
    results = {}
    for tag, d in (("СТАРЫЙ", old_dir), ("НОВЫЙ", new_dir)):
        files = synth(d, pool, tmp / tag)
        if len(files) < len(LINES):
            print(f"{tag}: синтезировано {len(files)} из {len(LINES)} — "
                  f"замер неполный, дальше не иду")
            return 1
        acc, heard = score(files[-len(LINES):], vosk)
        results[tag] = (acc, heard)
        print(f"{tag:<8} разобрано {acc * 100:5.1f}%   ({d.name})")

    print()
    old_acc, new_acc = results["СТАРЫЙ"][0], results["НОВЫЙ"][0]
    delta = (new_acc - old_acc) * 100
    print(f"разница: {delta:+.1f} процентных пункта")
    # Порог честности: меньше пяти пунктов на семи фразах — это шум, а не
    # улучшение. Прошлый раз именно так и вышло.
    if abs(delta) < 5:
        print("ВЫВОД: разницы нет, это шум замера.")
    else:
        print("ВЫВОД: разница есть." if delta > 0 else "ВЫВОД: стало ХУЖЕ.")

    print("\nчто услышал распознаватель (новый набор):")
    for line, said in zip(LINES, results["НОВЫЙ"][1]):
        mark = "  " if said.lower() == line.lower() else "≠ "
        print(f"  {mark}{said[:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
