"""
stt_shootout.py — очная ставка движков распознавания речи.

Одни и те же записи, одна метрика, четыре движка:
    whisper-medium  — что стоит сейчас (faster-whisper, процессор)
    vosk-small-ru   — «телефонная» модель, 45 МБ, потоковая
    vosk-ru-0.42    — большая потоковая, 1.8 ГБ
    gigaam-ctc      — русская модель Сбера

Мерим ДВЕ вещи, и вторая важнее:
    разбор  — сколько ушло на всю запись;
    хвост   — сколько остаётся ПОСЛЕ того, как игрок отпустил клавишу.
Потоковый движок жуёт звук по ходу речи, и хвост у него почти нулевой; Whisper
берётся за запись целиком только в конце, поэтому его разбор и есть хвост.

Точность — доля верно узнанных слов от эталона (WER наоборот). Записи
синтезированы, чтобы эталон был известен дословно; на живом голосе расклад
может отличаться, и это надо проверять микрофоном.

Запуск:  venv\\Scripts\\python.exe tests\\stt_shootout.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

WAVS = ROOT / "data" / "stt_shootout"
WISPER_PY = Path(r"D:\Wisper\Wisper\.venv\Scripts\python.exe")
VOSK_PY = ROOT / "tests" / "_vosk" / "Scripts" / "python.exe"
GIGA_PY = ROOT / "tests" / "_gigaam" / "Scripts" / "python.exe"

VOSK_SMALL = ROOT / "data" / "vosk" / "vosk-model-small-ru-0.22"
VOSK_BIG = ROOT / "data" / "vosk" / "vosk-model-ru-0.42"

# Что игрок реально говорит NPC: короткие фразы, имена собственные, числа.
PHRASES = [
    "Скажи, где здесь таможня и далеко ли до Балморы.",
    "Пойдём со мной, я заплачу двести золотых.",
    "Кто убил этого человека?",
    "Подожди меня здесь, я скоро вернусь.",
    "Что ты знаешь про контрабандистов на берегу?",
]


def words(t: str) -> list[str]:
    keep = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя-"
    out = []
    for w in (t or "").lower().split():
        w = "".join(c for c in w if c in keep)
        if w:
            out.append(w)
    return out


def accuracy(ref: str, got: str) -> float:
    """Доля слов эталона, найденных в ответе (порядок не учитываем)."""
    r, g = words(ref), words(got)
    pool = list(g)
    hit = 0
    for w in r:
        if w in pool:
            pool.remove(w)
            hit += 1
    return hit / max(1, len(r))


def make_wavs() -> list[Path]:
    from tts_piper import PIPER_EXE, MALE_MODEL
    WAVS.mkdir(parents=True, exist_ok=True)
    made = []
    for i, text in enumerate(PHRASES):
        p = WAVS / f"{i}.wav"
        if not p.exists():
            subprocess.run([str(PIPER_EXE), "-m", str(MALE_MODEL), "-f", str(p),
                            "--length_scale", "1.0", "--noise_w", "0.667"],
                           input=text.encode("utf-8"), capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW, timeout=60)
        made.append(p)
    return made


PROBE_WHISPER = r'''
import sys, json, time, wave
import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("sttd", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m._add_cuda_dlls(); model = m._load_model()
out = []
for path in sys.argv[2:]:
    with wave.open(path, "rb") as w:
        rate, n = w.getframerate(), w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    a = m._to_16k_mono(pcm, rate)
    m.transcribe(model, a, use_vad=False)          # прогрев
    t0 = time.time(); text = m.transcribe(model, a, use_vad=False); dt = time.time() - t0
    out.append({"file": path, "parse": round(dt, 2), "tail": round(dt, 2), "text": text})
print("@@" + json.dumps(out, ensure_ascii=False))
'''

PROBE_VOSK = r'''
import sys, json, time, wave
from vosk import Model, KaldiRecognizer, SetLogLevel
SetLogLevel(-1)
model = Model(sys.argv[1])
out = []
for path in sys.argv[2:]:
    with wave.open(path, "rb") as w:
        rate, n = w.getframerate(), w.getnframes(); pcm = w.readframes(n)
    rec = KaldiRecognizer(model, rate); rec.SetWords(False)
    step = int(rate * 0.2) * 2
    t0 = time.time()
    for i in range(0, len(pcm), step):
        rec.AcceptWaveform(pcm[i:i+step])
    parse = time.time() - t0
    t1 = time.time(); text = json.loads(rec.FinalResult()).get("text", ""); tail = time.time() - t1
    out.append({"file": path, "parse": round(parse, 2), "tail": round(tail, 2), "text": text})
print("@@" + json.dumps(out, ensure_ascii=False))
'''

PROBE_GIGA = r'''
import sys, json, time, wave
import torch, torchaudio
import gigaam.preprocess as pre

# GigaAM грузит звук через ffmpeg, которого в системе нет. Читаем wav сами:
# внешняя программа тут не нужна, а зависимость от неё только мешает.
def load_audio(audio_path, sample_rate=pre.SAMPLE_RATE, return_format="float"):
    with wave.open(audio_path, "rb") as w:
        rate, n = w.getframerate(), w.getnframes()
        pcm = torch.frombuffer(bytearray(w.readframes(n)), dtype=torch.int16).float()
    pcm = pcm / 32768.0
    if rate != sample_rate:
        pcm = torchaudio.functional.resample(pcm, rate, sample_rate)
    return pcm if return_format == "float" else (pcm * 32768.0).short()

pre.load_audio = load_audio
import gigaam
gigaam.preprocess.load_audio = load_audio
for mod in ("gigaam.model",):
    try:
        m = __import__(mod, fromlist=["*"])
        if hasattr(m, "load_audio"):
            m.load_audio = load_audio
    except Exception:
        pass

model = gigaam.load_model("ctc", device="cpu")
out = []
for path in sys.argv[1:]:
    model.transcribe(path)                          # прогрев
    t0 = time.time(); text = model.transcribe(path); dt = time.time() - t0
    out.append({"file": path, "parse": round(dt, 2), "tail": round(dt, 2), "text": text})
print("@@" + json.dumps(out, ensure_ascii=False))
'''


def run(py: Path, code: str, args: list[str], name: str) -> list[dict] | None:
    script = ROOT / "data" / f"_probe_{name}.py"
    script.write_text(code, encoding="utf-8")
    res = subprocess.run([str(py), str(script), *args], capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    for line in (res.stdout or "").splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    print(f"   !! {name} не отработал: {(res.stderr or '')[-300:]}")
    return None


def main() -> int:
    wavs = [str(p) for p in make_wavs()]
    total_audio = 0.0
    for p in wavs:
        with wave.open(p, "rb") as w:
            total_audio += w.getnframes() / w.getframerate()
    print(f"записей: {len(wavs)}, всего {total_audio:.1f}с речи\n")

    engines = [
        ("whisper-medium", lambda: run(WISPER_PY, PROBE_WHISPER,
                                       [str(ROOT / "python" / "stt_daemon.py"), *wavs],
                                       "whisper")),
        ("vosk-small-ru", lambda: run(VOSK_PY, PROBE_VOSK, [str(VOSK_SMALL), *wavs], "vosks")),
        ("vosk-ru-0.42", lambda: run(VOSK_PY, PROBE_VOSK, [str(VOSK_BIG), *wavs], "voskb")),
        ("gigaam-ctc", lambda: run(GIGA_PY, PROBE_GIGA, wavs, "giga")),
    ]

    table = []
    for name, fn in engines:
        print(f"— {name} …", flush=True)
        rows = fn()
        if not rows:
            continue
        acc = sum(accuracy(PHRASES[i], r["text"]) for i, r in enumerate(rows)) / len(rows)
        parse = sum(r["parse"] for r in rows)
        tail = sum(r["tail"] for r in rows) / len(rows)
        table.append({"engine": name, "acc": acc, "parse": parse, "tail": tail,
                      "texts": [r["text"] for r in rows]})
        for i, r in enumerate(rows):
            mark = "  " if accuracy(PHRASES[i], r["text"]) == 1.0 else "!!"
            print(f"   {mark} {r['text'][:76]}")

    print("\n" + "=" * 74)
    print(f" {'движок':<16}{'точность':>10}{'разбор':>10}{'хвост':>10}   (хвост = после клавиши)")
    for t in sorted(table, key=lambda x: -x["acc"]):
        print(f" {t['engine']:<16}{t['acc']*100:>9.0f}%{t['parse']:>9.1f}с{t['tail']:>9.2f}с")
    print("=" * 74)
    (ROOT / "data" / "stt_shootout.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
