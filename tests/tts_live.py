"""
tts_live.py — живая проверка озвучки: синтез длинной реплики и КОНТРОЛЬ того,
что получилось, распознаванием.

Ловит настоящую поломку: при куске длиннее лимита XTTS (182 символа для
русского) модель срывается в бесконечное «рррр» — файл есть, звук идёт,
а речи в нём нет. Тест синтезирует, распознаёт результат и сверяет с
эталоном, а затем проигрывает вслух.

Запуск:  venv\\Scripts\\python.exe tests\\tts_live.py
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

# Распознавание теперь на Vosk и живёт в нашем окружении.
WISPER_PY = ROOT / "venv" / "Scripts" / "python.exe"

REPLY = ("Ещё один чужак с вопросами. Иди прямо по мосткам до конца пристани, "
         "там будет таможня. И не вздумай задерживаться у моего дома, н'вах. "
         "В Балморе тебя ждут дела поважнее, чем болтовня со мной. "
         "Ступай себе, пока я добрый.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
fails: list[str] = []

CHECKER = r'''
import sys, json
import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("sttd", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
model = m._load_model()
out = []
for path in json.load(open(sys.argv[2], encoding="utf-8")):
    with open(path, "rb") as fh: raw = fh.read()
    i = raw.find(b"data"); body = raw[i+8:]
    bits = int.from_bytes(raw[34:36], "little")
    pcm = (np.frombuffer(body, dtype=np.float32) if bits == 32
           else np.frombuffer(body, dtype=np.int16).astype(np.float32) / 32768.0)
    a = m._to_16k_mono(pcm, 24000)
    out.append({"path": path, "sec": round(len(pcm) / 24000, 2),
                "text": m.transcribe(model, a, use_vad=False)})
print("@@RESULT@@" + json.dumps(out, ensure_ascii=False))
'''


def main() -> int:
    from tts_xtts import XttsTTS

    tts = XttsTTS(ROOT / "data" / "tts")
    t0 = time.time()
    while not tts.ready and time.time() - t0 < 300:
        time.sleep(0.5)
    if not tts.ready:
        print("!! демон XTTS не поднялся (см. data/xtts_daemon.log)")
        return 1
    print(f"   демон XTTS готов за {time.time() - t0:.1f}с")

    ref = tts._ref_for("live_test_npc", True, "Dunmer")
    out = ROOT / "data" / "tts" / "live_check.wav"
    print(f"   реплика: {len(REPLY)} символов, эталон голоса {Path(ref).name}")

    t0 = time.time()
    chunks: list[str] = []
    with tts._lock:
        tts._proc.stdin.write(json.dumps(
            {"cmd": "say", "text": REPLY, "ref": ref, "out": str(out)},
            ensure_ascii=False) + "\n")
        tts._proc.stdin.flush()
        while True:
            msg = tts._read_reply()
            if "chunk" in msg:
                if not chunks:
                    print(f"   первый фрагмент готов через {time.time() - t0:.1f}с")
                chunks.append(msg["chunk"])
                continue
            if not msg.get("ok"):
                print(f"!! синтез не удался: {msg.get('err')}")
                return 1
            break
    print(f"   синтез целиком: {time.time() - t0:.1f}с, фрагментов: {len(chunks)}")

    manifest = ROOT / "data" / "tts" / "_live_chunks.json"
    manifest.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    if not WISPER_PY.exists():
        print("!! нет venv Whisper — проверить содержимое нечем")
        return 1
    script = ROOT / "data" / "tts" / "_check.py"
    script.write_text(CHECKER, encoding="utf-8")
    res = subprocess.run(
        [str(WISPER_PY), str(script), str(ROOT / "python" / "stt_daemon.py"), str(manifest)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    marker = [l for l in (res.stdout or "").splitlines() if l.startswith("@@RESULT@@")]
    if not marker:
        print("!! проверка не отработала:", (res.stderr or "")[-400:])
        return 1

    print("\n   что реально произнесено:")
    total = 0.0
    for item in json.loads(marker[0][len("@@RESULT@@"):]):
        text, sec = item["text"], item["sec"]
        total += sec
        print(f"     [{sec:5.1f}с] {text[:70]}")
        letters = set(text.replace(" ", "").lower())
        if len(text) > 12 and len(letters) <= 3:
            fails.append(f"срыв в повтор: {text[:40]!r}")
        if not text.strip():
            fails.append("фрагмент без речи")
    speed = len(REPLY) / total if total else 0
    print(f"\n   всего {total:.1f}с на {len(REPLY)} символов ({speed:.1f} симв/с)")
    if speed < 8:
        fails.append(f"речь неестественно растянута ({speed:.1f} симв/с)")

    print("\n>>> ПРОИГРЫВАЮ ВСЛУХ — слушай колонки <<<")
    from audio_out import play
    for c in chunks:
        play(c, 120, wait=True)

    print("\n" + "=" * 58)
    if fails:
        print(" ОЗВУЧКА НЕИСПРАВНА:")
        for f in fails:
            print(f"   - {f}")
        return 1
    print(" ОЗВУЧКА ИСПРАВНА (проверено распознаванием)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
