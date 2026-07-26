"""
piper_daemon.py — синтез ДООБУЧЕННЫМИ голосами Morrowind.

Живёт в окружении обучения (piper_train_env), потому что модели, полученные
новым обучением, старый piper.exe не понимает: он падает на фонемах из двух
символов («aɪ»). Новый пакет piper-tts их читает.

Голоса держатся загруженными в памяти — иначе на каждую реплику уходило бы
полсекунды только на чтение модели с диска.

Протокол (строки JSON через stdin/stdout):
  in : {"cmd":"say","text":"...","voice":"dm","out":"C:/...wav","pitch":1.04}
  out: {"ok":true,"out":"...","sec":0.6}  |  {"ok":false,"err":"..."}
Печатает {"ready":true,"voices":[...]} после загрузки.
"""

from __future__ import annotations

import json
import os
import sys
import wave
from pathlib import Path

MOD_ROOT = Path(__file__).resolve().parent.parent
# Каталог голосов можно подменить — так сравнивают разные прогоны обучения
# между собой, не трогая рабочие модели.
VOICES_DIR = Path(os.environ.get("MWAI_VOICES_DIR")
                  or (MOD_ROOT / "piper" / "morrowind"))
SR = 22050

_PROTO = None      # захватывается в main(), см. комментарий там


def _out(obj) -> None:
    ch = _PROTO or sys.stdout
    ch.write(json.dumps(obj, ensure_ascii=False) + "\n")
    ch.flush()


def _claim_protocol_channel():
    """Отдаём протоколу отдельный канал.

    onnxruntime и espeak печатают предупреждения в общий вывод; одна такая
    строка посреди протокола — и мост считает демон сломанным. Стороннее уходит
    в лог, протокол остаётся чистым.
    """
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    return os.fdopen(proto_fd, "w", encoding="utf-8", buffering=1)


def shift_pitch(path: str, pitch: float) -> None:
    """Личная высота голоса поверх дообученного тембра.

    Голосов у нас четыре — по расе и полу, — а людей в игре сотни. Без сдвига
    все данмеры звучали бы одним человеком.
    """
    if abs(pitch - 1.0) < 0.005:
        return
    import numpy as np
    with wave.open(path, "rb") as w:
        params = w.getparams()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if params.nchannels > 1:
        pcm = pcm.reshape(-1, params.nchannels).mean(axis=1).astype(np.int16)
    n_out = max(1, int(len(pcm) / pitch))
    out = np.interp(np.linspace(0, len(pcm) - 1, n_out),
                    np.arange(len(pcm)), pcm.astype(np.float32))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(params.framerate)
        w.writeframes(np.clip(out, -32768, 32767).astype(np.int16).tobytes())


def main() -> None:
    global _PROTO
    _PROTO = _claim_protocol_channel()
    try:
        from piper import PiperVoice
    except ImportError as exc:
        _out({"ready": False, "err": f"нет пакета piper: {exc}"})
        return

    voices = {}
    for path in sorted(VOICES_DIR.glob("ru_RU-morrowind-*.onnx")):
        pool = path.stem.split("-")[-1]
        try:
            voices[pool] = PiperVoice.load(str(path))
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"голос {pool} не загрузился: {exc}\n")
    if not voices:
        _out({"ready": False, "err": f"в {VOICES_DIR} нет голосов"})
        return
    _out({"ready": True, "voices": sorted(voices)})

    import time
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            continue
        if cmd.get("cmd") == "quit":
            return
        if cmd.get("cmd") != "say":
            continue
        try:
            t0 = time.time()
            pool = str(cmd.get("voice") or "dm")
            voice = voices.get(pool) or next(iter(voices.values()))
            out_path = str(cmd["out"])
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with wave.open(out_path, "wb") as w:
                voice.synthesize_wav(str(cmd.get("text") or ""), w)
            shift_pitch(out_path, float(cmd.get("pitch") or 1.0))
            _out({"ok": True, "out": out_path, "sec": round(time.time() - t0, 2)})
        except Exception as exc:  # noqa: BLE001
            _out({"ok": False, "err": str(exc)})


if __name__ == "__main__":
    main()
