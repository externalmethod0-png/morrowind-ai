"""
vc_convert.py — перевод тембра в голос игрового актёра (шаг 2 из 2).

ЗАПУСКАТЬ ОКРУЖЕНИЕМ XTTS: xtts\\venv\\Scripts\\python.exe — там уже стоит
torch с CUDA, отдельного ставить не нужно.

Идея проверки: быстрый синтез (piper 0.6 с, silero 1.8 с) звучит чётко, но
чужим голосом. Преобразователь тембра берёт готовую речь и переливает её в
голос актёра из самой игры. Если это дёшево по времени — получаем родной тембр
за долю секунды вместо пяти секунд XTTS.

Движок — knn-vc: он не требует обучения под голос, ему нужен НАБОР ОБРАЗЦОВ
целевого актёра. У нас их 610 на голос — ровно то, для чего он сделан.

Запуск:  xtts\\venv\\Scripts\\python.exe tests\\vc_convert.py
"""

from __future__ import annotations

import glob
import json
import sys
import time
import wave
from pathlib import Path

import torch
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "vc_shootout"
VO = ROOT.parent / "OPENMW" / "Data Files" / "Sound" / "Vo" / "d" / "m"
REFS = 40                      # столько клипов актёра берём в образцы


def save(wav: torch.Tensor, path: Path, rate: int = 16000) -> float:
    pcm = (wav.squeeze().clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return len(pcm) / rate


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"устройство: {dev}")

    t0 = time.time()
    knn = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True,
                         trust_repo=True, pretrained=True, device=dev)
    print(f"   преобразователь поднялся за {time.time() - t0:.1f}с")

    clips = sorted(glob.glob(str(VO / "*.mp3")))
    if not clips:
        print(f"!! нет образцов голоса в {VO}")
        return 1
    # Берём клипы из середины списка: первые в озвучке часто самые короткие.
    picked = clips[len(clips) // 3:][:REFS]
    t0 = time.time()
    matching = knn.get_matching_set(picked)
    build = time.time() - t0
    print(f"   образцы актёра: {len(picked)} клипов, набор собран за {build:.1f}с")

    rows = []
    for src_name, tag in (("1_piper.wav", "5_piper+тембр_актёра.wav"),
                          ("2_silero.wav", "6_silero+тембр_актёра.wav")):
        src = OUT / src_name
        if not src.exists():
            print(f"   нет исходника {src_name}")
            continue
        t0 = time.time()
        q = knn.get_features(str(src))
        out = knn.match(q, matching, topk=4)
        dt = time.time() - t0
        dst = OUT / tag
        sec = save(out, dst)
        rows.append({"from": src_name, "file": tag, "convert": round(dt, 2),
                     "sec": round(sec, 1)})
        print(f"   {src_name} -> {tag}: перевод {dt:.2f}с")

    (OUT / "convert.json").write_text(json.dumps(
        {"refs": len(picked), "build": round(build, 1), "device": dev, "rows": rows},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
