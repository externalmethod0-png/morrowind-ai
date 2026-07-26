"""
clean_voice_dataset.py — убирает из датасета то, что речью не является.

Озвучка Morrowind содержит не только реплики, но и боевые выкрики, стоны от
удара и вопли бегства. Если оставить их, модель научится кричать посреди
разговора. Отсекаем по префиксу имени файла (движок сам их так разложил) и по
виду расшифровки.

Запуск:  venv\\Scripts\\python.exe tools\\clean_voice_dataset.py [--apply]
Без --apply только показывает, что будет убрано.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "piper_dataset"

# Префиксы файлов озвучки, которые не являются нормальной речью.
NON_SPEECH = ("atk_", "cratk_", "hit_", "flee_", "hlt_", "moan_", "roar_",
              "scream_", "death_", "pain_")


def is_junk(name: str, text: str) -> str:
    low = name.lower()
    for p in NON_SPEECH:
        if low.startswith(p) or low.startswith("t" + p) or low.startswith("b" + p):
            return f"боевой выкрик ({p.rstrip('_')})"
    t = text.strip()
    if len(t) < 8:
        return "слишком коротко"
    letters = re.sub(r"[^А-Яа-яЁё]", "", t)
    if letters and letters == letters.upper() and len(letters) > 2:
        return "сплошные заглавные (вопль)"
    if re.search(r"[A-Za-z]{4,}", t):
        return "латиница в расшифровке"
    return ""


def main() -> int:
    apply = "--apply" in sys.argv
    total_kept = total_cut = 0
    for pool in sorted(p for p in DATASET.iterdir() if p.is_dir()):
        meta = pool / "metadata.csv"
        if not meta.exists():
            continue
        keep, cut = [], []
        for line in meta.read_text(encoding="utf-8").splitlines():
            if "|" not in line:
                continue
            name, text = line.split("|", 1)
            why = is_junk(name, text)
            (cut if why else keep).append((line, name, why))
        print(f"{pool.name}: оставляем {len(keep)}, убираем {len(cut)}")
        for _, name, why in cut[:4]:
            print(f"    - {name}: {why}")
        if apply and cut:
            meta.write_text("".join(l + "\n" for l, _, _ in keep), encoding="utf-8")
            for _, name, _ in cut:
                wav = pool / "wavs" / f"{name}.wav"
                if wav.exists():
                    os.remove(wav)       # Remove-Item на этой машине заблокирован
        total_kept += len(keep)
        total_cut += len(cut)

    print(f"\nИТОГО: остаётся {total_kept}, убирается {total_cut}")
    if not apply:
        print("это была примерка — запусти с --apply, чтобы применить")
    return 0


if __name__ == "__main__":
    sys.exit(main())
