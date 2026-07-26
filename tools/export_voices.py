"""
export_voices.py — превращает обучающие чекпоинты в лёгкие модели для игры.

Обучающий файл весит 807 МБ и содержит состояние оптимизатора, которое в игре
не нужно. Экспорт оставляет только саму модель (~60 МБ) в формате, который
понимает piper.exe.

Рядом с каждой моделью кладётся её config.json — без него голос не заговорит.

Запуск:  piper_train_env\\venv\\Scripts\\python.exe tools\\export_voices.py
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / "piper_train_env"
PY = ENV / "venv" / "Scripts" / "python.exe"
RUNS = ENV / "runs"
OUT = ROOT / "piper" / "morrowind"          # рядом с базовыми голосами piper

POOLS = {
    "dm": "данмер мужской", "df": "данмер женский",
    "im": "имперец мужской", "if": "имперка женская",
}


def newest_checkpoint(pool: str) -> Path | None:
    """Берём last.ckpt — он на самом дальнем шаге обучения."""
    files = glob.glob(str(RUNS / pool / "**" / "last.ckpt"), recursive=True)
    if not files:
        return None
    return Path(max(files, key=os.path.getmtime))


def export(pool: str) -> bool:
    ckpt = newest_checkpoint(pool)
    if ckpt is None:
        print(f"  {pool}: нет обученного файла")
        return False
    OUT.mkdir(parents=True, exist_ok=True)
    onnx = OUT / f"ru_RU-morrowind-{pool}.onnx"

    proc = subprocess.run(
        [str(PY), "-m", "piper.train.export_onnx",
         "--checkpoint", str(ckpt), "--output-file", str(onnx)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT),
    )
    if proc.returncode != 0 or not onnx.exists():
        print(f"  {pool}: ПРОВАЛ экспорта")
        for line in (proc.stderr or proc.stdout or "").splitlines()[-6:]:
            print("      " + line)
        return False

    # Конфиг обучения кладём рядом: piper ищет <модель>.onnx.json
    src_cfg = RUNS / pool / "config.json"
    if not src_cfg.exists():
        print(f"  {pool}: нет config.json — голос не заработает")
        return False
    shutil.copyfile(src_cfg, onnx.with_suffix(".onnx.json"))

    size = onnx.stat().st_size / 1024 ** 2
    print(f"  {POOLS[pool]:<18} -> {onnx.name}  ({size:.0f} МБ)")
    return True


def main() -> int:
    print(f"экспорт в {OUT}\n")
    ok = {p: export(p) for p in POOLS}
    print()
    good = [p for p, v in ok.items() if v]
    print(f"готово голосов: {len(good)} из {len(POOLS)}")
    return 0 if len(good) == len(POOLS) else 1


if __name__ == "__main__":
    sys.exit(main())
