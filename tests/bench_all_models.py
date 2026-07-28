"""
bench_all_models.py — прогнать несколько моделей по одному правилу.

Модели надо мерить в ОДИНАКОВЫХ условиях, иначе сравнение врёт. Одна беда уже
случилась: две модели остались в памяти одновременно, вторая вытеснилась на
процессор, и её цифры оказались втрое хуже настоящих.

Поэтому здесь: выгрузить всё → загрузить одну → убедиться, что в памяти
только она → прогнать оба стенда → выгрузить. Загрузка одинаковая для всех:
полный сброс на видеокарту и один и тот же контекст.

Запуск:  venv\\Scripts\\python.exe tests\\bench_all_models.py [модель ...]
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LMS = Path(os.environ.get("LMS_EXE",
                          Path.home() / ".lmstudio" / "bin" / "lms.exe"))
PY = ROOT / "venv" / "Scripts" / "python.exe"
CONTEXT = os.environ.get("MWAI_BENCH_CTX", "8192")
TEMP = os.environ.get("MWAI_BENCH_TEMP", "0.3")


def lms(*args: str, timeout: int = 300) -> str:
    r = subprocess.run([str(LMS), *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    return (r.stdout or "") + (r.stderr or "")


def loaded_models() -> list[str]:
    """Имена загруженных моделей — и ничего кроме них.

    Когда в памяти пусто, `lms ps` печатает не пустоту, а подсказку («To load
    a model, run: lms load …»). Первый разбор принял её слова за имена
    моделей и пытался выгружать «To» и «lms».
    """
    out = lms("ps")
    if "No models are currently loaded" in out:
        return []
    names, started = [], False
    for line in out.splitlines():
        if line.startswith("IDENTIFIER"):
            started = True
            continue
        if not started or not line.strip():
            continue
        names.append(line.split()[0])
    return names


def swap_to(model: str) -> bool:
    """Оставить в памяти РОВНО одну модель. Возвращает False, если не вышло."""
    for m in loaded_models():
        print(f"  выгружаю {m}", flush=True)
        lms("unload", m)
    time.sleep(3)
    still = loaded_models()
    if still:
        print(f"  не выгрузились: {still}")
        return False

    print(f"  загружаю {model} (контекст {CONTEXT}, всё на видеокарту)", flush=True)
    out = lms("load", model, "--gpu", "max", "--context-length", CONTEXT,
              "--yes", timeout=600)
    time.sleep(3)
    now = loaded_models()
    if len(now) != 1:
        print(f"  в памяти оказалось {now}; вывод загрузчика:\n{out[-400:]}")
        return len(now) >= 1
    print(f"  в памяти: {now[0]}", flush=True)
    return True


def run_bench(script: str, model: str, rounds: str, out_file: Path) -> str:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MWAI_BENCH_TEMP=TEMP,
               MWAI_BENCH_MAXTOK="160")
    r = subprocess.run([str(PY), str(ROOT / "tests" / script), model, rounds],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=5400)
    text = (r.stdout or "") + (r.stderr or "")
    out_file.write_text(text, encoding="utf-8")
    for line in text.splitlines():
        if "ИТОГ" in line:
            return line.strip()
    return "итога нет — смотри " + out_file.name


def main() -> int:
    models = sys.argv[1:] or ["gigachat3.1-10b-a1.8b",
                              "saineko-hydra-remix-ru-12b-i1",
                              "vikhr-nemo-dostoevsky-saiga-12b-i1"]
    results = {}
    for model in models:
        print(f"\n{'=' * 74}\n{model}\n{'=' * 74}", flush=True)
        if not swap_to(model):
            results[model] = {"диалоги": "не загрузилась", "сцены": "—"}
            continue
        # ПОЛНОЕ имя, а не первое слово: под «saineko» подходят и q2_k, и
        # q3_k_s, и стенд намерил бы не ту квантизацию, о которой отчитался.
        safe = model.replace("@", "_").replace("/", "_")[:20]
        d = run_bench("bench_local_models.py", model, "4",
                      ROOT / "data" / f"bench_{safe}_dlg.txt")
        print(f"  диалоги: {d}", flush=True)
        s = run_bench("bench_local_scenes.py", model, "3",
                      ROOT / "data" / f"bench_{safe}_scn.txt")
        print(f"  сцены:   {s}", flush=True)
        results[model] = {"диалоги": d, "сцены": s}

    (ROOT / "data" / "bench_all.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{'=' * 74}\nСВОДКА\n{'=' * 74}")
    for m, r in results.items():
        print(f"{m}\n   {r['диалоги']}\n   {r['сцены']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
