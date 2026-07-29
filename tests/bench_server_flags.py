"""
bench_server_flags.py — подбор настроек llama-server под игру.

ЦЕЛЬ: реплика NPC за 3 секунды и меньше, 4 — потолок терпимого.

Замер идёт на РЕАЛИСТИЧНОМ сценарии, а не на удобном: между репликами
проходит сценка между NPC (она вытирает кеш разговорной записки), и каждый
раз спрашивают РАЗНОГО персонажа (у двух NPC общего начала 89%, остальное
считается заново). Мерить на одном и том же NPC подряд — обманывать себя:
так выходит 0.8 с вместо честных 2.5.

Время реплики складывается из двух частей, и лечатся они разным:
    разбор промпта  — сколько токенов в секунду читается
    генерация       — сколько токенов в секунду пишется
Поэтому обе части снимаются отдельно, из журнала сервера.

Запуск:  venv\\Scripts\\python.exe tests\\bench_server_flags.py
         venv\\Scripts\\python.exe tests\\bench_server_flags.py A B D   (только эти)
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

EXE = ROOT / "llamacpp" / "llama-server.exe"
MODEL = (Path.home() / ".lmstudio" / "models" / "ai-sage" /
         "GigaChat3.1-10B-A1.8B-GGUF" / "GigaChat3.1-10B-A1.8B-q4_K_M.gguf")
SLOTS = ROOT / "data" / "slots"
LOGS = ROOT / "data" / "flagbench"
PORT = 8090
URL = f"http://127.0.0.1:{PORT}"

BASE = ["-m", str(MODEL), "--alias", "gigachat3.1-10b-a1.8b",
        "-ngl", "999", "--slot-save-path", str(SLOTS),
        "--host", "127.0.0.1", "--port", str(PORT)]

# Настройки перебора. Каждая — гипотеза, а не «попробуем что-нибудь».
CONFIGS: dict[str, tuple[str, list[str]]] = {
    "A": ("как сейчас (основа)",
          ["-c", "8192", "-np", "1"]),
    "B": ("flash-attention включён",
          ["-c", "8192", "-np", "1", "-fa", "on"]),
    "C": ("крупный физический батч (разбор промпта)",
          ["-c", "8192", "-np", "1", "-fa", "on", "-ub", "2048", "-b", "4096"]),
    "D": ("KV-кеш сжат до q8_0 (память под второй слот)",
          ["-c", "8192", "-np", "1", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"]),
    "E": ("два слота: сценки не трогают разговорный кеш",
          ["-c", "16384", "-np", "2", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"]),
    "G": ("два слота, KV без сжатия",
          ["-c", "16384", "-np", "2", "-fa", "on"]),
    "H": ("два слота, контекст по нужде (4096 на слот)",
          ["-c", "8192", "-np", "2", "-fa", "on"]),
    "I": ("два слота, по нужде, KV сжат",
          ["-c", "8192", "-np", "2", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"]),
    "F": ("два слота + крупный батч",
          ["-c", "16384", "-np", "2", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
           "-ub", "2048", "-b", "4096"]),
}

NPCS = [("Фаргот", "Bosmer", "Commoner", "Сейда Нин", 60),
        ("Тидрал", "Dunmer", "Mercenary", "Балмора", 40),
        ("Арилл", "Dunmer", "Trader", "Сейда Нин", 70),
        ("Индрель", "Dunmer", "Guard", "Вивек", 30),
        ("Телери", "Dunmer", "Commoner", "Хла Оуд", 55),
        ("Раванус", "Imperial", "Pauper", "Балмора", 45)]
LINES = ["Здравствуй. Где тут работу найти?",
         "Слыхал про пропавшее кольцо?",
         "Идём со мной, дело есть.",
         "Стой тут и жди меня.",
         "Сколько возьмёшь за помощь?",
         "Что говорят про стражу в порту?"]
SCENE_TASK = "Фаргот и Водуниус переговариваются у причала. Три такта."


def notes():
    from agents.lore_agent import _build_system_prompt
    from agents.scene_agent import _schema
    talk = [_build_system_prompt(npc_name=n, npc_race=r, npc_class=k,
                                 npc_faction="", location=l, disposition=d,
                                 life_facts=[], lite=True)
            for n, r, k, l, d in NPCS]
    return talk, _schema(["Фаргот", "Водуниус Нуцциус", "Телери Хельви"])


def post(path: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        URL + path, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def ask(system: str, user: str, slot: int | None = None,
        max_tokens: int = 160) -> tuple[float, int]:
    payload = {"messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}],
               "temperature": 0.5, "max_tokens": max_tokens, "stream": False}
    if slot is not None:
        payload["id_slot"] = slot
    t = time.time()
    out = post("/v1/chat/completions", payload)
    dt = time.time() - t
    gen = (out.get("usage") or {}).get("completion_tokens") or 0
    return dt, gen


def start(flags: list[str], log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen([str(EXE), *BASE, *flags], stdout=fh,
                            stderr=subprocess.STDOUT, cwd=str(EXE.parent))
    for _ in range(180):
        try:
            urllib.request.urlopen(URL + "/health", timeout=3).read()
            return proc, fh
        except Exception:  # noqa: BLE001
            if proc.poll() is not None:
                fh.close()
                return None, None
            time.sleep(2)
    proc.kill()
    fh.close()
    return None, None


def stop(proc, fh) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except Exception:  # noqa: BLE001
        proc.kill()
    if fh:
        fh.close()
    time.sleep(3)


_PROMPT_RE = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
_EVAL_RE = re.compile(r"\|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens")


def speeds(log: Path) -> tuple[float, float]:
    """Скорость разбора промпта и генерации — из журнала самого сервера."""
    text = log.read_text(encoding="utf-8", errors="replace")
    pt = [(float(a), int(b)) for a, b in _PROMPT_RE.findall(text)]
    ev = [(float(a), int(b)) for a, b in _EVAL_RE.findall(text)]
    p = statistics.median([n / (ms / 1000) for ms, n in pt if ms > 0]) if pt else 0
    e = statistics.median([n / (ms / 1000) for ms, n in ev if ms > 0]) if ev else 0
    return p, e


def measure(key: str) -> dict | None:
    human, flags = CONFIGS[key]
    log = LOGS / f"{key}.log"
    print(f"\n{'=' * 70}\n{key}. {human}\n   {' '.join(flags)}\n{'=' * 70}", flush=True)

    proc, fh = start(flags, log)
    if proc is None:
        tail = log.read_text(encoding="utf-8", errors="replace")[-400:] if log.exists() else ""
        print(f"   сервер не поднялся:\n{tail}")
        return {"key": key, "human": human, "ok": False}

    try:
        talk, scene = notes()
        two = "-np" in flags and flags[flags.index("-np") + 1] != "1"
        dlg_slot, scn_slot = (0, 1) if two else (None, None)

        ask(talk[0], "разогрев", slot=dlg_slot)      # первый запрос всегда дорог
        times, gens = [], []
        for i in range(len(NPCS)):
            ask(scene, SCENE_TASK, slot=scn_slot, max_tokens=120)
            dt, gen = ask(talk[i], LINES[i], slot=dlg_slot)
            times.append(dt)
            gens.append(gen)
            print(f"   {NPCS[i][0]:10} {dt:5.1f}с  сгенерировано {gen} токенов", flush=True)

        med = statistics.median(times)
        p_sp, e_sp = speeds(log)
        verdict = ("ОТЛИЧНО" if med <= 3.0 else
                   "терпимо" if med <= 4.0 else "медленно")
        print(f"\n   медиана {med:.1f}с  [{verdict}]   "
              f"разбор {p_sp:.0f} ток/с, генерация {e_sp:.0f} ток/с")
        return {"key": key, "human": human, "ok": True, "median": round(med, 1),
                "times": [round(t, 1) for t in times],
                "gen_tokens": int(statistics.median(gens)),
                "prompt_tps": round(p_sp), "eval_tps": round(e_sp),
                "verdict": verdict, "flags": flags}
    finally:
        stop(proc, fh)


def main() -> int:
    keys = [k for k in (sys.argv[1:] or CONFIGS) if k in CONFIGS]
    out = []
    for k in keys:
        try:
            r = measure(k)
        except Exception as exc:  # noqa: BLE001
            print(f"   {k}: замер сорвался — {exc}")
            r = {"key": k, "human": CONFIGS[k][0], "ok": False}
        if r:
            out.append(r)
        (ROOT / "data" / "bench_server_flags.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'=' * 70}\nСВОДКА (цель: 3.0с, потолок 4.0с)\n{'=' * 70}")
    print(f"{'':2} {'настройка':<44} {'медиана':>8} {'разбор':>8} {'генер':>7}")
    for r in sorted([x for x in out if x.get("ok")], key=lambda x: x["median"]):
        print(f"{r['key']:2} {r['human']:<44} {r['median']:7.1f}с "
              f"{r['prompt_tps']:6d}т/с {r['eval_tps']:5d}т/с  {r['verdict']}")
    for r in [x for x in out if not x.get("ok")]:
        print(f"{r['key']:2} {r['human']:<44}   не поднялся")
    return 0


if __name__ == "__main__":
    sys.exit(main())
