"""
bench_state_cache.py — стоит ли «вечный кеш» ухода с LM Studio.

ЧТО ПРОВЕРЯЕМ. Модель, прочитав неизменную часть промпта (правила мира и
список команд), держит её разбор у себя. Второй запрос с тем же началом идёт
вшестеро быстрее: замерено 28.8 с против 3.5 с на гигачате.

Но записок у мода ДВЕ — разговор с NPC и сценка между NPC. Модель помнит
последнюю прочитанную, поэтому каждая сценка стирает разбор разговорной
записки, и следующая реплика игрока снова платит полную цену. В игре это и
выглядит как «иногда отвечает мгновенно, иногда думает двадцать секунд».

llama.cpp умеет сохранять этот разбор на диск и восстанавливать его за доли
секунды. LM Studio так не умеет — ради этого и вся затея.

Замер идёт в три захода:

  1. вперемешку, без сохранений  — как сейчас: разговор, сценка, разговор…
  2. только разговоры подряд     — потолок скорости, кеш не сбивается
  3. вперемешку, с сохранением   — сохранённый разбор восстанавливается
                                   перед каждым запросом

Если третий заход близок ко второму — «вечный кеш» стоит того. Если он ближе
к первому — не стоит, и вопрос закрыт.

Запуск (llama-server должен быть поднят):
    venv\\Scripts\\python.exe tests\\bench_state_cache.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

URL = os.environ.get("MWAI_LLAMA_URL", "http://127.0.0.1:8090")
SLOT_DIR = os.environ.get("MWAI_SLOT_DIR", str(ROOT / "data" / "slots"))
ROUNDS = int(os.environ.get("MWAI_STATE_ROUNDS", "6"))


def post(path: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        URL + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def ask(system: str, user: str, max_tokens: int = 120) -> tuple[float, dict]:
    """Один запрос. Возвращает секунды и служебные счётчики сервера."""
    t = time.time()
    out = post("/v1/chat/completions", {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.5, "max_tokens": max_tokens, "stream": False,
    })
    dt = time.time() - t
    usage = out.get("usage") or {}
    return dt, {"prompt": usage.get("prompt_tokens"),
                "gen": usage.get("completion_tokens")}


def slot_save(name: str, slot: int = 0) -> bool:
    """Сохранить разбор промпта на диск. Возвращает False, если сервер не умеет."""
    try:
        post(f"/slots/{slot}?action=save", {"filename": name}, timeout=120)
        return True
    except urllib.error.HTTPError as e:
        print(f"  сохранение не поддержано: HTTP {e.code} "
              f"(нужен ключ --slot-save-path у llama-server)")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"  сохранение не удалось: {exc}")
        return False


def slot_restore(name: str, slot: int = 0) -> bool:
    try:
        post(f"/slots/{slot}?action=restore", {"filename": name}, timeout=120)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  восстановление не удалось: {exc}")
        return False


def prompts() -> tuple[str, str]:
    """Настоящие записки мода — обе, какие уходят в модель в игре."""
    from agents.lore_agent import _build_system_prompt

    talk = _build_system_prompt(
        npc_name="Фаргот", npc_race="Bosmer", npc_class="Commoner",
        npc_faction="", location="Сейда Нин", disposition=60,
        life_facts=[], lite=True)

    # Записка сценок — та же функция, что зовёт игра (scene_agent.py:406).
    from agents.scene_agent import _schema
    scene = _schema(["Фаргот", "Водуниус Нуцциус", "Телери Хельви"])
    return talk, scene


LINES = [
    "Здравствуй. Не подскажешь, где тут работу найти?",
    "Слыхал что-нибудь про пропавшее кольцо?",
    "Идём со мной, дело есть.",
    "Стой тут и жди меня.",
    "Сколько возьмёшь за помощь?",
    "Что говорят про стражу в порту?",
]
SCENE_TASK = "Фаргот и Водуниус переговариваются у причала. Три такта."


def run() -> int:
    talk, scene = prompts()
    print(f"записка разговора: {len(talk)} знаков")
    print(f"записка сценок:   {len(scene)} знаков\n")

    try:
        post("/v1/chat/completions",
             {"messages": [{"role": "user", "content": "тест"}],
              "max_tokens": 1}, timeout=60)
    except Exception as exc:  # noqa: BLE001
        print(f"llama-server не отвечает на {URL}: {exc}")
        return 1

    # ── 1. вперемешку, как сейчас ────────────────────────────────────────
    print("1. вперемешку (как в игре сейчас)")
    mixed: list[float] = []
    for i in range(ROUNDS):
        dt, u = ask(talk, LINES[i % len(LINES)])
        mixed.append(dt)
        print(f"   разговор {dt:5.1f}с  промпт={u['prompt']}")
        ds, _ = ask(scene, SCENE_TASK)
        print(f"   сценка   {ds:5.1f}с")

    # ── 2. только разговоры ──────────────────────────────────────────────
    print("\n2. только разговоры подряд (потолок скорости)")
    solo: list[float] = []
    for i in range(ROUNDS):
        dt, _ = ask(talk, LINES[i % len(LINES)])
        solo.append(dt)
        print(f"   разговор {dt:5.1f}с")

    # ── 3. вперемешку, но с восстановлением состояния ────────────────────
    print("\n3. вперемешку, но состояние восстанавливается")
    os.makedirs(SLOT_DIR, exist_ok=True)
    ask(talk, LINES[0])                      # прогрев, чтобы было что сохранять
    if not slot_save("talk.bin"):
        print("\n   заход 3 пропущен — сервер не умеет сохранять состояние")
        cached = []
    else:
        cached = []
        for i in range(ROUNDS):
            slot_restore("talk.bin")
            dt, _ = ask(talk, LINES[i % len(LINES)])
            cached.append(dt)
            print(f"   разговор {dt:5.1f}с")
            ask(scene, SCENE_TASK)

    def med(xs: list[float]) -> str:
        return f"{statistics.median(xs):.1f}с" if xs else "—"

    print("\n" + "=" * 60)
    print(f"вперемешку сейчас     : {med(mixed)}")
    print(f"только разговоры      : {med(solo)}   <- потолок")
    print(f"вперемешку + сохранение: {med(cached)}")
    if cached and solo and mixed:
        gain = statistics.median(mixed) - statistics.median(cached)
        print(f"\nэкономия на реплику: {gain:+.1f}с")
    return 0


if __name__ == "__main__":
    sys.exit(run())
