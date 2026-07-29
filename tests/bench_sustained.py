"""
bench_sustained.py — держит ли сервер скорость в ДОЛГОЙ игре.

Короткий стенд отвечает на вопрос «как быстро сразу после запуска». Игроку
нужен ответ на другой: «как быстро через два часа». Разница оказалась
огромной — журнал сервера показал, что разбор промпта падает со 130 токенов
в секунду до 42 по ходу сессии, то есть втрое.

Именно на этом я дважды обвинил обвязку мода: стенд поднимал СВЕЖИЙ сервер и
мерил лучший случай, а проверки шли на сервере, отработавшем сотни запросов.

Здесь наоборот: один сервер, много реплик подряд, и время каждой десятки
отдельно. Если оно растёт — настройка непригодна, какой бы красивой ни
казалась на первых шести запросах.

Запуск (сервер должен быть поднят):
    venv\\Scripts\\python.exe tests\\bench_sustained.py [сколько_реплик]
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

URL = "http://127.0.0.1:8090"
# Имя модели — снаружи: стенд одинаково нужен и гигачату, и любой другой.
MODEL = __import__("os").environ.get("MWAI_MODEL", "gigachat3.1-10b-a1.8b")
TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 40

NPCS = [("Фаргот", "Bosmer", "Commoner", "Сейда Нин", 60),
        ("Тидрал", "Dunmer", "Mercenary", "Балмора", 40),
        ("Арилл", "Dunmer", "Trader", "Сейда Нин", 70),
        ("Индрель", "Dunmer", "Guard", "Вивек", 30),
        ("Телери", "Dunmer", "Commoner", "Хла Оуд", 55),
        ("Раванус", "Imperial", "Pauper", "Балмора", 45)]
LINES = ["Здравствуй. Где тут работу найти?", "Слыхал про пропавшее кольцо?",
         "Идём со мной, дело есть.", "Стой тут и жди меня.",
         "Сколько возьмёшь за помощь?", "Что говорят про стражу в порту?"]
SCENE_TASK = "Фаргот и Водуниус переговариваются у причала. Три такта."


def post(payload: dict, timeout: int = 300) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                 method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def ask(system: str, user: str, slot: int, max_tokens: int = 160) -> float:
    t = time.time()
    post({"model": MODEL,
          "messages": [{"role": "system", "content": system},
                       {"role": "user", "content": user}],
          "temperature": 0.5, "max_tokens": max_tokens,
          "stream": False, "id_slot": slot})
    return time.time() - t


def main() -> int:
    from agents.lore_agent import _build_system_prompt
    from agents.scene_agent import _schema

    talk = [_build_system_prompt(npc_name=n, npc_race=r, npc_class=k,
                                 npc_faction="", location=l, disposition=d,
                                 life_facts=[], lite=True)
            for n, r, k, l, d in NPCS]
    scene = _schema(["Фаргот", "Водуниус Нуцциус", "Телери Хельви"])

    try:
        urllib.request.urlopen(URL + "/health", timeout=10).read()
    except Exception as exc:  # noqa: BLE001
        print(f"сервер не отвечает: {exc}")
        return 1

    print(f"{TOTAL} реплик подряд, между каждой — сценка между NPC\n")
    ask(talk[0], "разогрев", 0)

    times: list[float] = []
    for i in range(TOTAL):
        ask(scene, SCENE_TASK, 1, max_tokens=120)
        dt = ask(talk[i % len(NPCS)], LINES[i % len(LINES)], 0)
        times.append(dt)
        if (i + 1) % 10 == 0:
            block = times[-10:]
            print(f"  реплики {i - 8:>3}-{i + 1:<3} медиана {statistics.median(block):5.1f}с "
                  f"(худшая {max(block):5.1f}с)", flush=True)

    first = statistics.median(times[:10])
    last = statistics.median(times[-10:])
    print(f"\nпервая десятка {first:.1f}с, последняя {last:.1f}с")
    if last > first * 1.5:
        print(f"ВЫВОД: скорость ПАДАЕТ по ходу сессии (в {last / first:.1f} раза) — "
              "настройка непригодна для долгой игры")
    else:
        print("ВЫВОД: скорость держится — настройку можно ставить в игру")
    print(f"общая медиана {statistics.median(times):.1f}с, "
          f"худшая реплика {max(times):.1f}с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
