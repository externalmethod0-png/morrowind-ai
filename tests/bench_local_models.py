"""
bench_local_models.py — сравнение локальных моделей на настоящих задачах мода.

Гоняет каждую модель через ТОТ ЖЕ путь, что и игра: настоящий системный промпт,
настоящие сцены, настоящий разбор ответа. Смотрим не «нравится ли текст», а то,
что реально ломает игру:

  формат   — теги разобрались, реплика не пустая, служебное не протекло в текст
  язык     — отвечает по-русски, а не по-английски
  действия — понимает прямые просьбы и НЕ выдумывает действий на пустом месте
  знание   — не сочиняет фактов, которых персонаж знать не может
  скорость — сколько ждать реплику в игре

Сырой вывод каждой модели пишется в data/bench_raw_<модель>.txt — оценка
оценкой, но глазами посмотреть надо.

Запуск:  venv\\Scripts\\python.exe tests\\bench_local_models.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

BASE_URL = "http://127.0.0.1:1234/v1"
MODELS = ["saineko-hydra-remix-ru-12b-i1", "gigachat3.1-10b-a1.8b", "t-lite-it-2.1",
          "gemma-4-e4b-uncensored-hauhaucs-aggressive"]

BASE = {
    "npc_race": "Dunmer", "npc_faction": "", "location": "Сейда Нин",
    "talkativeness": "normal", "conversation_history": [],
    "player_context": "время 13:00 (день); игрок: ЖЕНЩИНА, раса dark elf; уровень 5",
}

# Каждая сцена: что спрашиваем, чего ждём и что считать провалом.
SCENES = [
    {
        "name": "простой вопрос",
        "req": {"npc_name": "Фаргот", "npc_class": "Commoner", "npc_disposition": 50,
                "player_input": "Не подскажешь, где тут таможня?"},
        "want_action": {"none"},
        "why": "обычный вопрос — действий быть не должно",
    },
    {
        "name": "прямая просьба идти",
        "req": {"npc_name": "Тидрал", "npc_class": "Mercenary", "npc_disposition": 90,
                "player_input": "Идём со мной прямо сейчас. Плачу вперёд, "
                                "просто иди за мной следом."},
        "want_action": {"follow"},
        "why": "прямее некуда: наёмник, отношение 90, оплата вперёд",
    },
    {
        "name": "жди здесь",
        "req": {"npc_name": "Тидрал", "npc_class": "Mercenary", "npc_disposition": 90,
                "player_input": "Стой тут и жди меня. Никуда не уходи, я скоро вернусь.",
                "conversation_history": [
                    {"role": "npc", "content": "Хорошо, иду за тобой."}]},
        "want_action": {"wait_here"},
        "why": "прямой приказ спутнику остановиться",
    },
    {
        "name": "яд при страже",
        "req": {"npc_name": "Трактирщик", "npc_class": "Publican", "npc_disposition": 70,
                "player_input": "Двести золотых — подсыпь Драрену яду в кружку.",
                "risk_note": "ЧЕМ ТЫ РИСКУЕШЬ: стража рядом (3, ближайший в 4 шагах); "
                             "посторонних глаз поблизости: 5."},
        "forbid_action": {"poison"},
        "why": "при страже в четырёх шагах травить нельзя — риск очевиден",
    },
    {
        "name": "яд без свидетелей",
        "req": {"npc_name": "Трактирщик", "npc_class": "Publican", "npc_disposition": 75,
                "player_input": "Пятьсот золотых. Подсыпь Драрену Тираласу яду, никто не увидит.",
                "risk_note": "ЧЕМ ТЫ РИСКУЕШЬ: вокруг ни души — свидетелей не будет.",
                "npc_drives": "главное желание: скопить на вторую лавку; боится нищеты"},
        "want_action": {"poison", "none"},
        "need_target_if": "poison",
        "why": "согласие допустимо, но тогда обязана быть названа жертва",
    },
    {
        "name": "труп у ног",
        "req": {"npc_name": "Охранник", "npc_class": "Guard", "npc_disposition": 60,
                "player_input": "Ну что, идём дальше искать того, кто на меня напал?",
                "corpses": "РЯДОМ ЛЕЖАТ МЁРТВЫЕ — тело: Фаргот (убит(а) — Охранник), только что.",
                "conversation_history": [
                    {"role": "npc", "content": "(ФАКТ О СЕБЕ: ты своими руками убил Фаргота. "
                                               "Тело лежит рядом. Ты это помнишь и не отрицаешь.)"}]},
        "forbid_words": ("не видел", "не знаю никого", "во снах", "какое тело",
                         "повешен", "нашёл убийцу", "поймал убийцу", "убийца сбежал",
                         "кто-то другой"),
        # Приказы спутнику тут ни при чём — тело у ног, а не поход в пещеру.
        "want_action": {"none", "attack", "callguards", "threaten", "flee", "defend"},
        "why": "он сам только что убил — ни отрицать, ни выдумывать чужого убийцу нельзя",
    },
    {
        "name": "чего знать не может",
        "req": {"npc_name": "Раванус", "npc_class": "Trader", "npc_disposition": 55,
                "player_input": "Сколько за мою голову дают? Назови точную сумму.",
                "player_context": BASE["player_context"] +
                                  "; о чужаке ходит дурная слава — говорят, за ним ищет закон "
                                  "(точной суммы ты не знаешь)"},
        "forbid_number": True,
        "forbid_echo": True,
        "why": "торговец не служит закону и точной суммы знать не может",
    },
]


def cyrillic_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "а" <= c.lower() <= "я" or c.lower() == "ё") / len(letters)


async def run_model(model: str, raw_log: list, rounds: int = 1) -> dict:
    import agents.lore_agent as la
    from agents.lore_agent import LoreAgent
    from providers.local_provider import LocalProvider

    # Подсматриваем сырой текст модели до разбора: оценка оценкой, а понять,
    # ЧТО именно сломалось, можно только по тому, что модель реально написала.
    # Ставим перехват один раз на модуль, а словарь держим общий — иначе для
    # второй модели замыкание смотрело бы в словарь от первой.
    seen: dict[str, str] = getattr(la, "_bench_seen", None) or {}
    la._bench_seen = seen
    if not getattr(la, "_bench_spy", False):
        _orig = la._parse_response

        def _spy(raw_text: str):
            la._bench_seen["last"] = raw_text
            return _orig(raw_text)

        la._parse_response = _spy
        la._bench_spy = True

    agent = LoreAgent.__new__(LoreAgent)
    agent.llm = LocalProvider({"base_url": BASE_URL, "model": model, "timeout": 180})
    agent._temperature = 0.8
    agent._max_tokens = 300
    agent.model_name = model
    # Домашней модели — короткий промпт: 3 тысячи символов вместо 22.
    # Прошлый замер шёл на полном, и все модели поголовно проваливали действия
    # («прямая просьба идти» — 0 из 3 у каждой), а ответ шёл 13-17 секунд.
    # Проверять их на промпте, который они физически не удерживают, нечестно.
    # MWAI_BENCH_FULL=1 вернёт прежний режим для сравнения.
    agent._lite = os.environ.get("MWAI_BENCH_FULL", "") == ""

    print(f"\n{'=' * 70}\n{model}  "
          f"(промпт {'короткий' if agent._lite else 'полный'})\n{'=' * 70}",
          flush=True)

    # Прогрев: первый запрос грузит модель в память, его время не показательно.
    t0 = time.time()
    try:
        await agent.generate_response(
            {**BASE, "npc_id": "warm", "npc_name": "Х", "npc_class": "Commoner",
             "player_input": "Привет."}, memory_context=[])
        warm = time.time() - t0
    except Exception as exc:  # noqa: BLE001
        print(f"  модель не отвечает: {str(exc)[:140]}")
        return {"model": model, "ok": False, "error": str(exc)[:200]}
    print(f"  загрузка и прогрев: {warm:.1f}с")

    # Температура 0.8 — один прогон ничего не решает: та же модель на той же
    # сцене то ставит действие, то нет. Гоняем набор несколько раз и смотрим
    # на среднее, иначе выбираем шум.
    times, tps, score, notes = [], [], 0, []
    per_scene: dict[str, int] = {sc["name"]: 0 for sc in SCENES}
    for rnd in range(rounds):
        if rounds > 1:
            print(f"  --- заход {rnd + 1} из {rounds}")
        score += await _one_round(agent, seen, raw_log, model, times, tps,
                                  notes, per_scene)

    avg = sum(times) / len(times) if times else 0
    med = sorted(times)[len(times) // 2] if times else 0
    speed = sum(tps) / len(tps) if tps else 0
    total = len(SCENES) * rounds
    print(f"\n  ИТОГ: {score} из {total} сцен без нареканий; "
          f"среднее {avg:.1f}с, медиана {med:.1f}с, ~{speed:.0f} токенов/с")
    if rounds > 1:
        worst = [n for n, hits in per_scene.items() if hits < rounds]
        print("  спотыкается на: " + (", ".join(f"{n} ({per_scene[n]}/{rounds})"
                                                for n in worst) or "нигде"))
    return {"model": model, "score": score, "total": total, "ok": True,
            "avg": round(avg, 1), "median": round(med, 1), "warm": round(warm, 1),
            "tok_s": round(speed, 1), "per_scene": per_scene, "rounds": rounds,
            "notes": notes}


async def _one_round(agent, seen, raw_log, model, times, tps, notes,
                     per_scene) -> int:
    score = 0
    for sc in SCENES:
        req = {**BASE, **sc["req"], "npc_id": "bench"}
        seen["last"] = ""
        t0 = time.time()
        try:
            res = await agent.generate_response(
                req, memory_context=[])
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{sc['name']}: ошибка {str(exc)[:60]}")
            print(f"  [СБОЙ] {sc['name']}: {str(exc)[:80]}")
            continue
        dt = time.time() - t0
        times.append(dt)

        raw = seen.get("last", "")
        raw_log.append(f"\n### {model} / {sc['name']} ({dt:.1f}с)\n{raw}\n")

        text = str(res.get("response") or "")
        action = str(res.get("action") or "none")
        target = str(res.get("target") or "none")
        out_tok = int(res.get("tokens_used") or 0)
        if dt > 0 and out_tok:
            tps.append(out_tok / dt)
        problems = []

        if not text.strip():
            problems.append("пустая реплика" +
                            (" (модель вообще ничего не вернула)" if not raw.strip()
                             else " (текст есть, но разбор его не принял)"))
        elif cyrillic_share(text) < 0.5:
            problems.append("отвечает не по-русски")
        if any(t in text for t in ("ACTION:", "GOLD:", "EMOTION:", "TARGET:", "npc_response")):
            problems.append("служебное протекло в текст")
        # Модель может подменить маркер своим («<Tidral's Response>») — тогда он
        # уезжает игроку на экран и в озвучку.
        leak = re.search(r"<[^>]{2,40}>", text)
        if leak:
            problems.append(f"свой маркер протёк в текст: {leak.group(0)!r}")
        # Отношение и настроение обязаны смотреть в одну сторону: плюс к
        # отношению за реплику, сказанную в отвращении, портит связь с NPC
        # молча и надолго.
        emo, disp = str(res.get("emotion") or ""), int(res.get("disp") or 0)
        if emo in ("angry", "disgusted", "fearful") and disp > 2:
            problems.append(f"настроение {emo}, а отношение {disp:+d}")

        if "want_action" in sc and action not in sc["want_action"]:
            problems.append(f"действие {action}, ждали {'/'.join(sc['want_action'])}")
        if "forbid_action" in sc and action in sc["forbid_action"]:
            problems.append(f"ВЫДУМАЛ действие {action}")
        if sc.get("need_target_if") == action and target in ("none", ""):
            problems.append("действие без цели")
        for w in sc.get("forbid_words", ()):
            if w in text.lower():
                problems.append(f"противоречит очевидному («{w}»)")
        if sc.get("forbid_number") and re.search(r"\b\d{3,}\b", text):
            problems.append("СОЧИНИЛ сумму, которой знать не может")
        if sc.get("forbid_echo") and req["player_input"][:25].lower() in text.lower():
            problems.append("просто повторил вопрос игрока")

        ok = not problems
        score += ok
        per_scene[sc["name"]] += ok
        mark = "ok  " if ok else "ХУЖЕ"
        print(f"  [{mark}] {sc['name']:<22} {dt:5.1f}с  action={action:<10} "
              f"disp={disp:+3d}  {text[:44]}", flush=True)
        for p in problems:
            print(f"          -> {p}")
            notes.append(f"{sc['name']}: {p}")
    return score


async def main() -> int:
    results, raw_log = [], []
    args = [a.lower() for a in sys.argv[1:]]
    rounds = 1
    only = []
    for a in args:
        if a.isdigit():
            rounds = max(1, int(a))
        else:
            only.append(a)
    models = [m for m in MODELS if not only or any(a in m for a in only)]
    for m in models:
        results.append(await run_model(m, raw_log, rounds))
        (ROOT / "data" / "bench_raw.txt").write_text("".join(raw_log), encoding="utf-8")

    print(f"\n{'=' * 70}\nСВОДКА\n{'=' * 70}")
    good = [r for r in results if r.get("ok")]
    for r in sorted(good, key=lambda x: (-x["score"] / max(1, x["total"]), x["avg"])):
        print(f"  {r['model']:<44} {r['score']}/{r['total']}  "
              f"{r['avg']:>5.1f}с  ~{r['tok_s']:.0f} т/с")
    (ROOT / "data" / "bench_local.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n  сырые ответы: data/bench_raw.txt")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
