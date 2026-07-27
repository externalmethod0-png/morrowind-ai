"""
bench_local_scenes.py — потянет ли домашняя модель постановку сцен.

Диалог и сцена — разные задачи. В диалоге модель отвечает ОДНОЙ репликой за
себя; в сцене она пишет несколько тактов за РАЗНЫХ людей, соблюдая состав,
очерёдность и запрет на выдуманные имена. Это заметно тяжелее, и проверять
надо отдельно.

Смотрим то, на чём сцена ломается в игре:
  такты      — их вообще удалось разобрать, и их больше одного
  имена      — только из состава; выдуманное имя = реплика из ниоткуда
  действия   — только из разрешённого списка, иначе движок молча промолчит
  мирность   — в спокойной сцене не должно быть драки
  квестовые  — сюжетного персонажа не бьют
  скорость   — сколько ждать сцену в игре

Запуск:  venv\\Scripts\\python.exe tests\\bench_local_scenes.py <модель> [кругов]
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

BASE_URL = "http://127.0.0.1:1234/v1"

# Состав нарочно разный: имена русские, один персонаж сюжетный (его трогать
# нельзя), один — женщина, чтобы видеть, следит ли модель за родом.
CAST = [
    {"name": "Фаргот", "race": "bosmer", "class": "commoner",
     "is_male": True, "id": "0x1", "story": True},
    {"name": "Водуниус Нуцциус", "race": "imperial", "class": "commoner",
     "is_male": True, "id": "0x2", "story": False},
    {"name": "Телери Хельви", "race": "dark elf", "class": "commoner",
     "is_male": False, "id": "0x3", "story": False},
]

SCENES = [
    {"kind": "gossip_ring", "why": "мирные пересуды — драки быть не должно"},
    {"kind": "quarrel", "why": "ссора: резкие слова можно, увечья нет"},
    {"order": "Водуниус просит у Телери денег в долг, она отказывает",
     "why": "прямое указание режиссёра — исполнить буквально"},
]


async def run(model: str, rounds: int) -> dict:
    from agents.scene_agent import SCENE_ACTIONS, SceneAgent
    from providers.local_provider import LocalProvider

    agent = SceneAgent.__new__(SceneAgent)
    agent.llm = LocalProvider({"base_url": BASE_URL, "model": model, "timeout": 180})
    agent._temperature = 0.9

    names = {c["name"] for c in CAST}
    story = {c["name"] for c in CAST if c.get("story")}

    print(f"\n{'=' * 70}\n{model} — сцены\n{'=' * 70}", flush=True)
    t0 = time.time()
    try:
        await agent.stage({"kind": "gossip_ring", "cast": CAST,
                           "location": "Сейда Нин", "when": "13:00"})
    except Exception as exc:  # noqa: BLE001
        print(f"  модель не отвечает: {str(exc)[:140]}")
        return {"model": model, "ok": False, "error": str(exc)[:200]}
    print(f"  загрузка и прогрев: {time.time() - t0:.1f}с", flush=True)

    score = total = 0
    times: list[float] = []
    notes: list[str] = []

    for rnd in range(rounds):
        print(f"  --- заход {rnd + 1} из {rounds}", flush=True)
        for sc in SCENES:
            req = {"cast": CAST, "location": "Сейда Нин", "when": "19:00"}
            req.update({k: v for k, v in sc.items() if k in ("kind", "order")})
            label = sc.get("kind") or "указание режиссёра"

            t = time.time()
            try:
                res = await agent.stage(dict(req))
            except Exception as exc:  # noqa: BLE001
                total += 4
                notes.append(f"{label}: упало — {str(exc)[:70]}")
                print(f"  [ПАДЕНИЕ] {label:<22} {str(exc)[:50]}", flush=True)
                continue
            dt = time.time() - t
            times.append(dt)
            beats = res.get("beats") or []

            # Пустая сцена — это ПОЛНЫЙ провал, а не одно замечание из четырёх.
            # Первая версия оценки давала за неё 3 балла из 4, и итог 20/24
            # выглядел прилично при том, что половина сцен вообще не сыграла.
            if not beats:
                total += 4
                notes.append(f"{label}: сцена пустая, ни одного такта")
                print(f"  [ПУСТО] {label:<22} {dt:>5.1f}с  ни одного такта",
                      flush=True)
                continue

            bad = []
            if len(beats) < 2:
                bad.append(f"тактов всего {len(beats)}")
            for b in beats:
                if b.get("name") not in names:
                    bad.append(f"чужое имя «{b.get('name')}»")
                if b.get("action") not in SCENE_ACTIONS:
                    bad.append(f"действие «{b.get('action')}»")
                if b.get("action") == "attack" and b.get("name") in story:
                    bad.append("бьют сюжетного")
                if not str(b.get("line") or "").strip():
                    bad.append("пустая реплика")
            # Мирная сцена не должна кончаться дракой.
            if sc.get("kind") == "gossip_ring" and any(
                    b.get("action") == "attack" for b in beats):
                bad.append("драка в мирной сцене")

            total += 4
            got = 4 - min(4, len(set(bad)))
            score += got
            mark = "ok  " if not bad else "ХУЖЕ"
            first = (beats[0].get("line", "") if beats else "")[:42]
            print(f"  [{mark}] {label:<22} {dt:>5.1f}с  тактов {len(beats)}  {first}",
                  flush=True)
            for b in sorted(set(bad)):
                print(f"          -> {b}", flush=True)
                notes.append(f"{label}: {b}")

    times.sort()
    med = times[len(times) // 2] if times else 0.0
    print(f"\n  ИТОГ: {score} из {total} баллов; медиана {med:.1f}с", flush=True)
    if notes:
        print("  спотыкается на: " + "; ".join(sorted(set(notes))[:5]), flush=True)
    return {"model": model, "ok": True, "score": score, "total": total,
            "median": round(med, 1), "notes": sorted(set(notes))}


async def main() -> int:
    args = sys.argv[1:]
    model = args[0] if args else "gigachat3.1-10b-a1.8b"
    rounds = int(args[1]) if len(args) > 1 and args[1].isdigit() else 2
    res = await run(model, rounds)
    (ROOT / "data" / "bench_scenes.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
