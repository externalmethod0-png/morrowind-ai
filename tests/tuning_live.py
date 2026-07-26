"""
tuning_live.py — что делают ручки «опасность» и «нелепость» на живых сценах.

Проверяем не намерение, а результат: при низкой опасности злых поводов быть не
должно вовсе, при высокой они должны попадаться; при нелепости 100 сцена должна
стать фарсом, при 0 — остаться серьёзной.

Расходует запросы к модели (по одному на сцену).
Запуск:  venv\\Scripts\\python.exe tests\\tuning_live.py
"""

from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

CAST = [
    {"id": "t_a", "name": "Тидрал", "race": "Dunmer", "class": "Publican",
     "is_male": True, "story": False},
    {"id": "t_b", "name": "Раванус", "race": "Imperial", "class": "Trader",
     "is_male": True, "story": False},
    {"id": "t_c", "name": "Вида", "race": "Dunmer", "class": "Commoner",
     "is_male": False, "story": False},
]
# Что подходит по обстановке: вечер в трактире, все свои.
FIT = ["gossip_ring", "domestic", "merchant_row", "tavern_brawl"]


async def main() -> int:
    import yaml

    import world_tuning as wt
    from agents.scene_agent import (SCENE_KINDS, SceneAgent, is_absurd_roll,
                                    kinds_allowed_for)

    cfg = yaml.safe_load((ROOT / "python" / "config.yaml").read_text(encoding="utf-8"))
    agent = SceneAgent(cfg)

    print("── что вообще может случиться при разной опасности ──")
    for d in (0, 30, 60, 100):
        pool = kinds_allowed_for(d, FIT)
        rough = sum(1 for k in pool if not SCENE_KINDS[k].get("safe", True))
        print(f"  опасность {d:>3}: поводов в мешке {len(pool)}, из них злых "
              f"{rough} ({rough / len(pool) * 100:.0f}%)")

    print("\n── доля фарса при разной нелепости (1000 бросков) ──")
    for h in (0, 30, 100):
        hits = sum(is_absurd_roll(h, random.random()) for _ in range(1000))
        print(f"  нелепость {h:>3}: фарсом вышло {hits / 10:.0f}% событий")

    print("\n── живые сцены ──")
    for label, kind, absurd in (("серьёзная драка", "tavern_brawl", False),
                                ("драка как фарс", "tavern_brawl", True),
                                ("пересуды как фарс", "gossip_ring", True)):
        res = await agent.stage({"kind": kind, "absurd": absurd, "cast": CAST,
                                 "location": "Балмора, Клуб Совета",
                                 "when": "20:00", "used_jokes": []})
        beats = res.get("beats") or []
        print(f"\n  [{label}] тактов {len(beats)}:")
        for b in beats:
            act = b.get("action", "none")
            print(f"     {b.get('name')}: {str(b.get('line'))[:78]}"
                  + (f"   [{act}]" if act != "none" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
