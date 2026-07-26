"""
bench_actions_probe.py — умеет ли модель вообще ставить теги действий?

В общем прогоне t-lite не поставил НИ ОДНОГО действия за четырнадцать сцен.
Причин может быть две, и они лечатся по-разному:
  1) модель не умеет пользоваться тегами — тогда мод с ней бесполезен;
  2) модель играет всех подряд гордецами и просто всем отказывает — тогда
     дело в характере, и это правится промптом.

Здесь сцены, где отказ бессмыслен: торговец на просьбу показать товар, стража
при преступлении на глазах, обворованный хозяин. Если тегов нет и тут — вопрос
закрыт.

Запуск:  venv\\Scripts\\python.exe tests\\bench_actions_probe.py <часть-имени-модели>
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

BASE = {
    "npc_race": "Dunmer", "npc_faction": "", "location": "Балмора",
    "talkativeness": "normal", "conversation_history": [],
    "player_context": "время 12:00 (день); игрок: МУЖЧИНА, раса imperial; уровень 9",
}

PROBES = [
    ("торговля", {"npc_name": "Раванус", "npc_class": "Trader", "npc_disposition": 70,
                  "player_input": "Показывай товар, я при деньгах. Открывай сундук, "
                                  "будем торговать."},
     {"trade"}),
    ("кража на глазах", {"npc_name": "Раванус", "npc_class": "Trader", "npc_disposition": 45,
                         "player_input": "", "theft_item": "серебряный кубок"},
     {"callguards", "attack", "threaten"}),
    ("клинок у горла", {"npc_name": "Горожанин", "npc_class": "Commoner", "npc_disposition": 30,
                        "player_input": "Молчи и не дёргайся. Клинок видишь?",
                        "player_context": BASE["player_context"] +
                                          "; ОРУЖИЕ ОБНАЖЕНО и направлено на тебя"},
     {"flee", "callguards", "attack"}),
    # Сговорчивые действия — то, на чём держится вся партийная часть мода.
    ("наняться", {"npc_name": "Тидрал", "npc_class": "Mercenary", "npc_disposition": 90,
                  "player_input": "Нанимаю тебя. Двести золотых вперёд — идём, "
                                  "прикроешь меня в пещере."},
     {"follow"}),
    ("подождать", {"npc_name": "Тидрал", "npc_class": "Mercenary", "npc_disposition": 90,
                   "player_input": "Стой здесь и жди меня, я загляну внутрь один.",
                   "conversation_history": [{"role": "npc", "content": "Иду за тобой, наниматель."}]},
     {"wait_here"}),
]


async def main() -> int:
    from agents.lore_agent import LoreAgent
    from providers.local_provider import LocalProvider

    want = (sys.argv[1] if len(sys.argv) > 1 else "t-lite").lower()
    import urllib.request, json as _j
    with urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=20) as r:
        names = [m["id"] for m in _j.load(r)["data"]]
    model = next((n for n in names if want in n.lower()), None)
    if not model:
        print(f"не нашёл модель по «{want}» среди: {names}")
        return 1

    agent = LoreAgent.__new__(LoreAgent)
    agent.llm = LocalProvider({"base_url": "http://127.0.0.1:1234/v1",
                               "model": model, "timeout": 180})
    agent._temperature = 0.8
    agent._max_tokens = 300
    agent.model_name = model

    print(f"проверяю теги действий: {model}\n" + "-" * 62)
    hits = 0
    for name, req, want_any in PROBES:
        t0 = time.time()
        res = await agent.generate_response({**BASE, **req, "npc_id": "probe"},
                                            memory_context=[])
        act = str(res.get("action") or "none")
        ok = act in want_any
        hits += ok
        print(f"  [{'ok  ' if ok else 'мимо'}] {name:<18} {time.time()-t0:5.1f}с  "
              f"action={act:<12} ждали {'/'.join(sorted(want_any))}")
        print(f"          {str(res.get('response') or '')[:100]}")
    print("-" * 62)
    print(f"  теги поставлены в {hits} из {len(PROBES)} случаев, где отказ бессмыслен")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
