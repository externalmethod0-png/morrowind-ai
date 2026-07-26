"""
scene_live.py — живая проверка сцен БЕЗ запуска игры.

Поднимает настоящий мост, пишет в openmw.log запрос ровно так, как это делает
Lua, и читает такты ровно так, как их читает Lua. Проверяет то, на чём сцена
сломается в игре:
  - такты вообще доходят и лезут в слот постоянного размера;
  - имена и цели — только из состава (иначе реплика из ниоткуда);
  - действия только из списка (иначе движок молча ничего не сделает);
  - в мирной сцене нет ни драки, ни кражи;
  - квестового персонажа не бьют и не обкрадывают.

Запуск:  venv\\Scripts\\python.exe tests\\scene_live.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
BRIDGE_PY = ROOT / "venv" / "Scripts" / "python.exe"
LOG = Path(r"D:\Morrowind (ReBuild)\OPENMW\openmw.log")
SLOT = ROOT / "openmw-mod" / "ai_inbox" / "response.txt"

CAST = [
    {"id": "sc_a", "name": "Тидрал", "race": "Dunmer", "class": "Publican",
     "faction": "", "is_male": True, "story": False},
    {"id": "sc_b", "name": "Раванус", "race": "Imperial", "class": "Trader",
     "faction": "", "is_male": True, "story": False},
    {"id": "sc_c", "name": "Вида", "race": "Dunmer", "class": "Commoner",
     "faction": "", "is_male": False, "story": True},   # квестовая
]


def read_slot():
    try:
        raw = SLOT.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r"\}[^}]*$", raw)
    if m:
        raw = raw[:m.start() + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def send(req):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write("[MWAI_REQ] " + json.dumps(req, ensure_ascii=False) + "\n")


def wait_scene(rid, timeout=180.0):
    sizes, deadline = [], time.time() + timeout
    while time.time() < deadline:
        try:
            sizes.append(SLOT.stat().st_size)
        except OSError:
            pass
        rec = read_slot()
        if rec and rec.get("req_id") == rid and rec.get("scene"):
            return rec, sizes
        time.sleep(0.2)
    return None, sizes


def check(rec, kind, fails, directed=False):
    beats = rec.get("scene") or []
    names = {c["name"] for c in CAST}
    ids = {c["id"] for c in CAST}
    story = {c["id"] for c in CAST if c["story"]}
    from agents.scene_agent import SCENE_ACTIONS, SAFE_ACTIONS, is_safe

    print(f"\n   сцена «{kind or 'по указанию'}» — {len(beats)} тактов:")
    for b in beats:
        walk = f" -> к {b.get('walk_to')}" if b.get("walk_to") else ""
        act = b.get("action", "none")
        print(f"     {b.get('name')}{walk}: {str(b.get('line'))[:70]}"
              + (f"   [{act} -> {b.get('target') or '-'}]" if act != "none" else ""))

        if b.get("name") not in names:
            fails.append(f"{kind}: реплика от чужого — {b.get('name')}")
        if b.get("id") not in ids:
            fails.append(f"{kind}: неизвестный id {b.get('id')}")
        if act not in SCENE_ACTIONS:
            fails.append(f"{kind}: выдуманное действие {act}")
        if b.get("target") and b["target"] not in ids:
            fails.append(f"{kind}: цель вне состава {b['target']}")
        if b.get("walk_to") and b["walk_to"] not in ids:
            fails.append(f"{kind}: идёт к тому, кого нет — {b['walk_to']}")
        # Сцена по указанию игрока мирной не считается: он заказал её сам.
        if is_safe(kind, directed) and act not in SAFE_ACTIONS:
            fails.append(f"{kind}: в МИРНОЙ сцене действие {act}")
        if act not in SAFE_ACTIONS and (b.get("id") in story
                                        or b.get("target") in story):
            fails.append(f"{kind}: квестового задело действие {act}")
        if not str(b.get("line") or "").strip():
            fails.append(f"{kind}: пустая реплика")


def main() -> int:
    fails: list[str] = []
    # Старый признак готовности надо снести ДО запуска — иначе ждать нечего,
    # запрос уходит раньше, чем мост встал, и мост его не увидит: он читает
    # лог с конца. Ровно так первая сцена и потерялась.
    ready = ROOT / "openmw-mod" / "ai_inbox" / "bridge_ready.txt"
    if ready.exists():
        ready.unlink()
    proc = subprocess.Popen([str(BRIDGE_PY), "run_bridge_windows.py"],
                            cwd=str(ROOT / "python"),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("поднимаю мост (как ярлык)...")
    try:
        t0 = time.time()
        while time.time() - t0 < 120 and not ready.exists():
            time.sleep(0.3)
        time.sleep(2.0)

        stamp = int(time.time())
        for i, (kind, order) in enumerate((("gossip_ring", ""),
                                           ("tavern_brawl", ""),
                                           ("", "Тидрал и Раванус ссорятся из-за долга"))):
            rid = f"scenetest-{stamp}-{i}"
            send({"type": "scene", "req_id": rid, "kind": kind, "order": order,
                  "cast": CAST, "location": "Балмора, Клуб Совета",
                  "when": "19:00"})
            rec, sizes = wait_scene(rid)
            if not rec:
                print(f"!! сцена «{kind or order}» не пришла")
                fails.append(f"{kind or 'указание'}: тактов нет")
                continue
            check(rec, kind, fails, directed=bool(order))
            uniq = sorted(set(sizes))
            if len(uniq) > 1:
                fails.append(f"слот менял размер: {uniq}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("\n" + "=" * 58)
    if fails:
        print(" ПРОВАЛЕНО:")
        for f in dict.fromkeys(fails):
            print("   -", f)
        return 1
    print(" СЦЕНЫ РАБОТАЮТ: состав соблюдён, действия настоящие,\n"
          " мирная сцена без крови, квестовый персонаж цел.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
