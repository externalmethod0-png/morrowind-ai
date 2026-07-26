"""
e2e_live.py — сквозная проверка живой цепочки БЕЗ запуска игры.

Поднимает настоящий мост (тот же скрипт, что и ярлык), пишет запрос в
openmw.log ровно так, как это делает Lua, и читает ответ ровно так, как его
читает Lua. Проверяет то, на чём мод ломался в реальной игре:
  - ответ вообще доходит,
  - две реплики подряд не затирают друг друга,
  - слот-файл постоянного размера (VFS отдаёт размер со старта игры),
  - текст парсится после обрезки добивки,
  - в тексте нет служебных тегов,
  - озвучка создаёт wav.

Расходует запросы к Gemini (бесплатный тариф).
Запуск:  venv\\Scripts\\python.exe tests\\e2e_live.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRIDGE_PY = ROOT / "venv" / "Scripts" / "python.exe"
BRIDGE_DIR = ROOT / "python"
LOG = Path(r"D:\Morrowind (ReBuild)\OPENMW\openmw.log")
SLOT = ROOT / "openmw-mod" / "ai_inbox" / "response.txt"
TTS_DIR = ROOT / "data" / "tts"

TAGS = ("EMOTION:", "ACTION:", "TARGET:", "DISP:", "GOLD:", "ITEM:",
        "HEARD:", "LOAN:", "DEAL:", "COND:")

fails: list[str] = []


def read_slot_as_lua_does() -> dict | None:
    """Тот же путь, что в pollReply: прочитать всё, обрезать по '}', разобрать."""
    try:
        raw = SLOT.read_text(encoding="utf-8")
    except OSError:
        return None
    if not raw.strip():
        return None
    m = re.search(r"\}[^}]*$", raw)
    if m:
        raw = raw[:m.start() + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def send(req: dict) -> None:
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write("[MWAI_REQ] " + json.dumps(req, ensure_ascii=False) + "\n")


def make_req(rid: str, text: str, name: str, npc_id: str) -> dict:
    return {"type": "dialogue", "req_id": rid, "npc_id": npc_id, "npc_name": name,
            "npc_race": "Dunmer", "npc_class": "Commoner", "npc_faction": "",
            "location": "Сейда Нин", "npc_is_male": True, "distance": 120,
            "player_text": text, "conversation_history": []}


def wait_for(rid: str, timeout: float = 120.0) -> tuple[dict | None, list[int]]:
    """Ждём ГОТОВЫЙ ответ на rid, попутно собирая все размеры слот-файла.

    Потоковые порции (partial) в счёт не идут: игра их только показывает и
    затирает следующей. Однажды тест принял огрызок «<n» за реплику и отчитался
    об успехе — с тех пор ждём именно готовый ответ.
    """
    sizes: list[int] = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            sizes.append(SLOT.stat().st_size)
        except OSError:
            pass
        rec = read_slot_as_lua_does()
        if rec and rec.get("req_id") == rid and not rec.get("partial"):
            return rec, sizes
        time.sleep(0.2)
    return None, sizes


def main() -> int:
    stamp = int(time.time())
    tts_before = {p.name: p.stat().st_mtime for p in TTS_DIR.glob("*.wav")} \
        if TTS_DIR.exists() else {}

    print("запускаю мост (как ярлык)...")
    proc = subprocess.Popen([str(BRIDGE_PY), "run_bridge_windows.py"],
                            cwd=str(BRIDGE_DIR),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(20)   # загрузка ключей, XTTS-демона и прогрев голосов
        if proc.poll() is not None:
            print("!! мост умер на старте")
            return 1

        # 1) обычная реплика
        rid1 = f"e2e-{stamp}-1"
        send(make_req(rid1, "Приветствую. Как пройти к таможне?", "Хранитель Врат", "e2e_npc_1"))
        print(f"запрос 1 отправлен ({rid1}), жду...")
        rec1, sizes = wait_for(rid1)
        if not rec1:
            print("!! ОТВЕТА НЕТ — цепочка разорвана")
            fails.append("нет ответа на первый запрос")
        else:
            print(f"   ответ: {rec1.get('npc_response', '')[:70]}")

        # 2) две реплики подряд — вторая не должна съесть первую
        rid2, rid3 = f"e2e-{stamp}-2", f"e2e-{stamp}-3"
        send(make_req(rid2, "Сколько стоит комната?", "Трактирщик", "e2e_npc_2"))
        time.sleep(0.4)
        send(make_req(rid3, "А работа есть?", "Трактирщик", "e2e_npc_2"))
        print("запросы 2 и 3 отправлены подряд, жду обоих...")
        seen: dict[str, dict] = {}
        deadline = time.time() + 150
        while time.time() < deadline and len(seen) < 2:
            try:
                sizes.append(SLOT.stat().st_size)
            except OSError:
                pass
            rec = read_slot_as_lua_does()
            if rec and rec.get("req_id") in (rid2, rid3) and not rec.get("partial"):
                seen.setdefault(rec["req_id"], rec)
            time.sleep(0.15)
        if len(seen) < 2:
            got = list(seen) or "ничего"
            print(f"!! из двух реплик подряд дошло: {got}")
            fails.append("подряд идущие реплики теряются")
        else:
            print("   обе реплики дошли")

        # 3) размер слота постоянен
        uniq = sorted(set(sizes))
        if len(uniq) > 1:
            print(f"!! размер слот-файла плавает: {uniq}")
            fails.append(f"слот-файл меняет размер {uniq}")
        else:
            print(f"   размер слота постоянен: {uniq}")

        # 4) в репликах нет служебных тегов
        for rid, rec in list(seen.items()) + ([(rid1, rec1)] if rec1 else []):
            body = str(rec.get("npc_response") or "")
            leaked = [t for t in TAGS if t in body]
            if leaked:
                print(f"!! теги в реплике {rid}: {leaked}")
                fails.append(f"теги протекли в реплику ({leaked})")
        if not fails:
            print("   служебных тегов в тексте нет")

        # 5) Озвучка. Реплики 2 и 3 отправлены с разницей 0.4 с, то есть игрок
        #    заговорил снова — вторая ОТМЕНЯЕТСЯ намеренно. Обязательны две
        #    вещи: озвучена одиночная реплика и озвучена последняя из очереди,
        #    и ни одна не потеряна из-за переполнения.
        print("жду озвучку...")
        fresh: list[str] = []
        deadline = time.time() + 60
        while time.time() < deadline and len(fresh) < 2:
            now = {p.name: p.stat().st_mtime for p in TTS_DIR.glob("*.wav")} \
                if TTS_DIR.exists() else {}
            fresh = [n for n, m in now.items() if tts_before.get(n) != m]
            time.sleep(1)
        if len(fresh) >= 2:
            print(f"   озвучены одиночная и последняя реплики: {sorted(fresh)}")
        elif fresh:
            print(f"!! озвучена только одна реплика: {sorted(fresh)}")
            fails.append("озвучена лишь одна реплика из двух ожидаемых")
        else:
            print("!! новых wav нет — озвучка молчит")
            fails.append("TTS не создал звук")

        log = ROOT / "data" / "bridge.log"
        if log.exists():
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
            if any("очередь переполнена" in l for l in tail):
                print("!! очередь переполнена — реплики выбрасываются")
                fails.append("очередь озвучки переполняется")
            spoken = [l for l in tail if "TTS(" in l]
            if spoken:
                print(f"   последняя озвученная: {spoken[-1].split('INFO')[-1].strip()[:70]}")
    finally:
        proc.kill()

    print("\n" + "=" * 58)
    if fails:
        print(" СКВОЗНОЙ ТЕСТ ПРОВАЛЕН:")
        for f in fails:
            print(f"   - {f}")
        return 1
    print(" СКВОЗНОЙ ТЕСТ ПРОЙДЕН — можно запускать игру")
    return 0


if __name__ == "__main__":
    sys.exit(main())
