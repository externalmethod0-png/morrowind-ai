"""
build_fillers.py — заранее отрендеренные «хм…» голосами самой игры.

Зачем. XTTS клонирует голос на лету и тратит на это секунд девять: NPC стоит
столбом, пока считается его первая фраза. Заминка эту паузу закрывает — но
сейчас её произносит piper, то есть ЧУЖИМ голосом: персонаж мнётся одним
тембром, а отвечает другим, и обман становится виден.

Здесь заминки синтезируются заранее, тем же XTTS, и лежат готовыми на диске.
В игре они играются мгновенно, из файла.

Почему хватает одного клипа на пул: в озвучке Morrowind на расу и пол
приходится ОДИН актёр, поэтому любой клип пула клонируется в один и тот же
голос. Личная высота конкретного NPC накладывается уже при воспроизведении —
значит заминка звучит РОВНО его голосом, а не похожим.

Запуск:  venv\\Scripts\\python.exe tools\\build_fillers.py
         ... --force     перерисовать уже готовые
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

OUT_ROOT = ROOT / "data" / "fillers_xtts"

# Те же фразы, что мост произносит сейчас. Держим список здесь же, чтобы банк
# и мост нельзя было развести по разным наборам.
PHRASES = [
    "Хм…", "Так…", "Дай-ка подумать.", "Погоди.", "Ну-у…",
    "Это ты к чему?", "Тьфу…", "Ага…",
]

# Раса и пол -> пул. Берём весь список рас, который знает движок XTTS.
RACES = ["dark elf", "imperial", "nord", "breton", "redguard",
         "high elf", "wood elf", "khajiit", "argonian", "orc"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from tts_xtts import XttsTTS

    tts = XttsTTS(ROOT / "data" / "tts_fillerbuild")
    t0 = time.time()
    while not tts.ready and time.time() - t0 < 300:
        time.sleep(0.5)
    if not tts.ready:
        print("!! демон XTTS не поднялся (см. data/xtts_daemon.log)")
        return 1
    print(f"   демон готов за {time.time() - t0:.1f}с")

    # Пул -> его эталонный клип. Один на пул: внутри пула актёр один и тот же.
    pools: dict[str, str] = {}
    for race in RACES:
        for is_male in (True, False):
            clips = tts._pool(race, is_male)
            if not clips:
                continue
            key = tts._pool_key(race, is_male) if hasattr(tts, "_pool_key") else None
            if key is None:
                # Тот же ключ, что строит сам движок: буква расы + пол.
                from tts_xtts import RACE_DIR
                letter = RACE_DIR.get(race, "i")
                key = f"{letter}{'m' if is_male else 'f'}"
            pools.setdefault(key, clips[len(clips) // 2])

    print(f"   пулов голосов: {len(pools)} — {', '.join(sorted(pools))}")
    made, skipped, failed = 0, 0, 0
    for key, ref in sorted(pools.items()):
        d = OUT_ROOT / key
        d.mkdir(parents=True, exist_ok=True)
        for i, phrase in enumerate(PHRASES):
            out = d / f"{i}.wav"
            if out.exists() and out.stat().st_size > 1000 and not args.force:
                skipped += 1
                continue
            t1 = time.time()
            ok = _say(tts, phrase, ref, out)
            if ok:
                made += 1
                print(f"   [{key}] {phrase:<18} {time.time() - t1:4.1f}с -> {out.name}")
            else:
                failed += 1
                print(f"   [{key}] {phrase:<18} НЕ ВЫШЛО")
    (OUT_ROOT / "bank.json").write_text(
        json.dumps({"phrases": PHRASES, "pools": sorted(pools)},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n   готово: {made} новых, {skipped} уже были, {failed} не вышло")
    print(f"   банк: {OUT_ROOT}")
    return 1 if failed else 0


def _say(tts, text: str, ref: str, out: Path) -> bool:
    """Один синтез через демон, с ожиданием всех фрагментов."""
    try:
        with tts._lock:
            tts._proc.stdin.write(json.dumps(
                {"cmd": "say", "text": text, "ref": ref, "out": str(out)},
                ensure_ascii=False) + "\n")
            tts._proc.stdin.flush()
            chunks = []
            while True:
                msg = tts._read_reply()
                if "chunk" in msg:
                    chunks.append(msg["chunk"])
                    continue
                if not msg.get("ok"):
                    return False
                break
        # Короткая фраза — один фрагмент; если демон отдал его отдельным
        # файлом, переносим на место.
        if not out.exists() and chunks:
            Path(chunks[0]).replace(out)
        return out.exists() and out.stat().st_size > 1000
    except Exception as exc:  # noqa: BLE001
        print(f"      сбой: {exc}")
        return False


if __name__ == "__main__":
    sys.exit(main())
