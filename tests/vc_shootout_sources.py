"""
vc_shootout_sources.py — исходники для сравнения способов озвучки (шаг 1 из 2).

Синтезирует ОДНУ И ТУ ЖЕ реплику тремя движками и замеряет каждый. Дальше эти
файлы пойдут на перевод тембра в голос игрового актёра (шаг 2, vc_convert.py) —
идея в том, чтобы быстрый синтез получил родной тембр, не платя за это
секундами XTTS.

Запуск:  venv\\Scripts\\python.exe tests\\vc_shootout_sources.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

OUT = ROOT / "data" / "vc_shootout"
LINE = ("Ещё один чужак с вопросами. Иди прямо по мосткам до конца пристани, "
        "там будет таможня.")
NPC = "vc_probe_dunmer"


def secs(p: Path) -> float:
    with wave.open(str(p), "rb") as w:
        return w.getnframes() / w.getframerate()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    # ── piper: быстрый и чёткий, но голос не из игры ──────────────────────
    from tts_piper import PiperTTS
    piper = PiperTTS(OUT / "_work_piper")
    t0 = time.time()
    piper._speak_blocking(LINE, NPC, True, race="Dunmer", distance=0.0)
    dt = time.time() - t0
    src = sorted((OUT / "_work_piper").glob("voice_*.wav"))[-1]
    dst = OUT / "1_piper.wav"
    shutil.copyfile(src, dst)
    # Время синтеза = всё минус проигрывание: _speak_blocking ждёт конца звука.
    rows.append({"name": "piper", "file": dst.name,
                 "synth": round(max(0.0, dt - secs(dst)), 2), "sec": round(secs(dst), 1)})
    print(f"   piper: {rows[-1]['synth']}с синтеза, {rows[-1]['sec']}с звука")

    # ── silero: почти мгновенный, голос самый простой ─────────────────────
    try:
        from tts import SileroTTS
        sil = SileroTTS(OUT / "_work_silero")
        t1 = time.time()
        while not getattr(sil, "_ready", None) or not sil._ready.wait(timeout=0.1):
            if time.time() - t1 > 240:
                break
        t0 = time.time()
        sil._speak_blocking(LINE, NPC, True, race="Dunmer", distance=0.0)
        dt = time.time() - t0
        cand = sorted((OUT / "_work_silero").glob("voice_*.wav"))
        if cand:
            dst = OUT / "2_silero.wav"
            shutil.copyfile(cand[-1], dst)
            rows.append({"name": "silero", "file": dst.name,
                         "synth": round(max(0.0, dt - secs(dst)), 2),
                         "sec": round(secs(dst), 1)})
            print(f"   silero: {rows[-1]['synth']}с синтеза, {rows[-1]['sec']}с звука")
        else:
            print("   silero: файла нет — движок не отработал")
    except Exception as exc:  # noqa: BLE001
        print(f"   silero не поднялся: {str(exc)[:140]}")

    # ── morrowind: наши дообученные голоса ────────────────────────────────
    try:
        from tts_morrowind import MorrowindTTS
        mw = MorrowindTTS(OUT / "_work_mw")
        t1 = time.time()
        while not mw.ready and time.time() - t1 < 120:
            time.sleep(0.5)
        if mw.ready:
            t0 = time.time()
            mw._speak_blocking(LINE, NPC, True, race="Dunmer", distance=0.0)
            dt = time.time() - t0
            cand = sorted((OUT / "_work_mw").glob("mw_*.wav"))
            if cand:
                dst = OUT / "3_morrowind_обученный.wav"
                shutil.copyfile(cand[-1], dst)
                rows.append({"name": "morrowind", "file": dst.name,
                             "synth": round(max(0.0, dt - secs(dst)), 2),
                             "sec": round(secs(dst), 1)})
                print(f"   morrowind: {rows[-1]['synth']}с синтеза")
    except Exception as exc:  # noqa: BLE001
        print(f"   morrowind не поднялся: {str(exc)[:140]}")

    # ── xtts: эталон качества, но самый медленный ─────────────────────────
    try:
        from tts_xtts import XttsTTS
        x = XttsTTS(OUT / "_work_xtts")
        t1 = time.time()
        while not x.ready and time.time() - t1 < 400:
            time.sleep(0.5)
        if x.ready:
            ref = x._ref_for(NPC, True, "Dunmer")
            out = OUT / "_work_xtts" / "x.wav"
            t0 = time.time()
            with x._lock:
                x._proc.stdin.write(json.dumps(
                    {"cmd": "say", "text": LINE, "ref": ref, "out": str(out)},
                    ensure_ascii=False) + "\n")
                x._proc.stdin.flush()
                chunks = []
                first = 0.0
                while True:
                    msg = x._read_reply()
                    if "chunk" in msg:
                        if not first:
                            first = time.time() - t0
                        chunks.append(msg["chunk"])
                        continue
                    break
            dt = time.time() - t0
            got = Path(chunks[0]) if chunks and Path(chunks[0]).exists() else out
            if got.exists():
                dst = OUT / "4_xtts.wav"
                shutil.copyfile(got, dst)
                rows.append({"name": "xtts", "file": dst.name, "synth": round(dt, 2),
                             "first": round(first, 2), "sec": round(secs(dst), 1),
                             "ref": Path(ref).name})
                print(f"   xtts: {dt:.1f}с синтеза (первый кусок {first:.1f}с), "
                      f"эталон {Path(ref).name}")
    except Exception as exc:  # noqa: BLE001
        print(f"   xtts не поднялся: {str(exc)[:140]}")

    (OUT / "sources.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"\n   исходники в {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
