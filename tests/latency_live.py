"""
latency_live.py — сколько проходит от «отпустил клавишу» до «услышал ответ».

Меряем ПО ЗВЕНЬЯМ, а не общей цифрой: только так видно, что чинить. Микрофон
заменён готовой записью — всё остальное настоящее: тот же демон распознавания,
тот же промпт, та же модель, тот же синтез.

    распознавание — сколько Whisper думает над уже записанной фразой
    ответ модели  — от запроса до первого куска текста и до готовой реплики
    озвучка       — от готового текста до ПЕРВОГО звука
    заминка       — чем закрыта пауза, пока считается озвучка

Запуск:  venv\\Scripts\\python.exe tests\\latency_live.py
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

# Распознавание теперь на Vosk и живёт в нашем окружении.
WISPER_PY = ROOT / "venv" / "Scripts" / "python.exe"
SPEECH = ROOT / "data" / "latency_speech.wav"

# Фраза, которую «говорит» игрок. Синтезируем её сами, чтобы стенд не зависел
# от живого микрофона и повторялся дословно.
PLAYER_LINE = "Скажи, где здесь таможня и далеко ли до Балморы?"

STT_PROBE = r'''
import sys, json, wave, time
import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("sttd", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
t0 = time.time(); model = m._load_model(); load = time.time() - t0
with wave.open(sys.argv[2], "rb") as w:
    rate, n = w.getframerate(), w.getnframes()
    pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
a = m._to_16k_mono(pcm, rate)
times = []
for _ in range(3):
    t1 = time.time(); text = m.transcribe(model, a, use_vad=False); times.append(time.time() - t1)
print("@@" + json.dumps({"load": round(load, 1), "runs": [round(t, 2) for t in times],
                         "text": text, "audio_sec": round(n / rate, 1),
                         "device": m._DEVICE}, ensure_ascii=False))
'''


def make_player_speech() -> float:
    """Записать фразу игрока обычным piper — это просто входные данные."""
    from tts_piper import PIPER_EXE, MALE_MODEL
    SPEECH.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(PIPER_EXE), "-m", str(MALE_MODEL), "-f", str(SPEECH),
                    "--length_scale", "1.0", "--noise_w", "0.667"],
                   input=PLAYER_LINE.encode("utf-8"), capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW, timeout=60)
    with wave.open(str(SPEECH), "rb") as w:
        return w.getnframes() / w.getframerate()


async def main() -> int:
    import yaml
    cfg = yaml.safe_load((ROOT / "python" / "config.yaml").read_text(encoding="utf-8"))

    said = make_player_speech()
    print(f"фраза игрока: «{PLAYER_LINE}» — {said:.1f}с записи\n")

    # ── звено 1: распознавание ────────────────────────────────────────────
    script = ROOT / "data" / "_stt_probe.py"
    script.write_text(STT_PROBE, encoding="utf-8")
    res = subprocess.run([str(WISPER_PY), str(script),
                          str(ROOT / "python" / "stt_daemon.py"), str(SPEECH)],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    mark = [l for l in (res.stdout or "").splitlines() if l.startswith("@@")]
    if not mark:
        print("!! распознавание не отработало:", (res.stderr or "")[-300:])
        return 1
    stt = json.loads(mark[0][2:])
    stt_time = min(stt["runs"])
    print(f"1. РАСПОЗНАВАНИЕ ({stt['device']}): {stt_time:.1f}с "
          f"(прогоны {', '.join(str(x) for x in stt['runs'])})")
    print(f"   услышано: {stt['text']}")

    # ── звено 2: ответ модели ─────────────────────────────────────────────
    from agents.lore_agent import LoreAgent
    agent = LoreAgent(cfg)
    first_at: list[float] = []
    t0 = time.time()
    res2 = await agent.generate_response({
        "npc_id": "lat_npc", "npc_name": "Трактирщик", "npc_race": "Dunmer",
        "npc_class": "Publican", "npc_faction": "", "location": "Сейда Нин",
        "npc_is_male": True, "npc_disposition": 55,
        "player_input": stt["text"], "conversation_history": [],
        "player_context": "время 13:00 (день); игрок: ЖЕНЩИНА, раса dark elf",
        "on_partial": lambda t: first_at.append(time.time() - t0) if not first_at else None,
    }, memory_context=[])
    llm_total = time.time() - t0
    llm_first = first_at[0] if first_at else llm_total
    reply = str(res2.get("response") or "")
    print(f"\n2. ОТВЕТ МОДЕЛИ: первый кусок {llm_first:.1f}с, целиком {llm_total:.1f}с")
    print(f"   реплика ({len(reply)} знаков): {reply[:90]}")

    # ── звено 3: озвучка ──────────────────────────────────────────────────
    engine = str(cfg["tts"]["engine"]).lower()
    tts_first = tts_total = 0.0
    if engine == "xtts":
        from tts_xtts import XttsTTS
        tts = XttsTTS(ROOT / "data" / "tts_lat")
        t1 = time.time()
        while not tts.ready and time.time() - t1 < 300:
            time.sleep(0.5)
        warm = time.time() - t1
        print(f"\n   (демон XTTS поднимался {warm:.1f}с — один раз при старте моста)")
        ref = tts._ref_for("lat_npc", True, "Dunmer")
        t2 = time.time()
        with tts._lock:
            tts._proc.stdin.write(json.dumps(
                {"cmd": "say", "text": reply, "ref": ref,
                 "out": str(ROOT / "data" / "tts_lat" / "lat.wav")},
                ensure_ascii=False) + "\n")
            tts._proc.stdin.flush()
            while True:
                msg = tts._read_reply()
                if "chunk" in msg:
                    if not tts_first:
                        tts_first = time.time() - t2
                    continue
                break
        tts_total = time.time() - t2
    else:
        if engine in ("morrowind", "mw"):
            from tts_morrowind import MorrowindTTS
            tts = MorrowindTTS(ROOT / "data" / "tts_lat")
            t1 = time.time()
            while not tts.ready and time.time() - t1 < 120:
                time.sleep(0.5)
        else:
            from tts_piper import PiperTTS
            tts = PiperTTS(ROOT / "data" / "tts_lat")
        t2 = time.time()
        tts._speak_blocking(reply, "lat_npc", True, race="Dunmer", distance=0.0)
        spent = time.time() - t2
        # _speak_blocking ЖДЁТ конца звука: без вычета длительности реплики
        # «синтез» выходил в шесть секунд там, где он занимает две десятых.
        played = 0.0
        for pat in ("mw_*.wav", "voice_*.wav"):
            cand = sorted((ROOT / "data" / "tts_lat").glob(pat),
                          key=lambda p: p.stat().st_mtime)
            if cand:
                with wave.open(str(cand[-1]), "rb") as w:
                    played = w.getnframes() / w.getframerate()
                break
        tts_first = tts_total = max(0.0, spent - played)
        print(f"   (реплика звучит {played:.1f}с — из замера вычтено)")
    print(f"3. ОЗВУЧКА ({engine}): первый звук {tts_first:.1f}с, целиком {tts_total:.1f}с")

    # ── звено 4: чем закрыта пауза ────────────────────────────────────────
    from filler_bank import FillerBank
    bank = FillerBank(ROOT / "data" / "tts")
    filler_sec = 0.0
    if bank.available:
        clip = bank.pools.get("dm", [None])[0]
        if clip:
            with wave.open(str(clip), "rb") as w:
                filler_sec = w.getnframes() / w.getframerate()
    print(f"4. ЗАМИНКА: {filler_sec:.1f}с готового звука, играет сразу")

    # ── итог ──────────────────────────────────────────────────────────────
    silence = stt_time + llm_total + tts_first
    covered = max(0.0, silence - filler_sec)
    print("\n" + "=" * 62)
    print(f" ОТ «ОТПУСТИЛ КЛАВИШУ» ДО ПЕРВОГО ЗВУКА ОТВЕТА: {silence:.1f}с")
    print(f"   распознавание {stt_time:.1f} + модель {llm_total:.1f} + "
          f"синтез {tts_first:.1f}")
    print(f" СУБТИТРЫ появляются на {stt_time + llm_first:.1f}с "
          f"(первый кусок текста), готовый текст — на {stt_time + llm_total:.1f}с")
    print(f" ЧИСТОЙ ТИШИНЫ после заминки: {covered:.1f}с")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
