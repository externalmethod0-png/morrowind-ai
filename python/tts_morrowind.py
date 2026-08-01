"""
tts_morrowind.py — голоса, ДООБУЧЕННЫЕ на родной озвучке игры.

Четыре модели по расе и полу (данмеры и имперцы), полученные дообучением Piper
на клипах из Sound/Vo. Разница с XTTS принципиальная: тембр уже вшит в модель,
поэтому синтез идёт на процессоре за доли секунды и не отнимает видеокарту у
игры. Личная высота голоса разводит людей одной расы между собой.

Интерфейс тот же, что у остальных движков:
    speak_async(text, npc_id, is_male, distance=0.0, race="")
    stop()
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

from tts_queue import SerialSpeaker, pitch_for, race_pitch

logger = logging.getLogger(__name__)

MOD_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = MOD_ROOT / "piper_train_env" / "venv" / "Scripts" / "python.exe"
DAEMON = MOD_ROOT / "python" / "piper_daemon.py"
VOICES_DIR = MOD_ROOT / "piper" / "morrowind"

# Раса -> какой из обученных голосов ближе. Данмеры и имперцы получили свои;
# остальным достаётся тот, что ближе по звучанию, — это честнее, чем один
# синтезатор на всех.
# Раса -> какой из обученных голосов ближе ПО ТЕМБРУ. Выбор не на глаз:
# замерена высота родной озвучки каждой расы и наших четырёх голосов
# (dm 79 Гц, im 145, df 174, if 188). К данмерскому пулу идут только те, кто
# правда звучит низко; всех остальных ведём к имперскому, а разницу добирает
# поправка по расе — см. race_pitch в tts_queue.
#
# Раньше сюда были свалены босмеры, каджиты и аргониане, и босмер получал
# 80 Гц вместо своих 171. Это слышал игрок и это оказалось правдой.
# Раса -> буква своего пула озвучки. В игре есть все двадцать пулов, и
# каждому теперь соответствует свой обученный голос — по мере готовности.
#
# Пока пул не обучен, подставляется ЗАПАСНОЙ из FALLBACK: голос не тот, но
# персонаж говорит. Как только модель появится в piper/morrowind, она
# подхватится сама — код смотрит на то, что реально загружено.
RACE_TO_POOL = {
    "dark elf": "d", "dunmer": "d",
    "imperial": "i",
    "argonian": "a",
    "khajiit": "k",
    "nord": "n",
    "breton": "b",
    "orc": "o", "orsimer": "o",
    "redguard": "r",
    "high elf": "h", "altmer": "h",
    "wood elf": "w", "bosmer": "w",
}

# Кем подменить, пока свой голос не обучен. Выбрано по высоте основного тона
# родной озвучки: аргонианин (91 Гц) ближе к данмеру (80), остальные к
# имперцу (119). Это компромисс по высоте, а не по тембру — потому и слышно.
FALLBACK = {"a": "d", "k": "i", "n": "i", "b": "i",
            "o": "i", "r": "i", "h": "i", "w": "i"}

# Высота НАШИХ голосов — снята с того, что модель реально произносит, а не с
# клипов, на которых её учили. Разница заметная: по клипам выходило dm 79 / im 145 /
# df 174 / if 188, а живой синтез даёт 88 / 136 / 148 / 200. Обучение сдвигает
# голос, и если считать поправку от клипов, промах доходит до 18%.
#
# Замер: шесть фраз на пул, личный разброс и раса отключены, медиана по
# автокорреляции. Переснять после дообучения любого голоса.
#
# Разброс от фразы к фразе — dm 10%, im 3%, df 7%, if 7%. Это пол точности:
# точнее собственного дрожания модели попасть в расу нельзя, и лечится оно
# только дообучением, а не арифметикой.
POOL_HZ = {"dm": 84.2, "im": 135.7, "df": 162.5, "if": 183.4,
           # Дообученные пулы. НАБЛЮДЕНИЕ: свою расу по высоте они добирают
           # лишь отчасти — дообучение идёт с русского базового голоса, а
           # материала 8-16 минут, поэтому тембр перенимается, а высота тянется
           # к базовой. Родная озвучка игры против нашей:
           #     аргонианин  91 -> 79.8      хаджит   113 -> 119.7
           #     аргонианка 235 -> 185.3     хаджитка 178 -> 166.4
           #     норд       156 -> 148.6
           # Хаджиты и норд попали почти точно, аргониане нет. Поправка по
           # высоте поэтому нужна и своим пулам — но ложится она теперь на
           # СВОЙ тембр, а не на чужой имперский. Это и было целью.
           #
           # Переснимать после каждого дообучения: tools/measure_pool_hz.py
           "am": 79.8, "af": 185.3, "km": 119.7, "kf": 166.4,
           "nm": 148.6, "nf": 178.9, "bm": 168.7,
           "bf": 222.3, "om": 108.7, "of": 188.1,
           "rm": 113.8, "rf": 212.6, "hm": 111.8}


class MorrowindTTS(SerialSpeaker):
    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._slot = 0
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.ready = False
        self.voices: list[str] = []
        self._start_speech_queue("mw-voices")
        if not (TRAIN_PY.exists() and VOICES_DIR.is_dir()):
            logger.error("нет обученных голосов в %s — движок недоступен", VOICES_DIR)
            return
        threading.Thread(target=self._start, daemon=True, name="mw-voices-start").start()

    # ----------------------------------------------------------------- запуск

    def _start(self) -> None:
        try:
            err_log = self.out_dir.parent / "piper_daemon.log"
            self._err = err_log.open("w", encoding="utf-8", errors="replace")
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
            self._proc = subprocess.Popen(
                [str(TRAIN_PY), str(DAEMON)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
                text=True, encoding="utf-8", errors="replace", env=env,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            info = self._read_reply()
            self.ready = bool(info.get("ready"))
            self.voices = list(info.get("voices") or [])
            if self.ready:
                logger.info("голоса из озвучки игры готовы: %s", ", ".join(self.voices))
            else:
                logger.error("голоса не поднялись: %s (подробности в %s)",
                             info.get("err"), err_log)
        except Exception as exc:  # noqa: BLE001
            logger.error("демон голосов не запустился: %s", exc)

    def _read_reply(self) -> dict:
        """Читаем строку протокола, перешагивая посторонний вывод."""
        for _ in range(200):
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("демон голосов закрыл поток")
            line = line.strip()
            if not line.startswith("{"):
                logger.debug("голоса: посторонний вывод: %.80s", line)
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise RuntimeError("ответ демона голосов не найден")

    # ------------------------------------------------------------------ выбор

    def _pool_for(self, race: str, is_male: bool) -> str:
        letter = RACE_TO_POOL.get((race or "").strip().lower(), "d")
        sex = "m" if is_male else "f"
        for cand in (letter + sex, FALLBACK.get(letter, "i") + sex, "d" + sex):
            if cand in self.voices:
                return cand
        return self.voices[0] if self.voices else "dm"

    # ------------------------------------------------------------------ речь

    def _speak_blocking(self, text: str, npc_id: str, is_male: bool,
                        race: str = "", distance: float = 0.0) -> None:
        if not self.ready:
            waited = 0.0
            while not self.ready and waited < 60.0:
                if self._proc is not None and self._proc.poll() is not None:
                    break
                time.sleep(0.5)
                waited += 0.5
        if not (self.ready and self._proc and self._proc.poll() is None):
            logger.warning("голоса не готовы — реплика не озвучена")
            return

        self._slot = (self._slot + 1) % 6
        out = self.out_dir / f"mw_{self._slot}.wav"
        pool = self._pool_for(race, is_male)
        # Личная высота разводит соседей между собой, поправка по расе
        # ставит голос туда, где он звучит в самой игре.
        pitch = round(pitch_for(npc_id)
                      * race_pitch(race, is_male, POOL_HZ.get(pool, 0.0)), 3)
        from audio_out import play, volume_for_distance
        try:
            with self._lock:
                self._proc.stdin.write(json.dumps(
                    {"cmd": "say", "text": text, "voice": pool,
                     "out": str(out), "pitch": pitch}, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
                resp = self._read_reply()
        except Exception as exc:  # noqa: BLE001
            logger.error("запрос к голосам не прошёл: %s", exc)
            self.ready = False
            return
        if not resp.get("ok"):
            logger.warning("синтез не удался: %s", resp.get("err"))
            return
        logger.info("голос: '%s' — %s×%.2f (%.1fс, дист=%d, громк=%.2f)",
                    npc_id, pool, pitch, resp.get("sec", 0), int(distance or 0),
                    volume_for_distance(distance))
        play(str(out), distance, wait=True, npc_id=npc_id)
