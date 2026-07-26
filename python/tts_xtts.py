"""
tts_xtts.py — NPC voices cloned from the game's OWN Russian voice-over.

Each NPC gets a reference clip chosen by race + gender from Sound/Vo/, so a
Dunmer guard sounds like a Dunmer guard. Per-NPC variety comes from picking a
different clip out of that race/gender pool via a hash of the npc id — same
NPC, same voice, forever.

Synthesis runs in a separate process (xtts venv, CUDA on the CMP 70HX) so the
heavy model never touches the bridge's own environment.

Interface matches the other TTS backends:
    speak_async(text, npc_id, is_male, distance=0.0, race="")
    stop()
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

from tts_queue import SerialSpeaker

logger = logging.getLogger(__name__)

MOD_ROOT = Path(__file__).resolve().parent.parent
XTTS_PY = MOD_ROOT / "xtts" / "venv" / "Scripts" / "python.exe"
DAEMON = Path(__file__).resolve().parent / "xtts_daemon.py"
VO_ROOT = MOD_ROOT.parent / "OPENMW" / "Data Files" / "Sound" / "Vo"

# Race -> voice-over folder letter (the game's own layout)
RACE_DIR = {
    "argonian": "a", "breton": "b", "dark elf": "d", "dunmer": "d",
    "high elf": "h", "altmer": "h", "imperial": "i", "khajiit": "k",
    "nord": "n", "orc": "o", "orsimer": "o", "redguard": "r",
    "wood elf": "w", "bosmer": "w",
}

MIN_REF_BYTES = 24_000     # long enough to clone a timbre from
# Personal pitch per NPC. Measured: XTTS' own delivery varies by ~13% between
# takes, so a ±5% shift was inaudible — it drowned in that variance. Resampling
# moves the formants along with the pitch, which is what actually reads as a
# different person, and ±12% stops short of sounding processed.
PITCH_RANGE = (0.88, 0.92, 0.96, 1.0, 1.04, 1.08, 1.12)


class XttsTTS(SerialSpeaker):
    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._slot = 0
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.ready = False
        self._pools: dict[str, list[str]] = {}
        self._start_speech_queue("xtts")
        if not XTTS_PY.exists():
            logger.error("XTTS venv missing at %s — cloned voices disabled", XTTS_PY)
            return
        threading.Thread(target=self._start, daemon=True, name="xtts-start").start()

    # ------------------------------------------------------------- refs

    def _pool(self, race: str, is_male: bool) -> list[str]:
        letter = RACE_DIR.get((race or "").strip().lower(), "i")
        key = f"{letter}{'m' if is_male else 'f'}"
        if key in self._pools:
            return self._pools[key]
        d = VO_ROOT / letter / ("m" if is_male else "f")
        files = [p for p in glob.glob(str(d / "*.mp3"))
                 if os.path.getsize(p) >= MIN_REF_BYTES]
        files.sort()
        self._pools[key] = files
        logger.info("XTTS refs %s: %d clips", key, len(files))
        return files

    def _ref_for(self, npc_id: str, is_male: bool, race: str) -> str | None:
        pool = self._pool(race, is_male)
        if not pool:
            return None
        h = int(hashlib.md5(npc_id.encode("utf-8", "ignore")).hexdigest(), 16)
        return pool[h % len(pool)]

    @staticmethod
    def _pitch_for(npc_id: str) -> float:
        """A personal pitch for this NPC, the same one every time.

        Morrowind's voice-over uses ONE actor per race and gender, so cloning
        from different clips of the same pool produced the same voice — every
        Imperial guard was indistinguishable. A deterministic shift spreads
        them apart while keeping each character recognisable.
        """
        h = int(hashlib.md5(("pitch:" + npc_id).encode("utf-8", "ignore")).hexdigest(), 16)
        return round(PITCH_RANGE[h % len(PITCH_RANGE)], 3)

    # ------------------------------------------------------------ daemon

    def _start(self) -> None:
        try:
            # Keep the daemon's stderr: when it fails to load (missing CUDA,
            # dependency clash) the reason must not vanish into DEVNULL.
            err_log = self.out_dir.parent / "xtts_daemon.log"
            self._err = err_log.open("w", encoding="utf-8", errors="replace")
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
            self._proc = subprocess.Popen(
                [str(XTTS_PY), str(DAEMON)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._err, text=True, encoding="utf-8", errors="replace",
                env=env, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            info = self._read_reply()
            self.ready = bool(info.get("ready"))
            if not self.ready:
                logger.error("XTTS daemon не запустился: %s (подробности в %s)",
                             info.get("err"), err_log)
            logger.info("XTTS daemon ready=%s device=%s", self.ready, info.get("device"))
            if self.ready:
                self._warm_all()
        except Exception as exc:  # noqa: BLE001
            logger.error("XTTS daemon failed to start: %s", exc)

    def _read_reply(self) -> dict:
        """Read one protocol line, stepping over stray library output.

        Coqui/torch write their own messages into the pipe; a line that is not
        JSON is not our reply, and treating it as one silenced every voice.
        """
        for _ in range(200):
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("демон XTTS закрыл поток")
            line = line.strip()
            if not line.startswith("{"):
                logger.debug("XTTS: посторонний вывод: %.80s", line)
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                logger.debug("XTTS: нечитаемая строка: %.80s", line)
        raise RuntimeError("ответ демона XTTS не найден")

    def _warm_all(self) -> None:
        """Precompute one voice print per race+gender so the FIRST line of any
        conversation is as fast as the rest (cold synthesis costs ~2x)."""
        refs = []
        for race in ("Dunmer", "Imperial", "Nord", "Breton", "Redguard", "Altmer",
                     "Bosmer", "Khajiit", "Argonian", "Orsimer"):
            for male in (True, False):
                r = self._ref_for("warm-" + race, male, race)
                if r:
                    refs.append(r)
        if not refs:
            return
        try:
            with self._lock:
                self._proc.stdin.write(json.dumps({"cmd": "warm", "refs": refs}) + "\n")
                self._proc.stdin.flush()
                resp = self._read_reply()
            logger.info("XTTS: прогрето голосов — %s", resp.get("warmed"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("XTTS warm-up failed: %s", exc)

    # ------------------------------------------------------------- speak

    def _speak_blocking(self, text: str, npc_id: str, is_male: bool,
                        race: str = "", distance: float = 0.0) -> None:
        # Загрузка модели и прогрев голосов занимают ~40 секунд. Раньше реплика,
        # пришедшая в это окно, молча выбрасывалась — а это ровно первый разговор
        # после запуска игры. Теперь она ждёт своей очереди.
        if not self.ready:
            waited = 0.0
            while not self.ready and waited < 90.0:
                if self._proc is not None and self._proc.poll() is not None:
                    break                      # демон умер — ждать нечего
                if waited == 0.0:
                    logger.info("TTS: жду готовности XTTS, реплика в очереди")
                time.sleep(0.5)
                waited += 0.5
        if not (self.ready and self._proc and self._proc.poll() is None):
            logger.warning("TTS: демон XTTS так и не поднялся — реплика не озвучена")
            return
        ref = self._ref_for(npc_id, is_male, race)
        if not ref:
            logger.warning("no reference clip for race=%r male=%s", race, is_male)
            return
        self._slot = (self._slot + 1) % 6
        out = self.out_dir / f"xtts_{self._slot}.wav"
        from audio_out import play, volume_for_distance
        spoken = 0
        epoch = self.epoch()
        with self._lock:
            try:
                self._proc.stdin.write(json.dumps(
                    {"cmd": "say", "text": text, "ref": ref, "out": str(out),
                     "pitch": self._pitch_for(npc_id)},
                    ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
                # The daemon announces each finished sentence, then a final
                # ok/err line. Playing as they arrive means the NPC starts
                # talking after the FIRST sentence instead of after the last —
                # a long reply used to sit silent for ~15 seconds.
                while True:
                    msg = self._read_reply()
                    if "chunk" in msg:
                        if self.epoch() != epoch:
                            continue      # player left: drain, do not play
                        if spoken == 0:
                            logger.info(
                                "TTS(xtts): '%s' голосом %s×%.3f (дист=%d, громк=%.2f)",
                                npc_id, os.path.basename(ref),
                                self._pitch_for(npc_id), int(distance or 0),
                                volume_for_distance(distance))
                        spoken += 1
                        play(str(msg["chunk"]), distance, wait=True, npc_id=npc_id)
                        continue
                    if not msg.get("ok"):
                        logger.warning("XTTS synthesis failed: %s", msg.get("err"))
                    elif spoken == 0:
                        if self.epoch() != epoch:
                            logger.info("реплика отменена — игрок продолжил разговор")
                        else:
                            logger.warning("XTTS вернул 0 фрагментов — тишина")
                    break
            except Exception as exc:  # noqa: BLE001
                logger.error("XTTS request failed: %s", exc)
                self.ready = False
                return
