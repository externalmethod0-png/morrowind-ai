"""
filler_bank.py — заминка СВОИМ голосом, мгновенно, из готового файла.

XTTS клонирует голос на лету и тратит на первую фразу секунд девять. Всё это
время NPC стоит столбом. Заминка («Хм…», «Дай-ка подумать») паузу закрывает —
но только если звучит ГОЛОСОМ САМОГО ПЕРСОНАЖА: piper произносил её чужим
тембром, и подмена была слышна.

Банк заранее нарисован тем же XTTS (tools/build_fillers.py). Здесь остаётся
выбрать пул по расе и полу, наложить личную высоту NPC — и проиграть. Синтеза
нет вовсе, поэтому звук идёт сразу.

Почему хватает одного клипа на пул: в озвучке игры на расу и пол приходится
ОДИН актёр, значит любой клип пула клонируется в тот же голос. С личной
высотой это ровно голос конкретного NPC, а не похожий.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

MOD_ROOT = Path(__file__).resolve().parent.parent
BANK_ROOT = MOD_ROOT / "data" / "fillers_xtts"


class FillerBank:
    """Готовые заминки по пулам голосов."""

    def __init__(self, tmp_dir: str | Path, root: Path = BANK_ROOT) -> None:
        self.root = Path(root)
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._slot = 0
        self._lock = threading.Lock()
        self.pools: dict[str, list[Path]] = {}
        for d in sorted(self.root.glob("*")):
            if not d.is_dir():
                continue
            clips = sorted(d.glob("*.wav"))
            if clips:
                self.pools[d.name] = clips
        if self.pools:
            logger.info("заминки: банк на %d голосов, по %d фраз",
                        len(self.pools), len(next(iter(self.pools.values()))))
        else:
            logger.info("заминки: банка нет (%s) — соберите tools/build_fillers.py",
                        self.root)

    @property
    def available(self) -> bool:
        return bool(self.pools)

    def _key(self, race: str, is_male: bool) -> str | None:
        """Тот же ключ пула, что у движка XTTS: буква расы + пол."""
        try:
            from tts_xtts import RACE_DIR
        except Exception:  # noqa: BLE001
            return None
        letter = RACE_DIR.get((race or "").strip().lower(), "i")
        key = f"{letter}{'m' if is_male else 'f'}"
        if key in self.pools:
            return key
        # Раса без своего пула — берём имперский того же пола: он общий.
        alt = f"i{'m' if is_male else 'f'}"
        return alt if alt in self.pools else None

    def play_async(self, npc_id: str, is_male: bool, race: str,
                   distance: float = 0.0, salt: str = "",
                   until: threading.Event | None = None,
                   max_clips: int = 3) -> bool:
        """Мяться, пока не готов ответ. False — банка на этот голос нет.

        Одной фразы мало: от «отпустил клавишу» до первого звука ответа
        проходит секунд восемь (распознавание + модель + синтез), а «Хм…»
        длится одну-две. Поэтому персонаж мнётся НЕСКОЛЬКО раз подряд, пока
        не поднимется флажок «ответ пошёл», — тишины между ними не остаётся.
        """
        key = self._key(race, is_male)
        if not key:
            return False
        clips = self.pools[key]
        h = int(hashlib.md5((npc_id + salt).encode("utf-8", "ignore")).hexdigest(), 16)
        threading.Thread(target=self._cover, name="filler-bank", daemon=True,
                         args=(clips, h, npc_id, distance, until,
                               max(1, int(max_clips)))).start()
        return True

    def _cover(self, clips: list[Path], h: int, npc_id: str, distance: float,
               until: threading.Event | None, max_clips: int) -> None:
        for i in range(max_clips):
            if until is not None and until.is_set() and i > 0:
                # Ответ пошёл. Одну лишнюю фразу всё же доигрываем: между
                # «текст готов» и «звук пошёл» лежит ещё синтез.
                self._play_blocking(clips[(h + i) % len(clips)], npc_id, distance)
                return
            self._play_blocking(clips[(h + i) % len(clips)], npc_id, distance)

    def _play_blocking(self, src: Path, npc_id: str, distance: float) -> None:
        try:
            from tts_queue import pitch_for, shift_pitch_wav
            from audio_out import play
            # Копию, а не оригинал: сдвиг высоты правит файл на месте, и банк
            # после пары реплик съехал бы по тону навсегда.
            with self._lock:
                self._slot = (self._slot + 1) % 4
                dst = self.tmp_dir / f"filler_{self._slot}.wav"
                shutil.copyfile(src, dst)
            pitch = pitch_for(npc_id)
            shift_pitch_wav(str(dst), pitch)
            logger.debug("заминка: %s x%.2f (дист=%d)", src.name, pitch, int(distance or 0))
            play(str(dst), distance, wait=True, npc_id=npc_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("заминка не проигралась: %s", exc)
