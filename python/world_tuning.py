"""
world_tuning.py — две ручки, которыми игрок крутит характер мира.

    опасность  0-100   чем выше, тем злее поводы и тем чаще они случаются
    нелепость  0-100   доля событий, которые выходят откровенно дурацкими

Живут в data/настройки-мира.txt и читаются НА ЛЕТУ, как правила мира: правишь
между разговорами, ничего не перезапуская.

Ручки нужны сразу двум сторонам. Мост берёт их отсюда напрямую, а игре они
уезжают файлом постоянного размера в ai_inbox — тем же способом, что и ответы
NPC: VFS отдаёт скриптам размер, снятый при старте игры, поэтому файл обязан
существовать заранее и не меняться в размере.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

MOD_ROOT = Path(__file__).resolve().parent.parent
TUNING_FILE = MOD_ROOT / "data" / "настройки-мира.txt"
SLOT_FILE = MOD_ROOT / "openmw-mod" / "ai_inbox" / "tuning.txt"
SLOT_BYTES = 256

DEFAULTS = {"опасность": 30, "нелепость": 30}

# По-русски и по-английски: файл правит человек, пусть пишет как удобно.
_ALIASES = {
    "опасность": "опасность", "danger": "опасность", "жестокость": "опасность",
    "нелепость": "нелепость", "humour": "нелепость", "humor": "нелепость",
    "юмор": "нелепость", "абсурд": "нелепость",
}

_LINE_RE = re.compile(r"^\s*([A-Za-zА-Яа-яЁё]+)\s*[:=]\s*(\d{1,3})\s*$")

_cache: tuple[float, dict[str, int]] = (0.0, dict(DEFAULTS))

TEMPLATE = """\
# ХАРАКТЕР МИРА — две ручки от 0 до 100.
#
# Правки подхватываются на лету: сохранил — следующее событие уже по новым.
# Строки с решёткой — пояснения для тебя.

# ОПАСНОСТЬ — какие события вообще случаются вокруг и как часто.
#   0    только разговоры: пересуды, анекдоты, склоки торговцев, проповеди
#   30   в основном разговоры, изредка что-то резкое
#   100  драки, вымогательство, кражи, похищения, врываются в дом; и чаще
опасность: 30

# НЕЛЕПОСТЬ — доля событий, которые выходят откровенно дурацкими.
#   0    мир серьёзен, всё происходит по понятным причинам
#   30   каждое третье событие — фарс: идиотский повод, нелепая развязка
#   100  почти всё превращается в балаган, но люди в нём предельно серьёзны
#
# Нелепость НЕ смягчает опасность: на тебя всё так же нападут и обворуют,
# просто по дурацкой причине и с дурацким исходом. Зато дурацкий исход ничего
# не стоит — золото и вещи остаются при тебе.
нелепость: 30
"""


def _atomic(path: Path, blob: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def ensure_file() -> None:
    """Создать файл с пояснениями, если игрок его ещё не заводил."""
    if not TUNING_FILE.exists():
        TUNING_FILE.parent.mkdir(parents=True, exist_ok=True)
        TUNING_FILE.write_text(TEMPLATE, encoding="utf-8")
        logger.info("создан файл настроек мира: %s", TUNING_FILE)


def read() -> dict[str, int]:
    """Текущие значения ручек. Перечитывает файл, если тот изменился."""
    global _cache
    try:
        mtime = TUNING_FILE.stat().st_mtime
    except OSError:
        return dict(DEFAULTS)
    if mtime == _cache[0]:
        return dict(_cache[1])

    vals = dict(DEFAULTS)
    try:
        for line in TUNING_FILE.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("#"):
                continue
            m = _LINE_RE.match(line)
            if not m:
                continue
            key = _ALIASES.get(m.group(1).strip().lower())
            if key:
                vals[key] = max(0, min(100, int(m.group(2))))
    except OSError as exc:
        logger.warning("настройки мира не читаются: %s", exc)
        return dict(DEFAULTS)

    if vals != _cache[1]:
        logger.info("настройки мира: опасность %d, нелепость %d",
                    vals["опасность"], vals["нелепость"])
    _cache = (mtime, vals)
    return dict(vals)


def publish() -> dict[str, int]:
    """Отдать ручки игре. Размер файла постоянный — правило VFS."""
    vals = read()
    blob = json.dumps({"danger": vals["опасность"], "humour": vals["нелепость"]},
                      ensure_ascii=False).encode("utf-8")
    if len(blob) > SLOT_BYTES:
        blob = blob[:SLOT_BYTES]
    try:
        _atomic(SLOT_FILE, blob + b" " * (SLOT_BYTES - len(blob)))
    except OSError as exc:
        logger.warning("ручки не доехали до игры: %s", exc)
    return vals


# ── как ручки превращаются в решения ────────────────────────────────────────

def is_absurd(humour: int, roll: float) -> bool:
    """Выпал ли фарс на это конкретное событие.

    Ручка задаёт ШАНС, а не тон: событие либо честное, либо клоунада целиком.
    Постоянный полуабсурд читается как шум и портит оба жанра сразу.
    """
    return roll * 100.0 < max(0, min(100, int(humour)))


def kinds_allowed(danger: int, all_kinds: dict) -> list[str]:
    """Какие поводы вообще возможны при этой опасности.

    Мирные доступны всегда — без них при нуле не осталось бы ничего. Злые
    открываются с 40 и дальше просто случаются чаще.
    """
    peaceful = [k for k, v in all_kinds.items() if v.get("safe", True)]
    if int(danger) < 40:
        return peaceful
    return list(all_kinds)


def danger_weight(danger: int) -> float:
    """Во сколько раз злые поводы вероятнее мирных при этой опасности.

    40 -> почти никогда, 100 -> вдвое чаще мирных.
    """
    d = max(0, min(100, int(danger)))
    if d < 40:
        return 0.0
    return 0.15 + (d - 40) / 60.0 * 1.85
