"""Настоящие тексты озвучки — прямо из файлов игры.

Датасеты для обучения голосов собирались распознаванием: клип отдавали
Whisper'у и записывали, что он услышал. Но игра ХРАНИТ эти реплики сама —
в записях INFO лежит и имя mp3 (подзапись SNAM), и её текст (NAME).

Разница не косметическая. Распознавание отдаёт строчные буквы без знаков
препинания, а модели голоса интонацию берут именно из запятых и вопросов.
Плюс ошибки распознавания попадали в датасет как правда.

Формат Morrowind: запись = имя(4) + размер(4) + два поля по 4, дальше
подзаписи имя(4) + размер(4) + данные. Кодировка русской версии — cp1251.

Использование:
    python tools/extract_vo_text.py            # все esm/esp игры
    python tools/extract_vo_text.py --check    # сверить с текущими датасетами
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

MOD = Path(__file__).resolve().parent.parent
GAME = MOD.parent / "OPENMW" / "Data Files"
OUT = MOD / "data" / "vo_text.json"
ENCODING = "cp1251"          # русская 1С-локализация


def _records(data: bytes):
    """Пробегаем файл записями верхнего уровня."""
    pos, n = 0, len(data)
    while pos + 16 <= n:
        name = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 16:pos + 16 + size]
        pos += 16 + size
        yield name, body


def _subrecords(body: bytes):
    pos, n = 0, len(body)
    while pos + 8 <= n:
        name = body[pos:pos + 4]
        size = struct.unpack_from("<I", body, pos + 4)[0]
        if pos + 8 + size > n:
            return
        yield name, body[pos + 8:pos + 8 + size]
        pos += 8 + size


def _text(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode(ENCODING, "replace").strip()


def extract(paths: list[Path]) -> dict[str, str]:
    """{имя клипа без расширения: текст реплики}."""
    found: dict[str, str] = {}
    for path in paths:
        data = path.read_bytes()
        got = 0
        for name, body in _records(data):
            if name != b"INFO":
                continue
            text, sound = "", ""
            for sub, raw in _subrecords(body):
                if sub == b"NAME":
                    text = _text(raw)
                elif sub == b"SNAM":
                    sound = _text(raw)
            if sound and text:
                # SNAM хранит путь вида «Vo\n\m\Atk_NM001.mp3»
                stem = sound.replace("/", "\\").split("\\")[-1]
                if stem.lower().endswith(".mp3"):
                    stem = stem[:-4]
                found[stem] = text
                got += 1
        print(f"  {path.name}: {got} озвученных реплик", flush=True)
    return found


def check(found: dict[str, str]) -> int:
    """Сверяем то, что услышало распознавание, с тем, что написано в игре."""
    ds = MOD / "data" / "piper_dataset"
    print(f"\n{'пул':<5}{'клипов':>8}{'нашли текст':>13}{'совпало точно':>15}")
    worst: list[tuple[str, str, str]] = []
    for pool_dir in sorted(p for p in ds.iterdir() if p.is_dir()):
        meta = pool_dir / "metadata.csv"
        if not meta.exists():
            continue
        rows = [ln.split("|", 1) for ln in
                meta.read_text(encoding="utf-8").splitlines() if "|" in ln]
        hit = [(k, v) for k, v in rows if k in found]
        same = sum(1 for k, v in hit
                   if _norm(v) == _norm(found[k]))
        print(f"{pool_dir.name:<5}{len(rows):>8}{len(hit):>13}{same:>15}")
        for k, v in hit:
            if _norm(v) != _norm(found[k]) and len(worst) < 6:
                worst.append((k, v, found[k]))
    if worst:
        print("\nчто именно расходится:")
        for k, heard, real in worst:
            print(f"  {k}")
            print(f"    услышали: {heard}")
            print(f"    в игре:   {real}")
    return 0


# Насколько текст из игры должен совпадать с услышанным, чтобы ему поверить.
#
# Проверка независимым распознаванием показала: там, где текст игры расходится
# с расшифровкой СИЛЬНО, прав почти всегда звук — в русской версии дубляж и
# текст местами разошлись («Прочь! Убирайся отсюда!» в файлах против «Уходи!
# Я не представляю угрозы!» в записи). А там, где слова те же, текст игры даёт
# верное написание и знаки препинания, которых у распознавания нет вовсе.
#
# Поэтому берём его только при близком совпадении, а спорные строки оставляем
# как есть.
TRUST_AT = 0.70


def _sim(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def fix(found: dict[str, str]) -> int:
    """Уточняем разметку датасетов текстом из игры — там, где ему можно верить.

    Сами wav-файлы верные, речь только о подписях. Прежняя разметка
    сохраняется рядом в metadata.asr.csv.
    """
    ds = MOD / "data" / "piper_dataset"
    print(f"\n{'пул':<5}{'строк':>7}{'уточнено':>10}{'не поверили':>13}"
          f"{'без текста':>12}")
    for pool_dir in sorted(p for p in ds.iterdir() if p.is_dir()):
        meta = pool_dir / "metadata.csv"
        if not meta.exists():
            continue
        rows = [ln.split("|", 1) for ln in
                meta.read_text(encoding="utf-8").splitlines() if "|" in ln]
        backup = pool_dir / "metadata.asr.csv"
        if not backup.exists():
            backup.write_bytes(meta.read_bytes())

        changed = missing = refused = 0
        out = []
        for key, heard in rows:
            real = found.get(key)
            if real is None:
                missing += 1          # текста в игре нет — оставляем как было
                out.append(f"{key}|{heard}")
                continue
            if _sim(real, heard) < TRUST_AT:
                refused += 1          # дубляж и текст разошлись — верим звуку
                out.append(f"{key}|{heard}")
                continue
            if _norm(real) != _norm(heard) or real != heard:
                changed += 1
            out.append(f"{key}|{real}")
        meta.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"{pool_dir.name:<5}{len(rows):>7}{changed:>10}{refused:>13}"
              f"{missing:>12}")
    print("\nпрежняя разметка лежит рядом в metadata.asr.csv")
    return 0


def _norm(s: str) -> str:
    keep = [c.lower() for c in s if c.isalnum() or c.isspace()]
    return " ".join("".join(keep).split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="сверить с текущими датасетами, ничего не менять")
    ap.add_argument("--fix", action="store_true",
                    help="переписать разметку датасетов текстом из игры")
    args = ap.parse_args()

    paths = sorted(GAME.glob("*.esm")) + sorted(GAME.glob("*.esp"))
    if not paths:
        print(f"не нашёл файлов игры в {GAME}")
        return 1
    print(f"читаем {len(paths)} файлов игры:")
    found = extract(paths)
    print(f"\nвсего озвученных реплик с текстом: {len(found)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(found, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"сохранено: {OUT}")

    if args.check:
        return check(found)
    if args.fix:
        return fix(found)
    return 0


if __name__ == "__main__":
    sys.exit(main())
