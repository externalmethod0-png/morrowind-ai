"""
train_voices.py — дообучение Piper на РОДНЫХ голосах Morrowind.

Берёт готовый датасет (data/piper_dataset/<пул>/) и дообучает четыре голоса,
каждый со своего русского базового чекпоинта:

    dm, im  <- anton.ckpt  (мужской)
    df, if  <- mari.ckpt   (женский)

Почему дообучение, а не обучение с нуля: VITS с нуля требует десятки часов
речи, а у нас 18-32 минуты на голос. С русского чекпоинта этого хватает —
модель уже умеет говорить по-русски, ей остаётся перенять тембр.

Запуск (обучение занимает видеокарту на часы):
    piper_train_env\\venv\\Scripts\\python.exe tools\\train_voices.py
    ... --steps 3000        сколько шагов на голос
    ... --only dm,df        только эти пулы
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / "piper_train_env"
PY = ENV / "venv" / "Scripts" / "python.exe"
DATASET = ROOT / "data" / "piper_dataset"
RUNS = ENV / "runs"

# пул -> (базовый чекпоинт, человеческое имя)
# Все двадцать пулов озвучки игры. Обучены были четыре самых крупных —
# данмеры и имперцы, — а остальным восьми расам подставлялся имперский
# голос со сдвигом высоты. Высота совпадала, тембр нет: аргонианин
# говорил данмером, хаджит и орк — имперцем. Это и слышно как «не тот
# голос», и лечится только обучением своего пула.
VOICES = {
    "dm": ("anton.ckpt", "данмер мужской"),
    "df": ("mari.ckpt",  "данмер женский"),
    "im": ("anton.ckpt", "имперец мужской"),
    "if": ("mari.ckpt",  "имперец женский"),
    "am": ("anton.ckpt", "аргонианин мужской"),
    "af": ("mari.ckpt",  "аргонианин женский"),
    "km": ("anton.ckpt", "хаджит мужской"),
    "kf": ("mari.ckpt",  "хаджит женский"),
    "nm": ("anton.ckpt", "норд мужской"),
    "nf": ("mari.ckpt",  "норд женский"),
    "bm": ("anton.ckpt", "бретон мужской"),
    "bf": ("mari.ckpt",  "бретон женский"),
    "om": ("anton.ckpt", "орк мужской"),
    "of": ("mari.ckpt",  "орк женский"),
    "rm": ("anton.ckpt", "редгард мужской"),
    "rf": ("mari.ckpt",  "редгард женский"),
    "hm": ("anton.ckpt", "высокий эльф мужской"),
    "hf": ("mari.ckpt",  "высокий эльф женский"),
    "wm": ("anton.ckpt", "лесной эльф мужской"),
    "wf": ("mari.ckpt",  "лесной эльф женский"),
}

BATCH = 16          # 8 ГБ видеопамяти; при 12 занято было лишь 4 ГБ
# ЗАГРУЗЧИК ДАННЫХ. Было 6 — Lightning сам жаловался, что один поток узкое
# место. Но на Windows каждый рабочий процесс держит СВОЮ копию датасета, и на
# очереди из шестнадцати голосов память кончилась: три прогона подряд упали —
# бретонка и орк на записи чекпоинта (807 МБ не влезли), орчиха прямо на
# выделении 5.79 МиБ. Очередь при этом пошла дальше, оставив голоса
# недоученными и молча.
#
# Три потока — компромисс: загрузчик всё ещё не в один поток, но памяти
# хватает даже когда рядом работает REAPER на три гигабайта.
WORKERS = 3
VALIDATION_SPLIT = 0.02


def train_one(pool: str, steps: int, dry: bool = False) -> bool:
    base, human = VOICES[pool]
    data_dir = DATASET / pool
    csv_path = data_dir / "metadata.csv"
    if not csv_path.exists():
        print(f"[{pool}] нет датасета: {csv_path}")
        return False

    ckpt = ENV / "base" / base
    if not ckpt.exists():
        print(f"[{pool}] нет базового чекпоинта: {ckpt}")
        return False

    out = RUNS / pool
    out.mkdir(parents=True, exist_ok=True)
    n_clips = sum(1 for _ in csv_path.open(encoding="utf-8"))

    # max_steps у Lightning — АБСОЛЮТНЫЙ номер шага, а базовый чекпоинт уже
    # стоит на 85 тысячах. Задашь 3000 — обучение решит, что цель пройдена, и
    # выйдет, не сделав ни шага (и отрапортует успехом).
    base_step = checkpoint_step(ckpt)
    target = base_step + steps

    # ПРОДОЛЖЕНИЕ, А НЕ НАЧАЛО ЗАНОВО. Скрипт всегда стартовал с базового
    # чекпоинта — а если обучение уже шло и было прервано, это выбрасывает всю
    # проделанную работу молча, и со стороны выглядит как «просто медленно
    # учится». Цель по шагам считается от БАЗЫ, поэтому продолжение с середины
    # доводит до того же места, а не уезжает дальше.
    resumed = sorted(out.rglob("checkpoints/last.ckpt"))
    if resumed:
        latest = max(resumed, key=lambda f: checkpoint_step(f))
        done = checkpoint_step(latest)
        if done >= target:
            print(f"[{pool}] уже обучен: шаг {done} >= цель {target}")
            return True
        if done > base_step:
            print(f"[{pool}] продолжаю с шага {done} "
                  f"(вместо старта с {base_step})")
            ckpt = latest

    cmd = [
        str(PY), "-m", "piper.train", "fit",
        "--data.csv_path", str(csv_path),
        "--data.audio_dir", str(data_dir / "wavs"),
        "--data.cache_dir", str(out / "cache"),
        "--data.config_path", str(out / "config.json"),
        "--data.espeak_voice", "ru",
        "--data.voice_name", f"ru_RU-morrowind-{pool}",
        "--data.batch_size", str(BATCH),
        "--data.validation_split", str(VALIDATION_SPLIT),
        "--data.num_workers", str(WORKERS),
        "--model.sample_rate", "22050",
        "--model.num_speakers", "1",
        # Оценщик качества тянет 392 МБ по 120 КБ/с и нужен только для выбора
        # «самого приятного» чекпоинта — нам хватит точности реконструкции.
        "--model.mos_metric", "none",
        "--trainer.max_steps", str(target),
        "--trainer.default_root_dir", str(out),
        "--trainer.accelerator", "gpu",
        "--trainer.devices", "1",
        "--trainer.precision", "16-mixed",
        "--trainer.log_every_n_steps", "20",
        # Эпоха здесь всего ~44 шага, а проверка качества идёт после каждой и
        # съедает 90 секунд из 102 — скорость падала с 4.3 шага/с до 0.43.
        # Проверяем раз в 10 эпох: этого хватает, чтобы выбрать чекпоинт.
        "--trainer.check_val_every_n_epoch", "10",
        "--ckpt_path", str(ckpt),
    ]

    print(f"\n=== {pool} ({human}): {n_clips} клипов, "
          f"шаги {base_step} -> {target} (+{steps}) ===", flush=True)
    if dry:
        print("  " + " ".join(cmd))
        return True

    t0 = time.time()
    log_path = out / "train.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
    mins = (time.time() - t0) / 60

    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        print(f"  ПРОВАЛ (код {proc.returncode}, {mins:.1f} мин). Хвост лога:")
        for line in tail:
            print("    " + line)
        return False

    kept = prune_checkpoints(out)
    print(f"  готово за {mins:.1f} мин, оставлено чекпоинтов: {len(kept)}")
    for k in kept:
        print(f"    {k.name} ({k.stat().st_size / 1024**2:.0f} МБ)")
    return True


def checkpoint_step(path: Path) -> int:
    """Номер шага, на котором стоит чекпоинт (нужен, чтобы задать предел выше)."""
    try:
        import torch
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        return int(ck.get("global_step") or 0)
    except Exception as exc:  # noqa: BLE001
        print(f"  не смог прочитать шаг из {path.name}: {exc}")
        return 0


def prune_checkpoints(run_dir: Path) -> list[Path]:
    """Обучение сохраняет до 11 чекпоинтов по 845 МБ на голос. Оставляем
    лучший по качеству звучания, лучший по точности и последний — остальное
    только занимает диск и путает при экспорте."""
    found: list[Path] = []
    for ck_dir in run_dir.rglob("checkpoints"):
        files = sorted(ck_dir.glob("*.ckpt"))
        if not files:
            continue
        keep = set()
        mos = [f for f in files if "val_mos" in f.name]
        mel = [f for f in files if "val_mel" in f.name]
        last = [f for f in files if f.name == "last.ckpt"]
        # имена вида epoch=NN-val_mos=4.1234.ckpt: берём с наибольшим mos
        if mos:
            keep.add(max(mos, key=lambda f: _metric(f, "val_mos")))
        if mel:
            keep.add(min(mel, key=lambda f: _metric(f, "val_mel")))
        keep.update(last)
        for f in files:
            if f not in keep:
                try:
                    f.unlink()
                except OSError:
                    pass
        found.extend(sorted(keep))
    return found


def _metric(path: Path, key: str) -> float:
    try:
        part = path.stem.split(f"{key}=")[1]
        return float(part.split("-")[0])
    except (IndexError, ValueError):
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--only", default="")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    pools = [p.strip() for p in args.only.split(",") if p.strip()] or list(VOICES)
    print(f"голоса в очереди: {', '.join(pools)}")
    print(f"шагов на голос: {args.steps}, батч: {BATCH}\n")

    report = {}
    for pool in pools:
        if pool not in VOICES:
            print(f"неизвестный пул: {pool}")
            continue
        report[pool] = train_one(pool, args.steps, args.dry)

    print("\n" + "=" * 58)
    for pool, ok in report.items():
        print(f"  {pool}: {'готов' if ok else 'ПРОВАЛ'}")
    (RUNS / "report.json").parent.mkdir(parents=True, exist_ok=True)
    (RUNS / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    return 0 if all(report.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
