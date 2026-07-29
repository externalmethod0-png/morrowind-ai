"""
measure_pool_hz.py — высота голоса каждого обученного пула.

ЗАЧЕМ. Поправка по расе считается от высоты НАШЕГО голоса: чтобы поднять
имперский пул до босмера, надо знать, на какой частоте говорит имперский пул.
Ошибка в этой цифре напрямую уводит голос от расы.

Мерить надо по тому, что модель ПРОИЗНОСИТ, а не по клипам, на которых её
учили. Разница большая: по клипам выходило dm 79 / im 145 / df 174 / if 188, а
живой синтез дал 88 / 136 / 148 / 200. Обучение сдвигает голос, и если считать
поправку от клипов, промах доходит до 18%.

Личный разброс и поправка по расе при замере ОТКЛЮЧЕНЫ — иначе померяем не
голос, а сами поправки.

Запуск:  venv\\Scripts\\python.exe tools\\measure_pool_hz.py [пул ...]
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VOICES = ROOT / "piper" / "morrowind"

# Синтезируем ТЕМ ЖЕ демоном, которым говорит мод, а не отдельным piper.exe:
# иначе померяем чужой путь и получим цифру, к игре не относящуюся.
DAEMON = ROOT / "python" / "piper_daemon.py"
DAEMON_PY = ROOT / "piper_train_env" / "venv" / "Scripts" / "python.exe"

# Шесть фраз: разной длины и с разными гласными, чтобы не подстроиться под одну.
PHRASES = [
    "Здравствуй, чужеземец.",
    "Ступай своей дорогой, покуда цел.",
    "Говорят, на болотах опять видели пепельных тварей.",
    "Не советую туда ходить без хорошего клинка.",
    "Купец обещал заплатить, да снова тянет время.",
    "Стража сегодня злая, лучше не попадайся под руку.",
]


def f0(pcm: np.ndarray, rate: int) -> float | None:
    """Основной тон по автокорреляции. Возвращает None на тишине и шуме."""
    x = pcm.astype(np.float64)
    x -= x.mean()
    if np.sqrt((x ** 2).mean()) < 1e-4:
        return None

    # Человеческая речь живёт в 60-320 Гц; за краями ищем не голос, а мусор.
    lo, hi = int(rate / 320), int(rate / 60)
    best: list[float] = []
    win = int(rate * 0.04)          # 40 мс — примерно 2-3 периода низкого голоса
    for start in range(0, max(1, len(x) - win), win):
        frame = x[start:start + win]
        if len(frame) < win or np.sqrt((frame ** 2).mean()) < 1e-3:
            continue
        corr = np.correlate(frame, frame, mode="full")[win - 1:]
        seg = corr[lo:hi]
        if seg.size == 0 or corr[0] <= 0:
            continue
        lag = int(np.argmax(seg)) + lo
        # Слабый пик — это не периодичность, а шум: такой кадр отбрасываем.
        if corr[lag] / corr[0] < 0.3:
            continue
        best.append(rate / lag)
    return statistics.median(best) if best else None


class Speaker:
    """Демон озвучки: поднимается один раз на все замеры."""

    def __init__(self) -> None:
        import json
        import os
        self._json = json
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self.proc = subprocess.Popen(
            [str(DAEMON_PY), str(DAEMON)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            encoding="utf-8", errors="replace", env=env)
        hello = json.loads(self.proc.stdout.readline() or "{}")
        self.ready = bool(hello.get("ready"))
        self.voices = list(hello.get("voices") or [])
        if not self.ready:
            raise SystemExit(f"демон озвучки не поднялся: {hello.get('err')}")

    def say(self, pool: str, text: str, out: Path) -> bool:
        # pitch=1.0 нарочно: мерим САМ голос, без личного разброса и поправки
        # по расе, иначе замерим свои же поправки.
        self.proc.stdin.write(self._json.dumps(
            {"cmd": "say", "text": text, "voice": pool,
             "out": str(out), "pitch": 1.0}, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        try:
            reply = self._json.loads(self.proc.stdout.readline() or "{}")
        except Exception:  # noqa: BLE001
            return False
        return bool(reply.get("ok")) and out.exists() and out.stat().st_size > 1000

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def measure(sp: "Speaker", pool: str) -> float | None:
    vals: list[float] = []
    with tempfile.TemporaryDirectory() as td:
        for i, phrase in enumerate(PHRASES):
            wav = Path(td) / f"{pool}_{i}.wav"
            if not sp.say(pool, phrase, wav):
                continue
            with wave.open(str(wav), "rb") as w:
                rate = w.getframerate()
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            got = f0(pcm, rate)
            if got:
                vals.append(got)
    if not vals:
        return None
    return statistics.median(vals)


def main() -> int:
    sp = Speaker()
    pools = sys.argv[1:] or sp.voices
    if not pools:
        print(f"в {VOICES} нет обученных голосов")
        return 1

    print(f"мерю {len(pools)} голосов по {len(PHRASES)} фразам\n")
    got: dict[str, float] = {}
    for pool in pools:
        hz = measure(sp, pool)
        if hz is None:
            print(f"  {pool}: не замерился (нет модели или синтез молчит)")
            continue
        got[pool] = round(hz, 1)
        print(f"  {pool}: {hz:5.1f} Гц")

    if got:
        print("\nвставить в python/tts_morrowind.py:")
        body = ", ".join(f'"{k}": {v}' for k, v in sorted(got.items()))
        print(f"POOL_HZ = {{{body}}}")
    sp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
