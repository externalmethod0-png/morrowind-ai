"""
build_voice_dataset.py — датасет для дообучения Piper на РОДНЫХ голосах игры.

Берёт озвучку Morrowind (Sound/Vo/<раса>/<пол>/*.mp3), расшифровывает каждый
клип через Whisper и складывает пары «wav + текст» в формате LJSpeech, который
понимает обучение Piper.

Почему так: Piper не умеет клонировать голос на лету (это VITS, одна модель =
один голос). Зато его можно ДООБУЧИТЬ на конкретном дикторе — а в озвучке
Morrowind на расу и пол приходится ровно один актёр, то есть каждый пул это и
есть один диктор. Расшифровок к клипам нет, поэтому их делает Whisper.

Запуск (venv Wisper — там faster-whisper и декодер mp3):
    D:\\Wisper\\Wisper\\.venv\\Scripts\\python.exe tools\\build_voice_dataset.py [пулы]
Например:  ... build_voice_dataset.py d/m d/f i/m i/f
"""

from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

MOD = Path(__file__).resolve().parent.parent
VO = Path(r"D:\Morrowind (ReBuild)\OPENMW\Data Files\Sound\Vo")
OUT = MOD / "data" / "piper_dataset"
SR = 22050                 # частота, на которой обучается Piper
MIN_S, MAX_S = 0.6, 12.0   # слишком короткие и слишком длинные только мешают
MIN_CHARS = 4

POOLS_DEFAULT = ["d/m", "d/f", "i/m", "i/f"]   # данмеры и имперцы — их больше всего

# Мусор, который Whisper выдаёт на шуме: в датасет такое пускать нельзя.
JUNK = ("продолжение следует", "субтитры", "amara.org", "dimatorzok",
        "подписывайтесь", "спасибо за просмотр", "редактор субтитров")


def decode_mp3(path: Path) -> tuple[np.ndarray, int] | None:
    """mp3 -> моно float32. Декодер берём из av (он идёт с faster-whisper)."""
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            rate = stream.rate
            chunks = []
            for frame in container.decode(stream):
                arr = frame.to_ndarray()
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                chunks.append(arr.astype(np.float32))
        if not chunks:
            return None
        audio = np.concatenate(chunks)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        peak = float(np.max(np.abs(audio)))
        if peak > 1.5:                      # пришло в int-шкале
            audio = audio / 32768.0
        return audio, rate
    except Exception:  # noqa: BLE001
        return None


def resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return audio
    n = int(round(len(audio) * dst / src))
    return np.interp(np.linspace(0, len(audio) - 1, n),
                     np.arange(len(audio)), audio).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def clean(text: str) -> str:
    text = " ".join((text or "").split())
    low = text.lower()
    if any(j in low for j in JUNK):
        return ""
    if len(text) < MIN_CHARS:
        return ""
    return text


def _load_vo_text() -> dict[str, str]:
    """Настоящие тексты реплик из файлов игры; собирает extract_vo_text.py."""
    path = MOD / "data" / "vo_text.json"
    if not path.exists():
        print("нет data/vo_text.json — сначала tools/extract_vo_text.py")
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"текстов из игры: {len(data)}")
    return data


VO_TEXT: dict[str, str] = {}


class _Transcriber:
    """Vosk одной строкой: подать 16 кГц float, получить текст.

    Держим один экземпляр модели на весь прогон — она грузится 4 секунды, а
    клипов тут тысячи.
    """

    def __init__(self) -> None:
        sys.path.insert(0, str(MOD / "python"))
        from vosk import Model, SetLogLevel

        SetLogLevel(-1)
        self._model = Model(str(MOD / "data" / "vosk" / "vosk-ru-0.42-fast"))

    def transcribe(self, audio) -> str:
        import json as _json

        import numpy as _np
        from vosk import KaldiRecognizer

        pcm = (_np.clip(audio, -1.0, 1.0) * 32767).astype(_np.int16).tobytes()
        rec = KaldiRecognizer(self._model, 16000)
        rec.AcceptWaveform(pcm)
        return _json.loads(rec.FinalResult()).get("text", "")


def build_pool(model, pool: str) -> dict:
    letter, gender = pool.split("/")
    src_dir = VO / letter / gender
    if not src_dir.is_dir():
        return {"pool": pool, "error": f"нет каталога {src_dir}"}

    dst = OUT / f"{letter}{gender}"
    wavs = dst / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    meta_path = dst / "metadata.csv"

    done: set[str] = set()
    if meta_path.exists():          # продолжаем с места остановки
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if "|" in line:
                done.add(line.split("|", 1)[0])

    files = sorted(src_dir.glob("*.mp3"))
    kept, skipped, seconds = len(done), 0, 0.0
    from_asr = 0
    t0 = time.time()

    with meta_path.open("a", encoding="utf-8") as meta:
        for i, mp3 in enumerate(files, 1):
            name = mp3.stem
            if name in done:
                continue
            decoded = decode_mp3(mp3)
            if decoded is None:
                skipped += 1
                continue
            audio, rate = decoded
            dur = len(audio) / rate
            if not (MIN_S <= dur <= MAX_S):
                skipped += 1
                continue

            # Текст берём из самой игры: в записях INFO лежит и имя mp3, и
            # реплика. Распознавание врало в двух клипах из пяти — модель
            # училась произносить не то, что звучит, и это оказалось главной
            # причиной невнятности. Оно осталось только запасным путём.
            text = VO_TEXT.get(name, "")
            if not text:
                text = clean(model.transcribe(resample(audio, rate, 16000)))
                from_asr += 1
            if not text:
                skipped += 1
                continue

            write_wav(wavs / f"{name}.wav", resample(audio, rate, SR), SR)
            meta.write(f"{name}|{text}\n")
            meta.flush()
            kept += 1
            seconds += dur

            if i % 25 == 0:
                el = time.time() - t0
                print(f"  [{pool}] {i}/{len(files)}: годных {kept}, отсеяно {skipped}, "
                      f"речи {seconds/60:.1f} мин, прошло {el/60:.1f} мин", flush=True)

    return {"pool": pool, "kept": kept, "skipped": skipped,
            "minutes": round(seconds / 60, 1), "from_asr": from_asr,
            "dir": str(dst)}


def main() -> int:
    pools = sys.argv[1:] or POOLS_DEFAULT
    print(f"пулы: {', '.join(pools)}")
    print(f"источник: {VO}")
    print(f"результат: {OUT}\n", flush=True)

    # Расшифровку делает Vosk. Раньше здесь был Whisper, но он ушёл вместе с
    # переездом распознавания на Vosk, и этот путь молча сломался: сборщик
    # падал на вызове, которого больше нет. Замечено, когда понадобилось
    # собрать новые пулы.
    global VO_TEXT
    VO_TEXT = _load_vo_text()
    model = _Transcriber()
    print("модель распознавания загружена (Vosk, CPU)\n", flush=True)

    report = []
    for pool in pools:
        print(f"=== {pool} ===", flush=True)
        res = build_pool(model, pool)
        report.append(res)
        print(f"  ГОТОВО: {res}\n", flush=True)

    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ИТОГ:", json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
