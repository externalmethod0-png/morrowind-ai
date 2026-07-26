"""
stt_daemon.py — распознавание речи для голосового режима (клавиша V).

Движок — VOSK, русская модель vosk-model-ru-0.42. Выбран очной ставкой
четырёх движков на одних записях (tests/stt_shootout.py):

    vosk-ru-0.42     точность 96%, хвост после клавиши 0.31 с
    gigaam-ctc       93%, 0.10 с
    whisper-medium   84%, 2.70 с   <- было раньше
    vosk-small-ru    82%, 0.03 с

Whisper проиграл сразу по обоим показателям. И дело не только в модели:
Vosk разбирает речь ПО ХОДУ ГОВОРЕНИЯ, а Whisper брался за запись целиком
только после отпускания клавиши — отсюда почти три секунды тишины.

Работает на процессоре, видеокарта не нужна вовсе: отдельное окружение с
одолженными библиотеками CUDA больше не требуется, демон живёт в основном venv.

Протокол (строки JSON через stdin/stdout):
  in : {"cmd": "ptt_start"} / {"cmd": "ptt_stop"} / {"cmd": "listen", "max_s": 8}
       {"cmd": "transcribe", "path": "...wav"}   — прогнать файл, для диагностики
  out: {"ok": true, "text": "...", "sec": 0.3}
       {"ok": false, "err": "..."}
На старте печатает {"ready": true}.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

MOD_ROOT = Path(__file__).resolve().parent.parent
MOD_DATA = MOD_ROOT / "data"
MODEL_DIR = str(MOD_ROOT / "data" / "vosk" / "vosk-ru-0.42-fast")
SAMPLE_RATE = 16000
# Speech detection adapts to the microphone: a fixed threshold silently failed
# on a quiet USB mic (Fifine peaked at 0.0015 against a 0.012 gate). We sample
# the room for a moment and put the gate just above the noise floor instead.
NOISE_CALIB_S = 0.4
GATE_OVER_NOISE = 2.0    # speech must be this much louder than the room
GATE_FLOOR = 0.0012      # never gate below this (pure digital silence)
GATE_MAX = 0.015         # …and never above this: a hissy mic (measured floor
                         # 0.010 with speech peaking at 0.019) would otherwise
                         # gate out the very speech we are waiting for
SILENCE_STOP_S = 0.9     # this much trailing silence ends the take
MIN_SPEECH_S = 0.35      # shorter takes are treated as noise
TARGET_PEAK = 0.25       # normalise quiet recordings before transcription
# Scaling by the peak is wrong for a mic: one keyboard click owns the peak and
# the speech itself stays inaudible. Normalise by RMS, which tracks loudness.
TARGET_RMS = 0.06
MAX_GAIN = 60.0          # beyond this we would only be amplifying room hiss


def _claim_protocol_channel():
    """Give the JSON protocol a pipe nobody else can write to.

    PortAudio and the CUDA libraries print their own messages to stdout in the
    system codepage. One such line lands in the middle of the protocol and the
    bridge dies on it ("'utf-8' codec can't decode byte 0xcf"), so a recognised
    phrase never reaches the game. We keep the real stdout for ourselves and
    point file descriptor 1 at stderr, where stray output belongs anyway.
    """
    proto_fd = os.dup(1)
    os.dup2(2, 1)                       # print()/native writes -> log, not protocol
    return os.fdopen(proto_fd, "w", encoding="utf-8", buffering=1)


_PROTO = None       # claimed by main(); importing this module must not steal
                    # the stdout of whoever imports it (diagnostic scripts do)


def _out(obj) -> None:
    ch = _PROTO or sys.stdout
    ch.write(json.dumps(obj, ensure_ascii=False) + "\n")
    ch.flush()


def pick_device(name_hint: str = "") -> tuple[int | None, int, int]:
    """(device index, its native samplerate, channel count).

    Recording at the device's OWN rate/channels and converting afterwards is
    essential: forcing 16 kHz mono on a 44.1 kHz stereo USB mic (Fifine) yields
    garbage or near-silent input, which looked like "I can't hear you".
    """
    import sounddevice as sd
    idx = sd.default.device[0]
    if name_hint:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and name_hint.lower() in d["name"].lower():
                idx = i
                break
    try:
        info = sd.query_devices(idx)
    except Exception:  # noqa: BLE001
        return None, 44100, 1
    rate = int(info.get("default_samplerate") or 44100)
    ch = 1 if int(info.get("max_input_channels") or 1) >= 1 else 1
    return idx, rate, min(2, int(info.get("max_input_channels") or 1)) and ch or 1


def _to_16k_mono(buf: np.ndarray, rate: int) -> np.ndarray:
    """Downmix to mono and resample to the 16 kHz Whisper expects."""
    if buf.ndim > 1:
        buf = buf.mean(axis=1)
    if rate == SAMPLE_RATE:
        return buf.astype(np.float32)
    n_out = int(round(len(buf) * SAMPLE_RATE / rate))
    if n_out < 1:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(buf), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, buf).astype(np.float32)


def record(max_s: float, device_hint: str = "") -> np.ndarray | None:
    import sounddevice as sd

    idx, rate, ch = pick_device(device_hint)
    chunks: list[np.ndarray] = []
    heard_speech = False
    silence_run = 0.0
    peak = 0.0
    block = int(rate * 0.1)   # 100 ms blocks at the device's own rate

    with sd.InputStream(samplerate=rate, channels=ch, dtype="float32",
                        blocksize=block, device=idx) as stream:
        # 1) listen to the room to learn its noise floor
        noise = []
        for _ in range(max(1, int(NOISE_CALIB_S * 10))):
            data, _ = stream.read(block)
            mono = data.mean(axis=1) if data.ndim > 1 else data.copy()
            noise.append(float(np.sqrt(np.mean(mono ** 2))))
        # quietest calibration block, so a cough during calibration can't
        # raise the gate above normal speech
        floor = float(np.min(noise)) if noise else 0.0
        gate = min(max(floor * GATE_OVER_NOISE, GATE_FLOOR), GATE_MAX)
        sys.stderr.write(f"mic gate: floor={floor:.5f} -> gate={gate:.5f}\n")
        sys.stderr.flush()

        # 2) record until the speaker stops
        started = time.time()
        while True:
            data, _ = stream.read(block)
            mono = data.mean(axis=1) if data.ndim > 1 else data.copy()
            chunks.append(mono)
            rms = float(np.sqrt(np.mean(mono ** 2)))
            peak = max(peak, rms)
            if rms >= gate:
                heard_speech = True
                silence_run = 0.0
            else:
                silence_run += 0.1
            elapsed = time.time() - started
            if heard_speech and silence_run >= SILENCE_STOP_S:
                break
            if elapsed >= max_s:
                break
            if not heard_speech and elapsed >= 3.0:
                sys.stderr.write(f"no speech: peak {peak:.5f} < gate {gate:.5f}\n")
                sys.stderr.flush()
                return None

    audio = _to_16k_mono(np.concatenate(chunks), rate)
    if not heard_speech or len(audio) < SAMPLE_RATE * MIN_SPEECH_S:
        return None
    # Quiet mics make Whisper hallucinate or return nothing — lift the level.
    a_peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if 0 < a_peak < TARGET_PEAK:
        audio = (audio * (TARGET_PEAK / a_peak)).astype(np.float32)
        sys.stderr.write(f"normalised x{TARGET_PEAK / a_peak:.1f}\n")
        sys.stderr.flush()
    return audio


# ── push-to-talk ─────────────────────────────────────────────────────────────
# Auto-detecting when speech starts and stops means guessing a threshold, and
# a wrong guess is silent failure ("I can't hear you"). Holding a key removes
# the guess entirely: record while V is down, transcribe when it comes up.

_ptt = {"stream": None, "chunks": [], "rate": 44100}


def ptt_start(device_hint: str = "") -> None:
    import sounddevice as sd
    ptt_stop_stream()
    idx, rate, ch = pick_device(device_hint)
    _ptt["chunks"] = []
    _ptt["rate"] = rate

    def cb(indata, frames, time_info, status):  # noqa: ANN001
        _ptt["chunks"].append(
            indata.mean(axis=1) if indata.ndim > 1 else indata[:, 0].copy())

    stream = sd.InputStream(samplerate=rate, channels=ch, dtype="float32",
                            blocksize=int(rate * 0.05), device=idx, callback=cb)
    stream.start()
    _ptt["stream"] = stream


def ptt_stop_stream() -> None:
    st = _ptt.get("stream")
    if st is not None:
        try:
            st.stop(); st.close()
        except Exception:  # noqa: BLE001
            pass
        _ptt["stream"] = None


HALLUCINATIONS = (
    "продолжение следует", "субтитры сделал", "субтитры создавал",
    "редактор субтитров", "спасибо за просмотр", "dimatorzok", "amara.org",
    "подписывайтесь", "подписаться на канал", "ставьте лайк", "всем пока",
    "до новых встреч", "музыка играет", "аплодисменты",
)


def transcribe(model, audio: np.ndarray, use_vad: bool) -> str:
    """Run Whisper and drop its known silence-hallucinations.

    On near-silent input Whisper emits YouTube boilerplate ("Продолжение
    следует...") instead of nothing; sending that to an NPC as the player's
    line is worse than hearing nothing.
    """
    segments, _info = model.transcribe(
        audio, language="ru", beam_size=5, vad_filter=use_vad,
        condition_on_previous_text=False,
        temperature=[0.0, 0.2, 0.4],
        no_speech_threshold=0.8,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    low = text.lower()
    if any(h in low for h in HALLUCINATIONS):
        sys.stderr.write(f"отброшена галлюцинация тишины: {text!r}\n")
        sys.stderr.flush()
        return ""
    return text


def speech_level(audio: np.ndarray) -> float:
    """Loudness of the speech itself, immune to clicks and to leading silence.

    Plain RMS over the whole take is useless here: one keyboard click can
    dominate it, and the gain that follows leaves the speech as quiet as it
    was. Measuring per-block and taking a high percentile ignores both the
    single loud block and the silent ones.
    """
    if len(audio) == 0:
        return 0.0
    block = int(SAMPLE_RATE * 0.02)          # 20 ms
    n = len(audio) // block
    if n < 4:
        return float(np.sqrt(np.mean(audio ** 2)))
    blocks = audio[:n * block].reshape(n, block)
    rms = np.sqrt(np.mean(blocks ** 2, axis=1))
    return float(np.percentile(rms, 75))


def normalize(audio: np.ndarray) -> np.ndarray:
    """Lift a quiet take to a level Whisper can actually work with."""
    if len(audio) == 0:
        return audio
    rms = speech_level(audio)
    if rms <= 0:
        return audio
    out = audio * min(TARGET_RMS / rms, MAX_GAIN)
    # Clip the few over-range samples instead of scaling the take back down:
    # rescaling by the peak would let one keyboard click undo the whole gain
    # and leave the speech as quiet as it was.
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _dump_take(audio: np.ndarray) -> None:
    """Keep the last take on disk: "I spoke and it heard nothing" is only
    debuggable if the recording itself can be listened to afterwards."""
    try:
        import wave
        MOD_DATA.mkdir(parents=True, exist_ok=True)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(str(MOD_DATA / "last_ptt.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"dump failed: {exc}\n")


def ptt_finish() -> np.ndarray | None:
    ptt_stop_stream()
    if not _ptt["chunks"]:
        return None
    audio = _to_16k_mono(np.concatenate(_ptt["chunks"]), _ptt["rate"])
    _ptt["chunks"] = []
    if len(audio) < SAMPLE_RATE * 0.25:
        return None
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if peak < 0.0005:
        sys.stderr.write(f"ptt: тишина (peak={peak:.5f})\n")
        sys.stderr.flush()
        return None
    audio = normalize(audio)
    sys.stderr.write(f"ptt: {len(audio)/SAMPLE_RATE:.1f}s peak={peak:.4f} "
                     f"rms={rms:.5f} -> {float(np.sqrt(np.mean(audio**2))):.4f}\n")
    sys.stderr.flush()
    _dump_take(audio)
    return audio


# На чём считает распознавание. Vosk работает на процессоре — видеокарта ему
# не нужна вовсе, поэтому и одалживать библиотеки CUDA больше не у кого.
_DEVICE = "cpu"


def _load_model():
    """Поднять Vosk и УБЕДИТЬСЯ, что он вправду распознаёт.

    Конструктор модели отрабатывает и на битом каталоге, а падает всё уже на
    первой фразе — поэтому проверяем настоящим прогоном тишины: он должен
    вернуть пустой текст, а не исключение.
    """
    from vosk import Model, KaldiRecognizer, SetLogLevel
    SetLogLevel(-1)                       # kaldi болтлив, а stdout у нас занят
    if not Path(MODEL_DIR).is_dir():
        sys.stderr.write(f"нет модели распознавания: {MODEL_DIR}\n")
        sys.stderr.flush()
        return None
    try:
        t0 = time.time()
        model = Model(MODEL_DIR)
        rec = KaldiRecognizer(model, SAMPLE_RATE)
        rec.AcceptWaveform(np.zeros(SAMPLE_RATE // 2, dtype=np.int16).tobytes())
        json.loads(rec.FinalResult())
        sys.stderr.write(f"vosk готов за {time.time() - t0:.1f}с: "
                         f"{Path(MODEL_DIR).name}\n")
        sys.stderr.flush()
        return model
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"vosk не поднялся: {exc}\n")
        sys.stderr.flush()
        return None


def transcribe(model, audio: np.ndarray, use_vad: bool = False) -> str:
    """Распознать готовый кусок звука (16 кГц, float32 -1..1).

    Путь для диагностики и для проверки озвучки: живой разговор идёт другим,
    потоковым путём — см. ptt_start.

    use_vad оставлен ради совместимости с прежними вызовами и не используется:
    нажатая клавиша и ЕСТЬ решение игрока о том, что он говорит.
    """
    from vosk import KaldiRecognizer
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(False)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    step = SAMPLE_RATE * 2 // 5                       # порции по 200 мс
    for i in range(0, len(pcm), step):
        rec.AcceptWaveform(pcm[i:i + step])
    return str(json.loads(rec.FinalResult()).get("text", "")).strip()


def main() -> None:
    global _PROTO
    _PROTO = _claim_protocol_channel()
    model = _load_model()
    if model is None:
        _out({"ready": False, "err": f"vosk не поднялся, модель: {MODEL_DIR}"})
        return
    _out({"ready": True, "device": _DEVICE})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            continue
        c = cmd.get("cmd")
        if c == "quit":
            ptt_stop_stream()
            return
        if c == "transcribe":
            # Diagnostics: run a wav file through the exact recognition path.
            # "I spoke and it heard nothing" is otherwise unfalsifiable — this
            # replays data/last_ptt.wav (or any file) without the microphone.
            try:
                import wave
                t0 = time.time()
                path = str(cmd.get("path") or (MOD_DATA / "last_ptt.wav"))
                with wave.open(path, "rb") as w:
                    rate = w.getframerate()
                    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                audio = _to_16k_mono(pcm.astype(np.float32) / 32768.0, rate)
                text = transcribe(model, normalize(audio), use_vad=False)
                _out({"ok": True, "text": text, "sec": round(time.time() - t0, 2)})
            except Exception as exc:  # noqa: BLE001
                _out({"ok": False, "err": str(exc)})
            continue
        if c == "ptt_start":
            try:
                ptt_start(str(cmd.get("device") or ""))
                _out({"ok": True, "listening": True})
            except Exception as exc:  # noqa: BLE001
                _out({"ok": False, "err": str(exc)})
            continue
        if c == "ptt_stop":
            try:
                t0 = time.time()
                audio = ptt_finish()
                if audio is None:
                    _out({"ok": True, "text": "", "sec": 0.0})
                    continue
                # No VAD here: holding the key IS the voice activity decision.
                # Silero VAD was discarding whole quiet takes, and the player
                # got "I can't hear you" while clearly speaking.
                text = transcribe(model, audio, use_vad=False)
                _out({"ok": True, "text": text, "sec": round(time.time() - t0, 2)})
            except Exception as exc:  # noqa: BLE001
                _out({"ok": False, "err": str(exc)})
            continue
        if c != "listen":
            continue
        try:
            t0 = time.time()
            audio = record(float(cmd.get("max_s", 8)), str(cmd.get("device") or ""))
            if audio is None:
                _out({"ok": True, "text": "", "sec": 0.0})
                continue
            text = transcribe(model, audio, use_vad=True)
            _out({"ok": True, "text": text, "sec": round(time.time() - t0, 2)})
        except Exception as exc:  # noqa: BLE001
            _out({"ok": False, "err": str(exc)})


if __name__ == "__main__":
    main()
