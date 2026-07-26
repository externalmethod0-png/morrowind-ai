"""
xtts_daemon.py — voice cloning daemon (XTTS-v2), run under morrowind-ai/xtts/venv.

Speaks Russian in the voice of the game's OWN Russian voice-over: for each NPC
we pick a reference clip from Sound/Vo/<race>/<gender>/ and clone its timbre,
so a Dunmer sounds like the Dunmer you have heard for twenty years.

Speed note: the high-level API recomputes the speaker embedding from the
reference WAV on every single line (~2s of the ~4s total). We compute those
conditioning latents ONCE per reference clip, cache them in RAM and on disk,
and then only pay for the actual generation.

Protocol (JSON lines on stdin/stdout):
  in : {"cmd":"say","text":"...","ref":"C:/...mp3","out":"C:/...wav"}
       {"cmd":"warm","refs":["C:/a.mp3", ...]}      # precompute latents
  out: {"ok":true,"out":"C:/...wav","sec":1.8}   |  {"ok":false,"err":"..."}
Prints {"ready":true} once the model is loaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import sys
import time

os.environ.setdefault("COQUI_TOS_AGREED", "1")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "data", "xtts_latents")
SR = 24000        # XTTS output rate


def _claim_protocol_channel():
    """Keep the JSON protocol on a pipe nobody else writes to.

    Coqui/torch print progress and warnings to stdout in the system codepage;
    one such line inside the protocol breaks the reply parsing. Stray output
    goes to stderr (the log), the protocol keeps the real stdout.
    """
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    return os.fdopen(proto_fd, "w", encoding="utf-8", buffering=1)


_PROTO = None       # claimed by main(); importing this module must not steal
                    # the stdout of whoever imports it


def _out(obj) -> None:
    ch = _PROTO or sys.stdout
    ch.write(json.dumps(obj, ensure_ascii=False) + "\n")
    ch.flush()


class Engine:
    def __init__(self) -> None:
        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.utils.manage import ModelManager

        self.torch = torch
        name = "tts_models/multilingual/multi-dataset/xtts_v2"
        path, _, _ = ModelManager().download_model(name)
        cfg = XttsConfig()
        cfg.load_json(os.path.join(path, "config.json"))
        self.model = Xtts.init_from_config(cfg)
        self.model.load_checkpoint(cfg, checkpoint_dir=path, eval=True)
        # Устройство можно задать снаружи — так их и сравнивают замером.
        # MWAI_XTTS_DEVICE=cpu уводит синтез с ускорителя целиком.
        want = (os.environ.get("MWAI_XTTS_DEVICE") or "").strip().lower()
        if want in ("cpu", "cuda"):
            self.device = want if (want == "cpu" or torch.cuda.is_available()) else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self._lat: dict[str, tuple] = {}
        os.makedirs(CACHE_DIR, exist_ok=True)

    # ---------------------------------------------------------- latents

    def _cache_path(self, ref: str) -> str:
        h = hashlib.md5(ref.encode("utf-8", "ignore")).hexdigest()[:16]
        return os.path.join(CACHE_DIR, f"{h}.pkl")

    def latents(self, ref: str):
        """Speaker embedding for a reference clip — computed once, then reused."""
        if ref in self._lat:
            return self._lat[ref]
        p = self._cache_path(ref)
        if os.path.exists(p):
            try:
                with open(p, "rb") as fh:
                    gpt_lat, spk_emb = pickle.load(fh)
                gpt_lat = gpt_lat.to(self.device)
                spk_emb = spk_emb.to(self.device)
                self._lat[ref] = (gpt_lat, spk_emb)
                return self._lat[ref]
            except Exception:  # noqa: BLE001
                pass
        gpt_lat, spk_emb = self.model.get_conditioning_latents(audio_path=[ref])
        try:
            with open(p, "wb") as fh:
                pickle.dump((gpt_lat.cpu(), spk_emb.cpu()), fh)
        except OSError:
            pass
        self._lat[ref] = (gpt_lat, spk_emb)
        return self._lat[ref]

    # ------------------------------------------------------------- say

    @staticmethod
    def split_sentences(text: str, limit: int = 160, first: int = 90) -> list[str]:
        """Cut a reply into pieces XTTS can swallow in one call.

        XTTS' own splitter (enable_text_splitting) needs Spacy, which is not
        installed here: every reply longer than a sentence died with "requires
        Spacy" and the NPC stayed mute while its text sat on screen. Splitting
        here keeps the dependency out and the voice in.

        The limit must stay under XTTS' own Russian ceiling of 182 characters.
        Above it the model warns about truncation and generates rambling audio
        — one over-long chunk produced 19 seconds of speech, the queue backed
        up behind it and every later reply was dropped unspoken.

        The FIRST piece is deliberately shorter: the caller plays it while the
        rest is still being generated, so how soon the NPC starts talking is
        decided by that first piece alone.
        """
        text = " ".join((text or "").split())
        if not text:
            return []
        parts, cur = [], ""

        def cap() -> int:
            return first if not parts else limit      # пока собираем первый кусок

        for chunk in re.split(r"(?<=[.!?…])\s+", text):
            while len(chunk) > cap():                  # one giant sentence
                room = cap()
                cut = chunk.rfind(" ", 0, room)
                cut = cut if cut > 0 else room
                if cur:
                    parts.append(cur)
                    cur = ""
                parts.append(chunk[:cut].strip())
                chunk = chunk[cut:].strip()
            if not chunk:
                continue
            if cur and len(cur) + len(chunk) + 1 <= cap():
                cur = (cur + " " + chunk).strip()
            elif not cur:
                cur = chunk
            else:
                parts.append(cur)
                cur = chunk
        if cur:
            parts.append(cur)
        return [p for p in parts if p]

    def _shift(self, wav, pitch: float):
        """Move the voice up or down without changing how long it takes.

        Morrowind's voice-over has ONE actor per race+gender, so cloning from
        different clips of the same pool gives the same voice — every Imperial
        guard sounded identical. Resampling shifts the timbre; the generation
        speed was set to the inverse beforehand, so the duration stays put.
        """
        if abs(pitch - 1.0) < 0.005:
            return wav
        import torchaudio.functional as AF
        # Compress the waveform by `pitch` while still declaring SR in the
        # header: the file then plays back that much faster, which lifts pitch
        # and formants together. (Resampling the other way round preserves the
        # sound instead of shifting it — measured, the voices came out with
        # the shift inverted.)
        return AF.resample(wav, SR, int(round(SR / pitch)))

    def say(self, text: str, ref: str, out: str, emit=None,
            pitch: float = 1.0) -> float:
        """Synthesize sentence by sentence, announcing each piece as it lands.

        A four-sentence reply takes ~15 s to synthesize in full. Announcing
        every piece lets the caller start playing the first one after ~3 s
        while the rest are still being generated, so the NPC starts talking
        about as fast as it used to for a one-liner.
        """
        import torchaudio
        t0 = time.time()
        gpt_lat, spk_emb = self.latents(ref)
        pieces = self.split_sentences(text[:600])
        if not pieces:
            raise ValueError("нечего озвучивать")
        base = out[:-4] if out.lower().endswith(".wav") else out
        # Lower temperature than the XTTS default: measured delivery of the
        # same line varied by ~13% in pitch between takes, which both washed
        # out the per-NPC voice and made the "рррр" collapse more likely —
        # that failure is a sampling one.
        kw = dict(language="ru", gpt_cond_latent=gpt_lat, speaker_embedding=spk_emb,
                  temperature=0.55, enable_text_splitting=False)
        if abs(pitch - 1.0) >= 0.005:
            kw["speed"] = 1.0 / pitch     # компенсируем сдвиг темпа от resample
        for i, piece in enumerate(pieces):
            try:
                res = self.model.inference(text=piece, **kw)
            except TypeError:             # старая сборка XTTS без speed
                kw.pop("speed", None)
                res = self.model.inference(text=piece, **kw)
            path = out if len(pieces) == 1 else f"{base}_{i}.wav"
            # 16-bit PCM, not the float32 torchaudio writes by default: half
            # the size and playable by anything, including tools used to check
            # the output afterwards.
            wav = self.torch.tensor(res["wav"]).clamp(-1.0, 1.0).unsqueeze(0)
            wav = self._shift(wav, pitch)
            torchaudio.save(path, wav, SR, encoding="PCM_S", bits_per_sample=16)
            if emit is not None:
                emit({"chunk": path, "i": i, "of": len(pieces)})
        return time.time() - t0


def main() -> None:
    global _PROTO
    _PROTO = _claim_protocol_channel()
    try:
        eng = Engine()
    except Exception as exc:  # noqa: BLE001
        _out({"ready": False, "err": str(exc)})
        return
    _out({"ready": True, "device": eng.device})

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
            return
        if c == "warm":
            n = 0
            for ref in (cmd.get("refs") or []):
                try:
                    eng.latents(ref); n += 1
                except Exception:  # noqa: BLE001
                    pass
            _out({"ok": True, "warmed": n})
            continue
        if c != "say":
            continue
        try:
            # Each finished sentence is announced immediately; the terminal
            # {"ok":...} line always follows, so the caller can play as it goes
            # and still knows when the reply is complete.
            sec = eng.say(str(cmd.get("text") or ""), cmd["ref"], cmd["out"],
                          emit=_out, pitch=float(cmd.get("pitch") or 1.0))
            _out({"ok": True, "out": cmd["out"], "sec": round(sec, 2)})
        except Exception as exc:  # noqa: BLE001
            _out({"ok": False, "err": str(exc)})


if __name__ == "__main__":
    main()
