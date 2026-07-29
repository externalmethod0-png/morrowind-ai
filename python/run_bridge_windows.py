"""
run_bridge_windows.py — Native-Windows launcher for the morrowind-ai bridge.

The upstream openmw_log_bridge.py hardcodes WSL paths (/mnt/c/Users/rneeb/...).
This launcher overrides those module globals with the real paths of THIS
portable OpenMW install, swaps ChromaDB for the lightweight JSON memory, and
runs the core dialogue bridge (lore_agent + memory).

Run it (via start_morrowind_ai.bat, or directly):
    venv/Scripts/python.exe run_bridge_windows.py

Then launch OpenMW and press H near an NPC.
"""

from __future__ import annotations

import asyncio
import logging
import time
import os
import sys
from pathlib import Path

import yaml

# --- Resolve install paths (relative to this file) --------------------------
# This file lives at: <GAME>\morrowind-ai\python\run_bridge_windows.py
PYTHON_DIR = Path(__file__).resolve().parent
MOD_PKG_DIR = PYTHON_DIR.parent                      # ...\morrowind-ai
GAME_ROOT = MOD_PKG_DIR.parent                       # ...\Morrowind (ReBuild)
OPENMW_USER_DIR = GAME_ROOT / "OPENMW"               # portable user/config dir
MOD_DATA_DIR = MOD_PKG_DIR / "openmw-mod"            # the data= path in openmw.cfg
MEMORY_DIR = MOD_PKG_DIR / "data" / "memory"

# openmw.log location: portable install writes it next to the configs
# (user-data="../" -> OPENMW\). Fall back to the default Documents location.
_LOG_CANDIDATES = [
    OPENMW_USER_DIR / "openmw.log",
    Path.home() / "Documents" / "My Games" / "OpenMW" / "openmw.log",
]


def _pick_log() -> Path:
    for c in _LOG_CANDIDATES:
        if c.exists():
            return c
    # None exist yet (game never launched). Default to portable path; the
    # bridge waits for it to appear.
    return _LOG_CANDIDATES[0]


# Scripts that belong to a bridge session: the bridge itself plus the helper
# daemons it spawns (they survive their parent and would keep the GPU busy).
_OUR_SCRIPTS = ("run_bridge_windows.py", "xtts_daemon.py", "stt_daemon.py")


def _kill_previous_instances(log) -> None:
    """Terminate any bridge/daemon left over from an earlier launch."""
    try:
        import psutil
    except ImportError:
        log.warning("psutil не установлен — старые мосты не будут закрыты автоматически")
        return
    me = os.getpid()
    # Never kill ourselves or our own ancestors (the shell that launched us).
    safe = {me}
    try:
        safe.update(p.pid for p in psutil.Process(me).parents())
    except Exception:  # noqa: BLE001
        pass

    killed = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.info["pid"] in safe:
                continue
            if not (proc.info["name"] or "").lower().startswith("python"):
                continue
            argv = proc.info["cmdline"] or []
            # Match an ACTUAL script launch (python …\xxx.py), never a one-liner
            # that merely mentions the name inside a -c string.
            if len(argv) < 2 or "-c" in argv[:2]:
                continue
            script = os.path.basename(argv[-1]).lower()
            if script in _OUR_SCRIPTS:
                proc.kill()
                killed.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if killed:
        log.info("закрыты прошлые процессы моста: %s", killed)
        try:
            psutil.wait_procs([psutil.Process(p) for p in killed if psutil.pid_exists(p)], timeout=3)
        except Exception:  # noqa: BLE001
            pass


_LOG_LISTENER = None      # держим ссылку, иначе поток записи логов соберёт GC


def _disable_console_quick_edit() -> bool:
    """Stop a mouse click in the bridge window from freezing the whole bridge.

    Windows consoles ship with QuickEdit on: clicking inside the window starts
    a text selection and SUSPENDS the process at its next write to the console.
    The bridge then sits there alive but frozen — no error, no log line, and in
    game every NPC goes silent. Looking at the window was enough to break it.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-10)          # STD_INPUT_HANDLE
        if handle in (0, -1, None):
            return False
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False                              # нет консоли (pythonw и т.п.)
        QUICK_EDIT, EXTENDED_FLAGS = 0x0040, 0x0080
        new_mode = (mode.value & ~QUICK_EDIT) | EXTENDED_FLAGS
        return bool(kernel32.SetConsoleMode(handle, new_mode))
    except Exception:  # noqa: BLE001
        return False


def _setup_logging() -> None:
    """Console plus data/bridge.log, written from a dedicated thread.

    The console window disappears with the session, taking the only record of
    what went wrong with it, so everything also goes to a file.

    Writing goes through a queue on purpose: a console can stop accepting
    output (a frozen or blocked terminal), and with direct handlers that would
    stall whichever thread happened to be logging - including the loop that
    reads the game's requests. Behind a queue only the logging thread waits,
    and the bridge keeps answering.
    """
    import queue as _queue
    from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s",
                            datefmt="%H:%M:%S")
    targets: list[logging.Handler] = []
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    targets.append(console)
    try:
        (MOD_PKG_DIR / "data").mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(MOD_PKG_DIR / "data" / "bridge.log",
                                 maxBytes=2_000_000, backupCount=2,
                                 encoding="utf-8")
        fh.setFormatter(fmt)
        targets.append(fh)
    except OSError:
        pass

    global _LOG_LISTENER
    q: _queue.Queue = _queue.Queue(-1)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(QueueHandler(q))
    _LOG_LISTENER = QueueListener(q, *targets, respect_handler_level=True)
    _LOG_LISTENER.start()


def main() -> None:
    quick_edit_off = _disable_console_quick_edit()
    _setup_logging()
    log = logging.getLogger("run_bridge_windows")
    if quick_edit_off:
        log.info("выделение мышью в этом окне больше не морозит мост")

    # Ensure imports (agents.*, memory.*, providers.*) resolve.
    sys.path.insert(0, str(PYTHON_DIR))

    # Fail fast with a friendly message if the API key isn't set up yet.
    env_file = Path.home() / ".nemoclaw_env"
    key_ok = False
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GOOGLE_API_KEY=") and len(line.split("=", 1)[1].strip()) > 10:
                key_ok = True
                break
    if not key_ok:
        log.error(
            "Gemini API key not found. Put your key in:\n"
            "    %s\n"
            "as a single line:\n"
            "    GOOGLE_API_KEY=ваш_ключ_здесь\n"
            "Get a free key at https://aistudio.google.com/apikey",
            env_file,
        )
        sys.exit(2)

    # Exactly one bridge, always. Two bridges answer every line twice and burn
    # the quota twice as fast; the old .bat kill-line was silently broken, so
    # they used to pile up. The freshly launched bridge wins: it terminates any
    # older bridge AND its orphaned daemons, then takes over.
    _kill_previous_instances(log)

    import openmw_log_bridge as bridgemod
    from agents.lore_agent import LoreAgent
    from memory.json_memory import NPCMemory

    # --- Override the hardcoded WSL paths with real Windows paths ----------
    log_path = _pick_log()
    bridgemod.OPENMW_LOG = log_path
    bridgemod.MOD_ROOT = MOD_DATA_DIR
    bridgemod.INBOX_DIR = MOD_DATA_DIR / "ai_inbox"
    bridgemod.RESPONSE_FILE = bridgemod.INBOX_DIR / "response.txt"
    # The reply JOURNAL must be overridden too — miss it and every answer is
    # written into the unused WSL-style path while the game waits forever.
    bridgemod.JOURNAL_FILE = bridgemod.INBOX_DIR / "responses.ndjson"
    bridgemod.NPC_SPEECH_FILE = bridgemod.INBOX_DIR / "npc_speech.txt"
    bridgemod.PLAYER_TEXT_FILE = bridgemod.INBOX_DIR / "player_text.txt"

    cfg = yaml.safe_load((PYTHON_DIR / "config.yaml").read_text(encoding="utf-8"))

    log.info("=== morrowind-ai bridge (Windows) ===")
    log.info("openmw.log : %s%s", log_path, "" if log_path.exists() else "  (waiting for it to appear)")
    log.info("mod data   : %s", MOD_DATA_DIR)
    log.info("inbox      : %s", bridgemod.INBOX_DIR)
    log.info("журнал     : %s", bridgemod.JOURNAL_FILE)
    log.info("memory     : %s", MEMORY_DIR)

    # Clear any stale response from a previous session so a fresh game never
    # picks up (and acts on!) an old NPC reply.
    # It must go through _write_slot, i.e. padded to the same fixed size as a
    # real reply: OpenMW's VFS remembers the size the file had when the game
    # started, so a 2-byte "{}" at launch would truncate every later answer to
    # two bytes and the player would sit there staring at "жду ответ".
    try:
        bridgemod._write_slot(bridgemod.RESPONSE_FILE, "{}")
        bridgemod._write_slot(bridgemod.NPC_SPEECH_FILE, "{}")
        log.info("stale response.txt cleared (слот %d байт)", bridgemod.SLOT_BYTES)
        # Ручки характера мира — файл для игры обязан существовать ДО её
        # запуска: опись файлов VFS снимается один раз, на старте.
        try:
            import world_tuning
            world_tuning.ensure_file()
            dials = world_tuning.publish()
            log.info("характер мира: опасность %d, нелепость %d "
                     "(правится на лету в data/настройки-мира.txt)",
                     dials["опасность"], dials["нелепость"])
        except Exception as exc:  # noqa: BLE001
            log.warning("ручки характера мира не поднялись: %s", exc)

        # Запомнить конец лога ДО объявления готовности: дальше поднимаются
        # распознавание и озвучка (секунд десять), и всё, что игра успеет
        # сказать за это время, иначе пропало бы — цикл чтения стартовал бы уже
        # за этими строками.
        bridgemod.mark_log_position()
        # The launcher waits for this marker before starting OpenMW. The game
        # indexes its virtual file system once, at startup: a reply file that
        # does not exist by then stays invisible for the whole session.
        (bridgemod.INBOX_DIR / "bridge_ready.txt").write_text("ready", encoding="utf-8")
    except OSError as exc:
        log.warning("could not clear response.txt: %s", exc)

    memory = NPCMemory(str(MEMORY_DIR))
    lore = LoreAgent(cfg)

    # Social state (mood, life facts, rumors) is SAVE-SCOPED: it lives inside
    # the savegame on the Lua side and arrives with each request. The bridge
    # keeps no cross-save memory, so reloading a save rewinds everything.
    dispositions = None

    # Radiant NPC<->NPC ambient chatter (tamed Lua-side by cooldowns/radius).
    d2d = None
    if (cfg.get("radiant") or {}).get("enabled"):
        try:
            from agents.d2d_agent import D2DAgent
            d2d = D2DAgent(cfg)
            log.info("radiant D2D agent: ON")
        except Exception as exc:  # noqa: BLE001
            log.warning("radiant D2D disabled (failed to init): %s", exc)

    bridge = bridgemod.OpenMWLogBridge(cfg, lore, memory, d2d_agent=d2d, dispositions=dispositions)

    # Постановщик сцен: эпизоды на несколько человек — по клавише режиссёра и
    # сами собой. Не поднялся — мод живёт без сцен, разговоры не страдают.
    if (cfg.get("scenes") or {}).get("enabled", True):
        try:
            from agents.scene_agent import SceneAgent
            bridge.scene_agent = SceneAgent(cfg)
            log.info("сцены: включены (клавиша K — задать свою)")
        except Exception as exc:  # noqa: BLE001
            log.warning("сцены отключены (не поднялся постановщик): %s", exc)

    # NPC voices. engine: "edge" (Microsoft neural, natural, needs internet)
    # or "silero" (fully offline, more robotic). Both give per-NPC stable
    # gender-matched voices.
    tts_cfg = cfg.get("tts") or {}
    if tts_cfg.get("enabled", True):
        engine = str(tts_cfg.get("engine", "piper")).lower()
        try:
            if engine == "silero":
                from tts import SileroTTS
                bridge.tts = SileroTTS(str(MOD_PKG_DIR / "data" / "tts"))
                log.info("TTS: Silero (offline) loading in background")
            elif engine == "edge":
                from tts_edge import EdgeTTS
                bridge.tts = EdgeTTS(str(MOD_PKG_DIR / "data" / "tts"))
                log.info("TTS: Edge neural voices (per-NPC variants)")
            elif engine == "xtts":
                from tts_xtts import XttsTTS
                bridge.tts = XttsTTS(str(MOD_PKG_DIR / "data" / "tts"))
                log.info("TTS: XTTS — клонирование голосов из русской озвучки игры")
            elif engine in ("morrowind", "mw"):
                from tts_morrowind import MorrowindTTS
                bridge.tts = MorrowindTTS(str(MOD_PKG_DIR / "data" / "tts"))
                log.info("TTS: голоса, дообученные на родной озвучке игры")
            else:
                from tts_piper import PiperTTS
                bridge.tts = PiperTTS(str(MOD_PKG_DIR / "data" / "tts"))
                log.info("TTS: Piper (offline, ~0.5s per line)")
        except Exception as exc:  # noqa: BLE001
            log.warning("TTS disabled (failed to init): %s", exc)

        # Голос ИЗ САМОГО NPC: реплику играет движок игры, а не мы мимо неё —
        # слышно направление, глушат стены, замолкает вместе с паузой. Слоты
        # обязаны появиться СЕЙЧАС, до запуска игры: опись файлов VFS игра
        # составляет один раз на старте.
        if tts_cfg.get("spatial", False):
            from audio_out import enable_spatial
            if enable_spatial(True):
                log.info("звук: голоса идут через движок игры (из точки, где стоит NPC)")
            else:
                log.warning("звук: движковый режим не поднялся — играю мимо игры")

        # Заминка на время генерации. Нужна только медленным движкам: у xtts
        # между вопросом и первым звуком проходит секунд восемь, и NPC всё это
        # время выглядит зависшим. Piper успевает вставить «хм…» за 0.7 с.
        # Заминка нужна при ЛЮБОМ движке озвучки, а не только при медленном.
        # Даже когда синтез идёт за 0.3 с, ответ модели считается 2.3 — и это
        # ровно та пауза, в которую NPC стоит столбом. Банк нарисован голосами
        # самой игры, поэтому подходит и обученным голосам, и XTTS.
        if tts_cfg.get("filler", True):
            # Сперва банк готовых заминок ГОЛОСАМИ САМОЙ ИГРЫ: играется
            # мгновенно и тем же тембром, каким NPC потом ответит.
            try:
                from filler_bank import FillerBank
                bank = FillerBank(MOD_PKG_DIR / "data" / "tts")
                if bank.available:
                    bridge.filler_bank = bank
                    log.info("заминка перед ответом: готовый банк, %d голосов",
                             len(bank.pools))
            except Exception as exc:  # noqa: BLE001
                log.warning("банк заминок не поднялся: %s", exc)
            # Запасной вариант, если банк не собран (tools/build_fillers.py).
            if getattr(bridge, "filler_bank", None) is None:
                try:
                    from tts_piper import PiperTTS
                    bridge.filler = PiperTTS(str(MOD_PKG_DIR / "data" / "tts_filler"))
                    log.info("заминка перед ответом: включена (piper, чужой тембр)")
                except Exception as exc:  # noqa: BLE001
                    log.warning("заминка недоступна: %s", exc)

    # Voice input (V key in game): whisper via the Wisper venv daemon.
    # It runs on the CPU on purpose — see _load_model() in stt_daemon.py: on
    # this GPU recognition took anywhere from 6 to 71 seconds, while the CPU is
    # a steady ~3 s and leaves the card to the game and the voices.
    if (cfg.get("voice") or {}).get("enabled", True):
        try:
            from voice_stt import VoiceSTT
            vcfg = cfg.get("voice") or {}
            mic = str(vcfg.get("device") or "")
            where = str(vcfg.get("compute_device") or "cpu")
            bridge.stt = VoiceSTT(device_hint=mic, compute_device=where)
            log.info("Voice mode: распознавание на %s, микрофон=%s",
                     where.upper(), mic or "по умолчанию")
        except Exception as exc:  # noqa: BLE001
            log.warning("Voice mode disabled: %s", exc)

    # ПРОГРЕВ ДОМАШНЕЙ МОДЕЛИ. Первая реплика за сеанс стоит вшестеро дороже
    # остальных: модель разбирает неизменную часть промпта — правила мира и
    # список команд — и только потом отвечает. Замерено на гигачате: 28.8 с
    # первый запрос против 3.5 с следующего, разница в восемь раз.
    #
    # В игре это выглядело как «первый NPC думает почти минуту». Мост
    # запускается РАНЬШЕ игры, так что пусть он это время и потратит: пока
    # человек грузит сохранение, модель уже прочла правила.
    #
    # Только для СВОЕЙ модели. Облаку прогрев ничего не ускоряет, а запрос
    # стоил бы денег на пустом месте.
    if str((cfg.get("models", {}).get("lore_agent", {}) or {})
           .get("provider", "")).lower() in ("local", "ollama", "llamacpp", "lmstudio"):
        async def _warm() -> None:
            await lore.generate_response({
                "npc_name": "Прохожий", "npc_race": "Dunmer",
                "npc_class": "Commoner", "npc_disposition": 50,
                "player_input": "Здравствуй.", "conversation_history": [],
            }, [])

        try:
            t0 = time.time()
            log.info("прогреваю модель — читает правила, чтобы первый NPC не ждал")
            asyncio.run(_warm())
            log.info("модель прогрета за %.1f с", time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            # Не беда: прогрев — ускорение, а не условие работы.
            log.warning("прогрев не удался (%s) — играть можно, первая реплика "
                        "будет дольше", exc)

    log.info("Bridge ready. Launch OpenMW and press H near an NPC. Ctrl+C to stop.")
    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        log.info("Stopped by user.")
    except Exception:  # noqa: BLE001
        # Without this the traceback goes to a console window that is minimised
        # and then closed, and the log simply stops mid-session with no reason
        # given — which is exactly how a crash stayed invisible.
        log.exception("МОСТ УПАЛ — работа прервана")
        raise


if __name__ == "__main__":
    main()
