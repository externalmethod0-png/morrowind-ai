"""
main.py — Async orchestrator for the Morrowind AI NPC system.

Wires together:
  - NPCMemory  (ChromaDB)
  - LoreAgent  (Gemini dialogue generation)
  - PixelAgent (screen capture + vision)
  - OBSDirector (OBS websocket control)
  - IPCBridge  (file-based IPC with OpenMW Lua)
  - YouTubeChatListener + ChatCommandHandler (stream chat integration)

Agents that are not yet implemented are imported conditionally so the
bridge and memory remain functional while the rest of the system is built out.

API key is loaded at runtime from ~/.nemoclaw_env (GOOGLE_API_KEY).
Never hardcode secrets here.
"""

import asyncio
import logging
import os
import pathlib
import signal
import sys

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT = pathlib.Path("/home/nemoclaw/morrowind-ai")
_CONFIG_FILE = pathlib.Path(__file__).parent / "config.yaml"
_ENV_FILE = pathlib.Path.home() / ".nemoclaw_env"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(config: dict) -> None:
    logs_dir = pathlib.Path(config.get("logs", {}).get("dir", str(_PROJECT / "logs")))
    logs_dir.mkdir(parents=True, exist_ok=True)

    level_name = config.get("logs", {}).get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    log_file = logs_dir / "mw-bridge.log"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    logging.info("Logging initialised (level=%s, file=%s)", level_name, log_file)


# ---------------------------------------------------------------------------
# API key loader
# ---------------------------------------------------------------------------

def _load_api_key() -> str:
    """Read GOOGLE_API_KEY from ~/.nemoclaw_env. Raises on missing key."""
    if not _ENV_FILE.exists():
        raise FileNotFoundError(f"Environment file not found: {_ENV_FILE}")
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("GOOGLE_API_KEY="):
            key = line.split("=", 1)[1].strip()
            if key:
                return key
    raise ValueError("GOOGLE_API_KEY not found or empty in ~/.nemoclaw_env")


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

def _ensure_dirs(config: dict) -> None:
    ipc_cfg = config.get("ipc", {})
    dirs = [
        pathlib.Path(ipc_cfg.get("dir", str(_PROJECT / "ipc"))),
        pathlib.Path(ipc_cfg.get("events_dir", str(_PROJECT / "ipc" / "events"))),
        pathlib.Path(config.get("logs", {}).get("dir", str(_PROJECT / "logs"))),
        pathlib.Path(config.get("memory", {}).get("chroma_dir", str(_PROJECT / "chroma"))),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logging.debug("Ensured dir: %s", d)


# ---------------------------------------------------------------------------
# Optional agent imports
# ---------------------------------------------------------------------------

def _try_import_lore_agent():
    try:
        from agents.lore_agent import LoreAgent  # type: ignore
        return LoreAgent
    except ImportError as exc:
        logging.warning("LoreAgent not available (will use stub): %s", exc)
        return None


def _try_import_pixel_agent():
    try:
        from agents.pixel_agent import PixelAgent  # type: ignore
        return PixelAgent
    except ImportError as exc:
        logging.warning("PixelAgent not available: %s", exc)
        return None


def _try_import_d2d_agent():
    try:
        from agents.d2d_agent import D2DAgent  # type: ignore
        return D2DAgent
    except ImportError as exc:
        logging.warning("D2DAgent not available: %s", exc)
        return None


def _try_import_obs_director():
    try:
        from agents.obs_director import OBSDirector  # type: ignore
        return OBSDirector
    except ImportError as exc:
        logging.warning("OBSDirector not available: %s", exc)
        return None


def _try_import_youtube_chat():
    try:
        from stream.youtube_chat import YouTubeChatListener  # type: ignore
        return YouTubeChatListener
    except ImportError as exc:
        logging.warning("YouTubeChatListener not available: %s", exc)
        return None


def _try_import_chat_commands():
    try:
        from stream.chat_commands import ChatCommandHandler  # type: ignore
        return ChatCommandHandler
    except ImportError as exc:
        logging.warning("ChatCommandHandler not available: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Stub LoreAgent (used when the real one hasn't been written yet)
# ---------------------------------------------------------------------------

class _StubLoreAgent:
    """Placeholder lore agent that echoes the player's text until the real one exists."""

    async def generate(
        self,
        npc_id: str,
        history: list,
        player_text: str,
        location: str,
    ) -> str:
        logging.warning(
            "[StubLoreAgent] No real LoreAgent loaded. Returning placeholder response."
        )
        return f"(NPC '{npc_id}' has no words for you right now.)"

    async def generate_with_system(self, system_prompt: str, user_text: str) -> str:
        logging.warning("[StubLoreAgent] generate_with_system called on stub.")
        return "(Silence falls between the NPCs.)"


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

def _register_shutdown(loop: asyncio.AbstractEventLoop, tasks: list) -> None:
    """Cancel all tasks on SIGINT / SIGTERM."""

    def _handler(sig_name: str) -> None:
        logging.info("Received %s — initiating graceful shutdown", sig_name)
        for task in tasks:
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handler, sig.name)
        except NotImplementedError:
            # Windows / some environments don't support add_signal_handler
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    # 1. Load config
    if not _CONFIG_FILE.exists():
        print(f"ERROR: config.yaml not found at {_CONFIG_FILE}", file=sys.stderr)
        sys.exit(1)

    with _CONFIG_FILE.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    # 2. Set up logging (before any further output)
    _setup_logging(config)
    logger = logging.getLogger("main")
    logger.info("=== Morrowind AI starting ===")

    # 3. Load API key
    try:
        api_key = _load_api_key()
        os.environ.setdefault("GOOGLE_API_KEY", api_key)
        logger.info("GOOGLE_API_KEY loaded from ~/.nemoclaw_env")
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Cannot load API key: %s", exc)
        sys.exit(1)

    # 4. Ensure required directories exist
    _ensure_dirs(config)

    # 5. Initialise core components
    from memory.chroma_memory import NPCMemory, DispositionStore
    chroma_dir = config.get("memory", {}).get("chroma_dir", str(_PROJECT / "chroma"))
    memory = NPCMemory(persist_dir=chroma_dir)
    logger.info("NPCMemory initialised")

    # DispositionStore — sidecar JSON with opinion, mood residue, life facts.
    # Gated by features.disposition (default: on).
    dispositions = None
    if config.get("features", {}).get("disposition", True):
        try:
            dispositions = DispositionStore(
                pathlib.Path(chroma_dir) / "dispositions.json"
            )
            logger.info("DispositionStore initialised")
        except Exception as exc:  # noqa: BLE001
            logger.error("DispositionStore failed: %s — disposition features off", exc)
            dispositions = None
    else:
        logger.info("features.disposition=false — skipping DispositionStore")

    # LoreAgent — use stub if real agent not yet written
    LoreAgentClass = _try_import_lore_agent()
    if LoreAgentClass is not None:
        try:
            lore_agent = LoreAgentClass(config)
            logger.info("LoreAgent initialised")
        except Exception as exc:  # noqa: BLE001
            logger.error("LoreAgent failed to initialise: %s — using stub", exc)
            lore_agent = _StubLoreAgent()
    else:
        lore_agent = _StubLoreAgent()
        logger.info("Using StubLoreAgent")

    # PixelAgent
    pixel_agent = None
    pixel_enabled = config.get("pixel", {}).get("enabled", False)
    if pixel_enabled:
        PixelAgentClass = _try_import_pixel_agent()
        if PixelAgentClass is not None:
            try:
                pixel_agent = PixelAgentClass(config)
                logger.info("PixelAgent initialised")
            except Exception as exc:  # noqa: BLE001
                logger.error("PixelAgent failed to initialise: %s — disabling", exc)
                pixel_enabled = False
        else:
            logger.warning("PixelAgent class unavailable — pixel capture disabled")
            pixel_enabled = False

    # OBSDirector
    obs_director = None
    if config.get("obs", {}).get("enabled", False):
        OBSDirectorClass = _try_import_obs_director()
        if OBSDirectorClass is not None:
            try:
                obs_director = OBSDirectorClass(config)
                await obs_director.connect()
                logger.info("OBSDirector connected")
            except Exception as exc:  # noqa: BLE001
                logger.warning("OBSDirector failed to connect: %s — OBS disabled", exc)
                obs_director = None

    # D2DAgent (radiant ambient NPC-to-NPC dialogue)
    d2d_agent = None
    if config.get("radiant", {}).get("enabled", True):
        D2DAgentClass = _try_import_d2d_agent()
        if D2DAgentClass is not None:
            try:
                d2d_agent = D2DAgentClass(config)
                logger.info("D2DAgent initialised")
            except Exception as exc:  # noqa: BLE001
                logger.error("D2DAgent failed to initialise: %s — radiant D2D disabled", exc)

    # OpenMWLogBridge (Windows IPC via openmw.log tailing)
    from openmw_log_bridge import OpenMWLogBridge
    log_bridge = OpenMWLogBridge(
        config, lore_agent, memory,
        d2d_agent=d2d_agent,
        dispositions=dispositions,
    )
    logger.info("OpenMWLogBridge initialised")

    # IPCBridge (always runs — this is the core of the system)
    from bridge import IPCBridge
    bridge = IPCBridge(
        config=config,
        lore_agent=lore_agent,
        memory=memory,
        pixel_agent=pixel_agent,
    )
    logger.info("IPCBridge initialised")

    # YouTube chat listener
    chat_listener = None
    chat_handler = None
    video_id = config.get("stream", {}).get("youtube_video_id", "")
    if video_id:
        ChatCommandHandlerClass = _try_import_chat_commands()
        YouTubeChatListenerClass = _try_import_youtube_chat()
        if ChatCommandHandlerClass and YouTubeChatListenerClass:
            try:
                chat_handler = ChatCommandHandlerClass(config)
                chat_listener = YouTubeChatListenerClass(config, chat_handler)
                logger.info("YouTube chat listener initialised for video_id=%s", video_id)
            except Exception as exc:  # noqa: BLE001
                logger.error("Chat listener failed to initialise: %s", exc)
                chat_listener = None
    else:
        logger.info("stream.youtube_video_id not set — chat listener disabled")

    # 6. Build concurrent task list
    loop = asyncio.get_running_loop()
    tasks = []

    bridge_task = loop.create_task(bridge.run(), name="ipc-bridge")
    tasks.append(bridge_task)

    log_bridge_task = loop.create_task(log_bridge.run(), name="log-bridge")
    tasks.append(log_bridge_task)

    if pixel_enabled and pixel_agent is not None and hasattr(pixel_agent, "capture_loop"):
        pixel_task = loop.create_task(pixel_agent.capture_loop(), name="pixel-agent")
        tasks.append(pixel_task)
        logger.info("PixelAgent capture_loop scheduled")

    if chat_listener is not None and hasattr(chat_listener, "start"):
        chat_task = loop.create_task(chat_listener.start(video_id), name="chat-listener")
        tasks.append(chat_task)
        logger.info("YouTubeChatListener scheduled")

    _register_shutdown(loop, tasks)

    logger.info(
        "=== Morrowind AI running — %d task(s) active ===",
        len(tasks),
    )

    # 7. Wait for all tasks; handle cancellation gracefully
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Tasks cancelled — shutting down cleanly")
    except Exception as exc:  # noqa: BLE001
        logger.error("Unhandled error in task: %s", exc, exc_info=True)
    finally:
        if obs_director is not None and hasattr(obs_director, "disconnect"):
            try:
                await obs_director.disconnect()
            except Exception:  # noqa: BLE001
                pass
        logger.info("=== Morrowind AI stopped ===")


if __name__ == "__main__":
    asyncio.run(main())
