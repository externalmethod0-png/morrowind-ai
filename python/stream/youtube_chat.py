"""
youtube_chat.py — YouTube live chat listener using pytchat (no YouTube Data API).

Architecture:
  - A blocking pytchat loop runs in a ThreadPoolExecutor (can't be awaited directly).
  - Each message is placed on an asyncio.Queue via loop.call_soon_threadsafe().
  - A separate async consumer task drains the queue and calls command_handler.handle_message().
  - Graceful shutdown: stop() sets a threading.Event; the executor thread exits; the queue
    consumer is cancelled after the queue drains.
"""

import asyncio
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_LOGS_DIR = '/home/nemoclaw/morrowind-ai/logs'
_STATE_FILE = '/home/nemoclaw/morrowind-ai/ipc/stream_state.json'

_CHAT_LOG_PATH = os.path.join(_LOGS_DIR, 'chat.log')
_STREAM_LOG_PATH = os.path.join(_LOGS_DIR, 'stream.log')


def _setup_file_logger(name: str, path: str) -> logging.Logger:
    """Create (or fetch) a logger that also writes to a rolling log file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lg = logging.getLogger(name)
    if not lg.handlers:
        fh = logging.FileHandler(path, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        lg.addHandler(fh)
    return lg


_chat_log  = _setup_file_logger('morrowind.chat',   _CHAT_LOG_PATH)
_stream_log = _setup_file_logger('morrowind.stream', _STREAM_LOG_PATH)


def _load_state_file() -> dict:
    """Read ipc/stream_state.json. Returns {} on any error."""
    try:
        with open(_STATE_FILE, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError) as exc:
        _stream_log.warning("Could not read stream_state.json: %s", exc)
        return {}


def _detect_video_id(config: dict) -> Optional[str]:
    """
    Resolve video_id with priority:
      1. config dict ('video_id' key)
      2. stream_state.json on disk
    Returns None if not found.
    """
    vid = config.get('video_id')
    if vid:
        return vid
    state = _load_state_file()
    vid = state.get('video_id')
    if vid:
        _stream_log.info("video_id auto-detected from stream_state.json: %s", vid)
        return vid
    return None


def _log_chat_message(author: str, message: str, timestamp: str) -> None:
    """Append a single chat message to chat.log."""
    try:
        _chat_log.info("[%s] <%s> %s", timestamp, author, message)
    except Exception as exc:
        _stream_log.error("Failed to log chat message: %s", exc)


class YouTubeChatListener:
    """
    Listens to a YouTube live stream's chat via pytchat and dispatches messages
    to a ChatCommandHandler.

    Usage:
        handler = ChatCommandHandler(config)
        listener = YouTubeChatListener(config, handler)
        await listener.start()           # blocks until stop() or stream ends
        # ... from another coroutine:
        await listener.stop()
    """

    # Pause between pytchat polls (seconds) — pytchat has its own internal rate limit
    _POLL_INTERVAL = 1.0
    # How long to wait for queue to drain on shutdown
    _SHUTDOWN_DRAIN_TIMEOUT = 5.0

    def __init__(self, config: dict, command_handler):
        self.config = config
        self.command_handler = command_handler
        self._stop_event = threading.Event()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._executor_task: Optional[asyncio.Future] = None
        self._consumer_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def start(self, video_id: Optional[str] = None) -> None:
        """
        Start listening to chat. Blocks until the stream ends or stop() is called.

        video_id: explicit override. If None, falls back to config then stream_state.json.
        """
        resolved_id = video_id or _detect_video_id(self.config)
        if not resolved_id:
            _stream_log.error(
                "YouTubeChatListener: no video_id found. "
                "Set it in config.yaml or ipc/stream_state.json."
            )
            raise ValueError(
                "video_id is required. Provide it in config['video_id'] "
                "or ipc/stream_state.json {'video_id': '...'}."
            )

        _stream_log.info("YouTubeChatListener starting for video_id=%s", resolved_id)
        self._stop_event.clear()
        loop = asyncio.get_event_loop()

        # Spin up the async queue consumer
        self._consumer_task = asyncio.ensure_future(self._consume_queue())

        try:
            # Run the blocking pytchat loop in a thread
            self._executor_task = loop.run_in_executor(
                None,
                self._blocking_chat_loop,
                resolved_id,
                loop,
            )
            await self._executor_task
        except asyncio.CancelledError:
            _stream_log.info("YouTubeChatListener executor cancelled")
        except Exception as exc:
            _stream_log.error("YouTubeChatListener executor error: %s", exc)
        finally:
            # Signal stop and wait briefly for queue to drain
            self._stop_event.set()
            try:
                await asyncio.wait_for(
                    self._drain_queue(),
                    timeout=self._SHUTDOWN_DRAIN_TIMEOUT
                )
            except asyncio.TimeoutError:
                _stream_log.warning("Queue drain timed out on shutdown")
            if self._consumer_task and not self._consumer_task.done():
                self._consumer_task.cancel()
                try:
                    await self._consumer_task
                except asyncio.CancelledError:
                    pass
            _stream_log.info("YouTubeChatListener stopped")

    async def stop(self) -> None:
        """Signal the listener to shut down gracefully."""
        _stream_log.info("YouTubeChatListener stop requested")
        self._stop_event.set()
        if self._executor_task is not None:
            self._executor_task.cancel()
        if self._consumer_task is not None:
            self._consumer_task.cancel()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _blocking_chat_loop(self, video_id: str, loop: asyncio.AbstractEventLoop) -> None:
        """
        Blocking loop — runs in a ThreadPoolExecutor thread.
        Puts (author, message, timestamp) tuples on the asyncio queue via
        call_soon_threadsafe so they're safe to consume from the async side.
        """
        try:
            import pytchat
        except ImportError:
            _stream_log.error(
                "pytchat is not installed. Install it with: pip install pytchat"
            )
            return

        _stream_log.info("pytchat: connecting to video_id=%s", video_id)

        try:
            chat = pytchat.create(video_id=video_id)
        except Exception as exc:
            _stream_log.error("pytchat.create failed: %s", exc)
            return

        while chat.is_alive() and not self._stop_event.is_set():
            try:
                items = chat.get().sync_items()
                for c in items:
                    if self._stop_event.is_set():
                        break
                    try:
                        author    = c.author.name
                        message   = c.message
                        # c.datetime is a datetime-like string; normalise it
                        timestamp = str(c.datetime) if c.datetime else datetime.now(timezone.utc).isoformat()
                        _log_chat_message(author, message, timestamp)
                        # Hand off to async side
                        loop.call_soon_threadsafe(
                            self._queue.put_nowait,
                            (author, message, timestamp)
                        )
                    except Exception as exc:
                        _stream_log.error("Error processing chat item: %s", exc)
            except Exception as exc:
                _stream_log.error("pytchat get() error: %s", exc)

            # Brief sleep to avoid hammering the internal pytchat buffer
            self._stop_event.wait(timeout=self._POLL_INTERVAL)

        if not self._stop_event.is_set():
            _stream_log.info("pytchat: stream ended for video_id=%s", video_id)
        self._stop_event.set()

    async def _consume_queue(self) -> None:
        """
        Async consumer — drains the message queue and calls command_handler.
        Runs as a Task alongside the executor.
        """
        while True:
            try:
                author, message, timestamp = await self._queue.get()
                try:
                    await self.command_handler.handle_message(
                        author=author,
                        message=message,
                        timestamp=timestamp,
                    )
                except Exception as exc:
                    _stream_log.error(
                        "command_handler.handle_message error (author=%s): %s",
                        author, exc
                    )
                finally:
                    self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                _stream_log.error("_consume_queue unexpected error: %s", exc)

    async def _drain_queue(self) -> None:
        """Wait for the queue to empty (used during shutdown)."""
        await self._queue.join()
