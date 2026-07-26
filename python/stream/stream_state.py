"""
stream_state.py — Manages the shared stream state file read by game mod and other agents.

Atomic writes via tmp+replace so concurrent readers never see a partial write.
"""

import json
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

_LOGS_DIR = '/home/nemoclaw/morrowind-ai/logs'
_IPC_DIR = '/home/nemoclaw/morrowind-ai/ipc'


class StreamState:
    STATE_FILE = '/home/nemoclaw/morrowind-ai/ipc/stream_state.json'

    def __init__(self):
        os.makedirs(_IPC_DIR, exist_ok=True)
        os.makedirs(_LOGS_DIR, exist_ok=True)

    def _load(self) -> dict:
        """Read current state from disk. Returns empty dict on any read error."""
        try:
            with open(self.STATE_FILE, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("stream_state: could not load %s: %s", self.STATE_FILE, exc)
            return {}

    def _save(self, state: dict) -> None:
        """Atomically write state to disk using tmp+replace."""
        try:
            dir_path = os.path.dirname(self.STATE_FILE)
            os.makedirs(dir_path, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=dir_path, prefix='.stream_state_')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                    json.dump(state, fh, indent=2)
                os.replace(tmp_path, self.STATE_FILE)
            except Exception:
                # Clean up temp file if replace failed
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as exc:
            logger.error("stream_state: could not write %s: %s", self.STATE_FILE, exc)

    def update(self, **kwargs) -> None:
        """Merge keyword arguments into the current state file."""
        state = self._load()
        state.update(kwargs)
        self._save(state)

    def get(self, key: str, default=None):
        """Read a single key from state. Returns default if missing."""
        return self._load().get(key, default)

    def set_video_id(self, video_id: str) -> None:
        """Record the active YouTube video/stream ID."""
        self.update(video_id=video_id)
        logger.info("stream_state: video_id set to %s", video_id)

    def set_game_state(self, state: str) -> None:
        """
        Record the current game context. Expected values:
        'combat', 'dialogue', 'exploration'
        """
        valid = {'combat', 'dialogue', 'exploration'}
        if state not in valid:
            logger.warning("stream_state: unknown game_state '%s' (expected one of %s)", state, valid)
        self.update(game_state=state)
        logger.info("stream_state: game_state set to %s", state)
