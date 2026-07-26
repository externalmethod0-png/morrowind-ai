"""
screen_grab.py — on-demand screenshot for the Narrator's eyes.

The upstream pixel_agent captures every 3 seconds in the background, which
would burn the free-tier quota for nothing. The narrator only needs to LOOK
when the player actually asks (U key), so we grab a single downscaled frame
per request and hand it to the vision-capable provider.
"""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)

TARGET_W = 1024   # enough detail for "what is in this room", cheap in tokens


def grab_screen_png() -> bytes | None:
    """One downscaled PNG of the primary monitor, or None if unavailable."""
    try:
        import mss
        from PIL import Image

        with mss.mss() as sct:
            mon = sct.monitors[1]           # primary monitor (OpenMW borderless)
            raw = sct.grab(mon)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        if img.width > TARGET_W:
            h = max(1, int(img.height * TARGET_W / img.width))
            img = img.resize((TARGET_W, h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        logger.info("screen grab: %dx%d, %d KB", img.width, img.height, len(data) // 1024)
        return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("screen grab failed (%s) — narrator falls back to text", exc)
        return None
