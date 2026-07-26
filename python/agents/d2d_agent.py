"""
d2d_agent.py — Radiant ambient NPC-to-NPC dialogue generator.

Ported from kenshi-ai's D2D (dialogue-to-dialogue) radiant system.
Given two nearby NPCs, generates a short 2–4 line ambient exchange that
makes the world feel alive without requiring player involvement.

Response format expected from LLM:
    SPEAKER_A: <one sentence>
    SPEAKER_B: <one sentence>
    SPEAKER_A: <one sentence>   [optional]
    SPEAKER_B: <one sentence>   [optional]

Returns:
    {"exchanges": [{"speaker_id": str, "speaker_name": str, "text": str}, ...]}
"""

import logging
import re
from typing import Any

from providers.factory import get_provider
from providers.base import log_llm_response
from .base_agent import call_with_retry
from .lore_agent import RACE_PERSONALITIES, FACTION_NOTES

logger = logging.getLogger(__name__)

_SPEAKER_RE = re.compile(r"^SPEAKER_([AB]):\s*(.+)$", re.MULTILINE)

D2D_SCHEMA = """\
Write a short ambient exchange (2–4 lines) between these two NPCs.
Format each line EXACTLY as:
    SPEAKER_A: <one sentence of dialogue>
    SPEAKER_B: <one sentence of dialogue>
Alternate A and B. No other text, no stage directions, no scene descriptions.
Keep each line to one sentence (10–20 words). Stay in character for each NPC's race and faction.
"""


def _build_d2d_prompt(
    npc_a_name: str, npc_a_race: str, npc_a_faction: str,
    npc_b_name: str, npc_b_race: str, npc_b_faction: str,
    location: str,
    campaign_id: str,
) -> tuple[str, str]:
    race_a = RACE_PERSONALITIES.get(npc_a_race, f"You are a {npc_a_race}.")
    race_b = RACE_PERSONALITIES.get(npc_b_race, f"You are a {npc_b_race}.")
    fac_a  = FACTION_NOTES.get(npc_a_faction, "")
    fac_b  = FACTION_NOTES.get(npc_b_faction, "")

    system = "\n".join([
        "You are a Morrowind world-building engine. Generate ambient NPC dialogue.",
        "ВСЕ реплики ТОЛЬКО на русском языке. (All dialogue lines MUST be in Russian.)",
        "Setting: Third Era, ~3E 427, province of Morrowind.",
        f"Location: {location}",
        "",
        f"NPC A — {npc_a_name} ({npc_a_race}, {npc_a_faction or 'no faction'})",
        race_a,
        fac_a,
        "",
        f"NPC B — {npc_b_name} ({npc_b_race}, {npc_b_faction or 'no faction'})",
        race_b,
        fac_b,
        "",
        D2D_SCHEMA,
    ])

    user = (
        f"Generate a brief ambient conversation between {npc_a_name} and {npc_b_name} "
        f"who have just encountered each other in {location}. "
        f"The exchange should feel natural and reveal something about their personalities or the world."
    )

    return system, user


def _parse_d2d_response(
    raw: str,
    npc_a_id: str, npc_a_name: str,
    npc_b_id: str, npc_b_name: str,
) -> list[dict]:
    exchanges = []
    for m in _SPEAKER_RE.finditer(raw):
        speaker = m.group(1)   # 'A' or 'B'
        text    = m.group(2).strip()
        if not text:
            continue
        if speaker == "A":
            exchanges.append({"speaker_id": npc_a_id, "speaker_name": npc_a_name, "text": text})
        else:
            exchanges.append({"speaker_id": npc_b_id, "speaker_name": npc_b_name, "text": text})
    return exchanges


class D2DAgent:
    """Generates ambient NPC-to-NPC dialogue for the radiant dialogue system."""

    def __init__(self, config: dict[str, Any]) -> None:
        provider_cfg: dict = config.get("models", {}).get(
            "d2d_agent", {"provider": "gemini", "model": "gemini-3.1-flash-lite-preview"}
        )
        self.llm = get_provider(provider_cfg)
        self._temperature: float = provider_cfg.get("temperature", 0.9)
        self._max_tokens: int = config.get("max_output_tokens", 200)
        logger.info(
            "D2DAgent initialised: provider=%s model=%s",
            provider_cfg.get("provider"),
            provider_cfg.get("model"),
        )

    async def generate(self, req: dict[str, Any]) -> dict[str, Any]:
        """
        Generate ambient dialogue for two NPCs.

        Args:
            req: dict with npc_a_id, npc_a_name, npc_a_race, npc_a_faction,
                          npc_b_id, npc_b_name, npc_b_race, npc_b_faction,
                          location, campaign_id (all strings)
        Returns:
            {"req_id": str, "exchanges": [{"speaker_id", "speaker_name", "text"}, ...]}
        """
        npc_a_id      = req.get("npc_a_id", "npc_a")
        npc_a_name    = req.get("npc_a_name", "Stranger")
        npc_a_race    = req.get("npc_a_race", "Dunmer")
        npc_a_faction = req.get("npc_a_faction", "")
        npc_b_id      = req.get("npc_b_id", "npc_b")
        npc_b_name    = req.get("npc_b_name", "Traveller")
        npc_b_race    = req.get("npc_b_race", "Dunmer")
        npc_b_faction = req.get("npc_b_faction", "")
        location      = req.get("location", "Vvardenfell")
        req_id        = req.get("req_id", "d2d-unknown")

        system, user = _build_d2d_prompt(
            npc_a_name, npc_a_race, npc_a_faction,
            npc_b_name, npc_b_race, npc_b_faction,
            location, req.get("campaign_id", ""),
        )

        resp = await call_with_retry(
            lambda: self.llm.complete(
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        )

        log_llm_response("D2DAgent", resp)
        exchanges = _parse_d2d_response(resp.text, npc_a_id, npc_a_name, npc_b_id, npc_b_name)

        logger.info(
            "D2DAgent | %s <> %s | %d lines | tokens=%d | cost=$%.5f",
            npc_a_name, npc_b_name, len(exchanges), resp.tokens_in + resp.tokens_out, resp.cost_usd,
        )

        return {"req_id": req_id, "exchanges": exchanges}
