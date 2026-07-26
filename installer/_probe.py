"""Живая проверка: настоящий вопрос настоящему персонажу через настоящий мост.

Запускается мастером установки на шаге проверки. Печатает только то, что
человеку полезно увидеть, — без внутренностей.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


async def main() -> int:
    try:
        import yaml
    except ImportError:
        print("!! библиотеки не поставлены — вернись на шаг 2")
        return 1

    cfg = yaml.safe_load((ROOT / "python" / "config.yaml").read_text(encoding="utf-8"))
    who = cfg["models"]["lore_agent"]["provider"]

    if who == "gemini":
        from providers.gemini_provider import GeminiProvider
        keys = GeminiProvider._collect_keys({})
        if not keys:
            print("!! ключей Gemini нет — вернись на шаг 3 и вставь ключ")
            return 1
        print(f"ключей найдено: {len(keys)}"
              + (" (будут чередоваться)" if len(keys) > 1 else ""))

    from agents.lore_agent import LoreAgent
    agent = LoreAgent(cfg)
    try:
        res = await agent.generate_response({
            "npc_id": "setup_probe", "npc_name": "Стражник", "npc_race": "Imperial",
            "npc_class": "Guard", "npc_faction": "", "location": "Сейда Нин",
            "npc_is_male": True, "npc_disposition": 45,
            "player_input": "Скажи, далеко ли до Балморы?",
            "conversation_history": [],
        }, memory_context=[])
    except Exception as exc:  # noqa: BLE001
        print(f"!! модель не ответила: {str(exc)[:160]}")
        return 1

    line = str(res.get("response") or "").strip()
    if not line:
        print("!! модель ответила пустотой")
        return 1
    print(f"персонаж ответил: {line[:150]}")

    # Голос — если движок озвучки поднимается, скажем об этом.
    engine = str(cfg["tts"]["engine"]).lower()
    if engine in ("morrowind", "mw"):
        voices = list((ROOT / "piper" / "morrowind").glob("*.onnx"))
        if voices:
            print(f"голоса из озвучки игры: {len(voices)} шт.")
        else:
            print("голосов из игры нет — NPC заговорят базовым голосом piper "
                  "(см. ГОЛОСА.md, если хочешь родные)")
    model = ROOT / "data" / "vosk" / "vosk-ru-0.42-fast"
    print("распознавание речи: " + ("готово" if (model / "am").is_dir()
                                    else "не установлено (можно играть текстом)"))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
