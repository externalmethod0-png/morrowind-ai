"""
local_provider.py — локальная модель вместо облака.

Говорит по OpenAI-совместимому протоколу, поэтому подходит сразу к нескольким
запускалкам, которые его отдают:

    LM Studio   http://localhost:1234/v1   (модель выбирается в самом LM Studio)
    Ollama      http://localhost:11434/v1
    llama.cpp   http://localhost:8080/v1

Зачем: снимает зависимость от квот и интернета, а разговоры с NPC перестают
стоить денег. Расплата — качество слабее облачной модели и борьба за
видеокарту с озвучкой.

Главное здесь — ЗАПАСНОЙ ВАРИАНТ. Локальный сервер легко забыть запустить или
он падает вместе с моделью; без подстраховки NPC просто замолчат, и это будет
выглядеть поломкой мода. Поэтому при недоступности локальной модели провайдер
молча уходит на облачный, если тот настроен.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request

from .base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:1234/v1"


class LocalProvider(LLMProvider):
    supports_stream = True

    """
    Config keys:
        base_url   — адрес сервера (по умолчанию LM Studio)
        model      — имя модели; пусто = какая загружена в сервере
        timeout    — сколько ждать ответа, секунд (по умолчанию 90)
        fallback   — конфиг запасного провайдера, например
                     {"provider": "gemini", "model": "gemini-flash-lite-latest"}
    """

    def __init__(self, cfg: dict) -> None:
        self.base_url: str = str(cfg.get("base_url") or DEFAULT_URL).rstrip("/")
        self.model_name: str = str(cfg.get("model") or "")
        self.timeout: float = float(cfg.get("timeout") or 90)
        # НОМЕР СЛОТА у llama-server. Слот — это отдельная ячейка памяти, где
        # сервер держит разбор промпта. Записок у мода две (разговор с NPC и
        # сценка между NPC), и общего начала у них НОЛЬ знаков — поэтому на
        # одном слоте каждая сценка вытирала разбор разговорной записки, и
        # следующая реплика игрока читала правила заново.
        #
        # Замерено на гигачате, шесть реплик с разными NPC и сценкой между:
        #     один слот   13.4 с на реплику
        #     два слота    2.7 с
        # Разница пятикратная и никакими другими настройками не берётся.
        # LM Studio так не умеет — нужен llama-server с -np 2.
        slot = cfg.get("slot")
        self.slot: int | None = int(slot) if slot is not None else None
        self._fallback_cfg: dict | None = cfg.get("fallback") or None
        self._fallback: LLMProvider | None = None
        self._local_dead_until: float = 0.0

        if not self.model_name:
            self.model_name = self._first_available_model() or "local"
        logger.info("LocalProvider: %s, модель=%s%s%s", self.base_url, self.model_name,
                    f", слот {self.slot}" if self.slot is not None else "",
                    ", запасной: " + str((self._fallback_cfg or {}).get("provider"))
                    if self._fallback_cfg else ", без запасного")

    # ------------------------------------------------------------------ сеть

    def _get_json(self, path: str, timeout: float = 5.0):
        req = urllib.request.Request(self.base_url + path, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))

    def _first_available_model(self) -> str | None:
        try:
            data = self._get_json("/models")
            items = data.get("data") or []
            if items:
                return str(items[0].get("id") or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("локальный сервер не отвечает на /models: %s", exc)
        return None

    def is_up(self) -> bool:
        try:
            self._get_json("/models", timeout=3.0)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _post_chat(self, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer local"},   # LM Studio ключ не проверяет
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))

    # --------------------------------------------------------------- запасной

    def _get_fallback(self) -> LLMProvider | None:
        if self._fallback is None and self._fallback_cfg:
            try:
                from .factory import get_provider
                self._fallback = get_provider(self._fallback_cfg)
                logger.info("запасной провайдер поднят: %s",
                            self._fallback_cfg.get("provider"))
            except Exception as exc:  # noqa: BLE001
                logger.error("запасной провайдер недоступен: %s", exc)
                self._fallback_cfg = None
        return self._fallback

    # ---------------------------------------------------------------- запрос

    async def complete(
        self,
        system: str,
        messages: list[dict],
        image_bytes: bytes | None = None,
        **kwargs,
    ) -> LLMResponse:
        loop = asyncio.get_running_loop()
        now = loop.time()

        # Пока локальный сервер молчит, не долбимся в него каждой репликой —
        # иначе каждый ответ ждал бы таймаута.
        if now >= self._local_dead_until:
            try:
                return await asyncio.to_thread(self._complete_sync, system, messages, kwargs)
            except Exception as exc:  # noqa: BLE001
                self._local_dead_until = now + 60.0
                logger.warning("локальная модель не ответила (%s) — молчит минуту", exc)

        fb = self._get_fallback()
        if fb is None:
            raise RuntimeError(
                f"локальная модель по {self.base_url} недоступна, запасной не настроен")
        logger.info("отвечает запасной провайдер")
        return await fb.complete(system=system, messages=messages,
                                 image_bytes=image_bytes, **kwargs)

    async def complete_stream(
        self,
        system: str,
        messages: list[dict],
        on_text=None,
        image_bytes: bytes | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Отдаёт реплику по мере набора.

        Локальная модель выдаёт ~30 токенов в секунду, то есть полный ответ
        набирается секунд семь. Ждать его целиком — значит семь секунд смотреть
        на молчащего NPC. Наш формат для этого удачен: сама реплика идёт ПЕРВОЙ,
        а теги действий в конце, поэтому текст можно показывать и озвучивать
        сразу, а мир менять уже по готовым тегам.
        """
        loop = asyncio.get_running_loop()
        now = loop.time()
        if now >= self._local_dead_until:
            try:
                return await asyncio.to_thread(
                    self._stream_sync, system, messages, kwargs, on_text, loop)
            except Exception as exc:  # noqa: BLE001
                self._local_dead_until = now + 60.0
                logger.warning("поток от локальной модели оборвался (%s)", exc)

        fb = self._get_fallback()
        if fb is None:
            raise RuntimeError(
                f"локальная модель по {self.base_url} недоступна, запасной не настроен")
        return await fb.complete_stream(system=system, messages=messages,
                                        on_text=on_text, image_bytes=image_bytes,
                                        **kwargs)

    def _stream_sync(self, system: str, messages: list[dict], kwargs: dict,
                     on_text, loop) -> LLMResponse:
        chat = [{"role": "system", "content": system}]
        for m in messages:
            role = m.get("role", "user")
            chat.append({"role": "assistant" if role == "assistant" else "user",
                         "content": str(m.get("content", ""))})
        payload = {
            "model": self.model_name, "messages": chat,
            "temperature": float(kwargs.get("temperature", 0.8)),
            "max_tokens": int(kwargs.get("max_tokens", 400)),
            # Штраф за повторное употребление: домашняя модель открывала
            # разные реплики одним и тем же зачином («Я не слежу за…» четыре
            # раза из четырёх). Температура лечит это ценой формата, а штраф —
            # нет, он давит только повтор.
            "presence_penalty": float(kwargs.get("presence_penalty", 0.0)),
            "frequency_penalty": float(kwargs.get("frequency_penalty", 0.0)),
            **({"id_slot": self.slot} if self.slot is not None else {}),
            "stream": True,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer local"})

        chunks: list[str] = []
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    piece = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for ch in piece.get("choices") or []:
                    delta = (ch.get("delta") or {}).get("content")
                    if delta:
                        chunks.append(delta)
                        if on_text is not None:
                            # Callback уходит в цикл событий: он публикует
                            # промежуточную реплику в игру, а это не потокобезопасно.
                            loop.call_soon_threadsafe(on_text, "".join(chunks))

        text = "".join(chunks).strip()
        return LLMResponse(text=text, tokens_in=0, tokens_out=0, cost_usd=0.0,
                           model=self.model_name, provider="local")

    def _complete_sync(self, system: str, messages: list[dict], kwargs: dict) -> LLMResponse:
        chat = [{"role": "system", "content": system}]
        for m in messages:
            role = m.get("role", "user")
            chat.append({"role": "assistant" if role == "assistant" else "user",
                         "content": str(m.get("content", ""))})

        payload = {
            "model": self.model_name,
            "messages": chat,
            "temperature": float(kwargs.get("temperature", 0.8)),
            "presence_penalty": float(kwargs.get("presence_penalty", 0.0)),
            "frequency_penalty": float(kwargs.get("frequency_penalty", 0.0)),
            **({"id_slot": self.slot} if self.slot is not None else {}),
            "max_tokens": int(kwargs.get("max_tokens", 400)),
            "stream": False,
        }
        data = self._post_chat(payload)
        choices = data.get("choices") or []
        text = ""
        if choices:
            text = str((choices[0].get("message") or {}).get("content") or "")
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text.strip(),
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            cost_usd=0.0,                     # своё железо — платы нет
            model=self.model_name,
            provider="local",
        )
