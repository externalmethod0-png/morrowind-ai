"""
launcher.py — окно настроек и запуска morrowind-ai.

Всё, что раньше приходилось править в двух конфигах руками: правила мира для
отыгрыша, выбор модели (облако или своя), движок озвучки, микрофон. Плюс
кнопка запуска игры и проверка, что всё на месте.

Написано на tkinter — он идёт в комплекте с Python, ставить нечего.

Запуск:  venv\\Scripts\\python.exe tools\\launcher.py
или двойным кликом по «НАСТРОЙКИ И ЗАПУСК.bat».
"""

from __future__ import annotations

import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "python" / "config.yaml"
RULES = ROOT / "data" / "world_rules.txt"
LAUNCH_BAT = ROOT / "Morrowind AI (запуск).bat"

sys.path.insert(0, str(ROOT / "python"))

TTS_ENGINES = [
    ("morrowind — голоса из родной озвучки игры, быстрые (рекомендуется)", "morrowind"),
    ("piper — базовые русские голоса, тоже быстрые", "piper"),
    ("xtts — голоса из родной озвучки игры, нужна видеокарта", "xtts"),
    ("silero — запасной, офлайн", "silero"),
    ("edge — нейросетевые голоса, нужен интернет", "edge"),
]

PROVIDERS = [
    ("Облако Gemini — лучше держит характер и действия", "gemini"),
    ("Своя модель (LM Studio / Ollama) — без квот и интернета", "local"),
]

# Две раскладки железа под этот ПК. Разница ровно в одном: кому достаётся
# ускоритель NVIDIA. Всё остальное совпадает, потому что уже измерено:
# отрисовка игры идёт на видеоядре AMD (монитор подключён к нему), а синтез
# речи дообученными голосами — на процессоре за 0.2 с и карта ему не нужна.
# Раскладка железа. Распознавание речи с переходом на Vosk считает ВСЕГДА на
# процессоре — видеокарта ему не нужна вовсе, и прежний спор за неё отпал сам
# собой. На ускоритель претендуют только двое: своя модель и озвучка XTTS.
PROFILES = {
    "free": {
        "stt": "cpu",
        "text": ("Игра — видеоядро AMD · Распознавание — процессор (0.3 с) · "
                 "Синтез — процессор (0.1-0.2 с)\n"
                 "Ускоритель СВОБОДЕН: и распознавание, и дообученные голоса "
                 "считают на процессоре. Самая быстрая связка."),
    },
    "local": {
        "stt": "cpu",
        "text": ("Игра — видеоядро AMD · Своя модель — NVIDIA · "
                 "Распознавание и синтез — процессор\n"
                 "Ускоритель целиком под свою модель, и делить его не с кем."),
    },
    "xtts": {
        "stt": "cpu",
        "text": ("Игра — видеоядро AMD · Озвучка XTTS — NVIDIA · "
                 "Распознавание — процессор\n"
                 "Голоса клонируются с родной озвучки на ходу: первая фраза "
                 "готова за 4-5 с, паузу закрывает заранее нарисованная "
                 "заминка тем же голосом."),
    },
    "xtts+local": {
        "stt": "cpu",
        "text": ("Игра — видеоядро AMD · Своя модель И озвучка XTTS — обе на "
                 "NVIDIA · Распознавание — процессор\n"
                 "ТЕСНО: 8 ГБ на двоих, будут душить друг друга. Лучше выбрать "
                 "что-то одно."),
    },
}


def profile_for(provider: str, engine: str) -> dict:
    """Кто занимает ускоритель — тот и определяет раскладку.

    Распознавание в расчёте не участвует: Vosk всегда на процессоре.
    """
    local = provider == "local"
    xtts = engine == "xtts"
    if local and xtts:
        return PROFILES["xtts+local"]
    if xtts:
        return PROFILES["xtts"]
    return PROFILES["local"] if local else PROFILES["free"]



def load_yaml() -> dict:
    import yaml
    with CONFIG.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class Launcher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Morrowind AI — настройки")
        self.geometry("880x680")
        self.minsize(760, 560)

        try:
            self.cfg = load_yaml()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Конфиг не читается", str(exc))
            self.cfg = {}

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=(10, 0))
        self._tab_rules(nb)
        self._tab_model(nb)
        self._tab_voice(nb)
        self._tab_check(nb)
        # Раскладку применяем ПОСЛЕ сборки всех вкладок: поле «где считать
        # распознавание» живёт на вкладке голоса и к моменту сборки модели
        # ещё не существует.
        self._toggle_local()

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=10, pady=10)
        ttk.Button(bar, text="Сохранить", command=self.save).pack(side="left")
        ttk.Button(bar, text="Сохранить и запустить игру",
                   command=self.save_and_play).pack(side="left", padx=8)
        self.status = ttk.Label(bar, text="")
        self.status.pack(side="right")

    # ------------------------------------------------------------ правила мира

    def _tab_rules(self, nb: ttk.Notebook) -> None:
        f = ttk.Frame(nb)
        nb.add(f, text="Мир и отыгрыш")
        ttk.Label(f, wraplength=820, justify="left", text=(
            "Эти правила попадают в голову КАЖДОМУ NPC как указание высшего "
            "приоритета: тон мира, кто такой игрок, чего не упоминать. Строки с "
            "решёткой — пояснения для тебя, модель их не видит.\n"
            "Правки подхватываются на лету: сохранил — следующая реплика уже с ними."
        )).pack(anchor="w", padx=10, pady=(10, 6))

        self.rules_box = scrolledtext.ScrolledText(f, wrap="word", height=22,
                                                   font=("Consolas", 10))
        self.rules_box.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        try:
            self.rules_box.insert("1.0", RULES.read_text(encoding="utf-8"))
        except OSError:
            pass

        row = ttk.Frame(f)
        row.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(row, text="Готовые заготовки:").pack(side="left")
        for name, text in PRESETS:
            ttk.Button(row, text=name,
                       command=lambda t=text: self._add_preset(t)).pack(side="left", padx=4)

    def _add_preset(self, text: str) -> None:
        cur = self.rules_box.get("1.0", "end").rstrip()
        self.rules_box.delete("1.0", "end")
        self.rules_box.insert("1.0", cur + "\n\n" + text.strip() + "\n")
        self.rules_box.see("end")

    # ----------------------------------------------------------------- модель

    def _tab_model(self, nb: ttk.Notebook) -> None:
        f = ttk.Frame(nb)
        nb.add(f, text="Модель")
        lore = (self.cfg.get("models") or {}).get("lore_agent") or {}

        self.provider = tk.StringVar(value=str(lore.get("provider") or "gemini"))
        ttk.Label(f, text="Кто отвечает за NPC:", font=("", 10, "bold")).pack(
            anchor="w", padx=10, pady=(12, 4))
        for label, value in PROVIDERS:
            ttk.Radiobutton(f, text=label, value=value,
                            variable=self.provider,
                            command=self._toggle_local).pack(anchor="w", padx=24)

        self.local_frame = ttk.LabelFrame(f, text="Своя модель")
        self.local_frame.pack(fill="x", padx=10, pady=12)
        self.base_url = self._field(self.local_frame, "Адрес сервера",
                                    str(lore.get("base_url") or "http://localhost:1234/v1"))
        self.local_model = self._field(self.local_frame, "Имя модели (пусто = загруженная)",
                                       str(lore.get("model") if lore.get("provider") == "local"
                                           else "t-lite-it-2.1"))
        ttk.Label(self.local_frame, wraplength=780, justify="left", foreground="#555",
                  text=("В LM Studio: вкладка Developer → Start Server. Если сервер "
                        "не поднят, ответы молча пойдут через облако — NPC не замолчат.\n"
                        "Замер трёх установленных моделей (tests/bench_local_models.py, "
                        "семь сцен с известным правильным ответом):\n"
                        "  t-lite-it-2.1 — 5/7, ~9 с на реплику. Лучший русский и память "
                        "сцены, формат тегов безупречен. НО сговорчивых действий не даёт: "
                        "нанять спутника, велеть ждать и торговать через него нельзя.\n"
                        "  gigachat3.1-10b-a1.8b — 2/7, ~7.5 с. Быстрее всех, но половина "
                        "ответов вообще без реплики и с английскими вставками.\n"
                        "  gemma-4-e4b-uncensored — 0/7, 20–46 с. «Думающая» модель: весь "
                        "бюджет уходит в скрытые рассуждения, на ответ не остаётся ничего.\n"
                        "Партийная часть мода (спутники, интриги) пока живёт только на "
                        "облаке.")).pack(anchor="w", padx=10, pady=(0, 8))

        self.cloud_model = self._field(f, "Модель облака",
                                       str(lore.get("model") if lore.get("provider") != "local"
                                           else "gemini-flash-lite-latest"))

        box = ttk.LabelFrame(f, text="Раскладка железа при этом выборе")
        box.pack(fill="x", padx=10, pady=(14, 6))
        self.layout_label = ttk.Label(box, wraplength=800, justify="left")
        self.layout_label.pack(anchor="w", padx=10, pady=8)

        self._toggle_local()

    def _field(self, parent, label: str, value: str) -> tk.StringVar:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=10, pady=4)
        ttk.Label(row, text=label, width=34).pack(side="left")
        var = tk.StringVar(value=value)
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
        return var

    def _toggle_local(self) -> None:
        who = self.provider.get()
        state = "normal" if who == "local" else "disabled"
        for child in self.local_frame.winfo_children():
            for w in (child,) + tuple(getattr(child, "winfo_children", lambda: ())()):
                try:
                    w.configure(state=state)
                except tk.TclError:
                    pass

        # Выбор модели решает, кому достаётся ускоритель. Держать эти две
        # настройки порознь — верный способ однажды посадить распознавание и
        # свою модель на одну карту и не понять, почему всё встало.
        engine = self.tts_engine.get() if hasattr(self, "tts_engine") else ""
        profile = profile_for(who, engine)
        if hasattr(self, "layout_label"):
            self.layout_label.config(text=profile["text"])
        if hasattr(self, "stt_device"):
            self.stt_device.set(profile["stt"])

    # ------------------------------------------------------------------ голос

    def _tab_voice(self, nb: ttk.Notebook) -> None:
        f = ttk.Frame(nb)
        nb.add(f, text="Голос")
        tts = self.cfg.get("tts") or {}
        voice = self.cfg.get("voice") or {}

        self.tts_on = tk.BooleanVar(value=bool(tts.get("enabled", True)))
        ttk.Checkbutton(f, text="NPC говорят голосом", variable=self.tts_on).pack(
            anchor="w", padx=10, pady=(12, 6))

        self.tts_engine = tk.StringVar(value=str(tts.get("engine") or "piper"))
        ttk.Label(f, text="Чем озвучивать:", font=("", 10, "bold")).pack(
            anchor="w", padx=10, pady=(6, 4))
        for label, value in TTS_ENGINES:
            # Выбор озвучки меняет раскладку железа: XTTS забирает ускоритель,
            # и распознавание обязано уйти на процессор.
            ttk.Radiobutton(f, text=label, value=value,
                            variable=self.tts_engine,
                            command=self._toggle_local).pack(anchor="w", padx=24)

        self.voice_on = tk.BooleanVar(value=bool(voice.get("enabled", True)))
        ttk.Checkbutton(f, text="Голосовой ввод (клавиша V)",
                        variable=self.voice_on).pack(anchor="w", padx=10, pady=(14, 4))
        self.mic = self._field(f, "Микрофон (часть названия)",
                               str(voice.get("device") or ""))
        self.stt_device = tk.StringVar(value=str(voice.get("compute_device") or "cpu"))
        row = ttk.Frame(f)
        row.pack(fill="x", padx=10, pady=4)
        ttk.Label(row, text="Распознавание считать на", width=34).pack(side="left")
        ttk.Combobox(row, textvariable=self.stt_device, values=["cpu", "cuda"],
                     state="readonly", width=10).pack(side="left")
        ttk.Label(f, wraplength=800, justify="left", foreground="#555",
                  text=("На этой видеокарте распознавание давало от 6 до 71 секунды, "
                        "на процессоре — стабильные 3 секунды. Менять на cuda есть "
                        "смысл только с другой картой.")).pack(anchor="w", padx=10, pady=6)

    # --------------------------------------------------------------- проверка

    def _tab_check(self, nb: ttk.Notebook) -> None:
        f = ttk.Frame(nb)
        nb.add(f, text="Проверка")
        ttk.Button(f, text="Проверить, всё ли на месте",
                   command=self.run_checks).pack(anchor="w", padx=10, pady=10)
        self.check_box = scrolledtext.ScrolledText(f, wrap="word", height=24,
                                                   font=("Consolas", 9))
        self.check_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def run_checks(self) -> None:
        self.check_box.delete("1.0", "end")
        self.check_box.insert("end", "Проверяю...\n")
        threading.Thread(target=self._checks_worker, daemon=True).start()

    def _checks_worker(self) -> None:
        lines = []

        def ok(cond, good, bad):
            lines.append(("  [да] " if cond else "  [НЕТ] ") + (good if cond else bad))

        inbox = ROOT / "openmw-mod" / "ai_inbox"
        slot = inbox / "response.txt"
        ok(slot.exists(), f"файл ответов на месте", "нет файла ответов — запусти мост")
        if slot.exists():
            ok(slot.stat().st_size == 16384,
               "размер файла ответов верный",
               f"размер {slot.stat().st_size} вместо 16384 — игра не увидит ответы")
        ok(CONFIG.exists(), "конфиг на месте", "конфиг не найден")
        ok((ROOT / "venv" / "Scripts" / "python.exe").exists(),
           "окружение моста на месте", "нет окружения моста")

        try:
            from providers.gemini_provider import GeminiProvider
            n = len(GeminiProvider._collect_keys({}))
            ok(n > 0, f"ключей облака: {n}", "нет ключей облака")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"  [НЕТ] ключи облака: {exc}")

        if self.provider.get() == "local":
            try:
                from providers.local_provider import LocalProvider
                p = LocalProvider({"base_url": self.base_url.get(), "model": "x"})
                ok(p.is_up(), "своя модель отвечает",
                   "своя модель не отвечает — запусти сервер в LM Studio "
                   "(ответы пойдут через облако)")
            except Exception as exc:  # noqa: BLE001
                lines.append(f"  [НЕТ] своя модель: {exc}")

        rules = RULES.read_text(encoding="utf-8") if RULES.exists() else ""
        live = "\n".join(l for l in rules.splitlines()
                         if not l.lstrip().startswith("#")).strip()
        lines.append(f"  [да] правил мира: {len(live)} символов"
                     if live else "  [--] правила мира пустые (это нормально)")

        voices = ROOT / "piper_train_env" / "runs"
        if voices.exists():
            done = [p.name for p in voices.iterdir()
                    if p.is_dir() and list(p.rglob("*.ckpt"))]
            lines.append(f"  [да] обученных голосов: {', '.join(done) or 'нет'}")

        self.check_box.after(0, lambda: (
            self.check_box.delete("1.0", "end"),
            self.check_box.insert("end", "\n".join(lines) + "\n")))

    # -------------------------------------------------------------- сохранение

    def save(self) -> bool:
        try:
            RULES.write_text(self.rules_box.get("1.0", "end").rstrip() + "\n",
                             encoding="utf-8")
            self._patch_config()
            self.status.config(text="Сохранено")
            return True
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Не смог сохранить", str(exc))
            return False

    def _patch_config(self) -> None:
        """Правим конфиг ТЕКСТОМ, а не через yaml.dump: иначе слетят все
        комментарии, а в них половина знаний о том, почему что настроено так."""
        text = CONFIG.read_text(encoding="utf-8")
        text = _set_scalar(text, "tts", "engine", self.tts_engine.get())
        text = _set_scalar(text, "tts", "enabled", "true" if self.tts_on.get() else "false")
        text = _set_scalar(text, "voice", "enabled", "true" if self.voice_on.get() else "false")
        text = _set_scalar(text, "voice", "device", self.mic.get())
        text = _set_scalar(text, "voice", "compute_device", self.stt_device.get())
        text = _set_lore_agent(text, self.provider.get(), self.cloud_model.get(),
                               self.base_url.get(), self.local_model.get())
        CONFIG.write_text(text, encoding="utf-8")

    def save_and_play(self) -> None:
        if not self.save():
            return
        try:
            subprocess.Popen(["cmd", "/c", str(LAUNCH_BAT)], cwd=str(ROOT))
            self.status.config(text="Запускаю игру...")
            self.after(2500, self.destroy)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Не смог запустить", str(exc))


PRESETS = [
    ("Суровый мир",
     "Мир жёсткий и небогатый. Люди считают каждую монету, чужаку не доверяют,\n"
     "помощь просто так не оказывают. Вежливость — расчёт, а не доброта."),
    ("Без подсказок",
     "Не подсказывай игроку, куда идти и что делать, если он не спросил прямо.\n"
     "Никаких современных оборотов. Ругательства — только из мира игры."),
    ("Тайна пророчества",
     "Не упоминай Нереварина и пророчества, пока игрок сам не заговорит об этом."),
]


def _set_scalar(text: str, section: str, key: str, value: str) -> str:
    """Заменить `key: ...` внутри секции верхнего уровня, сохранив комментарии."""
    lines = text.splitlines()
    in_section = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not line.startswith((" ", "\t")) and stripped.endswith(":"):
            in_section = stripped[:-1] == section
            continue
        if in_section and stripped.startswith(key + ":"):
            indent = line[: len(line) - len(line.lstrip())]
            tail = ""
            if "#" in line:
                tail = "   " + line[line.index("#"):]
            shown = value if value != "" else '""'
            lines[i] = f"{indent}{key}: {shown}{tail}"
            break
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _set_lore_agent(text: str, provider: str, cloud_model: str,
                    base_url: str, local_model: str) -> str:
    """Переписать блок lore_agent целиком — у облака и своей модели разный
    набор полей, поточечная правка тут только запутает."""
    lines = text.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        if line.strip() == "lore_agent:" and line.startswith("  "):
            start = i
            continue
        if start is not None and i > start:
            if line.strip() and not line.startswith("    "):
                end = i
                break
    if start is None:
        return text
    end = end if end is not None else len(lines)

    if provider == "local":
        block = [
            "  lore_agent:",
            "    provider: local",
            f"    base_url: {base_url or 'http://localhost:1234/v1'}",
            f'    model: "{local_model}"',
            "    temperature: 0.8",
            "    timeout: 90",
            "    fallback:              # сервер не поднят — отвечает облако",
            "      provider: gemini",
            "      model: gemini-flash-lite-latest",
        ]
    else:
        block = [
            "  lore_agent:",
            "    provider: gemini",
            f"    model: {cloud_model or 'gemini-flash-lite-latest'}",
            "    temperature: 0.8",
        ]
    return "\n".join(lines[:start] + block + lines[end:]) + "\n"


if __name__ == "__main__":
    Launcher().mainloop()
