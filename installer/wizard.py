"""
wizard.py — мастер установки Morrowind AI.

Ведёт человека за руку от «скачал архив» до «играю»: находит игру, ставит
зависимости, качает голоса и распознавание речи, принимает ключи, помогает
задать характер мира и в конце САМ проверяет, что всё работает.

Ничего не требует знать заранее. Всё, что может определить сам, — определяет.

Запуск: УСТАНОВКА.bat (он сам найдёт Python и позовёт этот файл).
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import urllib.request
import zipfile
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / "venv"
VENV_PY = VENV / "Scripts" / "python.exe"
CONFIG = ROOT / "python" / "config.yaml"
RULES = ROOT / "data" / "world_rules.txt"
TUNING = ROOT / "data" / "настройки-мира.txt"
KEYS_FILE = Path.home() / ".nemoclaw_env"

VOSK_URL = "https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip"
VOSK_DIR = ROOT / "data" / "vosk"
VOICES_DIR = ROOT / "piper" / "morrowind"

PIP_PACKAGES = [
    "pyyaml", "google-genai", "chromadb", "pygame", "sounddevice",
    "numpy", "vosk", "psutil", "requests",
]

BG = "#1e1e1e"
FG = "#e8e8e8"
ACCENT = "#d8b56a"


# ── поиск игры ──────────────────────────────────────────────────────────────

def guess_openmw() -> tuple[Path | None, Path | None]:
    """(папка с openmw.exe, папка конфигов). Ищем в обычных местах.

    Папку зовут по-разному — «OpenMW», «Morrowind (ReBuild)», «Игры\\Морровинд».
    Поэтому не перебираем точные имена, а смотрим корни дисков и папки, в
    названии которых есть openmw или morrowind, и уже внутри ищем сам exe.
    """
    candidates: list[Path] = [
        Path(r"C:\Program Files\OpenMW"),
        Path(r"C:\Program Files (x86)\OpenMW"),
        Path.home() / "Documents" / "My Games" / "OpenMW",
    ]
    for letter in "CDEFGH":
        root = Path(f"{letter}:\\")
        if not root.is_dir():
            continue
        try:
            for entry in root.iterdir():
                if not entry.is_dir():
                    continue
                low = entry.name.lower()
                if "openmw" in low or "morrowind" in low:
                    candidates.append(entry)
                elif low in ("games", "игры", "program files", "steamlibrary"):
                    # На уровень глубже: игры часто лежат внутри сборников.
                    try:
                        for sub in entry.iterdir():
                            s = sub.name.lower()
                            if sub.is_dir() and ("openmw" in s or "morrowind" in s):
                                candidates.append(sub)
                    except OSError:
                        pass
        except OSError:
            pass

    exe = None
    for base in candidates:
        if not base.is_dir():
            continue
        try:
            for p in base.rglob("openmw.exe"):
                exe = p.parent
                break
        except OSError:
            continue
        if exe:
            break
    cfg = None
    for base in [Path.home() / "Documents" / "My Games" / "OpenMW",
                 exe.parent if exe else None]:
        if base and (base / "openmw.cfg").exists():
            cfg = base
            break
    return exe, cfg


def openmw_version(exe_dir: Path) -> str:
    try:
        res = subprocess.run([str(exe_dir / "openmw.exe"), "--version"],
                             capture_output=True, text=True, timeout=25,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        return (res.stdout or res.stderr or "").strip().splitlines()[0][:60]
    except Exception:  # noqa: BLE001
        return ""


class Wizard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Morrowind AI — установка")
        self.geometry("900x680")
        self.minsize(820, 600)
        self.configure(bg=BG)
        self.log_q: "queue.Queue[str]" = queue.Queue()

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8))
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("TCheckbutton", background=BG, foreground=FG)
        style.configure("TRadiobutton", background=BG, foreground=FG)
        style.configure("TLabelframe", background=BG, foreground=ACCENT)
        style.configure("TLabelframe.Label", background=BG, foreground=ACCENT)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=12, pady=(12, 0))
        self._page_game()
        self._page_install()
        self._page_brain()
        self._page_world()
        self._page_check()

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=12, pady=10)
        self.status = ttk.Label(bar, text="Шаг 1 из 5 — начнём с поиска игры")
        self.status.pack(side="left")
        ttk.Button(bar, text="Дальше →", command=self._next).pack(side="right")

        self.after(150, self._drain_log)

    # ── помощники ───────────────────────────────────────────────────────

    def say(self, text: str) -> None:
        self.log_q.put(text)

    def _drain_log(self) -> None:
        while True:
            try:
                line = self.log_q.get_nowait()
            except queue.Empty:
                break
            for box in (getattr(self, "install_log", None),
                        getattr(self, "check_log", None)):
                if box is not None:
                    box.configure(state="normal")
                    box.insert("end", line + "\n")
                    box.see("end")
                    box.configure(state="disabled")
        self.after(150, self._drain_log)

    def _next(self) -> None:
        i = self.nb.index(self.nb.select())
        if i < self.nb.index("end") - 1:
            self.nb.select(i + 1)
            self.status.config(text=f"Шаг {i + 2} из 5")

    def _hint(self, parent, text: str) -> None:
        ttk.Label(parent, text=text, wraplength=840, justify="left",
                  foreground="#9a9a9a").pack(anchor="w", padx=14, pady=(0, 8))

    # ── 1. игра ─────────────────────────────────────────────────────────

    def _page_game(self) -> None:
        f = ttk.Frame(self.nb)
        self.nb.add(f, text="1. Игра")
        ttk.Label(f, text="Где стоит игра", font=("", 13, "bold"),
                  foreground=ACCENT).pack(anchor="w", padx=14, pady=(14, 4))
        self._hint(f, "Мод работает с OpenMW — это бесплатный движок, который "
                      "запускает обычную Morrowind. Оригинальный лаунчер игры "
                      "(Morrowind.exe) не подойдёт. Если OpenMW ещё не стоит, "
                      "скачай его с openmw.org, поставь, один раз запусти игру "
                      "и вернись сюда.")

        row = ttk.Frame(f)
        row.pack(fill="x", padx=14)
        self.exe_var = tk.StringVar(value="")
        ttk.Entry(row, textvariable=self.exe_var, width=78).pack(side="left")
        ttk.Button(row, text="Выбрать папку…",
                   command=self._pick_exe).pack(side="left", padx=6)
        ttk.Button(row, text="Найти самому",
                   command=self._autodetect).pack(side="left")

        self.game_info = ttk.Label(f, text="", wraplength=840, justify="left")
        self.game_info.pack(anchor="w", padx=14, pady=10)
        self.after(400, self._autodetect)

    def _pick_exe(self) -> None:
        d = filedialog.askdirectory(title="Папка, где лежит openmw.exe")
        if d:
            self.exe_var.set(d)
            self._describe_game(Path(d))

    def _autodetect(self) -> None:
        exe, cfg = guess_openmw()
        if exe:
            self.exe_var.set(str(exe))
            self._describe_game(exe)
        else:
            self.game_info.config(
                text="Не нашёл openmw.exe сам. Укажи папку кнопкой слева — "
                     "это та, где лежит openmw.exe.")

    def _describe_game(self, exe_dir: Path) -> None:
        if not (exe_dir / "openmw.exe").exists():
            self.game_info.config(text="В этой папке нет openmw.exe. "
                                       "Выбери другую.")
            return
        ver = openmw_version(exe_dir)
        _, cfg = guess_openmw()
        cfg_txt = f"Настройки игры: {cfg}" if cfg else \
            "Файл openmw.cfg не найден — запусти игру один раз, он создастся."
        self.game_info.config(
            text=f"Игра найдена: {exe_dir}\n{ver or 'версию определить не вышло'}\n"
                 f"{cfg_txt}\n\nМод пропишется в настройки игры на шаге проверки.")

    # ── 2. установка ────────────────────────────────────────────────────

    def _page_install(self) -> None:
        f = ttk.Frame(self.nb)
        self.nb.add(f, text="2. Установка")
        ttk.Label(f, text="Что поставить", font=("", 13, "bold"),
                  foreground=ACCENT).pack(anchor="w", padx=14, pady=(14, 4))
        self._hint(f, "Отметь галочками. Если не уверен — оставь как есть, "
                      "это рабочий набор. Интернет нужен только сейчас.")

        self.opt_deps = tk.BooleanVar(value=True)
        self.opt_vosk = tk.BooleanVar(value=True)
        self.opt_voices = tk.BooleanVar(value=True)
        for var, title, why in (
            (self.opt_deps, "Библиотеки Python (обязательно)",
             "около 300 МБ. Без них ничего не запустится."),
            (self.opt_vosk, "Распознавание речи — чтобы говорить в микрофон",
             "1.8 ГБ. Работает на процессоре, видеокарта не нужна. "
             "Без этого остаётся ввод текстом."),
            (self.opt_voices, "Голоса NPC из родной озвучки игры",
             "240 МБ. Персонажи заговорят голосами настоящих актёров игры."),
        ):
            box = ttk.Frame(f)
            box.pack(fill="x", padx=14, pady=(6, 0))
            ttk.Checkbutton(box, text=title, variable=var).pack(anchor="w")
            ttk.Label(box, text="      " + why, foreground="#9a9a9a").pack(anchor="w")

        ttk.Button(f, text="Установить", command=self._do_install).pack(
            anchor="w", padx=14, pady=12)
        self.install_log = scrolledtext.ScrolledText(
            f, height=13, bg="#141414", fg=FG, insertbackground=FG,
            font=("Consolas", 9), state="disabled")
        self.install_log.pack(fill="both", expand=True, padx=14, pady=(0, 12))

    def _do_install(self) -> None:
        threading.Thread(target=self._install_worker, daemon=True).start()

    def _install_worker(self) -> None:
        try:
            if self.opt_deps.get():
                self.say("Создаю окружение Python…")
                if not VENV_PY.exists():
                    subprocess.run([sys.executable, "-m", "venv", str(VENV)],
                                   check=True)
                self.say("Ставлю библиотеки (это займёт несколько минут)…")
                subprocess.run([str(VENV_PY), "-m", "pip", "install", "--quiet",
                                "--disable-pip-version-check", *PIP_PACKAGES],
                               check=True)
                self.say("  библиотеки готовы")

            if self.opt_vosk.get():
                self._fetch_vosk()
            if self.opt_voices.get():
                self._fetch_voices()
            self.say("\nГОТОВО. Переходи к шагу 3.")
        except subprocess.CalledProcessError as exc:
            self.say(f"!! команда не отработала: {exc}")
        except Exception as exc:  # noqa: BLE001
            self.say(f"!! {exc}")

    def _download(self, url: str, dst: Path, label: str) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        self.say(f"Качаю {label}…")
        last = [0]

        def hook(blocks, size, total):  # noqa: ANN001
            if total > 0:
                pct = int(blocks * size * 100 / total)
                if pct >= last[0] + 10:
                    last[0] = pct
                    self.say(f"   {pct}%")

        urllib.request.urlretrieve(url, dst, reporthook=hook)

    def _fetch_vosk(self) -> None:
        target = VOSK_DIR / "vosk-ru-0.42-fast"
        if (target / "am").is_dir():
            self.say("Распознавание речи уже стоит — пропускаю.")
            return
        zip_path = VOSK_DIR / "_vosk.zip"
        self._download(VOSK_URL, zip_path, "модель распознавания речи (1.8 ГБ)")
        self.say("Распаковываю…")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(VOSK_DIR)
        src = VOSK_DIR / "vosk-model-ru-0.42"
        if src.is_dir():
            os.rename(src, target)
        # Блок rnnlm стоит 81 секунду загрузки при старте и на качество почти
        # не влияет — замер показал 85 с против 3.7 при той же точности.
        rnnlm = target / "rnnlm"
        if rnnlm.is_dir():
            self.say("Убираю тяжёлый блок — старт станет быстрее в 20 раз…")
            import shutil
            shutil.rmtree(rnnlm, ignore_errors=True)
        zip_path.unlink(missing_ok=True)
        self.say("  распознавание речи готово")

    def _fetch_voices(self) -> None:
        if VOICES_DIR.is_dir() and list(VOICES_DIR.glob("*.onnx")):
            self.say("Голоса уже на месте — пропускаю.")
            return
        self.say("Голоса NPC не входят в архив: их надо получить из твоей "
                 "копии игры (озвучка защищена авторским правом).")
        self.say("Открой ГОЛОСА.md рядом с этим установщиком — там написано, "
                 "как их собрать одной командой. Пока мод будет говорить "
                 "базовым голосом piper.")

    # ── 3. кто отвечает за NPC ──────────────────────────────────────────

    def _page_brain(self) -> None:
        f = ttk.Frame(self.nb)
        self.nb.add(f, text="3. Кто отвечает")
        ttk.Label(f, text="Кто будет говорить за NPC",
                  font=("", 13, "bold"), foreground=ACCENT).pack(
            anchor="w", padx=14, pady=(14, 4))
        self._hint(f, "Это «мозг» мода. Он придумывает, что скажет персонаж.")

        self.brain = tk.StringVar(value="gemini")
        for val, title, why in (
            ("gemini", "Облако Google Gemini — проще и быстрее (рекомендую)",
             "Ответ за 2–3 секунды. Нужен бесплатный ключ, берётся за минуту "
             "на aistudio.google.com — жми «Get API key». Денег не просят."),
            ("local", "Своя модель на этом компьютере — без интернета",
             "Нужна видеокарта и LM Studio или KoboldCpp. Ответ 10–25 секунд, "
             "качество заметно ниже. Наш замер: облако 4 из 4, своя 0 из 3."),
        ):
            box = ttk.Frame(f)
            box.pack(fill="x", padx=14, pady=(8, 0))
            ttk.Radiobutton(box, text=title, value=val,
                            variable=self.brain).pack(anchor="w")
            ttk.Label(box, text="      " + why, foreground="#9a9a9a",
                      wraplength=800, justify="left").pack(anchor="w")

        kf = ttk.LabelFrame(f, text="Ключи Gemini")
        kf.pack(fill="both", expand=True, padx=14, pady=12)
        ttk.Label(kf, wraplength=820, justify="left", text=(
            "Вставь ключ. Можно несколько — по одному в строке: мод будет "
            "брать их по очереди, и бесплатный лимит кончится нескоро. "
            "Ключи сохранятся в твою личную папку, в архив мода они не "
            "попадут и никуда не отправятся.")).pack(anchor="w", padx=10, pady=8)
        self.keys_box = scrolledtext.ScrolledText(
            kf, height=6, bg="#141414", fg=FG, insertbackground=FG,
            font=("Consolas", 10))
        self.keys_box.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self._load_keys()
        ttk.Button(kf, text="Сохранить ключи",
                   command=self._save_keys).pack(anchor="w", padx=10, pady=(0, 10))

    def _load_keys(self) -> None:
        if not KEYS_FILE.exists():
            return
        keys = []
        for line in KEYS_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().upper().startswith("GOOGLE_API_KEY"):
                _, _, val = line.partition("=")
                if val.strip():
                    keys.append(val.strip())
        self.keys_box.insert("1.0", "\n".join(keys))

    def _save_keys(self) -> None:
        keys = [k.strip() for k in self.keys_box.get("1.0", "end").splitlines()
                if k.strip()]
        if not keys:
            messagebox.showwarning("Пусто", "Ни одного ключа не вижу.")
            return
        old = []
        if KEYS_FILE.exists():
            old = [l for l in KEYS_FILE.read_text(encoding="utf-8",
                                                  errors="replace").splitlines()
                   if not l.strip().upper().startswith("GOOGLE_API_KEY")]
        lines = old + [f"GOOGLE_API_KEY{'' if i == 0 else f'_{i + 1}'}={k}"
                       for i, k in enumerate(keys)]
        KEYS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        messagebox.showinfo("Сохранено",
                            f"Ключей записано: {len(keys)}.\n"
                            f"Файл: {KEYS_FILE}\n\n"
                            "Мод будет перебирать их по кругу, когда у "
                            "очередного кончается дневной лимит.")

    # ── 4. характер мира ────────────────────────────────────────────────

    def _page_world(self) -> None:
        f = ttk.Frame(self.nb)
        self.nb.add(f, text="4. Характер мира")
        ttk.Label(f, text="Каким будет мир", font=("", 13, "bold"),
                  foreground=ACCENT).pack(anchor="w", padx=14, pady=(14, 4))
        self._hint(f, "Две ручки. Их можно менять когда угодно, даже во время "
                      "игры — правки подхватываются на лету.")

        self.danger = tk.IntVar(value=30)
        self.humour = tk.IntVar(value=30)
        for var, title, low, high in (
            (self.danger, "ОПАСНОСТЬ — что случается вокруг",
             "0 — только разговоры: пересуды, склоки, проповеди",
             "100 — драки, вымогательство, кражи, врываются в дом"),
            (self.humour, "НЕЛЕПОСТЬ — сколько событий выходят дурацкими",
             "0 — мир серьёзен, всё по понятным причинам",
             "100 — почти всё балаган, но люди в нём предельно серьёзны"),
        ):
            box = ttk.LabelFrame(f, text=title)
            box.pack(fill="x", padx=14, pady=8)
            ttk.Scale(box, from_=0, to=100, variable=var,
                      orient="horizontal").pack(fill="x", padx=10, pady=(8, 2))
            lab = ttk.Label(box, text="")
            lab.pack(anchor="w", padx=10, pady=(0, 8))
            ttk.Label(box, text=f"{low}\n{high}", foreground="#9a9a9a",
                      justify="left").pack(anchor="w", padx=10, pady=(0, 8))
            var.trace_add("write",
                          lambda *_, v=var, l=lab: l.config(text=f"сейчас: {v.get()}"))
            lab.config(text=f"сейчас: {var.get()}")

        rf = ttk.LabelFrame(f, text="Правила мира (попадают КАЖДОМУ персонажу)")
        rf.pack(fill="both", expand=True, padx=14, pady=8)
        self.rules_box = scrolledtext.ScrolledText(
            rf, height=8, bg="#141414", fg=FG, insertbackground=FG,
            font=("Consolas", 9))
        self.rules_box.pack(fill="both", expand=True, padx=10, pady=8)
        try:
            self.rules_box.insert("1.0", RULES.read_text(encoding="utf-8"))
        except OSError:
            pass
        ttk.Button(f, text="Сохранить настройки мира",
                   command=self._save_world).pack(anchor="w", padx=14, pady=(0, 12))

    def _save_world(self) -> None:
        TUNING.parent.mkdir(parents=True, exist_ok=True)
        TUNING.write_text(
            f"# Характер мира. Правится на лету.\n"
            f"опасность: {self.danger.get()}\n"
            f"нелепость: {self.humour.get()}\n", encoding="utf-8")
        RULES.write_text(self.rules_box.get("1.0", "end").rstrip() + "\n",
                         encoding="utf-8")
        messagebox.showinfo("Сохранено", "Характер мира записан.")

    # ── 5. проверка ─────────────────────────────────────────────────────

    def _page_check(self) -> None:
        f = ttk.Frame(self.nb)
        self.nb.add(f, text="5. Проверка")
        ttk.Label(f, text="Проверим, что всё работает",
                  font=("", 13, "bold"), foreground=ACCENT).pack(
            anchor="w", padx=14, pady=(14, 4))
        self._hint(f, "Мастер сам подключит мод к игре, прогонит тесты и "
                      "задаст живой вопрос персонажу. Если что-то не так — "
                      "скажет, что именно.")
        ttk.Button(f, text="Проверить всё",
                   command=lambda: threading.Thread(
                       target=self._check_worker, daemon=True).start()).pack(
            anchor="w", padx=14, pady=8)
        self.check_log = scrolledtext.ScrolledText(
            f, height=18, bg="#141414", fg=FG, insertbackground=FG,
            font=("Consolas", 9), state="disabled")
        self.check_log.pack(fill="both", expand=True, padx=14, pady=(0, 12))

    def _check_worker(self) -> None:
        ok = True
        self.say("── подключаю мод к игре ──")
        try:
            ok &= self._register_mod()
        except Exception as exc:  # noqa: BLE001
            self.say(f"!! не вышло: {exc}")
            ok = False

        self.say("\n── прогоняю тесты ──")
        if VENV_PY.exists():
            res = subprocess.run([str(VENV_PY), str(ROOT / "tests" / "test_all.py")],
                                 capture_output=True, text=True,
                                 encoding="utf-8", errors="replace",
                                 cwd=str(ROOT))
            tail = [l for l in (res.stdout or "").splitlines()
                    if "ПРОЙДЕНО" in l or "ПРОВАЛ" in l]
            for l in tail[:6]:
                self.say("  " + l.strip())
            ok &= res.returncode == 0
        else:
            self.say("!! окружение не создано — вернись на шаг 2")
            ok = False

        self.say("\n── спрашиваю персонажа по-настоящему ──")
        try:
            probe = ROOT / "installer" / "_probe.py"
            res = subprocess.run([str(VENV_PY), str(probe)], capture_output=True,
                                 text=True, encoding="utf-8", errors="replace",
                                 cwd=str(ROOT), timeout=180)
            for l in (res.stdout or "").splitlines():
                if l.strip():
                    self.say("  " + l.strip())
            ok &= res.returncode == 0
        except Exception as exc:  # noqa: BLE001
            self.say(f"!! {exc}")
            ok = False

        self.say("\n" + ("ВСЁ ГОТОВО. Запускай игру ярлыком «Morrowind AI»."
                         if ok else
                         "Есть неполадки — смотри строки с «!!» выше."))

    def _register_mod(self) -> bool:
        _, cfg_dir = guess_openmw()
        if not cfg_dir:
            self.say("!! не нашёл openmw.cfg — запусти игру один раз")
            return False
        cfg = cfg_dir / "openmw.cfg"
        text = cfg.read_text(encoding="utf-8", errors="replace")
        data_line = f'data="{ROOT / "openmw-mod"}"'
        script_line = "content=morrowind-ai.omwscripts"
        changed = False
        if data_line not in text:
            text = text.rstrip() + "\n" + data_line + "\n"
            changed = True
        if script_line not in text:
            text = text.rstrip() + "\n" + script_line + "\n"
            changed = True
        if changed:
            backup = cfg.with_suffix(".cfg.before-morrowind-ai")
            if not backup.exists():
                backup.write_text(cfg.read_text(encoding="utf-8", errors="replace"),
                                  encoding="utf-8")
            cfg.write_text(text, encoding="utf-8")
            self.say(f"  мод прописан в {cfg.name} (старый сохранён рядом)")
        else:
            self.say("  мод уже подключён")
        return True


if __name__ == "__main__":
    Wizard().mainloop()
