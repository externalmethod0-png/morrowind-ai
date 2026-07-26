@echo off
chcp 65001 >nul
title Morrowind AI - установка
rem ─────────────────────────────────────────────────────────────────────────
rem  Один клик. Находит Python, при необходимости предлагает поставить, и
rem  открывает окно мастера. Больше от человека ничего не требуется.
rem ─────────────────────────────────────────────────────────────────────────

echo.
echo   Morrowind AI — установка
echo   ------------------------
echo.

set "PY="
for %%C in (py python) do (
    if not defined PY (
        %%C -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
        if not errorlevel 1 set "PY=%%C"
    )
)

if not defined PY (
    echo   На компьютере не найден Python 3.10 или новее.
    echo.
    echo   Что делать:
    echo     1. Открой https://www.python.org/downloads/
    echo     2. Скачай последнюю версию и запусти установщик
    echo     3. ВАЖНО: на первом экране поставь галочку
    echo        "Add python.exe to PATH"
    echo     4. Дождись конца установки и запусти этот файл ещё раз
    echo.
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

echo   Python найден. Открываю мастер установки...
echo   (окно может появиться через несколько секунд)
echo.

%PY% "%~dp0installer\wizard.py"
if errorlevel 1 (
    echo.
    echo   Мастер закрылся с ошибкой. Если окно не открылось совсем —
    echo   напиши об этом в issues на странице проекта, приложив текст выше.
    pause
)
