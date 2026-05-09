"""Лаунчер с проверкой окружения и системных зависимостей."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _check_binary(name: str, install_hint: str) -> None:
    if shutil.which(name) is None:
        print(f"❌ Не найден бинарник '{name}'. {install_hint}")
        sys.exit(1)


def _check_python_module(module: str, install_hint: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"❌ Не установлен Python-модуль '{module}'. {install_hint}")
        sys.exit(1)


def _check_env_file() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        print("❌ Не найден файл .env. Скопируй .env.example -> .env и заполни.")
        sys.exit(1)


def main() -> None:
    _check_env_file()
    _check_binary("ffmpeg", "Windows: winget install Gyan.FFmpeg | Mac: brew install ffmpeg | Linux: apt install ffmpeg")
    _check_binary("ffprobe", "Идёт вместе с ffmpeg.")
    _check_python_module("yt_dlp", "Установи: pip install yt-dlp")

    bot_path = Path(__file__).parent / "bot.py"
    subprocess.run([sys.executable, str(bot_path)], check=False)


if __name__ == "__main__":
    main()
