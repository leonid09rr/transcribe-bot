"""Точка входа Telegram-бота транскрипции видео."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from aiogram import Bot, Dispatcher
from dotenv import load_dotenv

# Windows-консоль по умолчанию в cp1251 — emoji в логах падают с UnicodeEncodeError.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"❌ В .env не задана переменная {name}. См. .env.example")
    return value


async def main() -> None:
    token = _require_env("TELEGRAM_BOT_TOKEN")
    _require_env("GROQ_API_KEY")

    from src.handlers import router  # noqa: WPS433 — импорт после load_dotenv

    logger.info("🤖 Запуск transcribe-bot...")
    bot = Bot(token=token)
    dp = Dispatcher()
    dp.include_router(router)

    logger.info("✅ Хендлеры зарегистрированы. Polling.")
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("👋 Остановлен.")
