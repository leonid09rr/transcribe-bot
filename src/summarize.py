"""Генерация трёх уровней пересказа через Groq Llama 3.3 70B.

Все три уровня запускаются параллельно — три независимых API-запроса.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

from groq import AsyncGroq, GroqError

from .prompts import LEVELS

logger = logging.getLogger(__name__)


@dataclass
class Summaries:
    light: str
    medium: str
    full: str


class SummarizeError(Exception):
    pass


# Лимит контекста Llama 3.3 70B на Groq — 128k токенов на вход.
# 1 русский токен ≈ 2 символа, оставляем запас под промпт и ответ.
MAX_TRANSCRIPT_CHARS = 200_000


def _get_client() -> AsyncGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SummarizeError("GROQ_API_KEY не задан")
    return AsyncGroq(api_key=api_key)


def _get_model() -> str:
    return os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")


async def summarize_all_levels(transcript: str) -> Summaries:
    """Генерирует все три уровня параллельно."""
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        logger.warning(
            "Транскрипция %d символов > лимита %d, обрезаем",
            len(transcript), MAX_TRANSCRIPT_CHARS,
        )
        transcript = transcript[:MAX_TRANSCRIPT_CHARS] + "\n\n[…транскрипция обрезана для умещения в контекст…]"

    client = _get_client()
    model = _get_model()

    tasks = [
        _summarize_one(client, model, level_key, transcript)
        for level_key in ("light", "medium", "full")
    ]
    light, medium, full = await asyncio.gather(*tasks)
    return Summaries(light=light, medium=medium, full=full)


async def _summarize_one(client: AsyncGroq, model: str, level_key: str, transcript: str) -> str:
    _, prompt_template = LEVELS[level_key]
    prompt = prompt_template.format(transcript=transcript)
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Ты — редактор-конспектист. Отвечаешь на русском, кратко и по делу."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4000,
        )
    except GroqError as e:
        msg = str(e)
        if "rate_limit" in msg.lower() or "429" in msg:
            raise SummarizeError("Groq rate limit на Llama. Подожди минуту и попробуй снова.") from e
        raise SummarizeError(f"Groq Llama API: {msg}") from e

    content = response.choices[0].message.content if response.choices else ""
    return (content or "").strip()
