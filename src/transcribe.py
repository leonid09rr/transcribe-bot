"""Транскрипция через Groq Whisper API.

Принимает список AudioChunk, шлёт каждый в Whisper-large-v3-turbo, склеивает
результаты со сдвигом таймкодов, возвращает финальный текст.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

from groq import AsyncGroq, GroqError

from .media import AudioChunk

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    start: float  # абсолютные секунды от начала исходного файла
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str  # полный связный текст без таймкодов
    text_with_timecodes: str  # текст с таймкодами вида [MM:SS] перед каждым сегментом
    segments: list[TranscriptSegment]
    duration_seconds: float


class TranscribeError(Exception):
    """Ошибки Groq, которые показываем пользователю."""


def _get_client() -> AsyncGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise TranscribeError("GROQ_API_KEY не задан в .env")
    return AsyncGroq(api_key=api_key)


def _get_model() -> str:
    return os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")


async def transcribe_chunks(chunks: list[AudioChunk], language: str | None = None) -> TranscriptionResult:
    """Транскрибирует все чанки последовательно, склеивает с правильными таймкодами."""
    client = _get_client()
    model = _get_model()
    all_segments: list[TranscriptSegment] = []
    last_end = 0.0

    for i, chunk in enumerate(chunks):
        logger.info("Whisper: чанк %d/%d (offset=%ds)", i + 1, len(chunks), chunk.start_seconds)
        segments = await _transcribe_one(client, model, chunk, language)
        # Сдвигаем таймкоды на абсолютную позицию чанка.
        for seg in segments:
            seg.start += chunk.start_seconds
            seg.end += chunk.start_seconds
            # Защита от перекрытия: если сегмент попадает в зону overlap — пропускаем.
            if seg.end <= last_end:
                continue
            all_segments.append(seg)
            last_end = max(last_end, seg.end)

    text = " ".join(s.text.strip() for s in all_segments).strip()
    text_with_timecodes = _format_with_timecodes(all_segments)
    duration = all_segments[-1].end if all_segments else 0.0

    return TranscriptionResult(
        text=text,
        text_with_timecodes=text_with_timecodes,
        segments=all_segments,
        duration_seconds=duration,
    )


async def _transcribe_one(
    client: AsyncGroq,
    model: str,
    chunk: AudioChunk,
    language: str | None,
) -> list[TranscriptSegment]:
    try:
        with open(chunk.path, "rb") as f:
            response = await client.audio.transcriptions.create(
                file=(chunk.path.name, f.read()),
                model=model,
                response_format="verbose_json",
                language=language,
                temperature=0,
            )
    except GroqError as e:
        msg = str(e)
        if "rate_limit" in msg.lower() or "429" in msg:
            raise TranscribeError(
                "Groq rate limit. Подожди немного или попробуй короче видео."
            ) from e
        raise TranscribeError(f"Groq Whisper API: {msg}") from e

    segments_data = getattr(response, "segments", None) or []
    result: list[TranscriptSegment] = []
    for seg in segments_data:
        # API возвращает либо dict, либо pydantic-объект — поддержим оба.
        get = seg.get if isinstance(seg, dict) else lambda k, _s=seg: getattr(_s, k, None)
        start = float(get("start") or 0)
        end = float(get("end") or 0)
        text = (get("text") or "").strip()
        if text:
            result.append(TranscriptSegment(start=start, end=end, text=text))

    if not result:
        # Whisper иногда не отдаёт сегменты для совсем тихих/коротких записей —
        # берём общий text как один сегмент.
        full_text = getattr(response, "text", "") or ""
        if full_text.strip():
            result.append(TranscriptSegment(start=0.0, end=0.0, text=full_text.strip()))

    return result


def _format_with_timecodes(segments: list[TranscriptSegment]) -> str:
    """Группирует сегменты по ~30 секунд и ставит таймкод в начало каждой группы."""
    if not segments:
        return ""

    lines: list[str] = []
    buffer: list[str] = []
    group_start = segments[0].start
    last_flushed = -999.0

    def flush() -> None:
        if buffer:
            lines.append(f"[{_fmt_tc(group_start)}] {' '.join(buffer).strip()}")
            buffer.clear()

    for seg in segments:
        if seg.start - last_flushed >= 30:
            flush()
            group_start = seg.start
            last_flushed = seg.start
        buffer.append(seg.text)

    flush()
    return "\n".join(lines)


def _fmt_tc(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"
