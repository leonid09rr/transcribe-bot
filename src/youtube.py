"""YouTube-субтитры через youtube-transcript-api.

YouTube блокирует yt-dlp с дата-центровых IP (Railway, AWS, GCP), требуя
авторизации через cookies. Вместо борьбы с этим — берём готовые субтитры
напрямую через YouTube API. Если у видео есть субтитры (ручные или
авто-сгенерированные), их можно получить без авторизации и без скачивания
самого видео.

Если субтитров нет — поднимаем понятную ошибку, чтобы хендлер показал её
пользователю.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

logger = logging.getLogger(__name__)

YOUTUBE_HOSTS = ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "music.youtube.com")
PREFERRED_LANGUAGES = ["ru", "en", "uk", "be"]

# Регулярка для извлечения YouTube video ID из любых форматов ссылок:
# https://youtu.be/XXXX, https://youtube.com/watch?v=XXXX, /shorts/XXXX, /embed/XXXX
_VIDEO_ID_RE = re.compile(
    r"(?:v=|/)([0-9A-Za-z_-]{11})(?:[?&/]|$)"
)


class YouTubeNotApplicable(Exception):
    """URL не относится к YouTube — нужно использовать обычный пайплайн."""


class YouTubeTranscriptError(Exception):
    """Ошибки получения субтитров, которые показываем пользователю."""


@dataclass
class YouTubeTranscript:
    text: str  # сплошной текст без таймкодов
    text_with_timecodes: str  # текст с таймкодами вида [MM:SS]
    duration_seconds: float
    language: str  # из какого языка взяли (ru, en, ...)
    is_generated: bool  # авто-сгенерированы ли


def is_youtube_url(url: str) -> bool:
    if not url:
        return False
    return any(host in url for host in YOUTUBE_HOSTS)


def extract_video_id(url: str) -> str | None:
    match = _VIDEO_ID_RE.search(url or "")
    return match.group(1) if match else None


async def fetch_transcript(url: str) -> YouTubeTranscript:
    """Берёт субтитры YouTube через API. Бросает YouTubeTranscriptError,
    если ссылка валидная YouTube, но субтитров нет / видео недоступно."""
    if not is_youtube_url(url):
        raise YouTubeNotApplicable("Не YouTube-ссылка")

    video_id = extract_video_id(url)
    if not video_id:
        raise YouTubeTranscriptError("Не смог распознать YouTube video ID в ссылке.")

    logger.info("YouTube subtitles: video_id=%s", video_id)

    # Сетевой вызов синхронный — оборачиваем в to_thread.
    try:
        transcript_list = await asyncio.to_thread(
            YouTubeTranscriptApi.list_transcripts, video_id
        )
    except TranscriptsDisabled as e:
        raise YouTubeTranscriptError(
            "У этого видео отключены субтитры. Скачай видео и пришли файлом до 20 МБ."
        ) from e
    except VideoUnavailable as e:
        raise YouTubeTranscriptError(f"YouTube: видео недоступно ({e}).") from e
    except Exception as e:
        # YouTube периодически блокирует API с дата-центров — даём понятное сообщение.
        msg = str(e).lower()
        if "ip" in msg or "blocked" in msg or "captcha" in msg or "forbidden" in msg:
            raise YouTubeTranscriptError(
                "YouTube блокирует доступ к субтитрам с этого сервера. "
                "Скачай видео и пришли файлом."
            ) from e
        raise YouTubeTranscriptError(f"Не смог получить субтитры YouTube: {e}") from e

    # Выбираем лучший доступный язык: сначала ручные, потом авто-сгенерированные.
    chosen = _pick_best_transcript(transcript_list)
    if chosen is None:
        raise YouTubeTranscriptError(
            "У этого видео нет субтитров на поддерживаемых языках (ru, en, uk, be)."
        )

    try:
        entries = await asyncio.to_thread(chosen.fetch)
    except NoTranscriptFound as e:
        raise YouTubeTranscriptError("Субтитры есть в списке, но не загрузились.") from e
    except Exception as e:
        raise YouTubeTranscriptError(f"Сбой загрузки субтитров: {e}") from e

    # entries: list[{'text': str, 'start': float, 'duration': float}]
    plain_text = " ".join(e["text"].strip() for e in entries if e.get("text")).strip()
    text_with_timecodes = _format_with_timecodes(entries)
    duration = (entries[-1]["start"] + entries[-1].get("duration", 0)) if entries else 0.0

    return YouTubeTranscript(
        text=plain_text,
        text_with_timecodes=text_with_timecodes,
        duration_seconds=duration,
        language=chosen.language_code,
        is_generated=chosen.is_generated,
    )


def _pick_best_transcript(transcript_list):
    """Сначала ищем ручные субтитры на наших языках, потом авто-сгенерированные."""
    # Manually created — приоритет.
    for lang in PREFERRED_LANGUAGES:
        try:
            t = transcript_list.find_manually_created_transcript([lang])
            return t
        except Exception:
            continue
    # Auto-generated — fallback.
    for lang in PREFERRED_LANGUAGES:
        try:
            t = transcript_list.find_generated_transcript([lang])
            return t
        except Exception:
            continue
    # Любые имеющиеся — последний шанс.
    for t in transcript_list:
        return t
    return None


def _format_with_timecodes(entries: list[dict]) -> str:
    """Группирует субтитры по 30-секундным окнам, ставит таймкод в начало."""
    if not entries:
        return ""

    lines: list[str] = []
    buffer: list[str] = []
    group_start = entries[0]["start"]
    last_flushed = -999.0

    def flush() -> None:
        if buffer:
            lines.append(f"[{_fmt_tc(group_start)}] {' '.join(buffer).strip()}")
            buffer.clear()

    for e in entries:
        start = float(e.get("start") or 0)
        text = (e.get("text") or "").strip()
        if not text:
            continue
        if start - last_flushed >= 30:
            flush()
            group_start = start
            last_flushed = start
        buffer.append(text)

    flush()
    return "\n".join(lines)


def _fmt_tc(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"
