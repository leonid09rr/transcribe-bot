"""Скачивание видео по ссылке (yt-dlp) + извлечение и нарезка аудио (ffmpeg).

Whisper API Groq принимает файлы до 25 МБ. Чтобы обрабатывать длинные видео,
аудио конвертируется в mp3 64 kbps mono 16 кГц и режется на чанки по 10 минут
с перекрытием 2 секунды (чтобы не терялись слова на границе).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# Параметры аудио для Whisper.
# 16 кГц mono mp3 64 kbps ≈ 0.48 МБ/мин → 10 мин ≈ 4.8 МБ (с запасом до лимита 25 МБ).
AUDIO_BITRATE = "64k"
AUDIO_SAMPLE_RATE = 16000
CHUNK_SECONDS = 600
CHUNK_OVERLAP_SECONDS = 2

URL_PATTERN = re.compile(r"https?://\S+")


@dataclass
class AudioChunk:
    path: Path
    start_seconds: int  # абсолютный таймкод начала чанка в исходном файле


@dataclass
class MediaInfo:
    audio_path: Path  # полное аудио (один файл)
    duration_seconds: int
    title: str | None = None


class MediaError(Exception):
    """Любые ошибки скачивания/конвертации, которые показываем пользователю."""


def is_url(text: str) -> bool:
    return bool(URL_PATTERN.search(text or ""))


def extract_url(text: str) -> str | None:
    match = URL_PATTERN.search(text or "")
    return match.group(0) if match else None


def make_workdir() -> Path:
    """Создаёт временную папку для одной задачи. Чистится извне."""
    return Path(tempfile.mkdtemp(prefix="transcribe_"))


def cleanup_workdir(workdir: Path) -> None:
    shutil.rmtree(workdir, ignore_errors=True)


async def download_from_url(url: str, workdir: Path) -> MediaInfo:
    """Скачивает видео/аудио по ссылке через yt-dlp, конвертирует в mp3.

    Возвращает MediaInfo с путём к полному mp3 и длительностью.
    """
    output_template = str(workdir / "source.%(ext)s")
    # python -m yt_dlp вместо "yt-dlp" — работает и в venv, и без активации,
    # и в Docker без правильно настроенного PATH.
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio/best",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "5",
        "--postprocessor-args", f"-ar {AUDIO_SAMPLE_RATE} -ac 1 -b:a {AUDIO_BITRATE}",
        "--no-playlist",
        "--no-warnings",
        "--print-json",
        "-o", output_template,
        url,
    ]
    logger.info("yt-dlp: скачиваю %s", url)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip().splitlines()[-1:]
        raise MediaError(f"Не удалось скачать видео по ссылке. yt-dlp: {' / '.join(err) or 'unknown error'}")

    title = None
    duration = 0
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        try:
            meta = json.loads(line)
        except json.JSONDecodeError:
            continue
        title = meta.get("title") or title
        duration = int(meta.get("duration") or duration)

    audio_path = workdir / "source.mp3"
    if not audio_path.exists():
        raise MediaError("yt-dlp скачал файл, но mp3 не появился. Проверь ffmpeg в системе.")

    if duration <= 0:
        duration = await _ffprobe_duration(audio_path)

    return MediaInfo(audio_path=audio_path, duration_seconds=duration, title=title)


async def convert_to_audio(input_path: Path, workdir: Path) -> MediaInfo:
    """Конвертирует локальный файл (видео/аудио) в mp3 16 кГц mono."""
    audio_path = workdir / "source.mp3"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", "1",
        "-b:a", AUDIO_BITRATE,
        str(audio_path),
    ]
    logger.info("ffmpeg: конвертирую %s -> mp3", input_path.name)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        tail = stderr.decode("utf-8", errors="replace").strip().splitlines()[-2:]
        raise MediaError(f"ffmpeg не смог извлечь аудио: {' / '.join(tail)}")

    duration = await _ffprobe_duration(audio_path)
    return MediaInfo(audio_path=audio_path, duration_seconds=duration)


async def split_into_chunks(audio_path: Path, duration_seconds: int, workdir: Path) -> list[AudioChunk]:
    """Режет mp3 на чанки по CHUNK_SECONDS с перекрытием CHUNK_OVERLAP_SECONDS.

    Если файл короче CHUNK_SECONDS — возвращает один чанк (исходный файл).
    """
    if duration_seconds <= CHUNK_SECONDS:
        return [AudioChunk(path=audio_path, start_seconds=0)]

    chunks_dir = workdir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    chunks: list[AudioChunk] = []
    start = 0
    index = 0
    while start < duration_seconds:
        chunk_path = chunks_dir / f"chunk_{index:03d}.mp3"
        # Берём с перекрытием в начало (кроме самого первого чанка).
        seek = max(0, start - CHUNK_OVERLAP_SECONDS) if index > 0 else 0
        length = CHUNK_SECONDS + (CHUNK_OVERLAP_SECONDS if index > 0 else 0)

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(seek),
            "-i", str(audio_path),
            "-t", str(length),
            "-c", "copy",
            str(chunk_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            tail = stderr.decode("utf-8", errors="replace").strip().splitlines()[-2:]
            raise MediaError(f"ffmpeg не смог нарезать аудио: {' / '.join(tail)}")

        chunks.append(AudioChunk(path=chunk_path, start_seconds=seek))
        start += CHUNK_SECONDS
        index += 1

    logger.info("Нарезано чанков: %d", len(chunks))
    return chunks


async def _ffprobe_duration(audio_path: Path) -> int:
    """Возвращает длительность аудио в секундах через ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    try:
        return int(float(stdout.decode("utf-8").strip()))
    except (ValueError, AttributeError):
        return 0
