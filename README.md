# Transcribe Bot — транскрипция видео в три уровня

Telegram-бот: кидаешь видео или ссылку — получаешь три уровня пересказа (темы → тезисы → детальный) + полную транскрипцию файлом. Работает полностью на бесплатных лимитах Groq (Whisper + Llama 3.3 70B).

## Что внутри

| Файл | Зачем |
|------|-------|
| [bot.py](bot.py) | Точка входа: загружает `.env`, запускает aiogram polling |
| [run_bot.py](run_bot.py) | Лаунчер с проверкой ffmpeg/ffprobe/yt-dlp |
| [src/handlers.py](src/handlers.py) | Хендлеры `/start`, `/help`, `/limits`, `/myid` и приём медиа |
| [src/media.py](src/media.py) | yt-dlp + ffmpeg: скачивание ссылок, конвертация, нарезка на чанки |
| [src/transcribe.py](src/transcribe.py) | Groq Whisper-large-v3-turbo: распознавание речи с таймкодами |
| [src/summarize.py](src/summarize.py) | Groq Llama 3.3 70B: три уровня пересказа параллельно |
| [src/prompts.py](src/prompts.py) | Промпты для уровней `light` / `medium` / `full` |
| [src/rate_limit.py](src/rate_limit.py) | Per-user лимит минут аудио в сутки + blacklist |
| [requirements.txt](requirements.txt) | Зависимости |
| [Procfile](Procfile) / [nixpacks.toml](nixpacks.toml) | Конфиг для Railway |
| [.env.example](.env.example) | Шаблон секретов |

## Что умеет бот

| Источник | Поддержка | Лимит |
|---|---|---|
| 🎥 Видео из Telegram | ✅ | до 20 МБ (ограничение Telegram Bot API) |
| 🎙 Voice / video note | ✅ | до 20 МБ |
| 🎵 Аудио-файл | ✅ | до 20 МБ |
| 🔗 YouTube / Reels / TikTok / VK Video / Rutube / прямые .mp4 | ✅ через `yt-dlp` | без лимита размера |

Длинные видео (часовые лекции, эфиры) — **только ссылкой**. Telegram не отдаёт ботам файлы > 20 МБ.

## Локальный запуск (Windows / Mac / Linux)

```bash
# 1. Системные зависимости
# Windows: winget install Gyan.FFmpeg
# Mac: brew install ffmpeg
# Linux: sudo apt install ffmpeg

# 2. Python зависимости
cd transcribe-bot
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt    # Windows
# .venv/bin/pip install -r requirements.txt        # Mac/Linux

# 3. Заполнить .env
copy .env.example .env                             # Windows
# cp .env.example .env                             # Mac/Linux
# Открыть .env и вписать TELEGRAM_BOT_TOKEN и GROQ_API_KEY

# 4. Запустить
.venv\Scripts\python run_bot.py                    # Windows
# .venv/bin/python run_bot.py                       # Mac/Linux
```

После запуска — найди бота в Telegram, нажми `/start`, кинь видео или ссылку.

## Деплой на Railway ($5/мес)

1. Зайди в Railway → **New Project** → **Deploy from GitHub repo**.
2. Выбери этот репозиторий.
3. В настройках сервиса укажи **Root Directory:** `transcribe-bot`.
4. В **Variables** добавь:
   - `TELEGRAM_BOT_TOKEN`
   - `GROQ_API_KEY`
   - `DAILY_USER_LIMIT_MINUTES` = `30` (по желанию)
   - `BLACKLIST_USER_IDS` = пусто (заполняется по мере нужды)
5. Railway сам подхватит [nixpacks.toml](nixpacks.toml) — установит Python 3.11 + ffmpeg.
6. Стартовая команда из `nixpacks.toml`: `python bot.py`.

После деплоя смотри логи в Railway → твой сервис → **Deployments** → **View Logs**.

## Команды бота

| Команда | Что делает |
|---------|-----------|
| `/start`, `/help` | Приветствие и инструкция |
| `/limits` | Сколько минут аудио использовано/осталось за 24ч |
| `/myid` | Показать свой Telegram ID |
| Видео / ссылка | Запустить транскрипцию |

## Лимиты Groq Free Tier (на API-ключ, общий для всех ботов на нём)

- **Whisper-large-v3-turbo:** ~7200 секунд аудио в день (≈2 часа), 20 запросов/мин
- **Llama 3.3 70B:** ~6000 запросов/день, 12k токенов/мин

Если у тебя несколько ботов на одном Groq-ключе — лимит общий. Заведи отдельный аккаунт под этот бот, если будет упираться.

## Per-user лимит (антиспам)

Бот публичный — любой может писать. Чтобы посторонний не сжёг твой Groq-лимит за минуту:

- `DAILY_USER_LIMIT_MINUTES=30` — каждый юзер не больше 30 мин аудио в сутки.
- `BLACKLIST_USER_IDS=12345,67890` — забанить конкретные ID.

Узнать ID злоупотребляющего пользователя: посмотри в логах (`bot.log` или Railway logs) после его сообщения.

## Стоимость

| Что | Цена |
|---|---|
| Railway Hobby ($5/мес) | твоя текущая подписка |
| Groq Whisper + Llama (Free Tier) | **$0** |
| **Итого** | **$5/мес** (то, что ты уже платишь) |

Если упрёшься в лимиты Groq — переход на Dev Tier стоит ~$0.04 за час аудио. Час видео в день = $1.20/мес.

## Траблшутинг

| Симптом | Решение |
|---|---|
| `❌ Не найден бинарник 'ffmpeg'` | Установи ffmpeg (см. раздел «Локальный запуск»). На Railway — должен подтянуться из `nixpacks.toml`. |
| `Файл больше 20 МБ` | Загрузи на YouTube/VK/облако и пришли ссылкой. Telegram Bot API не отдаёт большие файлы. |
| `Groq rate limit` | Лимит исчерпан, жди 24ч или переключи Groq на платный Dev Tier. |
| Бот не отвечает | Смотри `bot.log` локально или Railway → Logs. |
| `yt-dlp: Unsupported URL` | Сайт не поддерживается. Список: https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md |
