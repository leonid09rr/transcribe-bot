"""Per-user лимит на минуты аудио в сутки + blacklist.

Хранится в памяти (без БД). При рестарте бота счётчики обнуляются — это ок:
лимит и так суточный, окно скользящее по 24 часам.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque

WINDOW_SECONDS = 24 * 3600

# user_id -> deque[(timestamp, minutes_used)]
_usage: dict[int, deque[tuple[float, float]]] = defaultdict(deque)


def _limit_minutes() -> int:
    try:
        return int(os.getenv("DAILY_USER_LIMIT_MINUTES", "30"))
    except ValueError:
        return 30


def _blacklist() -> set[int]:
    raw = os.getenv("BLACKLIST_USER_IDS", "").strip()
    if not raw:
        return set()
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            out.add(int(part))
    return out


def is_blacklisted(user_id: int) -> bool:
    return user_id in _blacklist()


def used_minutes(user_id: int) -> float:
    """Сколько минут юзер использовал за последние 24 часа."""
    _evict_old(user_id)
    return sum(m for _, m in _usage[user_id])


def remaining_minutes(user_id: int) -> float:
    limit = _limit_minutes()
    if limit <= 0:
        return float("inf")
    return max(0.0, limit - used_minutes(user_id))


def can_process(user_id: int, planned_minutes: float) -> tuple[bool, str]:
    """Проверка перед обработкой. Возвращает (можно, причина_отказа)."""
    if is_blacklisted(user_id):
        return False, "Доступ к боту ограничен."

    limit = _limit_minutes()
    if limit <= 0:
        return True, ""

    remaining = remaining_minutes(user_id)
    if planned_minutes > remaining:
        return (
            False,
            f"⏳ Дневной лимит — {limit} мин аудио. "
            f"Осталось {remaining:.0f} мин, видео — {planned_minutes:.0f} мин. "
            f"Попробуй короче видео или подожди до завтра.",
        )
    return True, ""


def record_usage(user_id: int, minutes: float) -> None:
    _usage[user_id].append((time.time(), minutes))
    _evict_old(user_id)


def _evict_old(user_id: int) -> None:
    cutoff = time.time() - WINDOW_SECONDS
    q = _usage[user_id]
    while q and q[0][0] < cutoff:
        q.popleft()
