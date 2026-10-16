"""Read timezone-aware review timestamps, including browser ISO strings."""

from datetime import datetime


def parse_review_time(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Review time must be a nonempty ISO timestamp")
    timestamp = value.strip()
    if timestamp.endswith("Z"):
        timestamp = timestamp[:-1] + "+00:00"
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo is None:
        raise ValueError("Review time must include a timezone")
    return parsed
