"""Shared fixtures/helpers for detector tests.

Provides candle-building helpers so tests can construct synthetic OHLCV series
with controlled structure. All helpers return RAW (unvalidated) candle dicts;
tests that need validation call ``validate_candles`` explicitly.
"""

from __future__ import annotations

from typing import Any

__all__ = ["mkc", "mkseries", "epoch_dt"]


def mkc(o: float, h: float, l: float, c: float, ts: Any, vol: float = 1000.0) -> dict:
    """Build a single candle dict. Does NOT validate."""
    return {"open": o, "high": h, "low": l, "close": c, "volume": vol, "timestamp": ts}


def mkseries(rows: list[tuple], start_ts: int = 0, step: int = 60) -> list[dict]:
    """Build a candle series from (o,h,l,c) tuples with epoch timestamps.

    Timestamps start at ``start_ts`` and increment by ``step`` seconds.
    """
    out = []
    ts = start_ts
    for (o, h, l, c) in rows:
        out.append(mkc(o, h, l, c, ts))
        ts += step
    return out


def epoch_dt(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    """Epoch seconds for a UTC datetime — handy for tz-sensitive tests."""
    from datetime import datetime, timezone

    return int(datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp())
