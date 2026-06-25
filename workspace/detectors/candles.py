"""candles.py — OHLCV ingestion and validation (Phase 1).

Single ingestion gate. All other detectors consume the output of
``validate_candles`` and assume clean data. Pure / stateless: identical
input always produces identical output. No I/O, no broker, no DB.

Candle contract (§0.1 of blueprint):
    open, high, low, close : float
    volume                 : float (>= 0; 0 allowed for synthetic/FX)
    timestamp              : int (epoch seconds) OR ISO-8601 str.
                             Must be monotonic non-decreasing across the list.

Methodological notes (flagged, not silently resolved):
    - Timestamps are NOT mutated by validate_candles (values preserved).
      Time-dependent detectors normalize internally via _to_epoch /
      _to_datetime so both epoch-int and ISO-8601 inputs are supported.
    - Unsorted (strictly decreasing) timestamps raise ValueError; we do NOT
      silently re-sort (could mask feed corruption).
    - Duplicate (equal) timestamps are permitted (non-decreasing allows
      equal) and flagged with ``duplicate_timestamp: True`` on the duplicate.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

__all__ = ["validate_candles", "candle_metrics", "_to_epoch", "_to_datetime"]

_REQUIRED_KEYS = ("open", "high", "low", "close", "volume", "timestamp")


def _is_nan(x: Any) -> bool:
    """True if x is a float NaN. None and non-numbers are handled by callers."""
    try:
        return isinstance(x, float) and math.isnan(x)
    except (TypeError, ValueError):
        return False


def _to_epoch(ts: Any) -> int:
    """Normalize a timestamp (int epoch seconds OR ISO-8601 str) to epoch int.

    ISO-8601 strings may carry an explicit offset; naive strings are assumed
    UTC. Raises ValueError on unparseable input.
    """
    if isinstance(ts, bool):  # bool is an int subclass — reject explicitly
        raise ValueError(f"timestamp must be int or ISO-8601 str, got bool: {ts!r}")
    if isinstance(ts, int):
        return ts
    if isinstance(ts, float) and ts.is_integer():
        return int(ts)
    if isinstance(ts, str):
        s = ts.strip()
        # Try ISO-8601. fromisoformat handles offsets in 3.11+.
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"unparseable ISO-8601 timestamp: {ts!r}") from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    raise ValueError(f"timestamp must be int or ISO-8601 str, got {type(ts).__name__}: {ts!r}")


def _to_datetime(ts: Any) -> datetime:
    """Normalize a timestamp to an aware datetime (UTC if naive)."""
    if isinstance(ts, bool):
        raise ValueError(f"timestamp must be int or ISO-8601 str, got bool: {ts!r}")
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    if isinstance(ts, str):
        s = ts.strip()
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    raise ValueError(f"timestamp must be int or ISO-8601 str, got {type(ts).__name__}: {ts!r}")


def _coerce_float(value: Any, key: str) -> float:
    """Coerce an OHLCV field to float, rejecting None/NaN/non-numeric."""
    if value is None:
        raise ValueError(f"{key} is None")
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric, got bool: {value!r}")
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} is non-numeric: {value!r}") from exc
    if _is_nan(f):
        raise ValueError(f"{key} is NaN")
    return f


def validate_candles(candles: list[dict]) -> list[dict]:
    """Validate and normalize a list of OHLCV candle dicts.

    Returns a list of cleaned candle dicts (value-preserved copies with
    OHLCV coerced to float and a ``duplicate_timestamp`` flag added when a
    timestamp repeats the previous one). Raises ``ValueError`` on
    structurally unusable input: missing keys, non-numeric OHLCV, NaN,
    OHLC sanity violations (high < low, high < body, low > body), negative
    volume, or strictly decreasing timestamps.

    Edge cases:
        - empty input -> []
        - single candle -> validated single-element list
        - duplicate timestamps -> permitted, flagged (not raised)
        - unsorted timestamps -> ValueError (not silently re-sorted)
    """
    if candles is None:
        return []
    if not isinstance(candles, list):
        raise ValueError(f"candles must be a list, got {type(candles).__name__}")

    cleaned: list[dict] = []
    prev_epoch: int | None = None

    for i, c in enumerate(candles):
        if not isinstance(c, dict):
            raise ValueError(f"candle[{i}] is not a dict: {type(c).__name__}")

        for key in _REQUIRED_KEYS:
            if key not in c:
                raise ValueError(f"candle[{i}] missing required key: {key!r}")

        o = _coerce_float(c["open"], "open")
        h = _coerce_float(c["high"], "high")
        l = _coerce_float(c["low"], "low")
        cl = _coerce_float(c["close"], "close")
        v = _coerce_float(c["volume"], "volume")

        if v < 0:
            raise ValueError(f"candle[{i}] volume is negative: {v}")

        # OHLC sanity: high >= low, high >= max(open,close), low <= min(open,close)
        body_high = o if o >= cl else cl
        body_low = o if o <= cl else cl
        if h < l:
            raise ValueError(f"candle[{i}] high < low: {h} < {l}")
        if h < body_high:
            raise ValueError(f"candle[{i}] high < body high: {h} < {body_high}")
        if l > body_low:
            raise ValueError(f"candle[{i}] low > body low: {l} > {body_low}")

        ts = c["timestamp"]
        epoch = _to_epoch(ts)  # validates format; raises on bad input

        # Monotonic non-decreasing: equal allowed (flagged), decreasing -> error.
        is_dup = False
        if prev_epoch is not None:
            if epoch < prev_epoch:
                raise ValueError(
                    f"candle[{i}] timestamp {epoch} < previous {prev_epoch} "
                    f"(unsorted; not silently re-sorted)"
                )
            if epoch == prev_epoch:
                is_dup = True

        out: dict = {
            "open": o,
            "high": h,
            "low": l,
            "close": cl,
            "volume": v,
            "timestamp": ts,  # value preserved (not mutated)
            "duplicate_timestamp": is_dup,
        }
        cleaned.append(out)
        prev_epoch = epoch

    return cleaned


def candle_metrics(candle: dict) -> dict:
    """Derive range, body, wicks, midpoint, direction for one candle.

    Returns:
        body        : abs(close - open)
        range       : high - low
        upper_wick  : high - max(open, close)
        lower_wick  : min(open, close) - low
        midpoint    : (high + low) / 2
        direction   : "bull" (close > open), "bear" (close < open), "doji" (==)

    Edge case: range == 0 (flat/doji) -> wicks 0, direction "doji".
    Assumes a validated candle (does not re-validate).
    """
    o = float(candle["open"])
    h = float(candle["high"])
    l = float(candle["low"])
    cl = float(candle["close"])

    rng = h - l
    body = abs(cl - o)
    body_high = o if o >= cl else cl
    body_low = o if o <= cl else cl
    upper_wick = h - body_high
    lower_wick = body_low - l
    midpoint = (h + l) / 2.0

    if cl > o:
        direction = "bull"
    elif cl < o:
        direction = "bear"
    else:
        direction = "doji"

    return {
        "body": body,
        "range": rng,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "midpoint": midpoint,
        "direction": direction,
    }
