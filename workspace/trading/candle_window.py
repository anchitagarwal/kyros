"""candle_window.py — bounded sliding window of candles per timeframe.

Maintains a ``deque`` per timeframe with a configurable ``maxlen``. The
trading loop calls ``update()`` with the candles returned by the source;
``to_list()`` returns oldest→newest for passing to Phase 1 detectors.

Window sizes (default, per the module layout spec):
    4h → 60, 1h → 100, 15m → 200, 5m → 300, 1m → 500
"""

from __future__ import annotations

from collections import deque

__all__ = ["CandleWindow", "DEFAULT_SIZES", "TIMEFRAMES"]

TIMEFRAMES = ("4h", "1h", "15m", "5m", "1m")

DEFAULT_SIZES: dict[str, int] = {
    "4h": 60,
    "1h": 100,
    "15m": 200,
    "5m": 300,
    "1m": 500,
}


class CandleWindow:
    """Per-timeframe bounded deque of candles.

    ``update()`` appends only the timeframes present in the incoming dict
    (no synthetic fills). ``to_list()`` returns a chronological copy (never
    the live deque). Unknown timeframe keys raise ``KeyError``.
    """

    def __init__(self, sizes: dict[str, int] | None = None):
        sizes = sizes if sizes is not None else dict(DEFAULT_SIZES)
        self.sizes: dict[str, int] = dict(sizes)
        self._data: dict[str, deque] = {
            tf: deque(maxlen=sizes[tf]) for tf in sizes
        }

    def update(self, candles: dict[str, dict]) -> None:
        """Append candles for the timeframes present in ``candles``."""
        for tf, candle in candles.items():
            if tf not in self._data:
                raise KeyError(f"unknown timeframe: {tf!r}")
            self._data[tf].append(candle)

    def to_list(self, timeframe: str) -> list[dict]:
        """Return oldest→newest candles for ``timeframe`` (a copy)."""
        if timeframe not in self._data:
            raise KeyError(f"unknown timeframe: {timeframe!r}")
        return list(self._data[timeframe])

    def is_warm(self, timeframe: str) -> bool:
        """True when the window for ``timeframe`` is at full capacity."""
        if timeframe not in self._data:
            raise KeyError(f"unknown timeframe: {timeframe!r}")
        return len(self._data[timeframe]) == self.sizes[timeframe]

    def __len__(self, timeframe: str) -> int:  # pragma: no cover - convenience
        return len(self._data[timeframe])
