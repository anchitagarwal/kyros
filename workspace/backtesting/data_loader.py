"""data_loader.py — materialize 1m NQ historical data into a canonical parquet file.

Three backends behind one interface:
  - csv:     read a local 1m CSV (offline TWS export), normalize, write parquet.
  - yfinance: download /NQ=F in 7-day 1m chunks, stitch, write parquet.
  - alpaca:  page Alpaca Markets v2 bars, write parquet.

All backends normalize to the CANONICAL PARQUET SCHEMA:
    timestamp : datetime64[ns, UTC]   (tz-aware, UTC, sorted ascending, unique)
    open      : float64
    high      : float64
    low       : float64
    close     : float64
    volume    : int64

Cache path: workspace/data/NQ_1m_{start}_{end}_{backend}.parquet
The backend segment is part of the cache key so that a csv-derived cache and
a yfinance/alpaca-derived cache for the same date range never alias to the
same file. Downloads only on cache miss; the second load() of the same
(start, end, backend) returns the cached parquet without re-reading the source.

Range semantics: ``start`` and ``end`` are ISO date strings (YYYY-MM-DD). The
range is INCLUSIVE of both bounds at date granularity — the full ``end`` day
is included. Internally the download loops treat the end as exclusive
(end_dt + 1 day) so a single-day request (start == end) still fetches that
day's bars; ``_normalize`` then filters to [start, end+1day).

No broker, no IBKR, no live market data, no order placement. The csv backend
never imports yfinance/alpaca/any broker client. yfinance/alpaca backends
exist for completeness but are mocked in tests (no real HTTP in CI).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

__all__ = ["DataLoader", "CANONICAL_COLUMNS"]

# Canonical parquet column order and dtypes.
CANONICAL_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# Default cache directory.
_CACHE_DIR = Path("workspace/data")


def _parse_date_range(start: str, end: str) -> tuple[datetime, datetime]:
    """Parse (start, end) ISO date strings to UTC datetimes.

    Returns (start_dt, end_exclusive) where end_exclusive = end + 1 day, so a
    single-day request (start == end) covers the full day. Both are tz-aware
    UTC at midnight.
    """
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    return start_dt, end_dt + timedelta(days=1)


class DataLoader:
    """Load 1m NQ data into a canonical parquet file, abstracting the backend.

    Args:
        backend: "yfinance" | "alpaca" | "csv". Defaults to the
            KYROS_DATA_BACKEND env var, or "yfinance" if unset.
    """

    def __init__(self, backend: str | None = None):
        if backend is None:
            backend = os.getenv("KYROS_DATA_BACKEND", "yfinance")
        if backend not in ("yfinance", "alpaca", "csv"):
            raise ValueError(
                f"unknown backend {backend!r}; choose from yfinance|alpaca|csv"
            )
        self.backend = backend

    # ── public API ──────────────────────────────────────────────────────────

    def load(self, start: str, end: str) -> Path:
        """Return the path to the cached parquet for [start, end].

        Downloads/reads the source only on cache miss. ``start`` and ``end``
        are ISO date strings (YYYY-MM-DD); the range is inclusive of both
        bounds at date granularity (the full ``end`` day is included).
        """
        cache_path = self._cache_path(start, end)
        if cache_path.exists():
            return cache_path

        if self.backend == "csv":
            df = self._load_csv(start, end)
        elif self.backend == "yfinance":
            df = self._load_yfinance(start, end)
        elif self.backend == "alpaca":
            df = self._load_alpaca(start, end)
        else:  # pragma: no cover — guarded by __init__
            raise ValueError(f"unknown backend {self.backend!r}")

        df = self._normalize(df, start, end)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        return cache_path

    # ── cache path ──────────────────────────────────────────────────────────

    def _cache_path(self, start: str, end: str) -> Path:
        """Cache key includes the backend so different backends never collide.

        e.g. NQ_1m_2024-06-10_2024-06-10_csv.parquet
        """
        return _CACHE_DIR / f"NQ_1m_{start}_{end}_{self.backend}.parquet"

    # ── normalization ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize(df, start: str, end: str):
        """Coerce a raw DataFrame to the canonical schema.

        - timestamp → UTC tz-aware datetime64.
        - sort ascending, drop duplicate timestamps.
        - filter to [start, end] inclusive (date granularity).
        - column order + dtypes per CANONICAL_COLUMNS.
        """
        import pandas as pd

        if len(df) == 0:
            # Empty frame — still coerce column order/dtypes, including a
            # tz-aware datetime64[ns, UTC] timestamp per the canonical schema.
            empty = pd.DataFrame(columns=CANONICAL_COLUMNS).astype(
                {
                    "open": "float64",
                    "high": "float64",
                    "low": "float64",
                    "close": "float64",
                    "volume": "int64",
                }
            )
            empty["timestamp"] = pd.to_datetime(empty["timestamp"], utc=True)
            return empty

        # Ensure timestamp is tz-aware UTC.
        ts = pd.to_datetime(df["timestamp"], utc=True)
        df = df.copy()
        df["timestamp"] = ts

        # Sort ascending and drop duplicate timestamps (keep first).
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

        # Filter to [start, end] inclusive at date granularity.
        start_dt = pd.to_datetime(start, utc=True)
        # end is inclusive: include the full end day.
        end_dt = pd.to_datetime(end, utc=True) + timedelta(days=1)
        mask = (df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)
        df = df.loc[mask].reset_index(drop=True)

        # Select + coerce canonical columns.
        df = df[CANONICAL_COLUMNS].copy()
        df["open"] = df["open"].astype("float64")
        df["high"] = df["high"].astype("float64")
        df["low"] = df["low"].astype("float64")
        df["close"] = df["close"].astype("float64")
        df["volume"] = df["volume"].fillna(0).astype("int64")
        return df

    # ── csv backend ─────────────────────────────────────────────────────────

    def _load_csv(self, start: str, end: str):
        """Read a local 1m CSV exported offline (e.g. TWS).

        Source columns: date (UTC string), open, high, low, close, volume,
        plus optional extras (e.g. contract) which are dropped. The ``date``
        column is parsed as UTC tz-aware and renamed to ``timestamp``.

        No network access — never imports a broker/IBKR client.
        """
        import pandas as pd

        csv_path = os.getenv("KYROS_CSV_PATH", "workspace/data/nq_1min_data.csv")
        df = pd.read_csv(csv_path)

        # Drop extra columns (e.g. contract) — keep only what we need.
        # The source uses ``date`` for the timestamp column.
        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})

        # Keep only canonical-relevant columns; drop extras like 'contract'.
        keep = [c for c in CANONICAL_COLUMNS if c in df.columns]
        df = df[keep]
        return df

    # ── yfinance backend ────────────────────────────────────────────────────

    def _load_yfinance(self, start: str, end: str):
        """Download /NQ=F in 7-day 1m chunks and stitch into one DataFrame.

        yfinance limits 1m history to the most recent 7 days (30 calendar days
        with a 7-day lookback window). We download in 7-day chunks and stitch.

        This path is NEVER exercised in CI (no network). Tests mock the
        download helper. The import of ``yfinance`` is local so the csv
        backend never pulls it in.
        """
        import pandas as pd

        start_dt, end_exclusive = _parse_date_range(start, end)

        frames = []
        chunk_start = start_dt
        while chunk_start < end_exclusive:
            chunk_end = min(chunk_start + timedelta(days=7), end_exclusive)
            chunk = self._download_chunk_yf(chunk_start, chunk_end)
            if chunk is not None and len(chunk) > 0:
                frames.append(chunk)
            chunk_start = chunk_end

        if not frames:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        df = pd.concat(frames, ignore_index=True)
        return df

    def _download_chunk_yf(self, chunk_start: datetime, chunk_end: datetime):
        """Download one 7-day chunk of /NQ=F 1m data via yfinance.

        Local import so the csv backend never imports yfinance.
        """
        import yfinance as yf

        ticker = yf.Ticker("NQ=F")
        hist = ticker.history(
            start=chunk_start.strftime("%Y-%m-%d"),
            end=chunk_end.strftime("%Y-%m-%d"),
            interval="1m",
        )
        if hist is None or len(hist) == 0:
            return None

        import pandas as pd

        # yfinance returns a tz-aware index (America/New_York for 1m).
        # Normalize to UTC.
        idx = hist.index.tz_convert("UTC") if hist.index.tz else hist.index.tz_localize("UTC")
        df = pd.DataFrame(
            {
                "timestamp": idx,
                "open": hist["Open"].values,
                "high": hist["High"].values,
                "low": hist["Low"].values,
                "close": hist["Close"].values,
                "volume": hist["Volume"].values,
            }
        )
        return df

    # ── alpaca backend ──────────────────────────────────────────────────────

    def _load_alpaca(self, start: str, end: str):
        """Alpaca backend — not implemented for NQ futures.

        Alpaca has no bars endpoint we can rely on for NQ index futures, and
        this path is never exercised offline. Fail loudly so a misconfigured
        KYROS_DATA_BACKEND surfaces immediately at load() time instead of
        silently returning wrong/empty data. Use the 'csv' backend.
        """
        raise NotImplementedError(
            "alpaca backend not implemented for NQ futures; use the 'csv' backend"
        )
