"""test_data_loader.py — DataLoader unit tests (all offline).

yfinance/alpaca backends are mocked (no real HTTP). The csv backend reads a
fixture CSV, normalizes date→timestamp (UTC), drops extra columns (e.g.
contract), and writes parquet with the correct schema.
"""

import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Ensure workspace/ is importable.
_WORKSPACE = Path(__file__).resolve().parent.parent.parent / "workspace"
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from backtesting.data_loader import DataLoader, CANONICAL_COLUMNS


# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_fixture_csv(path, rows, extra_col=True):
    """Write a 1m CSV with optional extra 'contract' column."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["date", "open", "high", "low", "close", "volume"]
        if extra_col:
            header.append("contract")
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _make_rows(n=10, start="2024-06-10 13:30:00"):
    """Build n 1m candle rows starting at ``start`` (UTC)."""
    base = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        ts = (base + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
        o = 20000.0 + i
        h = o + 5
        l = o - 5
        c = o + 2
        rows.append([ts, o, h, l, c, 100 + i, "NQU4"])
    return rows


# ── CSV backend ───────────────────────────────────────────────────────────────


def test_csv_backend_writes_correct_schema(tmp_path, monkeypatch):
    """CSV backend: parquet has correct columns and UTC tz-aware timestamps."""
    csv_path = tmp_path / "nq.csv"
    _write_fixture_csv(csv_path, _make_rows(10))
    monkeypatch.setenv("KYROS_CSV_PATH", str(csv_path))

    # Redirect cache dir to tmp_path so we don't pollute workspace/data.
    import backtesting.data_loader as dl_mod

    monkeypatch.setattr(dl_mod, "_CACHE_DIR", tmp_path / "cache")

    loader = DataLoader(backend="csv")
    out = loader.load("2024-06-10", "2024-06-10")

    assert out.exists()
    df = pd.read_parquet(out)
    assert list(df.columns) == CANONICAL_COLUMNS
    # timestamp is UTC tz-aware.
    assert df["timestamp"].dt.tz is not None
    assert str(df["timestamp"].dt.tz) == "UTC"
    # Sorted ascending.
    assert df["timestamp"].is_monotonic_increasing
    # No duplicates.
    assert df["timestamp"].is_unique
    # Extra 'contract' column dropped.
    assert "contract" not in df.columns
    # volume is int64.
    assert str(df["volume"].dtype) == "int64"


def test_csv_backend_drops_extra_columns(tmp_path, monkeypatch):
    """The 'contract' extra column is dropped from the parquet output."""
    csv_path = tmp_path / "nq.csv"
    _write_fixture_csv(csv_path, _make_rows(5), extra_col=True)
    monkeypatch.setenv("KYROS_CSV_PATH", str(csv_path))

    import backtesting.data_loader as dl_mod

    monkeypatch.setattr(dl_mod, "_CACHE_DIR", tmp_path / "cache")

    loader = DataLoader(backend="csv")
    out = loader.load("2024-06-10", "2024-06-10")
    df = pd.read_parquet(out)
    assert "contract" not in df.columns
    assert len(df) == 5


def test_csv_backend_filters_to_range(tmp_path, monkeypatch):
    """Rows outside [start, end] are absent from the output."""
    # Write rows spanning two days.
    rows = _make_rows(5, start="2024-06-10 13:30:00")
    rows += _make_rows(5, start="2024-06-11 13:30:00")
    csv_path = tmp_path / "nq.csv"
    _write_fixture_csv(csv_path, rows)
    monkeypatch.setenv("KYROS_CSV_PATH", str(csv_path))

    import backtesting.data_loader as dl_mod

    monkeypatch.setattr(dl_mod, "_CACHE_DIR", tmp_path / "cache")

    loader = DataLoader(backend="csv")
    # Request only 2024-06-10.
    out = loader.load("2024-06-10", "2024-06-10")
    df = pd.read_parquet(out)
    # All rows should be on 2024-06-10.
    assert all(ts.strftime("%Y-%m-%d") == "2024-06-10" for ts in df["timestamp"])
    assert len(df) == 5


def test_csv_backend_cache_hit(tmp_path, monkeypatch):
    """Second load() returns cached parquet without re-reading the source."""
    csv_path = tmp_path / "nq.csv"
    _write_fixture_csv(csv_path, _make_rows(10))
    monkeypatch.setenv("KYROS_CSV_PATH", str(csv_path))

    import backtesting.data_loader as dl_mod

    monkeypatch.setattr(dl_mod, "_CACHE_DIR", tmp_path / "cache")

    loader = DataLoader(backend="csv")

    # Patch _load_csv to count calls.
    call_count = {"n": 0}
    original = loader._load_csv

    def counting_load(start, end):
        call_count["n"] += 1
        return original(start, end)

    monkeypatch.setattr(loader, "_load_csv", counting_load)

    out1 = loader.load("2024-06-10", "2024-06-10")
    out2 = loader.load("2024-06-10", "2024-06-10")

    assert out1 == out2
    assert call_count["n"] == 1, "source should be read only once (cache hit)"


def test_csv_backend_normalizes_date_to_utc_timestamp(tmp_path, monkeypatch):
    """The 'date' column is renamed to 'timestamp' and parsed as UTC."""
    csv_path = tmp_path / "nq.csv"
    _write_fixture_csv(csv_path, _make_rows(3))
    monkeypatch.setenv("KYROS_CSV_PATH", str(csv_path))

    import backtesting.data_loader as dl_mod

    monkeypatch.setattr(dl_mod, "_CACHE_DIR", tmp_path / "cache")

    loader = DataLoader(backend="csv")
    out = loader.load("2024-06-10", "2024-06-10")
    df = pd.read_parquet(out)
    # First timestamp should be 2024-06-10 13:30:00 UTC.
    first = df["timestamp"].iloc[0]
    assert first == pd.Timestamp("2024-06-10 13:30:00", tz="UTC")


# ── yfinance backend (mocked) ─────────────────────────────────────────────────


def test_yfinance_backend_mocked_writes_parquet(tmp_path, monkeypatch):
    """yfinance backend with mocked download writes correct parquet schema."""
    import backtesting.data_loader as dl_mod

    monkeypatch.setattr(dl_mod, "_CACHE_DIR", tmp_path / "cache")

    # Build a fake chunk DataFrame in the yfinance shape.
    idx = pd.date_range("2024-06-10 13:30", periods=5, freq="1min", tz="UTC")
    fake_hist = pd.DataFrame(
        {
            "Open": [20000.0, 20002.0, 20004.0, 20006.0, 20008.0],
            "High": [20005.0, 20007.0, 20009.0, 20011.0, 20013.0],
            "Low": [19995.0, 19997.0, 19999.0, 20001.0, 20003.0],
            "Close": [20002.0, 20004.0, 20006.0, 20008.0, 20010.0],
            "Volume": [100, 101, 102, 103, 104],
        },
        index=idx,
    )

    def fake_download_chunk(self, chunk_start, chunk_end):
        return pd.DataFrame(
            {
                "timestamp": fake_hist.index,
                "open": fake_hist["Open"].values,
                "high": fake_hist["High"].values,
                "low": fake_hist["Low"].values,
                "close": fake_hist["Close"].values,
                "volume": fake_hist["Volume"].values,
            }
        )

    monkeypatch.setattr(dl_mod.DataLoader, "_download_chunk_yf", fake_download_chunk)

    loader = DataLoader(backend="yfinance")
    out = loader.load("2024-06-10", "2024-06-10")
    df = pd.read_parquet(out)
    assert list(df.columns) == CANONICAL_COLUMNS
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert len(df) == 5


def test_yfinance_backend_cache_hit(tmp_path, monkeypatch):
    """yfinance: second load() does not re-download (cache hit)."""
    import backtesting.data_loader as dl_mod

    monkeypatch.setattr(dl_mod, "_CACHE_DIR", tmp_path / "cache")

    idx = pd.date_range("2024-06-10 13:30", periods=3, freq="1min", tz="UTC")
    fake_hist = pd.DataFrame(
        {"Open": [1.0, 2.0, 3.0], "High": [1.5, 2.5, 3.5], "Low": [0.5, 1.5, 2.5],
         "Close": [1.2, 2.2, 3.2], "Volume": [10, 20, 30]},
        index=idx,
    )

    call_count = {"n": 0}

    def fake_download_chunk(self, chunk_start, chunk_end):
        call_count["n"] += 1
        return pd.DataFrame(
            {
                "timestamp": fake_hist.index,
                "open": fake_hist["Open"].values,
                "high": fake_hist["High"].values,
                "low": fake_hist["Low"].values,
                "close": fake_hist["Close"].values,
                "volume": fake_hist["Volume"].values,
            }
        )

    monkeypatch.setattr(dl_mod.DataLoader, "_download_chunk_yf", fake_download_chunk)

    loader = DataLoader(backend="yfinance")
    out1 = loader.load("2024-06-10", "2024-06-10")
    out2 = loader.load("2024-06-10", "2024-06-10")
    assert out1 == out2
    assert call_count["n"] == 1


# ── Import isolation ──────────────────────────────────────────────────────────


def test_csv_backend_does_not_import_yfinance(tmp_path, monkeypatch):
    """The csv backend never imports yfinance/alpaca/ib_insync."""
    csv_path = tmp_path / "nq.csv"
    _write_fixture_csv(csv_path, _make_rows(3))
    monkeypatch.setenv("KYROS_CSV_PATH", str(csv_path))

    import backtesting.data_loader as dl_mod

    monkeypatch.setattr(dl_mod, "_CACHE_DIR", tmp_path / "cache")

    # Remove any cached yfinance/alpaca/ib_insync modules and install a guard.
    guarded = {"yfinance", "alpaca", "ib_insync", "ibapi"}
    for mod in list(sys.modules):
        top = mod.split(".")[0]
        if top in guarded:
            del sys.modules[mod]

    import builtins

    real_import = builtins.__import__

    def guard_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if top in guarded:
            raise AssertionError(
                f"csv backend must not import {name!r} (broker/network client)"
            )
        return real_import(name, *args, **kwargs)

    builtins.__import__ = guard_import
    try:
        loader = DataLoader(backend="csv")
        out = loader.load("2024-06-10", "2024-06-10")
        assert out.exists()
    finally:
        builtins.__import__ = real_import


# ── Default backend ───────────────────────────────────────────────────────────


def test_default_backend_from_env(tmp_path, monkeypatch):
    """DataLoader reads KYROS_DATA_BACKEND from env."""
    monkeypatch.setenv("KYROS_DATA_BACKEND", "csv")
    loader = DataLoader()
    assert loader.backend == "csv"


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        DataLoader(backend="bogus")


# ── alpaca backend (fail-loud) ────────────────────────────────────────────────


def test_alpaca_backend_not_implemented(tmp_path, monkeypatch):
    """The alpaca backend fails loudly (NotImplementedError) at load() time."""
    import backtesting.data_loader as dl_mod

    monkeypatch.setattr(dl_mod, "_CACHE_DIR", tmp_path / "cache")
    loader = DataLoader(backend="alpaca")  # accepted name...
    with pytest.raises(NotImplementedError):
        loader.load("2024-06-10", "2024-06-10")  # ...but unusable, surfaced here


# ── _normalize empty frame ────────────────────────────────────────────────────


def test_normalize_empty_frame_has_utc_timestamp_dtype():
    """An empty result still carries a tz-aware datetime64[ns, UTC] timestamp."""
    empty = DataLoader._normalize(pd.DataFrame(), "2024-06-10", "2024-06-10")
    assert list(empty.columns) == CANONICAL_COLUMNS
    assert len(empty) == 0
    assert str(empty["timestamp"].dt.tz) == "UTC"
