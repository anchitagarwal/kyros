"""Tests for sessions.py — session classification and kill zones."""

from datetime import time, timezone, timedelta
from zoneinfo import ZoneInfo

import pytest

from detectors.sessions import detect_sessions, detect_kill_zones
from tests._fixtures import mkc, epoch_dt


def _candle_at(year, month, day, hour, minute, price=100.0, tz="America/New_York"):
    """Build a candle whose timestamp is the given wall-clock time in `tz`."""
    from datetime import datetime
    zone = ZoneInfo(tz)
    dt = datetime(year, month, day, hour, minute, tzinfo=zone)
    ts = int(dt.timestamp())
    return mkc(price, price + 1, price - 1, price, ts)


# ── detect_sessions ───────────────────────────────────────────────────────────

def test_sessions_empty():
    assert detect_sessions([], tz="America/New_York") == []


def test_sessions_tz_required():
    # Calling without tz should raise TypeError (no default).
    with pytest.raises(TypeError):
        detect_sessions([mkc(10, 11, 9, 10, 1000)])


def test_sessions_assignment_and_aggregation():
    # Two candles in NY AM session (07:00-10:00), one outside.
    candles = [
        _candle_at(2024, 3, 4, 8, 0, price=100),   # ny_am
        _candle_at(2024, 3, 4, 8, 30, price=110),  # ny_am
        _candle_at(2024, 3, 4, 12, 0, price=105),  # gap (no session in defaults)
    ]
    sessions = detect_sessions(candles, tz="America/New_York")
    am = [s for s in sessions if s["session_name"] == "ny_am"]
    assert len(am) == 1
    assert am[0]["start_index"] == 0
    assert am[0]["end_index"] == 1
    assert am[0]["session_high"] == 111  # max high of the two
    assert am[0]["session_low"] == 99    # min low


def test_sessions_asian_range_high_low():
    # Asian session 20:00-00:00 (wraps midnight). Two candles.
    candles = [
        _candle_at(2024, 3, 4, 20, 0, price=100),
        _candle_at(2024, 3, 4, 21, 0, price=120),
    ]
    sessions = detect_sessions(candles, tz="America/New_York")
    asian = [s for s in sessions if s["session_name"] == "asian"]
    assert len(asian) == 1
    assert asian[0]["session_high"] == 121
    assert asian[0]["session_low"] == 99


def test_sessions_asian_midnight_wrap():
    # The Asian session wraps midnight (20:00-00:00). A candle at 23:00 is in
    # the Asian session; a candle at exactly 00:00 is the exclusive end and is
    # NOT in the Asian session (consistent with the exclusive-end convention).
    candles = [
        _candle_at(2024, 3, 4, 23, 0, price=100),   # in asian (>= 20:00)
        _candle_at(2024, 3, 5, 0, 0, price=110),    # 00:00 -> exclusive end, not asian
    ]
    sessions = detect_sessions(candles, tz="America/New_York")
    asian = [s for s in sessions if s["session_name"] == "asian"]
    assert len(asian) == 1
    assert asian[0]["start_index"] == 0
    assert asian[0]["end_index"] == 0
    # The 00:00 candle is not assigned to asian.
    assert sessions[0]["session_name"] == "asian"


def test_sessions_gap_candles_skipped():
    # Candle at 12:00 (lunch gap) is not in any default session.
    candles = [_candle_at(2024, 3, 4, 12, 0, price=100)]
    sessions = detect_sessions(candles, tz="America/New_York")
    assert sessions == []


def test_sessions_dst_spring_forward():
    # 2024-03-10 is DST spring-forward in America/New_York (2:00 -> 3:00).
    # A candle at 03:00 on that day should still be classifiable (london 02-05
    # but 02:00 doesn't exist; 03:00 is in london window 02:00-05:00).
    candles = [_candle_at(2024, 3, 10, 3, 0, price=100)]
    sessions = detect_sessions(candles, tz="America/New_York")
    # 03:00 falls in london (02:00-05:00).
    assert any(s["session_name"] == "london" for s in sessions)


def test_sessions_invalid_tz_raises():
    with pytest.raises(ValueError, match="invalid timezone"):
        detect_sessions([mkc(10, 11, 9, 10, 1000)], tz="Not/A/Zone")


def test_sessions_custom_windows():
    # Override with a single custom window.
    candles = [_candle_at(2024, 3, 4, 6, 0, price=100)]
    custom = [("early", time(5, 0), time(7, 0))]
    sessions = detect_sessions(candles, tz="America/New_York",
                               session_windows=custom)
    assert len(sessions) == 1
    assert sessions[0]["session_name"] == "early"


# ── detect_kill_zones ─────────────────────────────────────────────────────────

def test_kill_zones_empty():
    assert detect_kill_zones([], tz="America/New_York") == []


def test_kill_zones_tz_required():
    with pytest.raises(TypeError):
        detect_kill_zones([mkc(10, 11, 9, 10, 1000)])


def test_kill_zones_ny_am_window():
    # NY AM kill zone 09:30-11:00.
    candles = [
        _candle_at(2024, 3, 4, 9, 30, price=100),
        _candle_at(2024, 3, 4, 10, 0, price=101),
        _candle_at(2024, 3, 4, 11, 30, price=102),  # outside
    ]
    kz = detect_kill_zones(candles, tz="America/New_York")
    am = [k for k in kz if k["window_name"] == "ny_am_kz"]
    assert len(am) == 1
    assert am[0]["start_index"] == 0
    assert am[0]["end_index"] == 1


def test_kill_zones_silver_bullet_ny_am():
    # Silver Bullet NY AM 10:00-11:00.
    candles = [
        _candle_at(2024, 3, 4, 10, 0, price=100),
        _candle_at(2024, 3, 4, 10, 30, price=101),
    ]
    kz = detect_kill_zones(candles, tz="America/New_York")
    sb = [k for k in kz if k["type"] == "silver_bullet" and k["window_name"] == "ny_am_sb"]
    assert len(sb) == 1
    assert sb[0]["start_index"] == 0
    assert sb[0]["end_index"] == 1


def test_kill_zones_no_candles_in_window_skipped():
    # Candle at 12:00 (lunch) — not in any default kill zone or silver bullet.
    candles = [_candle_at(2024, 3, 4, 12, 0, price=100)]
    kz = detect_kill_zones(candles, tz="America/New_York")
    assert kz == []


def test_kill_zones_dst_day():
    # On DST spring-forward day, 10:00 still exists and is in ny_am_kz/sb.
    candles = [_candle_at(2024, 3, 10, 10, 0, price=100)]
    kz = detect_kill_zones(candles, tz="America/New_York")
    assert any(k["window_name"] == "ny_am_kz" for k in kz)
    assert any(k["window_name"] == "ny_am_sb" for k in kz)
