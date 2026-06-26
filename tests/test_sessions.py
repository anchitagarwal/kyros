"""Tests for sessions.py — session classification and kill zones."""

from datetime import time, timezone, timedelta
from zoneinfo import ZoneInfo

import pytest

from detectors.sessions import detect_sessions, detect_kill_zones, detect_session_levels
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


# ── detect_session_levels ─────────────────────────────────────────────────────

def test_session_levels_empty():
    result = detect_session_levels([], tz="America/New_York")
    assert all(v is None for v in result.values())
    assert set(result) == {
        "midnight_open", "true_day_open", "london_open", "open_830", "open_930",
        "asia_high", "asia_low", "london_high", "london_low",
        "nyam_high", "nyam_low", "nylunch_high", "nylunch_low",
        "nypm_high", "nypm_low",
    }


def test_session_levels_tz_required():
    with pytest.raises(TypeError):
        detect_session_levels([mkc(10, 11, 9, 10, 1000)])


def test_session_levels_invalid_tz():
    with pytest.raises(ValueError, match="invalid timezone"):
        detect_session_levels([mkc(10, 11, 9, 10, 1000)], tz="Bad/Zone")


def test_session_levels_midnight_open():
    candles = [_candle_at(2024, 3, 4, 0, 0, price=100)]
    result = detect_session_levels(candles, tz="America/New_York")
    assert result["midnight_open"] == 100.0


def test_session_levels_true_day_open():
    candles = [_candle_at(2024, 3, 4, 18, 0, price=200)]
    result = detect_session_levels(candles, tz="America/New_York")
    assert result["true_day_open"] == 200.0


def test_session_levels_london_open():
    candles = [_candle_at(2024, 3, 4, 2, 0, price=150)]
    result = detect_session_levels(candles, tz="America/New_York")
    assert result["london_open"] == 150.0


def test_session_levels_open_830():
    candles = [_candle_at(2024, 3, 4, 8, 30, price=175)]
    result = detect_session_levels(candles, tz="America/New_York")
    assert result["open_830"] == 175.0


def test_session_levels_open_930():
    candles = [_candle_at(2024, 3, 4, 9, 30, price=180)]
    result = detect_session_levels(candles, tz="America/New_York")
    assert result["open_930"] == 180.0


def test_session_levels_asia_high_low():
    # Asia window 20:00-00:00; candle at 10:00 is outside.
    candles = [
        _candle_at(2024, 3, 4, 20, 0, price=100),  # high=101, low=99
        _candle_at(2024, 3, 4, 21, 0, price=120),  # high=121, low=119
        _candle_at(2024, 3, 4, 10, 0, price=200),  # outside
    ]
    result = detect_session_levels(candles, tz="America/New_York")
    assert result["asia_high"] == 121
    assert result["asia_low"] == 99


def test_session_levels_london_high_low():
    # London window 02:00-05:00; candle at 05:00 is exclusive end.
    candles = [
        _candle_at(2024, 3, 4, 2, 0, price=100),   # high=101, low=99
        _candle_at(2024, 3, 4, 4, 0, price=110),   # high=111, low=109
        _candle_at(2024, 3, 4, 5, 0, price=200),   # excluded
    ]
    result = detect_session_levels(candles, tz="America/New_York")
    assert result["london_high"] == 111
    assert result["london_low"] == 99


def test_session_levels_nyam_high_low():
    # NY AM window 07:00-10:00; candle at 10:00 is exclusive end.
    candles = [
        _candle_at(2024, 3, 4, 7, 0, price=100),   # high=101, low=99
        _candle_at(2024, 3, 4, 9, 0, price=110),   # high=111, low=109
        _candle_at(2024, 3, 4, 10, 0, price=200),  # excluded
    ]
    result = detect_session_levels(candles, tz="America/New_York")
    assert result["nyam_high"] == 111
    assert result["nyam_low"] == 99


def test_session_levels_nylunch_high_low():
    # NY lunch window 12:00-13:00; candle at 13:00 is exclusive end.
    candles = [
        _candle_at(2024, 3, 4, 12, 0, price=100),  # high=101, low=99
        _candle_at(2024, 3, 4, 12, 30, price=105), # high=106, low=104
        _candle_at(2024, 3, 4, 13, 0, price=200),  # excluded
    ]
    result = detect_session_levels(candles, tz="America/New_York")
    assert result["nylunch_high"] == 106
    assert result["nylunch_low"] == 99


def test_session_levels_nypm_high_low():
    # NY PM window 13:00-16:00; candle at 16:00 is exclusive end.
    candles = [
        _candle_at(2024, 3, 4, 13, 0, price=100),  # high=101, low=99
        _candle_at(2024, 3, 4, 15, 0, price=110),  # high=111, low=109
        _candle_at(2024, 3, 4, 16, 0, price=200),  # excluded
    ]
    result = detect_session_levels(candles, tz="America/New_York")
    assert result["nypm_high"] == 111
    assert result["nypm_low"] == 99


def test_session_levels_missing_returns_none():
    # Only nylunch candles — all other levels should be None.
    candles = [_candle_at(2024, 3, 4, 12, 0, price=100)]
    result = detect_session_levels(candles, tz="America/New_York")
    assert result["midnight_open"] is None
    assert result["asia_high"] is None
    assert result["nypm_high"] is None
    assert result["nylunch_high"] == 101
    assert result["nylunch_low"] == 99


def test_session_levels_open_not_found_returns_none():
    # Candle not at any target open time.
    candles = [_candle_at(2024, 3, 4, 10, 0, price=100)]
    result = detect_session_levels(candles, tz="America/New_York")
    assert result["midnight_open"] is None
    assert result["open_930"] is None


def test_session_levels_first_candle_wins_for_opens():
    # Two candles at 09:30 on different days — first one's open is returned.
    candles = [
        _candle_at(2024, 3, 4, 9, 30, price=100),
        _candle_at(2024, 3, 5, 9, 30, price=200),
    ]
    result = detect_session_levels(candles, tz="America/New_York")
    assert result["open_930"] == 100.0


def test_session_levels_dst_spring_forward():
    # On DST spring-forward day (2024-03-10) 09:30 still resolves correctly.
    candles = [_candle_at(2024, 3, 10, 9, 30, price=100)]
    result = detect_session_levels(candles, tz="America/New_York")
    assert result["open_930"] == 100.0
