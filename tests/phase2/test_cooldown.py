"""test_cooldown.py — tiered cooldown behaviour."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from trading.cooldown import CooldownState
from trading.alert import AlertPayload

_NY = ZoneInfo("America/New_York")


def _snap(ts, killzone=None):
    """Minimal snapshot stub with timestamp + current_killzone."""
    class _S:
        pass
    s = _S()
    s.timestamp = ts
    s.current_killzone = killzone
    return s


def test_no_prior_alert_never_cooling():
    cd = CooldownState()
    snap = _snap(datetime(2026, 6, 15, 10, 0, tzinfo=_NY), "ny_am_kz")
    assert cd.is_cooling_down(snap) is False


def test_no_trade_blocks_for_5_min():
    cd = CooldownState()
    t0 = datetime(2026, 6, 15, 10, 0, tzinfo=_NY)
    cd.update(AlertPayload(bias="no_trade"), _snap(t0, "ny_am_kz"))
    # +4 min → still cooling.
    assert cd.is_cooling_down(_snap(t0 + timedelta(minutes=4), "ny_am_kz")) is True
    # +5 min → clear.
    assert cd.is_cooling_down(_snap(t0 + timedelta(minutes=5), "ny_am_kz")) is False
    # +6 min → clear.
    assert cd.is_cooling_down(_snap(t0 + timedelta(minutes=6), "ny_am_kz")) is False


def test_long_blocks_same_killzone():
    cd = CooldownState()
    t0 = datetime(2026, 6, 15, 10, 0, tzinfo=_NY)
    cd.update(AlertPayload(bias="long"), _snap(t0, "ny_am_kz"))
    # +30 min, same killzone → still blocked.
    assert cd.is_cooling_down(_snap(t0 + timedelta(minutes=30), "ny_am_kz")) is True


def test_long_allows_when_killzone_changes():
    cd = CooldownState()
    t0 = datetime(2026, 6, 15, 10, 0, tzinfo=_NY)
    cd.update(AlertPayload(bias="long"), _snap(t0, "ny_am_kz"))
    # Different killzone → allowed.
    assert cd.is_cooling_down(_snap(t0 + timedelta(minutes=5), "ny_pm_kz")) is False


def test_long_allows_when_killzone_becomes_none():
    cd = CooldownState()
    t0 = datetime(2026, 6, 15, 10, 0, tzinfo=_NY)
    cd.update(AlertPayload(bias="long"), _snap(t0, "ny_am_kz"))
    # Killzone transitions to None → allowed.
    assert cd.is_cooling_down(_snap(t0 + timedelta(minutes=5), None)) is False


def test_short_blocks_same_killzone():
    cd = CooldownState()
    t0 = datetime(2026, 6, 15, 14, 0, tzinfo=_NY)
    cd.update(AlertPayload(bias="short"), _snap(t0, "ny_pm_kz"))
    assert cd.is_cooling_down(_snap(t0 + timedelta(minutes=20), "ny_pm_kz")) is True


def test_20min_gap_same_killzone_still_blocked():
    """No flat time threshold for directional alerts — same killzone stays blocked."""
    cd = CooldownState()
    t0 = datetime(2026, 6, 15, 9, 35, tzinfo=_NY)
    cd.update(AlertPayload(bias="long"), _snap(t0, "ny_am_kz"))
    # 20 minutes later, still in ny_am_kz (09:30-11:00) → blocked.
    assert cd.is_cooling_down(_snap(t0 + timedelta(minutes=20), "ny_am_kz")) is True


def test_update_records_all_fields():
    cd = CooldownState()
    t0 = datetime(2026, 6, 15, 10, 0, tzinfo=_NY)
    snap = _snap(t0, "ny_am_kz")
    cd.update(AlertPayload(bias="short"), snap)
    assert cd.last_alert_time == t0
    assert cd.last_alert_bias == "short"
    assert cd.last_alert_killzone == "ny_am_kz"
