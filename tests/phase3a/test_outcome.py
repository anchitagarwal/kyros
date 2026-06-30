"""test_outcome.py — OutcomeSimulator unit tests.

All tests use hand-built candle sequences with known outcomes. The simulator
is a pure function of (alert, candles): no I/O, no clock, no LLM.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

# Ensure workspace/ is importable.
_WORKSPACE = Path(__file__).resolve().parent.parent.parent / "workspace"
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from trading.alert import AlertPayload
from backtesting.outcome import OutcomeSimulator, TradeOutcome


# ── Helpers ───────────────────────────────────────────────────────────────────


def _candle(ts, o, h, l, c):
    """Build a candle dict with an ISO timestamp."""
    return {"open": o, "high": h, "low": l, "close": c, "volume": 100,
            "timestamp": ts.isoformat() if isinstance(ts, datetime) else ts}


def _long_alert(entry_zone=(100.0, 102.0), stop=95.0, target=115.0,
                valid_until="", bias="long"):
    """A long alert with entry_mid=101, risk=6, target=115 (rr=14/6≈2.33)."""
    return AlertPayload(
        bias=bias, model="2022", conviction=70,
        entry_zone=entry_zone, stop=stop, target=target,
        dol={"level": 115.0, "type": "bsl", "timeframe": "1h"},
        risk_reward=2.33, rationale="long", killzone="ny_am_kz",
        valid_until=valid_until,
    )


def _short_alert(entry_zone=(100.0, 102.0), stop=110.0, target=85.0,
                 valid_until="", bias="short"):
    """A short alert with entry_mid=101, risk=9, target=85 (rr=16/9≈1.78)."""
    return AlertPayload(
        bias=bias, model="2022", conviction=70,
        entry_zone=entry_zone, stop=stop, target=target,
        dol={"level": 85.0, "type": "ssl", "timeframe": "1h"},
        risk_reward=1.78, rationale="short", killzone="ny_am_kz",
        valid_until=valid_until,
    )


_BASE = datetime(2026, 6, 15, 10, 0, tzinfo=timezone.utc)


def _subsequent(n, start=None):
    """Build n placeholder subsequent candles (timestamps after _BASE)."""
    s = start or _BASE + timedelta(minutes=1)
    return [_candle(s + timedelta(minutes=i), 100, 102, 98, 101) for i in range(n)]


# ── Long win ──────────────────────────────────────────────────────────────────


def test_long_win():
    """Long: subsequent candles reach target before stop → win, actual_rr > 0."""
    alert = _long_alert()
    # Candle 0: fills (low=98 <= 102, high=103 >= 100). fill_price=101.
    # Candle 1: high=116 >= target 115 → win.
    candles = [
        _candle(_BASE + timedelta(minutes=1), 100, 103, 98, 101),   # fill
        _candle(_BASE + timedelta(minutes=2), 101, 116, 100, 115),  # win
    ]
    sim = OutcomeSimulator()
    out = sim.simulate(alert, candles)
    assert out.result == "win"
    assert out.fill_price == 101.0
    assert out.exit_price == 115.0
    assert out.actual_rr > 0
    # actual_rr = (115 - 101) / |101 - 95| = 14/6
    assert out.actual_rr == pytest.approx(14 / 6)
    assert out.candles_to_fill == 1
    assert out.candles_to_resolution == 2


def test_long_loss():
    """Long: subsequent candles reach stop before target → loss, actual_rr < 0."""
    alert = _long_alert()
    # Candle 0: fills. Candle 1: low=94 <= stop 95 → loss.
    candles = [
        _candle(_BASE + timedelta(minutes=1), 100, 103, 98, 101),  # fill
        _candle(_BASE + timedelta(minutes=2), 101, 102, 94, 95),   # loss
    ]
    sim = OutcomeSimulator()
    out = sim.simulate(alert, candles)
    assert out.result == "loss"
    assert out.fill_price == 101.0
    assert out.exit_price == 95.0
    assert out.actual_rr < 0
    # actual_rr = (95 - 101) / |101 - 95| = -6/6 = -1.0
    assert out.actual_rr == pytest.approx(-1.0)


# ── Short win / loss ──────────────────────────────────────────────────────────


def test_short_win():
    """Short: subsequent candles reach target before stop → win."""
    alert = _short_alert()
    # Candle 0: fills (low=98 <= 102, high=103 >= 100). fill_price=101.
    # Candle 1: low=84 <= target 85 → win.
    candles = [
        _candle(_BASE + timedelta(minutes=1), 102, 103, 98, 100),  # fill
        _candle(_BASE + timedelta(minutes=2), 100, 101, 84, 85),   # win
    ]
    sim = OutcomeSimulator()
    out = sim.simulate(alert, candles)
    assert out.result == "win"
    assert out.fill_price == 101.0
    assert out.exit_price == 85.0
    assert out.actual_rr > 0
    # actual_rr = (101 - 85) / |101 - 110| = 16/9
    assert out.actual_rr == pytest.approx(16 / 9)


def test_short_loss():
    """Short: subsequent candles reach stop before target → loss."""
    alert = _short_alert()
    # Candle 0: fills. Candle 1: high=111 >= stop 110 → loss.
    candles = [
        _candle(_BASE + timedelta(minutes=1), 102, 103, 98, 100),  # fill
        _candle(_BASE + timedelta(minutes=2), 100, 111, 99, 110),  # loss
    ]
    sim = OutcomeSimulator()
    out = sim.simulate(alert, candles)
    assert out.result == "loss"
    assert out.exit_price == 110.0
    assert out.actual_rr < 0
    # actual_rr = (101 - 110) / |101 - 110| = -9/9 = -1.0
    assert out.actual_rr == pytest.approx(-1.0)


# ── Realized gap-through-stop losses ──────────────────────────────────────────


def test_long_loss_gap_through_stop():
    """Long loss: a candle that gaps below the stop fills at the open (< -1R)."""
    alert = _long_alert()  # entry_mid=101, stop=95, risk=6
    candles = [
        _candle(_BASE + timedelta(minutes=1), 100, 103, 98, 101),  # fill
        _candle(_BASE + timedelta(minutes=2), 90, 92, 88, 89),     # gap below stop 95
    ]
    sim = OutcomeSimulator()
    out = sim.simulate(alert, candles)
    assert out.result == "loss"
    # Realized exit is the gap open (90), not the nominal stop (95).
    assert out.exit_price == 90.0
    # actual_rr = (90 - 101) / 6 = -11/6 ≈ -1.83
    assert out.actual_rr == pytest.approx(-11 / 6)
    assert out.actual_rr < -1.0


def test_short_loss_gap_through_stop():
    """Short loss: a candle that gaps above the stop fills at the open (< -1R)."""
    alert = _short_alert()  # entry_mid=101, stop=110, risk=9
    candles = [
        _candle(_BASE + timedelta(minutes=1), 102, 103, 98, 100),    # fill
        _candle(_BASE + timedelta(minutes=2), 115, 117, 113, 116),   # gap above stop 110
    ]
    sim = OutcomeSimulator()
    out = sim.simulate(alert, candles)
    assert out.result == "loss"
    # Realized exit is the gap open (115), not the nominal stop (110).
    assert out.exit_price == 115.0
    # actual_rr = (101 - 115) / 9 = -14/9 ≈ -1.56
    assert out.actual_rr == pytest.approx(-14 / 9)
    assert out.actual_rr < -1.0


# ── Ambiguous (same candle hits both) ─────────────────────────────────────────


def test_ambiguous_same_candle_is_loss():
    """A single post-fill candle straddling both stop and target → loss."""
    alert = _long_alert(entry_zone=(100.0, 102.0), stop=95.0, target=115.0)
    # Candle 0: fills. Candle 1: high=116 >= 115 AND low=94 <= 95 → loss.
    candles = [
        _candle(_BASE + timedelta(minutes=1), 100, 103, 98, 101),  # fill
        _candle(_BASE + timedelta(minutes=2), 101, 116, 94, 100),  # ambiguous
    ]
    sim = OutcomeSimulator()
    out = sim.simulate(alert, candles)
    assert out.result == "loss"
    assert out.exit_price == 95.0
    assert out.actual_rr < 0


# ── Fill candle does not resolve ──────────────────────────────────────────────


def test_fill_candle_does_not_resolve():
    """The fill candle's high/low must NOT be used for resolution.

    The fill candle fills (but does NOT reach target/stop — that would cancel
    pre-fill). Resolution must wait for the next candle; here the next candle
    also has no hit, so the trade is still unresolved (expired) — proving the
    fill candle was not used to resolve.
    """
    alert = _long_alert(entry_zone=(100.0, 102.0), stop=95.0, target=115.0)
    # Candle 0: fills (low=98<=102, high=104>=100); high=104 < target 115 and
    # low=98 > stop 95 → no pre-fill cancel. Resolution starts at candle 1,
    # which does not hit target or stop → expired.
    candles = [
        _candle(_BASE + timedelta(minutes=1), 100, 104, 98, 103),  # fill, no target/stop
        _candle(_BASE + timedelta(minutes=2), 103, 110, 100, 108),  # no hit
    ]
    sim = OutcomeSimulator()
    out = sim.simulate(alert, candles)
    assert out.result == "expired"
    assert out.fill_price == 101.0


# ── No fill ───────────────────────────────────────────────────────────────────


def test_no_fill_before_valid_until():
    """No fill before valid_until → no_fill, actual_rr=None."""
    vu = _BASE + timedelta(minutes=5)
    alert = _long_alert(valid_until=vu.isoformat())
    # Candles never enter entry_zone (all above 102) and never reach target
    # 115 (highs stay <= 110, so no pre-fill cancel). The 5th candle is
    # at/after valid_until → no_fill.
    candles = [
        _candle(_BASE + timedelta(minutes=1), 106, 108, 104, 107),
        _candle(_BASE + timedelta(minutes=2), 107, 109, 105, 108),
        _candle(_BASE + timedelta(minutes=3), 108, 110, 106, 109),
        _candle(_BASE + timedelta(minutes=4), 107, 109, 105, 108),
        _candle(_BASE + timedelta(minutes=5), 108, 110, 106, 109),  # >= valid_until
    ]
    sim = OutcomeSimulator()
    out = sim.simulate(alert, candles)
    assert out.result == "no_fill"
    assert out.actual_rr is None
    assert out.fill_price is None
    assert out.candles_to_fill is None


# ── Pre-fill cancel (target/stop hit before entry) ─────────────────────────────


def test_long_cancel_target_before_entry():
    """Long: price reaches target before the entry zone fills → cancelled."""
    alert = _long_alert(entry_zone=(100.0, 102.0), stop=95.0, target=115.0)
    # Candle 0 never enters the zone (low=108 > 102) but high=116 >= target 115
    # → the targeted move happened without us → cancelled (no fill).
    candles = [
        _candle(_BASE + timedelta(minutes=1), 110, 116, 108, 114),
        _candle(_BASE + timedelta(minutes=2), 101, 101, 100, 100),  # would have filled
    ]
    out = OutcomeSimulator().simulate(alert, candles)
    assert out.result == "cancelled"
    assert out.fill_price is None
    assert out.actual_rr is None
    assert out.candles_to_fill is None


def test_short_cancel_target_before_entry():
    """Short: price reaches target (below) before the entry zone fills → cancelled."""
    alert = _short_alert(entry_zone=(100.0, 102.0), stop=110.0, target=85.0)
    # Candle 0 never enters the zone (high=92 < 100) but low=84 <= target 85
    # → cancelled.
    candles = [
        _candle(_BASE + timedelta(minutes=1), 90, 92, 84, 86),
        _candle(_BASE + timedelta(minutes=2), 101, 102, 100, 101),  # would have filled
    ]
    out = OutcomeSimulator().simulate(alert, candles)
    assert out.result == "cancelled"
    assert out.fill_price is None


def test_long_cancel_stop_before_entry():
    """Long: price breaches the stop before filling → cancelled (invalidated)."""
    alert = _long_alert(entry_zone=(100.0, 102.0), stop=95.0, target=115.0)
    # Candle 0 gaps below: low=94 <= stop 95 → invalidation level traded
    # through before fill → cancelled. (It also overlaps the zone, but the
    # conservative cancel check precedes the fill check.)
    candles = [
        _candle(_BASE + timedelta(minutes=1), 101, 101, 94, 96),
        _candle(_BASE + timedelta(minutes=2), 101, 116, 100, 115),
    ]
    out = OutcomeSimulator().simulate(alert, candles)
    assert out.result == "cancelled"


def test_cancel_same_candle_target_and_entry_is_conservative():
    """Target tagged AND entry zone overlapped in the same bar → cancelled.

    Conservative tie-break: we cannot assume we filled before the run to
    target, so the same-bar overlap cancels rather than fills.
    """
    alert = _long_alert(entry_zone=(100.0, 102.0), stop=95.0, target=115.0)
    # Candle 0: low=100 (fills) AND high=116 >= target 115 → cancelled, not win.
    candles = [
        _candle(_BASE + timedelta(minutes=1), 101, 116, 100, 114),
    ]
    out = OutcomeSimulator().simulate(alert, candles)
    assert out.result == "cancelled"
    assert out.fill_price is None


def test_no_cancel_on_normal_fill():
    """A normal pull-back fill (no target/stop touch) is unaffected by cancel."""
    alert = _long_alert(entry_zone=(100.0, 102.0), stop=95.0, target=115.0)
    # Candle 0 fills at 101, no target/stop touch; candle 1 reaches target → win.
    candles = [
        _candle(_BASE + timedelta(minutes=1), 103, 104, 100, 101),
        _candle(_BASE + timedelta(minutes=2), 101, 116, 100, 115),
    ]
    out = OutcomeSimulator().simulate(alert, candles)
    assert out.result == "win"
    assert out.fill_price == 101.0


def test_no_fill_ran_out_of_candles():
    """No fill and no valid_until → no_fill (ran out of data)."""
    alert = _long_alert(valid_until="")
    candles = [
        _candle(_BASE + timedelta(minutes=1), 110, 112, 108, 111),
        _candle(_BASE + timedelta(minutes=2), 111, 113, 109, 112),
    ]
    sim = OutcomeSimulator()
    out = sim.simulate(alert, candles)
    assert out.result == "no_fill"
    assert out.actual_rr is None


# ── Naive valid_until (timezone robustness) ───────────────────────────────────


def test_naive_valid_until_does_not_crash():
    """A naive valid_until must not crash against tz-aware candle timestamps.

    Production: candles arrive tz-aware ET and the LLM-produced valid_until may
    be a naive ISO string. The simulator coerces naive → ET before comparing,
    so the comparison never raises "offset-naive vs offset-aware".
    """
    ny = ZoneInfo("America/New_York")
    base_et = datetime(2026, 6, 15, 10, 0, tzinfo=ny)
    # Naive valid_until at 10:03 ET (no offset in the string).
    vu_naive = datetime(2026, 6, 15, 10, 3).isoformat()  # "2026-06-15T10:03:00"
    alert = _long_alert(valid_until=vu_naive)
    candles = [
        _candle(base_et + timedelta(minutes=1), 100, 103, 98, 101),   # fill (10:01 ET)
        _candle(base_et + timedelta(minutes=2), 101, 105, 99, 103),   # no hit (10:02 ET)
        _candle(base_et + timedelta(minutes=3), 103, 106, 100, 104),  # 10:03 ET ≥ valid_until
    ]
    sim = OutcomeSimulator()
    out = sim.simulate(alert, candles)  # must not raise
    assert out.result == "expired"
    assert out.fill_price == 101.0


# ── Expired ───────────────────────────────────────────────────────────────────


def test_filled_but_unresolved_before_valid_until():
    """Filled but neither stop nor target before valid_until → expired."""
    vu = _BASE + timedelta(minutes=3)
    alert = _long_alert(valid_until=vu.isoformat())
    # Candle 0: fills. Candle 1: no hit. Candle 2: at valid_until → expired.
    candles = [
        _candle(_BASE + timedelta(minutes=1), 100, 103, 98, 101),  # fill
        _candle(_BASE + timedelta(minutes=2), 101, 105, 99, 103),  # no hit
        _candle(_BASE + timedelta(minutes=3), 103, 106, 100, 104),  # >= valid_until
    ]
    sim = OutcomeSimulator()
    out = sim.simulate(alert, candles)
    assert out.result == "expired"
    assert out.fill_price == 101.0
    assert out.actual_rr is None
    assert out.candles_to_fill == 1
    assert out.candles_to_resolution is None


def test_filled_but_unresolved_no_valid_until():
    """Filled but ran out of candles with no valid_until → expired."""
    alert = _long_alert(valid_until="")
    # Candle 0: fills. Candle 1: no hit. No more candles.
    candles = [
        _candle(_BASE + timedelta(minutes=1), 100, 103, 98, 101),  # fill
        _candle(_BASE + timedelta(minutes=2), 101, 105, 99, 103),  # no hit
    ]
    sim = OutcomeSimulator()
    out = sim.simulate(alert, candles)
    assert out.result == "expired"
    assert out.fill_price == 101.0
    assert out.actual_rr is None


# ── no_trade ──────────────────────────────────────────────────────────────────


def test_no_trade_returns_immediately():
    """A no_trade alert returns immediately without iterating candles."""
    alert = AlertPayload(bias="no_trade", no_trade_reason="rr_below_1")
    candles = _subsequent(5)
    sim = OutcomeSimulator()
    out = sim.simulate(alert, candles)
    assert out.result == "no_trade"
    assert out.actual_rr is None
    assert out.fill_price is None
    assert out.candles_to_fill is None
    assert out.candles_to_resolution is None


def test_no_trade_with_empty_candles():
    """no_trade with an empty candle list still returns no_trade."""
    alert = AlertPayload(bias="no_trade")
    sim = OutcomeSimulator()
    out = sim.simulate(alert, [])
    assert out.result == "no_trade"


# ── CRITICAL: alert candle as subsequent ──────────────────────────────────────


def test_alert_candle_as_subsequent_gives_wrong_outcome():
    """Passing the alert candle as the first subsequent candle is WRONG.

    This test documents WHY the caller must not include the alert candle in
    subsequent_candles. The alert candle's high/low are known at decision
    time, but using them for fill detection leaks the alert bar into the
    outcome — a lookahead bias that shifts the fill index and can flip an
    EXPIRED trade into a WIN.

    Setup (entry_zone 100-102, stop 95, target 115):
      - alert candle: enters entry zone (fills) but does NOT reach target.
      - candle A:     enters entry zone (fills) AND reaches target.
      - candle B:     no hit.

    CORRECT (alert candle excluded, subsequent = [A, B]):
      A is the first post-alert candle and reaches target (116 >= 115) before
      any fill → pre-fill invalidation → CANCELLED (the targeted move happened
      without us).

    WRONG (alert candle prepended, subsequent = [alert, A, B]):
      alert fills (index 0; high=110 < target so no pre-fill cancel);
      resolution starts at A (index 1); A reaches target → WIN.

    The two outcomes DIVERGE (cancelled vs win) because the alert candle's
    range was (incorrectly) used for fill detection, shifting the resolution
    window by one bar. The caller MUST exclude the alert candle.
    """
    alert = _long_alert(entry_zone=(100.0, 102.0), stop=95.0, target=115.0)

    # alert candle: low=99 <= 102, high=110 >= 100 → fills; high=110 < 115 (no target).
    alert_candle = _candle(_BASE, 100, 110, 99, 105)
    # candle A: low=100 <= 102, high=116 >= 100 → fills; high=116 >= 115 → target.
    candle_a = _candle(_BASE + timedelta(minutes=1), 105, 116, 100, 114)
    # candle B: high=114 < 115, low=100 > 95 → no hit.
    candle_b = _candle(_BASE + timedelta(minutes=2), 114, 114, 100, 110)

    correct = [candle_a, candle_b]
    wrong = [alert_candle, candle_a, candle_b]

    sim = OutcomeSimulator()
    correct_out = sim.simulate(alert, correct)
    wrong_out = sim.simulate(alert, wrong)

    # The correct outcome is cancelled (A reaches target before any fill).
    assert correct_out.result == "cancelled", (
        f"expected cancelled, got {correct_out.result}"
    )
    # The WRONG outcome is a win (alert fills, A reaches target).
    assert wrong_out.result == "win", (
        f"expected win (lookahead), got {wrong_out.result}"
    )
    # The divergence proves the alert candle must be excluded.
    assert correct_out.result != wrong_out.result


# ── Idempotence ───────────────────────────────────────────────────────────────


def test_idempotence():
    """Repeated calls with the same inputs produce identical outcomes."""
    alert = _long_alert()
    candles = [
        _candle(_BASE + timedelta(minutes=1), 100, 103, 98, 101),
        _candle(_BASE + timedelta(minutes=2), 101, 116, 100, 115),
    ]
    sim = OutcomeSimulator()
    out1 = sim.simulate(alert, candles)
    out2 = sim.simulate(alert, candles)
    assert out1 == out2


# ── fill_price clamping ───────────────────────────────────────────────────────


def test_fill_price_clamped_to_candle_when_entry_mid_above_high():
    """A short fill that only grazes the zone bottom fills at the candle high,
    never at entry_mid (a price the bar never traded)."""
    # entry_zone [102, 104] → entry_mid 103. Candle grazes the bottom edge:
    # high=102.1 (>= entry_low 102 → overlap) but never reaches 103.
    alert = _short_alert(entry_zone=(102.0, 104.0), stop=110.0, target=85.0)
    candles = [_candle(_BASE + timedelta(minutes=1), 100.0, 102.1, 99.0, 100.0)]
    out = OutcomeSimulator().simulate(alert, candles)
    assert out.result == "expired"  # filled, unresolved by end of candles
    assert out.fill_price == 102.1  # clamped to candle high, not entry_mid 103


def test_fill_price_clamped_to_candle_when_entry_mid_below_low():
    """A long fill that only grazes the zone top fills at the candle low,
    never below it."""
    # entry_zone [98, 100] → entry_mid 99. Candle grazes the top edge:
    # low=100.0 (<= entry_high 100 → overlap) but never trades down to 99.
    alert = _long_alert(entry_zone=(98.0, 100.0), stop=90.0, target=120.0)
    candles = [_candle(_BASE + timedelta(minutes=1), 101.0, 103.0, 100.0, 102.0)]
    out = OutcomeSimulator().simulate(alert, candles)
    assert out.result == "expired"
    assert out.fill_price == 100.0  # clamped to candle low, not entry_mid 99


def test_fill_price_unclamped_when_entry_mid_inside_candle():
    """When entry_mid lies inside the fill candle range, fill is entry_mid."""
    alert = _long_alert(entry_zone=(100.0, 102.0))  # entry_mid 101
    candles = [_candle(_BASE + timedelta(minutes=1), 100.5, 103.0, 99.0, 100.0)]
    out = OutcomeSimulator().simulate(alert, candles)
    assert out.fill_price == 101.0  # inside [99, 103] → unchanged


# ── TradeOutcome.to_dict ──────────────────────────────────────────────────────


def test_trade_outcome_to_dict():
    out = TradeOutcome(result="win", candles_to_fill=1, candles_to_resolution=2,
                       fill_price=101.0, exit_price=115.0, actual_rr=2.33)
    d = out.to_dict()
    assert d["result"] == "win"
    assert d["fill_price"] == 101.0
    assert d["actual_rr"] == 2.33
    assert set(d.keys()) == {"result", "candles_to_fill", "candles_to_resolution",
                             "fill_price", "exit_price", "actual_rr"}
