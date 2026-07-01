"""Tests for fibonacci.py — direction-aware Fibonacci grid (Phase 2B).

Asserts the EXACT canonical numbers (100→200 range, R=100) for both
directions, edge cases, purity, and convention parity with
detect_premium_discount.
"""

import copy

import pytest

from detectors.fibonacci import detect_fibonacci
from detectors.premium_discount import detect_premium_discount


def _mkseries(rows, start_ts=0, step=60):
    out = []
    ts = start_ts
    for (o, h, l, c) in rows:
        out.append({"open": o, "high": h, "low": l, "close": c,
                    "volume": 1000.0, "timestamp": ts})
        ts += step
    return out


# Swing low at 100, swing high at 200, low precedes high → direction up.
_ROWS_UP = [
    (150, 155, 145, 150),  # 0
    (150, 152, 120, 130),  # 1
    (130, 135, 100, 110),  # 2  low=100 (trough)
    (110, 130, 105, 125),  # 3
    (125, 140, 120, 135),  # 4
    (135, 200, 130, 190),  # 5  high=200 (peak)
    (190, 195, 185, 188),  # 6
    (188, 190, 185, 186),  # 7
]

# Swing high at 200, swing low at 100, high precedes low → direction down.
_ROWS_DOWN = [
    (150, 155, 145, 150),  # 0
    (150, 180, 145, 175),  # 1
    (175, 200, 170, 190),  # 2  high=200 (peak)
    (190, 195, 160, 162),  # 3
    (162, 165, 135, 138),  # 4
    (138, 145, 100, 110),  # 5  low=100 (trough)
    (110, 115, 105, 112),  # 6
    (112, 115, 108, 110),  # 7
]


# ── Edge cases → [] ───────────────────────────────────────────────────────────


def test_fib_empty_input():
    assert detect_fibonacci([]) == []


def test_fib_no_swing_pair():
    # Monotonic rise → no swings → [].
    rows = [(i, i + 1, i - 1, i) for i in range(10)]
    assert detect_fibonacci(_mkseries(rows)) == []


def test_fib_single_swing_no_pair():
    # Only one swing type present (a single peak, no trough) → [].
    rows = [
        (100, 100, 100, 100),
        (100, 200, 99, 190),   # peak at index 1
        (190, 191, 189, 190),
        (190, 191, 189, 190),
        (190, 191, 189, 190),
    ]
    # With lookback=2 we need 5 candles; a single high with no low → [].
    assert detect_fibonacci(_mkseries(rows)) == []


def test_fib_degenerate_range():
    # high == low → degenerate → [].
    rows = [
        (100, 100, 100, 100),
        (100, 100, 100, 100),
        (100, 100, 100, 100),
        (100, 100, 100, 100),
        (100, 100, 100, 100),
    ]
    assert detect_fibonacci(_mkseries(rows)) == []


# ── Canonical numbers: UP ─────────────────────────────────────────────────────


def test_fib_up_canonical_numbers():
    r = detect_fibonacci(_mkseries(_ROWS_UP))[0]
    assert r["direction"] == "up"
    assert r["range_high"] == 200
    assert r["range_low"] == 100
    assert r["equilibrium"] == pytest.approx(150.0, abs=1e-9)
    assert r["golden_pocket"] == pytest.approx([134.0, 138.2], abs=1e-9)
    assert r["ote"]["primary"] == pytest.approx(129.5, abs=1e-9)
    assert r["ote"]["zone"] == pytest.approx([121.0, 138.0], abs=1e-9)
    assert r["retracement_target"] == pytest.approx(161.8, abs=1e-9)
    assert r["extensions"] == pytest.approx(
        {"-0.5": 250.0, "-1.0": 300.0, "-1.5": 350.0, "-2.0": 400.0, "-2.5": 450.0},
        abs=1e-9,
    )


def test_fib_up_ote_grid_keys():
    r = detect_fibonacci(_mkseries(_ROWS_UP))[0]
    ote = r["ote"]
    # Grid ratios present as string keys.
    for k in ("0.5", "0.62", "0.705", "0.79"):
        assert k in ote
    assert ote["0.5"] == pytest.approx(150.0, abs=1e-9)
    assert ote["0.62"] == pytest.approx(138.0, abs=1e-9)
    assert ote["0.705"] == pytest.approx(129.5, abs=1e-9)
    assert ote["0.79"] == pytest.approx(121.0, abs=1e-9)


def test_fib_up_retracements_dict():
    r = detect_fibonacci(_mkseries(_ROWS_UP))[0]
    retr = r["retracements"]
    # One entry per default retracement ratio.
    assert set(retr.keys()) == {"0.382", "0.5", "0.618", "0.66", "0.705", "0.79"}
    assert retr["0.382"] == pytest.approx(161.8, abs=1e-9)
    assert retr["0.5"] == pytest.approx(150.0, abs=1e-9)
    assert retr["0.618"] == pytest.approx(138.2, abs=1e-9)


# ── Canonical numbers: DOWN ───────────────────────────────────────────────────


def test_fib_down_canonical_numbers():
    r = detect_fibonacci(_mkseries(_ROWS_DOWN))[0]
    assert r["direction"] == "down"
    assert r["range_high"] == 200
    assert r["range_low"] == 100
    assert r["equilibrium"] == pytest.approx(150.0, abs=1e-9)
    assert r["golden_pocket"] == pytest.approx([161.8, 166.0], abs=1e-9)
    assert r["ote"]["primary"] == pytest.approx(170.5, abs=1e-9)
    assert r["ote"]["zone"] == pytest.approx([162.0, 179.0], abs=1e-9)
    assert r["retracement_target"] == pytest.approx(138.2, abs=1e-9)
    assert r["extensions"] == pytest.approx(
        {"-0.5": 50.0, "-1.0": 0.0, "-1.5": -50.0, "-2.0": -100.0, "-2.5": -150.0},
        abs=1e-9,
    )


def test_fib_down_ote_grid_keys():
    r = detect_fibonacci(_mkseries(_ROWS_DOWN))[0]
    ote = r["ote"]
    for k in ("0.5", "0.62", "0.705", "0.79"):
        assert k in ote
    assert ote["0.5"] == pytest.approx(150.0, abs=1e-9)
    assert ote["0.62"] == pytest.approx(162.0, abs=1e-9)
    assert ote["0.705"] == pytest.approx(170.5, abs=1e-9)
    assert ote["0.79"] == pytest.approx(179.0, abs=1e-9)


# ── premium_array ─────────────────────────────────────────────────────────────


def test_fib_premium_array_true_above_equilibrium():
    # Append a candle closing above equilibrium (150) → premium_array True.
    rows = _ROWS_UP + [(186, 210, 185, 205)]
    r = detect_fibonacci(_mkseries(rows))[0]
    assert r["premium_array"] is True


def test_fib_premium_array_false_below_equilibrium():
    # Append a candle closing below equilibrium (150) → premium_array False.
    rows = _ROWS_UP + [(186, 188, 185, 140)]
    r = detect_fibonacci(_mkseries(rows))[0]
    assert r["premium_array"] is False


# ── Convention parity with premium_discount ───────────────────────────────────


@pytest.mark.parametrize("rows", [_ROWS_UP, _ROWS_DOWN])
def test_fib_convention_parity_with_premium_discount(rows):
    """The 0.5/0.62/0.79 prices must equal premium_discount's OTE band."""
    candles = _mkseries(rows)
    fib = detect_fibonacci(candles)[0]
    pd = detect_premium_discount(candles)[0]
    assert fib["direction"] == pd["direction"]
    assert fib["equilibrium"] == pytest.approx(pd["equilibrium"], abs=1e-9)
    # premium_discount ote_zone == [price@0.62, price@0.79] (low, high) for up,
    # and [price@0.62, price@0.79] for down too (it sorts low/high). Our fib
    # ote.zone is [price@0.79, price@0.62] ascending → same two endpoints.
    fib_zone = sorted(fib["ote"]["zone"])
    pd_zone = sorted(pd["ote_zone"])
    assert fib_zone == pytest.approx(pd_zone, abs=1e-9)
    # The 0.62 and 0.79 grid entries match premium_discount's OTE bounds.
    assert fib["ote"]["0.62"] == pytest.approx(pd["ote_zone"][0], abs=1e-9) or \
           fib["ote"]["0.62"] == pytest.approx(pd["ote_zone"][1], abs=1e-9)
    assert fib["ote"]["0.79"] == pytest.approx(pd["ote_zone"][0], abs=1e-9) or \
           fib["ote"]["0.79"] == pytest.approx(pd["ote_zone"][1], abs=1e-9)


# ── Purity / determinism ──────────────────────────────────────────────────────


def test_fib_purity_no_mutation():
    candles = _mkseries(_ROWS_UP)
    before = copy.deepcopy(candles)
    detect_fibonacci(candles)
    assert candles == before


def test_fib_purity_same_input_same_output():
    candles = _mkseries(_ROWS_UP)
    r1 = detect_fibonacci(candles)
    r2 = detect_fibonacci(candles)
    assert r1 == r2


# ── Parameterization honored ──────────────────────────────────────────────────


def test_fib_golden_pocket_parameterized():
    # Custom golden pocket ratios → reflected in output.
    r = detect_fibonacci(_mkseries(_ROWS_UP), golden_pocket=(0.5, 0.79))[0]
    # up: price(0.5)=150, price(0.79)=121 → ascending [121, 150].
    assert r["golden_pocket"] == pytest.approx([121.0, 150.0], abs=1e-9)


def test_fib_ote_primary_parameterized():
    r = detect_fibonacci(_mkseries(_ROWS_UP), ote_primary=0.5)[0]
    assert r["ote"]["primary"] == pytest.approx(150.0, abs=1e-9)


def test_fib_extensions_parameterized():
    r = detect_fibonacci(_mkseries(_ROWS_UP), extensions=(-1.0,))[0]
    assert set(r["extensions"].keys()) == {"-1.0"}
    assert r["extensions"]["-1.0"] == pytest.approx(300.0, abs=1e-9)


def test_fib_retracement_target_parameterized():
    r = detect_fibonacci(_mkseries(_ROWS_UP), retracement_target=0.5)[0]
    assert r["retracement_target"] == pytest.approx(150.0, abs=1e-9)


# ── Output shape ──────────────────────────────────────────────────────────────


def test_fib_output_type_and_index():
    r = detect_fibonacci(_mkseries(_ROWS_UP))[0]
    assert r["type"] == "fibonacci"
    # confirm index = max of the two swing indices (5 for the high here).
    assert r["index"] == 5
    assert r["timestamp"] == 5 * 60  # step=60, index 5


def test_fib_returns_list_of_zero_or_one():
    assert isinstance(detect_fibonacci([]), list)
    assert isinstance(detect_fibonacci(_mkseries(_ROWS_UP)), list)
    assert len(detect_fibonacci(_mkseries(_ROWS_UP))) == 1
