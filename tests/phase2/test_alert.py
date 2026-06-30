"""test_alert.py — R:R validator + LLM parse-error handling."""

import pytest

from trading.alert import AlertPayload, validate_rr, parse_llm_json


# ── validate_rr ───────────────────────────────────────────────────────────────


def test_rr_above_1_passes_unchanged():
    alert = AlertPayload(
        bias="long", model="2022", conviction=70,
        entry_zone=(100.0, 102.0), stop=98.0, target=110.0,
    )
    out = validate_rr(alert)
    assert out.bias == "long"
    # entry_mid=101, risk=|101-98|=3, rr=|110-101|/3=3.0
    assert out.risk_reward == 3.0
    assert out.no_trade_reason is None


def test_rr_exactly_1_passes_unchanged():
    alert = AlertPayload(
        bias="long", conviction=60,
        entry_zone=(100.0, 100.0), stop=95.0, target=105.0,
    )
    out = validate_rr(alert)
    assert out.bias == "long"
    # entry_mid=100, risk=5, rr=5/5=1.0
    assert out.risk_reward == 1.0
    assert out.no_trade_reason is None


def test_rr_below_1_overrides_to_no_trade():
    alert = AlertPayload(
        bias="long", conviction=80,
        entry_zone=(100.0, 100.0), stop=95.0, target=99.0,
    )
    out = validate_rr(alert)
    assert out.bias == "no_trade"
    assert out.no_trade_reason == "rr_below_1"
    # entry_mid=100, risk=5, rr=|99-100|/5=0.2
    assert out.risk_reward == 0.2


def test_degenerate_stop_zero_risk():
    alert = AlertPayload(
        bias="long", conviction=90,
        entry_zone=(100.0, 100.0), stop=100.0, target=200.0,
    )
    out = validate_rr(alert)
    assert out.bias == "no_trade"
    assert out.no_trade_reason == "degenerate_stop"
    assert out.risk_reward == 0.0


def test_no_trade_reason_preserved_through_validation():
    """An LLM no_trade alert keeps its own reason — not clobbered to degenerate_stop.

    no_trade alerts carry placeholder 0/0 geometry (risk == 0), which would
    otherwise trip the degenerate_stop branch and destroy the LLM's real
    rationale code. validate_rr must short-circuit on no_trade.
    """
    alert = AlertPayload(
        bias="no_trade", model="none", conviction=0,
        entry_zone=(0.0, 0.0), stop=0.0, target=0.0,
        no_trade_reason="intermediate liquidity in path",
    )
    out = validate_rr(alert)
    assert out.bias == "no_trade"
    assert out.no_trade_reason == "intermediate liquidity in path"


def test_llm_risk_reward_overridden_by_python():
    """LLM says rr=5.0 but geometry implies 1.5 → output 1.5."""
    alert = AlertPayload(
        bias="long", conviction=70,
        entry_zone=(100.0, 100.0), stop=98.0, target=103.0,
        risk_reward=5.0,  # LLM's wrong value
    )
    out = validate_rr(alert)
    # entry_mid=100, risk=2, rr=|103-100|/2=1.5
    assert out.risk_reward == 1.5
    assert out.bias == "long"


def test_validate_rr_pure_no_side_effects():
    alert = AlertPayload(
        bias="long", conviction=70,
        entry_zone=(100.0, 100.0), stop=95.0, target=110.0,
        risk_reward=99.0,
    )
    original_rr = alert.risk_reward
    out = validate_rr(alert)
    # Original alert is not mutated.
    assert alert.risk_reward == original_rr
    assert out.risk_reward == 2.0


def test_short_bias_rr_above_1_passes():
    alert = AlertPayload(
        bias="short", conviction=65,
        entry_zone=(100.0, 100.0), stop=105.0, target=90.0,
    )
    out = validate_rr(alert)
    assert out.bias == "short"
    # entry_mid=100, risk=|100-105|=5, rr=|90-100|/5=2.0
    assert out.risk_reward == 2.0


# ── parse_llm_json ────────────────────────────────────────────────────────────


def test_parse_llm_json_valid():
    data = {
        "bias": "long", "model": "2022", "conviction": 75,
        "entry_zone": [100.0, 102.0], "stop": 98.0, "target": 110.0,
        "dol": {"level": 110.0, "type": "bsl", "timeframe": "1h"},
        "risk_reward": 3.0, "rationale": "sweep + FVG",
        "killzone": "ny_am_kz", "valid_until": "2026-06-15T11:00:00-04:00",
        "no_trade_reason": None,
    }
    alert = parse_llm_json(data)
    assert alert.bias == "long"
    assert alert.model == "2022"
    assert alert.entry_zone == (100.0, 102.0)
    assert alert.dol["type"] == "bsl"


def test_parse_llm_json_malformed_returns_no_trade():
    alert = parse_llm_json("not a dict")
    assert alert.bias == "no_trade"
    assert alert.no_trade_reason == "llm_parse_error"


def test_parse_llm_json_missing_keys_returns_no_trade():
    alert = parse_llm_json({"bias": "long"})  # missing entry_zone etc.
    assert alert.bias == "no_trade"
    assert alert.no_trade_reason == "llm_parse_error"


def test_parse_llm_json_invalid_enum_normalized():
    data = {
        "bias": "INVALID", "model": "FAKE",
        "entry_zone": [100.0, 102.0], "stop": 98.0, "target": 110.0,
    }
    alert = parse_llm_json(data)
    assert alert.bias == "no_trade"  # invalid bias → no_trade
    assert alert.model == "none"  # invalid model → none


def test_parse_llm_json_extra_prose_ignored():
    """If the LLM wraps JSON in prose, the caller must extract JSON first;
    parse_llm_json itself only accepts a parsed dict. A non-dict → parse error."""
    alert = parse_llm_json("Here is the alert: {...}")
    assert alert.bias == "no_trade"
    assert alert.no_trade_reason == "llm_parse_error"
