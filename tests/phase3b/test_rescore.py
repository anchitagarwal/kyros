"""test_rescore.py — rescore reuses recorded outcomes; filters downgrade to
no_trade; no re-simulation; idempotence; no input mutation."""

import copy
import math

import pytest

from trading.alert import AlertPayload, validate_rr
from tuning.params import ALL, PostLLMParams
from tuning.rescore import compute_rr, rescore_trace, rescore_traces


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _taken_trace(
    bias="long",
    model="2022",
    conviction=70,
    entry_zone=(100.0, 102.0),
    stop=98.0,
    target=110.0,
    killzone="ny_am_kz",
    result="win",
    actual_rr=2.0,
    timestamp="2026-06-01T10:00:00-04:00",
):
    """A recorded trace of a TAKEN trade (directional, resolved fill)."""
    return {
        "trace_id": f"{timestamp}_{bias}_abc12345",
        "timestamp": timestamp,
        "instrument": "NQ",
        "killzone": killzone,
        "trigger_reason": "fvg",
        "snapshot_summary": {"instrument": "NQ"},
        "raw_llm_output": "{}",
        "alert": {
            "bias": bias,
            "model": model,
            "conviction": conviction,
            "entry_zone": list(entry_zone),
            "stop": stop,
            "target": target,
            "dol": {"level": target, "type": "bsl", "timeframe": "1h"},
            "risk_reward": 0.0,  # stale; rescore recomputes
            "rationale": "fixture",
            "killzone": killzone,
            "valid_until": "",
            "no_trade_reason": None,
        },
        "rr_validated": True,
        "outcome": {
            "result": result,
            "candles_to_fill": 1,
            "candles_to_resolution": 2,
            "fill_price": 101.0,
            "exit_price": 110.0,
            "actual_rr": actual_rr,
        },
    }


def _no_trade_trace(timestamp="2026-06-01T13:30:00-04:00", reason="degenerate_stop"):
    """A recorded no_trade trace (filters must NOT touch it)."""
    return {
        "trace_id": f"{timestamp}_no_trade_abc12345",
        "timestamp": timestamp,
        "instrument": "NQ",
        "killzone": "ny_pm_kz",
        "trigger_reason": "fvg",
        "snapshot_summary": {"instrument": "NQ"},
        "raw_llm_output": "{}",
        "alert": {
            "bias": "no_trade",
            "model": "none",
            "conviction": 0,
            "entry_zone": [0.0, 0.0],
            "stop": 0.0,
            "target": 0.0,
            "dol": {"level": 0.0, "type": "", "timeframe": ""},
            "risk_reward": 0.0,
            "rationale": "no setup",
            "killzone": "ny_pm_kz",
            "valid_until": "",
            "no_trade_reason": reason,
        },
        "rr_validated": False,
        "outcome": {
            "result": "no_trade",
            "candles_to_fill": None,
            "candles_to_resolution": None,
            "fill_price": None,
            "exit_price": None,
            "actual_rr": None,
        },
    }


KEEPING = PostLLMParams(conviction_min=40, rr_min=1.0, allowed_models=ALL, allowed_killzones=ALL)


# ── Surviving trades keep their recorded outcome (no re-simulation) ───────────


def test_keeping_params_return_trace_unchanged():
    """With keeping params, a taken trade is returned with its recorded outcome
    verbatim — no re-simulation, byte-identical outcome."""
    t = _taken_trace(result="win", actual_rr=2.0)
    out = rescore_trace(t, KEEPING)
    assert out["outcome"]["result"] == "win"
    assert out["outcome"]["actual_rr"] == 2.0
    assert out["alert"]["bias"] == "long"
    assert out["alert"]["no_trade_reason"] is None


def test_surviving_trade_outcome_byte_identical():
    """The outcome dict of a surviving trade is identical to the input's."""
    t = _taken_trace(result="loss", actual_rr=-1.0)
    out = rescore_trace(t, KEEPING)
    assert out["outcome"] == t["outcome"]


def test_no_input_mutation():
    """rescore_trace returns a NEW dict; the input is not mutated."""
    t = _taken_trace()
    original = copy.deepcopy(t)
    out = rescore_trace(t, PostLLMParams(
        conviction_min=80, rr_min=1.0, allowed_models=ALL, allowed_killzones=ALL))
    # The input is untouched.
    assert t == original
    # The output is a different object.
    assert out is not t
    assert out["alert"] is not t["alert"]


# ── Each filter independently downgrades a taken trade to no_trade ────────────


def test_conviction_filter_downgrades():
    t = _taken_trace(conviction=50)
    p = PostLLMParams(conviction_min=60, rr_min=1.0, allowed_models=ALL, allowed_killzones=ALL)
    out = rescore_trace(t, p)
    assert out["alert"]["bias"] == "no_trade"
    assert out["alert"]["no_trade_reason"] == "conviction_below_min"
    assert out["outcome"]["result"] == "no_trade"
    assert out["outcome"]["actual_rr"] is None


def test_rr_filter_downgrades():
    """rr_min above the trade's recomputed R:R → no_trade."""
    # entry_mid=101, risk=|101-98|=3, rr=|110-101|/3=3.0
    t = _taken_trace(entry_zone=(100.0, 102.0), stop=98.0, target=110.0)
    p = PostLLMParams(conviction_min=40, rr_min=4.0, allowed_models=ALL, allowed_killzones=ALL)
    out = rescore_trace(t, p)
    assert out["alert"]["bias"] == "no_trade"
    assert out["alert"]["no_trade_reason"] == "rr_below_min"
    assert out["outcome"]["actual_rr"] is None


def test_model_filter_downgrades():
    t = _taken_trace(model="2022")
    p = PostLLMParams(conviction_min=40, rr_min=1.0,
                      allowed_models=frozenset({"unicorn"}), allowed_killzones=ALL)
    out = rescore_trace(t, p)
    assert out["alert"]["bias"] == "no_trade"
    assert out["alert"]["no_trade_reason"] == "model_filtered"


def test_killzone_filter_downgrades():
    t = _taken_trace(killzone="ny_am_kz")
    p = PostLLMParams(conviction_min=40, rr_min=1.0, allowed_models=ALL,
                      allowed_killzones=frozenset({"london_kz"}))
    out = rescore_trace(t, p)
    assert out["alert"]["bias"] == "no_trade"
    assert out["alert"]["no_trade_reason"] == "killzone_filtered"


def test_empty_model_set_rejects_everything():
    """An empty frozenset (not ALL) means 'reject everything'."""
    t = _taken_trace(model="2022")
    p = PostLLMParams(conviction_min=40, rr_min=1.0,
                      allowed_models=frozenset(), allowed_killzones=ALL)
    out = rescore_trace(t, p)
    assert out["alert"]["bias"] == "no_trade"
    assert out["alert"]["no_trade_reason"] == "model_filtered"


def test_all_sentinel_allows_any_model():
    """ALL (frozenset({'*'})) means 'no model filter'."""
    t = _taken_trace(model="silver_bullet")
    p = PostLLMParams(conviction_min=40, rr_min=1.0, allowed_models=ALL, allowed_killzones=ALL)
    out = rescore_trace(t, p)
    assert out["alert"]["bias"] == "long"


# ── R:R recompute matches validate_rr ─────────────────────────────────────────


def test_rr_recompute_matches_validate_rr():
    """compute_rr uses the SAME formula as validate_rr.

    For several geometries, the recomputed rr equals validate_rr's
    risk_reward (after it overwrites the LLM value).
    """
    geometries = [
        ((100.0, 102.0), 98.0, 110.0),   # entry_mid=101, risk=3, rr=3.0
        ((100.0, 100.0), 95.0, 105.0),   # rr=1.0
        ((100.0, 100.0), 95.0, 99.0),    # rr=0.2
        ((100.0, 104.0), 98.0, 110.0),   # entry_mid=102, risk=4, rr=2.0
    ]
    for ez, stop, target in geometries:
        alert = AlertPayload(bias="long", conviction=70,
                             entry_zone=ez, stop=stop, target=target)
        validated = validate_rr(alert)
        recomputed = compute_rr(ez, stop, target)
        assert round(recomputed, 2) == validated.risk_reward, (
            f"rr mismatch for {ez},{stop},{target}: "
            f"validate_rr={validated.risk_reward} compute_rr={recomputed}"
        )


def test_degenerate_stop_recompute_is_zero():
    """risk == 0 (stop == entry_mid) → compute_rr returns 0.0 → downgrade."""
    t = _taken_trace(entry_zone=(100.0, 100.0), stop=100.0, target=110.0)
    # rr_min=1.0; compute_rr returns 0.0 < 1.0 → downgrade.
    out = rescore_trace(t, KEEPING)
    assert out["alert"]["bias"] == "no_trade"
    assert out["alert"]["no_trade_reason"] == "rr_below_min"


# ── Non-taken traces are untouched by any filter ──────────────────────────────


def test_no_trade_trace_untouched_by_filters():
    """A recorded no_trade stays no_trade under aggressive filters."""
    t = _no_trade_trace(reason="intermediate liquidity in path")
    aggressive = PostLLMParams(
        conviction_min=100, rr_min=10.0,
        allowed_models=frozenset(), allowed_killzones=frozenset())
    out = rescore_trace(t, aggressive)
    assert out["outcome"]["result"] == "no_trade"
    # The LLM's own no_trade_reason is preserved (not clobbered).
    assert out["alert"]["no_trade_reason"] == "intermediate liquidity in path"


def test_no_fill_trace_untouched():
    """A no_fill (not a taken trade) is untouched by filters."""
    t = _taken_trace(result="no_fill", actual_rr=None)
    out = rescore_trace(t, PostLLMParams(
        conviction_min=100, rr_min=10.0, allowed_models=ALL, allowed_killzones=ALL))
    assert out["outcome"]["result"] == "no_fill"
    assert out["outcome"]["actual_rr"] is None


def test_expired_trace_untouched():
    t = _taken_trace(result="expired", actual_rr=None)
    out = rescore_trace(t, PostLLMParams(
        conviction_min=100, rr_min=10.0, allowed_models=ALL, allowed_killzones=ALL))
    assert out["outcome"]["result"] == "expired"


# ── Idempotence ───────────────────────────────────────────────────────────────


def test_idempotent_keeping():
    t = _taken_trace()
    once = rescore_trace(t, KEEPING)
    twice = rescore_trace(once, KEEPING)
    assert once == twice


def test_idempotent_filtering():
    """rescore_trace(rescore_trace(t, p), p) == rescore_trace(t, p)."""
    t = _taken_trace(conviction=50)
    p = PostLLMParams(conviction_min=60, rr_min=1.0, allowed_models=ALL, allowed_killzones=ALL)
    once = rescore_trace(t, p)
    twice = rescore_trace(once, p)
    assert once == twice


# ── Order preservation ────────────────────────────────────────────────────────


def test_rescore_traces_preserves_order():
    traces = [
        _taken_trace(timestamp="2026-06-01T10:00:00-04:00"),
        _no_trade_trace(timestamp="2026-06-01T11:00:00-04:00"),
        _taken_trace(timestamp="2026-06-01T12:00:00-04:00", conviction=30),
    ]
    out = rescore_traces(traces, PostLLMParams(
        conviction_min=40, rr_min=1.0, allowed_models=ALL, allowed_killzones=ALL))
    assert len(out) == 3
    # Order preserved.
    assert out[0]["timestamp"] == "2026-06-01T10:00:00-04:00"
    assert out[1]["timestamp"] == "2026-06-01T11:00:00-04:00"
    assert out[2]["timestamp"] == "2026-06-01T12:00:00-04:00"
    # The third (conviction 30 < 40) was downgraded.
    assert out[2]["alert"]["bias"] == "no_trade"


# ── Property: every output outcome ∈ {original, no_trade} ─────────────────────


def test_property_outcome_is_original_or_no_trade():
    """For any params, every output trace's outcome is either its original
    outcome or no_trade — filters only ever downgrade."""
    import random
    traces = [
        _taken_trace(result="win", actual_rr=2.0, conviction=70, model="2022", killzone="ny_am_kz"),
        _taken_trace(result="loss", actual_rr=-1.0, conviction=50, model="unicorn", killzone="london_kz"),
        _no_trade_trace(),
        _taken_trace(result="expired", actual_rr=None, conviction=80, model="ifvg", killzone="ny_pm_kz"),
    ]
    original_outcomes = [t["outcome"]["result"] for t in traces]
    param_variants = [
        PostLLMParams(40, 1.0, ALL, ALL),
        PostLLMParams(60, 1.0, ALL, ALL),
        PostLLMParams(40, 2.0, ALL, ALL),
        PostLLMParams(40, 1.0, frozenset({"2022"}), ALL),
        PostLLMParams(40, 1.0, ALL, frozenset({"ny_am_kz"})),
        PostLLMParams(80, 5.0, frozenset(), frozenset()),
    ]
    for p in param_variants:
        out = rescore_traces(traces, p)
        for orig_t, out_t, orig_result in zip(traces, out, original_outcomes):
            if out_t["outcome"]["result"] != "no_trade":
                # If not downgraded, the outcome is the original.
                assert out_t["outcome"]["result"] == orig_result
                assert out_t["outcome"] == orig_t["outcome"]
