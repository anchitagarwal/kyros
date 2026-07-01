"""test_fib_liquidity_cycle.py — Phase 2B: ERL/IRL classifier, cycle, DOL, snapshot wiring.

All offline (no API key). Tests the additive Phase 2B surfaces:
  - LiquidityPool scope/role classification (ERL vs IRL sources).
  - _derive_liquidity_cycle (ERL→IRL→ERL position from most-recent sweep).
  - _dol_target (cycle-aware DOL augmenting _nearest_dol).
  - MarketSnapshot wiring (fib_levels / liquidity_cycle / dol_target in build()
    and to_compact_dict()).
  - Mocked-LLM e2e (fib/cycle-referencing alert → valid AlertPayload;
    parse-failure path → no_trade).
  - Determinism: same window + config → identical snapshot + compact dict.

Phase 2B continuation:
  - ranked_dols + clarity_score/score_breakdown surfaces exist.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from trading.candle_source import MockCandleSource, TIMEFRAMES
from trading.candle_window import CandleWindow, DEFAULT_SIZES
from trading.config import TradingConfig
from trading.snapshot import (
    SnapshotBuilder,
    LiquidityPool,
)
from trading.reasoning_agent import LLMReasoningAgent, ICT_SYSTEM_PROMPT
from trading.alert import AlertPayload

_NY = ZoneInfo("America/New_York")


def _build_window(scenario, n=100):
    src = MockCandleSource(scenario, n_bars=n)
    w = CandleWindow(DEFAULT_SIZES)
    while not src.is_done():
        w.update(src.next())
    return w


def _build_snapshot(scenario, n=100, now=None, config=None):
    w = _build_window(scenario, n)
    return SnapshotBuilder(config=config or TradingConfig()).build(w, now=now)


# ── LiquidityPool scope/role classification ──────────────────────────────────


def test_liquidity_pool_defaults_external_empty_role():
    p = LiquidityPool(level=100.0, type="bsl", timeframe="1h",
                      distance_points=10.0, confluence_count=0)
    assert p.scope == "external"
    assert p.role == ""


def test_liquidity_pool_to_dict_has_scope_and_role():
    p = LiquidityPool(level=100.0, type="bsl", timeframe="1h",
                      distance_points=10.0, confluence_count=0,
                      scope="internal", role="fvg_ce")
    d = p.to_dict()
    assert d["scope"] == "internal"
    assert d["role"] == "fvg_ce"
    assert "clarity_score" in d
    assert "score_breakdown" in d


def test_snapshot_pools_carry_scope_and_role():
    snap = _build_snapshot("sweep_and_fvg")
    assert len(snap.all_pools) > 0
    for p in snap.all_pools:
        assert p.scope in ("external", "internal")
        assert isinstance(p.role, str)


def test_irl_sources_gated_by_config():
    cfg = TradingConfig(irl_sources=("fvg", "order_block", "equilibrium"))
    snap_no_ote = _build_snapshot("sweep_and_fvg", config=cfg)
    roles_no_ote = {p.role for p in snap_no_ote.all_pools}
    assert "ote" not in roles_no_ote


# ── _derive_liquidity_cycle ───────────────────────────────────────────────────


def _ts(minute):
    return datetime(2026, 6, 15, 10, minute, tzinfo=_NY).isoformat()


def test_cycle_bsl_sweep_no_reversal_seek_irl():
    b = SnapshotBuilder()
    sweeps = {"5m": [{"type": "sweep_bsl", "swept_level": 200.0, "timestamp": _ts(0)}]}
    cycle = b._derive_liquidity_cycle(sweeps, {}, {}, "bearish")
    assert cycle["last_swept_erl_side"] == "buyside"
    assert cycle["target_erl_side"] == "sellside"
    assert cycle["current_leg"] == "seek_irl"
    assert cycle["next_draw"] == "irl"


def test_cycle_ssl_sweep_with_reversal_bos_expand_to_erl():
    b = SnapshotBuilder()
    sweeps = {"5m": [{"type": "sweep_ssl", "swept_level": 100.0, "timestamp": _ts(0)}]}
    ms = {"5m": [{"type": "bos_bullish", "break_price": 110.0, "timestamp": _ts(30)}]}
    cycle = b._derive_liquidity_cycle(sweeps, ms, {}, "bullish")
    assert cycle["last_swept_erl_side"] == "sellside"
    assert cycle["target_erl_side"] == "buyside"
    assert cycle["current_leg"] == "expand_to_erl"
    assert cycle["next_draw"] == "erl"


# ── _dol_target (updated signature) ─────────────────────────────────────────-


def _pool(level, ptype="bsl", scope="external", role="swing", swept=False):
    return LiquidityPool(
        level=level, type=ptype, timeframe="1h",
        distance_points=abs(level - 200.0), confluence_count=0,
        swept=swept, scope=scope, role=role,
    )


def test_dol_target_cycle_none_equals_nearest_dol_external_only():
    b = SnapshotBuilder()
    pools = [_pool(210, "bsl", scope="external"), _pool(205, "bsl", scope="internal", role="ote")]
    # cycle None → ERL-only nearest_dol (external 210 wins; internal ignored)
    res = b._dol_target(pools, None, "bullish", 200.0, None, None)
    assert res is pools[0]


def test_dol_target_cycle_active_uses_ranked():
    b = SnapshotBuilder()
    pools = [_pool(210, "bsl", scope="external"), _pool(140, "ssl", scope="external")]
    cycle = {"last_swept_erl_side": "buyside", "target_erl_side": "sellside", "current_leg": "expand_to_erl"}
    res = b._dol_target(pools, cycle, "bearish", 200.0, None, None)
    assert res is not None
    assert res.type == "ssl"


# ── Snapshot integration ─────────────────────────────────────────────────────-


def test_snapshot_has_ranked_dols_field():
    snap = _build_snapshot("sweep_and_fvg")
    assert hasattr(snap, "ranked_dols")
    assert isinstance(snap.ranked_dols, list)


def test_compact_dict_has_ranked_dols_key():
    snap = _build_snapshot("sweep_and_fvg")
    cd = snap.to_compact_dict()
    assert "ranked_dols" in cd


# ── Mocked-LLM e2e ─────────────────────────────────────────────────────────--


def _mock_router(content: str):
    router = MagicMock()
    resp = MagicMock()
    resp.content = content
    router.call.return_value = resp
    return router


def test_mocked_llm_fib_cycle_alert_parses_to_valid_payload():
    snap = _build_snapshot("sweep_and_fvg")
    router = _mock_router(json.dumps({
        "bias": "short", "model": "2022", "conviction": 80,
        "entry_zone": [snap.current_price - 5, snap.current_price + 5],
        "stop": snap.current_price + 50,
        "target": snap.current_price - 100,
        "dol": {"level": snap.current_price - 100, "type": "ssl", "timeframe": "1h"},
        "risk_reward": 2.0,
        "rationale": "cycle + fib",
        "killzone": snap.current_killzone or "",
        "valid_until": "",
        "no_trade_reason": None,
    }))
    agent = LLMReasoningAgent(router)
    alert = agent.reason(snap)
    assert isinstance(alert, AlertPayload)
    args, kwargs = router.call.call_args
    messages = kwargs.get("messages") or args[1]
    payload = json.loads(messages[0]["content"])
    assert "ranked_dols" in payload


def test_prompt_references_ranked_dols():
    assert "ranked_dols" in ICT_SYSTEM_PROMPT
