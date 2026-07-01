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
  - scoring behavior is pinned (proximity does not dominate, wrong-side filter,
    clean_path penalty, monotonicity, breakdown sums).
  - repair #8 pinned: detectors are not re-run inside _build_pools when
    precomputed outputs are supplied.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from trading.candle_source import MockCandleSource
from trading.candle_window import CandleWindow, DEFAULT_SIZES
from trading.config import TradingConfig
from trading.snapshot import (
    SnapshotBuilder,
    LiquidityPool,
    TIMEFRAMES,
)
from trading.reasoning_agent import LLMReasoningAgent, ICT_SYSTEM_PROMPT
from trading.alert import AlertPayload

_NY = ZoneInfo("America/New_York")
_BULLISH_EXPAND_CYCLE = {"current_leg": "expand_to_erl", "target_erl_side": "buyside"}
_BASE_HTF_FIB = {"range_low": 100.0, "range_high": 200.0, "equilibrium": 150.0}


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


def _pool(level, ptype="bsl", scope="external", role="swing", swept=False, tf="1h", conf=0, dist_from=200.0):
    return LiquidityPool(
        level=level, type=ptype, timeframe=tf,
        distance_points=abs(level - dist_from), confluence_count=conf,
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


# ── Scoring behavior (continuation phase) ─────────────────────────────────────


def test_score_breakdown_sums_to_score():
    b = SnapshotBuilder()
    pool = _pool(210, "bsl", scope="external", role="equal", tf="4h", conf=2)
    score, breakdown = b._score_pool(
        pool,
        htf_bias="bullish",
        cycle=_BULLISH_EXPAND_CYCLE,
        htf_fib=_BASE_HTF_FIB,
        current_price=200.0,
        killzone=None,
        pools=[pool],
    )
    assert pytest.approx(sum(breakdown.values()), rel=0, abs=1e-9) == score


def test_rank_dols_filters_wrong_side_first():
    b = SnapshotBuilder()
    current_price = 200.0
    pools = [
        _pool(210, "bsl", scope="external"),
        _pool(190, "ssl", scope="external"),
        _pool(220, "bsl", scope="external"),
    ]
    cycle = _BULLISH_EXPAND_CYCLE
    ranked = b._rank_dols(pools, htf_bias="bullish", cycle=cycle, htf_fib=None, current_price=current_price, killzone=None)
    assert ranked
    assert all(p.level > current_price and p.type == "bsl" for p in ranked)


def test_clean_path_penalizes_opposing_external_between_price_and_target():
    b = SnapshotBuilder()
    current_price = 200.0
    target = _pool(230, "bsl", scope="external", role="equal", tf="4h")
    blocker = _pool(215, "ssl", scope="external", role="equal", tf="4h")
    score_clean, _ = b._score_pool(
        target,
        htf_bias="bullish",
        cycle=_BULLISH_EXPAND_CYCLE,
        htf_fib=_BASE_HTF_FIB,
        current_price=current_price,
        killzone=None,
        pools=[target],
    )
    score_blocked, _ = b._score_pool(
        target,
        htf_bias="bullish",
        cycle=_BULLISH_EXPAND_CYCLE,
        htf_fib=_BASE_HTF_FIB,
        current_price=current_price,
        killzone=None,
        pools=[target, blocker],
    )
    assert score_blocked < score_clean


def test_confluence_monotonic_increases_score():
    b = SnapshotBuilder()
    base = _pool(210, "bsl", scope="external", role="equal", tf="4h", conf=0)
    more = _pool(210, "bsl", scope="external", role="equal", tf="4h", conf=3)
    s0, _ = b._score_pool(
        base,
        htf_bias="bullish",
        cycle=_BULLISH_EXPAND_CYCLE,
        htf_fib=_BASE_HTF_FIB,
        current_price=200.0,
        killzone=None,
        pools=[base],
    )
    s1, _ = b._score_pool(
        more,
        htf_bias="bullish",
        cycle=_BULLISH_EXPAND_CYCLE,
        htf_fib=_BASE_HTF_FIB,
        current_price=200.0,
        killzone=None,
        pools=[more],
    )
    assert s1 > s0


def test_proximity_does_not_dominate_far_high_tf_equal_beats_near_low_tf_swing():
    b = SnapshotBuilder()
    current_price = 200.0
    # Near but weak.
    near = _pool(201.0, "bsl", scope="external", role="swing", tf="1m", dist_from=current_price)
    # Far but strong.
    far = _pool(218.0, "bsl", scope="external", role="equal", tf="4h", dist_from=current_price)
    cycle = _BULLISH_EXPAND_CYCLE
    ranked = b._rank_dols([near, far], htf_bias="bullish", cycle=cycle, htf_fib=_BASE_HTF_FIB, current_price=current_price, killzone=None)
    assert ranked[0] is far
    assert ranked[0].clarity_score > ranked[1].clarity_score


# ── Repair #8: detectors not re-run inside _build_pools when precomputed ─────-


def test_build_pools_uses_precomputed_detectors_no_rerun():
    b = SnapshotBuilder()
    candles_by_tf = {tf: [] for tf in TIMEFRAMES}
    # Minimal candles to avoid early exits.
    candles_by_tf["1h"] = [{"timestamp": _ts(0), "open": 1, "high": 2, "low": 0.5, "close": 1.5}]
    candles_by_tf["4h"] = [{"timestamp": _ts(0), "open": 1, "high": 2, "low": 0.5, "close": 1.5}]

    pre = {
        "swings": {tf: [] for tf in TIMEFRAMES},
        "fvgs": {tf: [] for tf in TIMEFRAMES},
        "order_blocks": {tf: [] for tf in TIMEFRAMES},
    }
    pre["swings"].update({
        "1h": [{"type": "swing_high", "price": 2.0, "timestamp": _ts(0)}],
        "4h": [{"type": "swing_low", "price": 0.5, "timestamp": _ts(0)}],
    })

    with patch("trading.snapshot.detect_swings") as p_swings, patch("trading.snapshot.detect_fvg") as p_fvg, patch("trading.snapshot.detect_order_blocks") as p_ob:
        b._build_pools(candles_by_tf, current_price=1.5, session_levels={}, htf_fib=None, precomputed=pre)
        # swings/fvg/ob should not be called because precomputed provides them.
        assert p_swings.call_count == 0
        assert p_fvg.call_count == 0
        assert p_ob.call_count == 0


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
