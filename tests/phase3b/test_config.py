"""test_config.py — TradingConfig defaults reproduce current snapshot/trigger/
cooldown/alert behavior, and config_hash() is stable/deterministic."""

import json
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from trading.config import TradingConfig
from trading.alert import AlertPayload, validate_rr
from trading.cooldown import CooldownState, NO_TRADE_COOLDOWN_MINUTES
from trading.trigger import TriggerEngine
from trading.snapshot import SnapshotBuilder, _KILLZONES


# ── Default values match today's literals (audited per call site) ─────────────


def test_default_rr_min():
    assert TradingConfig().rr_min == 1.0


def test_default_conviction_min_is_40_byte_preserving():
    """O0: conviction enforced in the LLM prompt at 40; default 40 is a no-op
    for every current input (all taken trades have conviction >= 40)."""
    assert TradingConfig().conviction_min == 40


def test_default_no_trade_cooldown_minutes():
    assert TradingConfig().no_trade_cooldown_minutes == 5
    assert NO_TRADE_COOLDOWN_MINUTES == 5


def test_default_confluence_band_pct():
    assert TradingConfig().confluence_band_pct == 0.001


def test_default_pools_to_llm():
    assert TradingConfig().pools_to_llm == 5


def test_default_htf_tf_order():
    assert TradingConfig().htf_tf_order == ("4h", "1h")


def test_default_soft_trigger_order():
    assert TradingConfig().soft_trigger_order == ("fvg", "ifvg", "sweep", "displacement")


def test_default_recency_caps_match_snapshot_literals():
    """recency_caps defaults equal the [-N:] slices in SnapshotBuilder.build."""
    caps = TradingConfig().recency_caps_dict()
    # The slices audited from snapshot.py (pre-threading).
    assert caps["recent_sweeps"] == 10
    assert caps["displacements"] == 10
    assert caps["recent_inducements"] == 10
    assert caps["market_structure"] == 10
    assert caps["recent_swings"] == 5
    assert caps["fvgs"] == 5
    assert caps["ifvgs"] == 5
    assert caps["order_blocks"] == 5
    assert caps["breaker_blocks"] == 5
    assert caps["volume_imbalances"] == 5
    assert caps["opening_gaps"] == 3
    assert caps["po3_phase"] == 3


def test_default_killzone_windows_match_snapshot_literals():
    """killzone_windows defaults equal snapshot._KILLZONES."""
    kz = TradingConfig().killzone_windows_list()
    # Same names, start times, end times as _KILLZONES.
    assert kz == _KILLZONES


def test_default_soft_trigger_tf_map():
    m = TradingConfig().soft_trigger_tf_map_dict()
    assert m["fvg"] == ("5m", "15m")
    assert m["ifvg"] == ("5m", "15m")
    assert m["sweep"] == ("15m", "5m")
    assert m["displacement"] == ("5m", "1m")


# ── config_hash: deterministic, stable, differs on change ─────────────────────


def test_config_hash_stable_across_constructions():
    assert TradingConfig().config_hash() == TradingConfig().config_hash()


def test_config_hash_does_not_use_builtin_hash():
    """config_hash must be stable across Python hash-seed randomization.

    We can't change PYTHONHASHSEED in-process, but we assert the hash is a
    64-hex-char sha256 (not the builtin int hash) and is reproducible.
    """
    h = TradingConfig().config_hash()
    assert isinstance(h, str)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_config_hash_differs_on_scalar_change():
    base = TradingConfig().config_hash()
    changed = TradingConfig(rr_min=2.0).config_hash()
    assert changed != base


def test_config_hash_differs_on_conviction_change():
    base = TradingConfig().config_hash()
    changed = TradingConfig(conviction_min=50).config_hash()
    assert changed != base


def test_config_hash_differs_on_recency_change():
    base = TradingConfig().config_hash()
    changed = TradingConfig(
        recency_caps=TradingConfig().recency_caps + (("extra_field", 7),)
    ).config_hash()
    assert changed != base


def test_config_hash_differs_on_killzone_change():
    base = TradingConfig().config_hash()
    changed = TradingConfig(
        killzone_windows=(("custom_kz", ("01:00", "02:00")),)
    ).config_hash()
    assert changed != base


def test_config_hash_insensitive_to_mapping_insertion_order():
    """Two configs with the same logical mappings (different tuple order) hash
    identically — canonicalization sorts keys."""
    caps_a = (("fvgs", 5), ("obs", 5))
    caps_b = (("obs", 5), ("fvgs", 5))
    # Build via the same field with reordered tuples.
    cfg_a = TradingConfig(recency_caps=caps_a)
    cfg_b = TradingConfig(recency_caps=caps_b)
    assert cfg_a.config_hash() == cfg_b.config_hash()


def test_short_hash_is_prefix_of_config_hash():
    cfg = TradingConfig()
    assert cfg.short_hash() == cfg.config_hash()[:12]


# ── Behavior-preserving: default config == no config ──────────────────────────


def test_validate_rr_default_config_matches_no_config():
    """validate_rr(alert) == validate_rr(alert, TradingConfig())."""
    alert = AlertPayload(
        bias="long", conviction=70,
        entry_zone=(100.0, 100.0), stop=95.0, target=110.0,
    )
    assert validate_rr(alert).risk_reward == validate_rr(alert, TradingConfig()).risk_reward
    assert validate_rr(alert).bias == validate_rr(alert, TradingConfig()).bias


def test_validate_rr_conviction_gate_default_is_noop_for_valid_trades():
    """A trade with conviction >= 40 passes the default conviction gate."""
    alert = AlertPayload(
        bias="long", conviction=60,
        entry_zone=(100.0, 100.0), stop=95.0, target=110.0,
    )
    out = validate_rr(alert, TradingConfig())
    assert out.bias == "long"
    assert out.no_trade_reason is None


def test_validate_rr_conviction_gate_activates_below_default():
    """A trade with conviction < 40 is downgraded (the gate is real, just a
    no-op for current inputs where the LLM already enforces >= 40)."""
    alert = AlertPayload(
        bias="long", conviction=30,
        entry_zone=(100.0, 100.0), stop=95.0, target=110.0,
    )
    out = validate_rr(alert, TradingConfig())
    assert out.bias == "no_trade"
    assert out.no_trade_reason == "conviction_below_min"


def test_cooldown_default_config_matches_module_constant():
    """CooldownState with default config uses 5 minutes (the module constant)."""
    cd = CooldownState()
    assert cd.config.no_trade_cooldown_minutes == NO_TRADE_COOLDOWN_MINUTES


# ── SnapshotBuilder threading: default config == no config (byte-identical) ────


def test_snapshot_builder_accepts_config():
    """SnapshotBuilder(config=TradingConfig()) constructs without error."""
    b = SnapshotBuilder(config=TradingConfig())
    assert b.config is not None
    assert b.config.confluence_band_pct == 0.001


def test_snapshot_builder_default_config_killzones_match_module():
    """The builder's cached killzones equal _KILLZONES under the default config."""
    b = SnapshotBuilder()
    assert b._killzones == _KILLZONES


def test_snapshot_builder_nondefault_pools_to_llm_changes_compact_dict():
    """A non-default pools_to_llm actually changes the compact dict's all_pools
    length — proving the knob is wired (not dead)."""
    from trading.candle_source import MockCandleSource, TIMEFRAMES
    from trading.candle_window import CandleWindow, DEFAULT_SIZES

    src = MockCandleSource("trending_up", n_bars=100)
    w = CandleWindow(DEFAULT_SIZES)
    while not src.is_done():
        w.update(src.next())

    snap_default = SnapshotBuilder().build(w)
    snap_capped = SnapshotBuilder(config=TradingConfig(pools_to_llm=2)).build(w)

    cd_default = snap_default.to_compact_dict()
    cd_capped = snap_capped.to_compact_dict()
    # The capped version sends at most 2 pools; the default at most 5.
    assert len(cd_capped["all_pools"]) <= 2
    # If there are >2 pools, the cap actually bit.
    if len(cd_default["all_pools"]) > 2:
        assert len(cd_capped["all_pools"]) == 2
        assert len(cd_capped["all_pools"]) < len(cd_default["all_pools"])


def test_snapshot_builder_nondefault_recency_cap_changes_output():
    """A non-default recency_cap actually changes the detector output length."""
    from trading.candle_source import MockCandleSource, TIMEFRAMES
    from trading.candle_window import CandleWindow, DEFAULT_SIZES

    src = MockCandleSource("sweep_and_fvg", n_bars=100)
    w = CandleWindow(DEFAULT_SIZES)
    while not src.is_done():
        w.update(src.next())

    snap_default = SnapshotBuilder().build(w)
    # Override recent_sweeps cap to 2.
    cfg = TradingConfig(
        recency_caps=TradingConfig().recency_caps + (("recent_sweeps", 2),)
    )
    # Replace the recent_sweeps entry rather than append a duplicate.
    caps = {k: v for k, v in TradingConfig().recency_caps}
    caps["recent_sweeps"] = 2
    cfg = TradingConfig(recency_caps=tuple(sorted(caps.items())))
    snap_capped = SnapshotBuilder(config=cfg).build(w)

    # The capped recent_sweeps["5m"] is at most 2; default at most 10.
    assert len(snap_capped.recent_sweeps["5m"]) <= 2
    if len(snap_default.recent_sweeps["5m"]) > 2:
        assert len(snap_capped.recent_sweeps["5m"]) < len(snap_default.recent_sweeps["5m"])
