"""snapshot.py — assemble a complete MarketSnapshot from a CandleWindow.

Phase 2B continuation: weighted DOL scoring.
- Adds LiquidityPool.clarity_score + score_breakdown.
- Adds MarketSnapshot.ranked_dols.
- Adds _score_pool (pure) and _rank_dols (direction-filtered ranking).
- Rewrites _dol_target to use ranking when cycle active, else ERL-only nearest.
- Dedup repair: merge coincident ERL sources on (level,type) keeping richest role.

Hard constraints:
- _nearest_dol must remain byte-identical.
- Deterministic: no wall-clock, no RNG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from .config import TradingConfig

from detectors.displacement import detect_displacement
from detectors.fair_value_gaps import detect_fvg, detect_ifvg
from detectors.fibonacci import detect_fibonacci
from detectors.inducement import detect_inducement
from detectors.liquidity import (
    detect_equal_levels,
    detect_liquidity_sweeps,
    detect_prior_levels,
)
from detectors.market_structure import detect_bos, detect_choch, detect_swings
from detectors.order_blocks import detect_breaker_blocks, detect_order_blocks
from detectors.power_of_three import detect_power_of_three
from detectors.premium_discount import detect_premium_discount
from detectors.sessions import detect_session_levels
from detectors.volume_imbalance import detect_opening_gaps, detect_volume_imbalance

__all__ = [
    "MarketSnapshot",
    "LiquidityPool",
    "SnapshotBuilder",
    "TIMEFRAMES",
    "TZ",
]

TIMEFRAMES = ("4h", "1h", "15m", "5m", "1m")
TZ = "America/New_York"
_NY = ZoneInfo(TZ)

_KILLZONES = [
    ("london_kz", time(2, 0), time(5, 0)),
    ("ny_am_kz", time(9, 30), time(11, 0)),
    ("ny_pm_kz", time(13, 30), time(15, 0)),
]


def _parse_dt(ts: Any) -> datetime:
    if isinstance(ts, datetime):
        dt = ts
    else:
        dt = datetime.fromisoformat(str(ts))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_NY)
    return dt


def _in_window(local_dt: datetime, start: time, end: time) -> bool:
    t = local_dt.time()
    if start <= end:
        return start <= t < end
    return t >= start or t < end


def _killzone_at(dt: datetime, killzones) -> str | None:
    local = dt.astimezone(_NY)
    for name, start, end in killzones:
        if _in_window(local, start, end):
            return name
    return None


def _session_at(dt: datetime) -> str | None:
    local = dt.astimezone(_NY)
    sessions = [
        ("asian", time(20, 0), time(0, 0)),
        ("london", time(2, 0), time(5, 0)),
        ("ny_am", time(7, 0), time(10, 0)),
        ("ny_pm", time(13, 0), time(16, 0)),
    ]
    for name, start, end in sessions:
        if _in_window(local, start, end):
            return name
    return None


def _bias_to_side(htf_bias: str | None) -> str | None:
    if htf_bias == "bullish":
        return "buyside"
    if htf_bias == "bearish":
        return "sellside"
    return None


@dataclass
class LiquidityPool:
    level: float
    type: str
    timeframe: str
    distance_points: float
    confluence_count: int
    swept: bool = False
    timestamp: Any = None
    scope: str = "external"
    role: str = ""
    clarity_score: float = 0.0
    score_breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "type": self.type,
            "timeframe": self.timeframe,
            "distance_points": round(self.distance_points, 2),
            "confluence_count": self.confluence_count,
            "swept": self.swept,
            "timestamp": str(self.timestamp) if self.timestamp is not None else None,
            "scope": self.scope,
            "role": self.role,
            "clarity_score": round(self.clarity_score, 6),
            "score_breakdown": self.score_breakdown,
        }


@dataclass
class MarketSnapshot:
    instrument: str
    timestamp: datetime
    current_price: float
    current_killzone: str | None
    current_session: str | None
    session_levels: dict[str, float | None]
    htf_bias: str | None
    htf_bias_source: dict | None
    recent_swings: dict[str, list[dict]]
    market_structure: dict[str, list[dict]]
    premium_discount: dict[str, list[dict]]
    fvgs: dict[str, list[dict]]
    ifvgs: dict[str, list[dict]]
    order_blocks: dict[str, list[dict]]
    breaker_blocks: dict[str, list[dict]]
    volume_imbalances: dict[str, list[dict]]
    opening_gaps: dict[str, list[dict]]
    recent_sweeps: dict[str, list[dict]]
    displacements: dict[str, list[dict]]
    recent_inducements: dict[str, list[dict]]
    po3_phase: dict[str, list[dict]]
    all_pools: list[LiquidityPool]
    nearest_dol: LiquidityPool | None
    pools_to_llm: int = 5
    fib_levels: dict[str, list[dict]] = field(default_factory=dict)
    liquidity_cycle: dict | None = None
    ranked_dols: list[LiquidityPool] = field(default_factory=list)
    dol_target: LiquidityPool | None = None

    def to_compact_dict(self) -> dict:
        return _compact_dict(self)


class SnapshotBuilder:
    def __init__(self, instrument: str = "NQ", tz: str = TZ, config: TradingConfig = TradingConfig()):
        self.instrument = instrument
        self.tz = tz
        self.config = config
        self._recency_caps = config.recency_caps_dict()
        self._killzones = config.killzone_windows_list()
        self._irl_sources = config.irl_sources_set()
        self._dol_w = config.dol_weights_dict()
        self._tf_w = config.tf_weights_dict()
        self._role_w = config.role_weights_dict()

    def _cap(self, field_name: str, default: int) -> int:
        return self._recency_caps.get(field_name, default)

    def build(self, window, now: datetime | None = None) -> MarketSnapshot:
        candles_by_tf: dict[str, list[dict]] = {}
        for tf in TIMEFRAMES:
            try:
                candles_by_tf[tf] = window.to_list(tf)
            except KeyError:
                candles_by_tf[tf] = []

        if now is None:
            one_m = candles_by_tf.get("1m", [])
            now = one_m[-1]["timestamp"] if one_m else datetime.now(tz=_NY)
        now_dt = _parse_dt(now)
        current_price = candles_by_tf.get("1m", [{}])[-1].get("close", 0.0) if candles_by_tf.get("1m") else 0.0

        kz = _killzone_at(now_dt, self._killzones)
        sess = _session_at(now_dt)
        session_levels = detect_session_levels(candles_by_tf.get("1m", []), self.tz)

        htf_bias, htf_bias_source = self._derive_htf_bias(candles_by_tf)

        recent_swings: dict[str, list[dict]] = {}
        market_structure: dict[str, list[dict]] = {}
        premium_discount: dict[str, list[dict]] = {}
        fib_levels: dict[str, list[dict]] = {}
        fvgs: dict[str, list[dict]] = {}
        ifvgs: dict[str, list[dict]] = {}
        order_blocks: dict[str, list[dict]] = {}
        breaker_blocks: dict[str, list[dict]] = {}
        volume_imbalances: dict[str, list[dict]] = {}
        opening_gaps: dict[str, list[dict]] = {}
        recent_sweeps: dict[str, list[dict]] = {}
        displacements: dict[str, list[dict]] = {}
        recent_inducements: dict[str, list[dict]] = {}
        po3_phase: dict[str, list[dict]] = {}

        for tf in TIMEFRAMES:
            candles = candles_by_tf[tf]
            if not candles:
                for d in (recent_swings, market_structure, premium_discount, fib_levels, fvgs, ifvgs,
                          order_blocks, breaker_blocks, volume_imbalances, opening_gaps, recent_sweeps,
                          displacements, recent_inducements, po3_phase):
                    d[tf] = []
                continue

            swings = detect_swings(candles)
            recent_swings[tf] = [_swing_dict(s) for s in swings[-self._cap("recent_swings", 5):]]

            breaks = detect_bos(candles) + detect_choch(candles)
            breaks.sort(key=lambda b: b["break_index"])
            market_structure[tf] = [_structure_dict(b) for b in breaks[-self._cap("market_structure", 10):]]

            pd = detect_premium_discount(candles)
            premium_discount[tf] = [_pd_dict(p) for p in pd]

            fib = detect_fibonacci(
                candles,
                lookback=self.config.fib_anchor_lookback,
                retracements=self.config.fib_retracements,
                ote_grid=self.config.fib_ote_grid,
                ote_primary=self.config.fib_ote_primary,
                golden_pocket=self.config.fib_golden_pocket,
                retracement_target=self.config.fib_retracement_target,
                extensions=self.config.fib_extensions,
                swings=swings,
            )
            fib_levels[tf] = fib[-self._cap("fib_levels", 1):]

            fvg_list = detect_fvg(candles)
            fvgs[tf] = [_fvg_dict(f) for f in fvg_list[-self._cap("fvgs", 5):]]
            ifvgs[tf] = [_ifvg_dict(f) for f in detect_ifvg(candles)[-self._cap("ifvgs", 5):]]

            obs = detect_order_blocks(candles)
            order_blocks[tf] = [_ob_dict(o) for o in obs if not o.get("mitigated")][-self._cap("order_blocks", 5):]

            breaker_blocks[tf] = [_breaker_dict(b) for b in detect_breaker_blocks(candles)[-self._cap("breaker_blocks", 5):]]
            volume_imbalances[tf] = [_vi_dict(v) for v in detect_volume_imbalance(candles)[-self._cap("volume_imbalances", 5):]]
            opening_gaps[tf] = [_gap_dict(g) for g in detect_opening_gaps(candles, "day", self.tz)[-self._cap("opening_gaps", 3):]]

            sweeps = detect_liquidity_sweeps(candles)
            recent_sweeps[tf] = [_sweep_dict(s) for s in sweeps[-self._cap("recent_sweeps", 10):]]

            disp = detect_displacement(candles)
            displacements[tf] = [_disp_dict(d) for d in disp[-self._cap("displacements", 10):]]

            idm = detect_inducement(candles)
            recent_inducements[tf] = [_idm_dict(d) for d in idm[-self._cap("recent_inducements", 10):]]

            po3 = detect_power_of_three(candles, period="day", tz=self.tz)
            po3_phase[tf] = [_po3_dict(p) for p in po3[-self._cap("po3_phase", 3):]]

        htf_fib = self._htf_fib(fib_levels)

        all_pools = self._build_pools(
            candles_by_tf,
            current_price,
            session_levels,
            htf_fib,
            precomputed={
                "swings": recent_swings,
                "fvgs": fvgs,
                "order_blocks": order_blocks,
            },
        )
        nearest_dol = self._nearest_dol(all_pools, htf_bias, current_price)

        liquidity_cycle = self._derive_liquidity_cycle(recent_sweeps, market_structure, displacements, htf_bias)

        ranked_dols: list[LiquidityPool] = []
        if liquidity_cycle is not None and self.config.dol_use_cycle:
            ranked_dols = self._rank_dols(
                all_pools,
                htf_bias=htf_bias,
                cycle=liquidity_cycle,
                htf_fib=htf_fib,
                current_price=current_price,
                killzone=kz,
            )

        dol_target = self._dol_target(all_pools, liquidity_cycle, htf_bias, current_price, kz, htf_fib)

        return MarketSnapshot(
            instrument=self.instrument,
            timestamp=now_dt,
            current_price=current_price,
            current_killzone=kz,
            current_session=sess,
            session_levels=session_levels,
            htf_bias=htf_bias,
            htf_bias_source=htf_bias_source,
            recent_swings=recent_swings,
            market_structure=market_structure,
            premium_discount=premium_discount,
            fvgs=fvgs,
            ifvgs=ifvgs,
            order_blocks=order_blocks,
            breaker_blocks=breaker_blocks,
            volume_imbalances=volume_imbalances,
            opening_gaps=opening_gaps,
            recent_sweeps=recent_sweeps,
            displacements=displacements,
            recent_inducements=recent_inducements,
            po3_phase=po3_phase,
            all_pools=all_pools,
            nearest_dol=nearest_dol,
            pools_to_llm=self.config.pools_to_llm,
            fib_levels=fib_levels,
            liquidity_cycle=liquidity_cycle,
            ranked_dols=ranked_dols,
            dol_target=dol_target,
        )

    def _derive_htf_bias(self, candles_by_tf: dict[str, list[dict]]) -> tuple[str | None, dict | None]:
        for tf in self.config.htf_tf_order:
            candles = candles_by_tf.get(tf, [])
            if not candles:
                continue
            bos = detect_bos(candles)
            choch = detect_choch(candles)
            events: list[dict] = []
            for b in bos:
                events.append({"type": b["type"], "index": b["break_index"], "timestamp": b["timestamp"]})
            for c in choch:
                events.append({"type": c["type"], "index": c["break_index"], "timestamp": c["timestamp"]})
            if not events:
                continue
            events.sort(key=lambda e: e["index"])
            latest = events[-1]
            bias = "bullish" if "bullish" in latest["type"] else "bearish"
            source = {
                "timeframe": tf,
                "type": latest["type"],
                "index": latest["index"],
                "timestamp": str(latest["timestamp"]),
            }
            return bias, source
        return None, None

    def _htf_fib(self, fib_levels: dict[str, list[dict]]) -> dict | None:
        for tf in self.config.htf_tf_order:
            lst = fib_levels.get(tf, [])
            if lst:
                return lst[-1]
        return None

    def _build_pools(
        self,
        candles_by_tf: dict[str, list[dict]],
        current_price: float,
        session_levels: dict | None = None,
        htf_fib: dict | None = None,
        precomputed: dict | None = None,
    ) -> list[LiquidityPool]:
        raw: list[tuple[float, str, str, Any, str, str]] = []

        for tf in TIMEFRAMES:
            candles = candles_by_tf.get(tf, [])
            if not candles:
                continue
            for eq in detect_equal_levels(candles):
                ptype = "bsl" if eq["type"] == "equal_highs" else "ssl"
                raw.append((eq["level"], ptype, tf, eq["timestamp"], "external", "equal"))
            for pl in detect_prior_levels(candles, "day", self.tz):
                ptype = "bsl" if pl["type"] in ("pdh", "pwh") else "ssl"
                raw.append((pl["level"], ptype, tf, pl["timestamp"], "external", "prior"))
            for pl in detect_prior_levels(candles, "week", self.tz):
                ptype = "bsl" if pl["type"] in ("pdh", "pwh") else "ssl"
                raw.append((pl["level"], ptype, tf, pl["timestamp"], "external", "prior"))

            swings = detect_swings(candles)
            highs = [s for s in swings if s["type"] == "swing_high"]
            lows = [s for s in swings if s["type"] == "swing_low"]
            if highs:
                sh = highs[-1]
                raw.append((sh["price"], "bsl", tf, sh["timestamp"], "external", "swing"))
            if lows:
                sl = lows[-1]
                raw.append((sl["price"], "ssl", tf, sl["timestamp"], "external", "swing"))

        sl = session_levels if session_levels is not None else detect_session_levels(candles_by_tf.get("1m", []), self.tz)
        for key, val in sl.items():
            if val is None:
                continue
            if key.endswith("_high"):
                raw.append((val, "bsl", "1m", None, "external", "session"))
            elif key.endswith("_low"):
                raw.append((val, "ssl", "1m", None, "external", "session"))
            elif key.endswith("_open"):
                raw.append((val, "bsl", "1m", None, "external", "session"))
                raw.append((val, "ssl", "1m", None, "external", "session"))

        htf_tf = None
        for tf in self.config.htf_tf_order:
            if candles_by_tf.get(tf):
                htf_tf = tf
                break
        if htf_tf is not None:
            htf_candles = candles_by_tf[htf_tf]
            if "fvg" in self._irl_sources:
                for fvg in detect_fvg(htf_candles):
                    raw.append((fvg["midpoint"], _pool_type_for(fvg["midpoint"], current_price), htf_tf, fvg["timestamp"], "internal", "fvg_ce"))
            if "order_block" in self._irl_sources:
                for ob in detect_order_blocks(htf_candles):
                    if ob.get("mitigated"):
                        continue
                    centre = (ob["top"] + ob["bottom"]) / 2.0
                    raw.append((centre, _pool_type_for(centre, current_price), htf_tf, ob["timestamp"], "internal", "ob_ce"))
            if htf_fib is not None:
                if "equilibrium" in self._irl_sources:
                    eq = htf_fib["equilibrium"]
                    raw.append((eq, _pool_type_for(eq, current_price), htf_tf, htf_fib.get("timestamp"), "internal", "equilibrium"))
                if "ote" in self._irl_sources:
                    ote = htf_fib["ote"]["primary"]
                    raw.append((ote, _pool_type_for(ote, current_price), htf_tf, htf_fib.get("timestamp"), "internal", "ote"))

        if not raw:
            return []

        role_priority = {"equal": 4, "prior": 3, "session": 2, "swing": 1}

        merged: dict[tuple[float, str], tuple[float, str, str, Any, str, str]] = {}
        for level, ptype, tf, ts, scope, role in raw:
            key = (round(level, 4), ptype)
            if key not in merged:
                merged[key] = (level, ptype, tf, ts, scope, role)
                continue
            prev = merged[key]
            prev_role = prev[5]
            if role_priority.get(role, 0) > role_priority.get(prev_role, 0):
                merged[key] = (level, ptype, tf, ts, scope, role)

        unique = list(merged.values())

        pools: list[LiquidityPool] = []
        for level, ptype, tf, ts, scope, role in unique:
            dist = abs(level - current_price)
            band = abs(level) * self.config.confluence_band_pct
            conf = sum(
                1 for (l2, t2, _tf2, _ts2, _sc2, _rl2) in unique
                if (t2 == ptype) and (l2 is not level) and abs(l2 - level) <= band
            )
            pools.append(LiquidityPool(level=level, type=ptype, timeframe=tf, distance_points=dist, confluence_count=conf, swept=False, timestamp=ts, scope=scope, role=role))

        if htf_fib is not None:
            rlo = htf_fib["range_low"]
            rhi = htf_fib["range_high"]
            for p in pools:
                p.scope = "internal" if (rlo < p.level < rhi) else "external"

        pools.sort(key=lambda p: p.distance_points)
        return pools

    def _nearest_dol(self, pools: list[LiquidityPool], htf_bias: str | None, current_price: float) -> LiquidityPool | None:
        """Nearest unswept pool in the htf_bias direction.

        bullish → nearest BSL pool ABOVE current_price (target liquidity above).
        bearish → nearest SSL pool BELOW current_price (target liquidity below).
        None if htf_bias is None or no qualifying pool exists.
        """
        if htf_bias is None:
            return None
        if htf_bias == "bullish":
            candidates = [p for p in pools if p.type == "bsl" and p.level > current_price]
        else:  # bearish
            candidates = [p for p in pools if p.type == "ssl" and p.level < current_price]
        if not candidates:
            return None
        return candidates[0]

    def _derive_liquidity_cycle(self, recent_sweeps: dict[str, list[dict]], market_structure: dict[str, list[dict]], displacements: dict[str, list[dict]], htf_bias: str | None) -> dict | None:
        sweep_entries: list[tuple[datetime, int, str, dict]] = []
        for tf, lst in recent_sweeps.items():
            tf_rank = TIMEFRAMES.index(tf) if tf in TIMEFRAMES else len(TIMEFRAMES)
            for s in lst:
                try:
                    ts = _parse_dt(s["timestamp"])
                except (ValueError, TypeError):
                    continue
                sweep_entries.append((ts, tf_rank, tf, s))
        if not sweep_entries:
            return None

        latest_ts, _rank, _tf, latest_sweep = max(sweep_entries, key=lambda e: (e[0], -e[1]))
        is_bsl = latest_sweep["type"] == "sweep_bsl"
        last_swept_erl_side = "buyside" if is_bsl else "sellside"
        target_erl_side = "sellside" if is_bsl else "buyside"

        reversal_tokens = (
            ("displacement_bearish", "bos_bearish", "choch_bearish") if is_bsl else ("displacement_bullish", "bos_bullish", "choch_bullish")
        )

        has_reversal = False
        for tf, lst in displacements.items():
            for d in lst:
                try:
                    dts = _parse_dt(d["timestamp"])
                except (ValueError, TypeError):
                    continue
                if dts > latest_ts and d["type"] in reversal_tokens:
                    has_reversal = True
                    break
            if has_reversal:
                break
        if not has_reversal:
            for tf, lst in market_structure.items():
                for b in lst:
                    try:
                        bts = _parse_dt(b["timestamp"])
                    except (ValueError, TypeError):
                        continue
                    if bts > latest_ts and b["type"] in reversal_tokens:
                        has_reversal = True
                        break
                if has_reversal:
                    break

        current_leg = "expand_to_erl" if has_reversal else "seek_irl"
        next_draw = "erl" if current_leg == "expand_to_erl" else "irl"

        bias_side = _bias_to_side(htf_bias)
        agrees_with_htf_bias = bias_side is not None and bias_side == target_erl_side

        return {
            "last_swept_erl_side": last_swept_erl_side,
            "last_swept_level": latest_sweep.get("swept_level"),
            "last_swept_timestamp": latest_sweep.get("timestamp"),
            "current_leg": current_leg,
            "next_draw": next_draw,
            "target_erl_side": target_erl_side,
            "agrees_with_htf_bias": agrees_with_htf_bias,
        }

    def _score_pool(
        self,
        pool: LiquidityPool,
        *,
        htf_bias: str | None,
        cycle: dict | None,
        htf_fib: dict | None,
        current_price: float,
        killzone: str | None,
        pools: list[LiquidityPool],
    ) -> tuple[float, dict]:
        w = self._dol_w
        breakdown: dict[str, float] = {}

        tf_term = self._tf_w.get(pool.timeframe, 0.0) * w.get("timeframe", 0.0)
        breakdown["timeframe"] = tf_term

        role_term = self._role_w.get(pool.role, 0.0) * w.get("role", 0.0)
        breakdown["role"] = role_term

        bias_term = 0.0
        if htf_bias == "bullish" and pool.type == "bsl":
            bias_term = 1.0 * w.get("bias_align", 0.0)
        elif htf_bias == "bearish" and pool.type == "ssl":
            bias_term = 1.0 * w.get("bias_align", 0.0)
        breakdown["bias_align"] = bias_term

        cycle_term = 0.0
        if cycle is not None:
            leg = cycle.get("current_leg")
            if leg == "seek_irl":
                last_side = cycle.get("last_swept_erl_side")
                reversal_ok = (last_side == "buyside" and pool.level < current_price) or (last_side == "sellside" and pool.level > current_price)
                if pool.scope == "internal" and reversal_ok:
                    cycle_term = 1.0 * w.get("cycle_align", 0.0)
            elif leg == "expand_to_erl":
                target_side = cycle.get("target_erl_side")
                if pool.scope == "external" and ((target_side == "buyside" and pool.type == "bsl") or (target_side == "sellside" and pool.type == "ssl")):
                    cycle_term = 1.0 * w.get("cycle_align", 0.0)
        breakdown["cycle_align"] = cycle_term

        conf_term = pool.confluence_count * w.get("confluence", 0.0)
        breakdown["confluence"] = conf_term

        pd_term = 0.0
        if htf_fib is not None:
            eq = htf_fib.get("equilibrium")
            if eq is not None:
                if current_price > eq and pool.level < eq:
                    pd_term = 1.0 * w.get("pd_align", 0.0)
                elif current_price < eq and pool.level > eq:
                    pd_term = 1.0 * w.get("pd_align", 0.0)
        breakdown["pd_align"] = pd_term

        clean_pen = 0.0
        opp_type = "ssl" if pool.type == "bsl" else "bsl"
        lo = min(current_price, pool.level)
        hi = max(current_price, pool.level)
        opposing = [p for p in pools if (not p.swept) and p.scope == "external" and p.type == opp_type and lo < p.level < hi]
        clean_pen = -len(opposing) * w.get("clean_path", 0.0)
        breakdown["clean_path"] = clean_pen

        prox_term = 0.0
        if htf_fib is not None:
            r = abs(htf_fib.get("range_high", 0.0) - htf_fib.get("range_low", 0.0))
            if r:
                prox_term = -(pool.distance_points / r) * w.get("proximity", 0.0)
        breakdown["proximity"] = prox_term

        kz_term = (1.0 * w.get("killzone", 0.0)) if killzone else 0.0
        breakdown["killzone"] = kz_term

        score = sum(breakdown.values())
        return score, breakdown

    def _rank_dols(
        self,
        pools: list[LiquidityPool],
        *,
        htf_bias: str | None,
        cycle: dict | None,
        htf_fib: dict | None,
        current_price: float,
        killzone: str | None,
    ) -> list[LiquidityPool]:
        if cycle is None:
            return []
        target_side = cycle.get("target_erl_side")
        if target_side == "buyside":
            candidates = [p for p in pools if p.level > current_price and p.type == "bsl"]
        else:
            candidates = [p for p in pools if p.level < current_price and p.type == "ssl"]

        ranked: list[LiquidityPool] = []
        for p in candidates:
            score, breakdown = self._score_pool(
                p,
                htf_bias=htf_bias,
                cycle=cycle,
                htf_fib=htf_fib,
                current_price=current_price,
                killzone=killzone,
                pools=pools,
            )
            p.clarity_score = score
            p.score_breakdown = breakdown
            ranked.append(p)

        ranked.sort(key=lambda p: (-p.clarity_score, p.distance_points))
        return ranked

    def _dol_target(
        self,
        pools: list[LiquidityPool],
        cycle: dict | None,
        htf_bias: str | None,
        current_price: float,
        killzone: str | None,
        htf_fib: dict | None,
    ) -> LiquidityPool | None:
        if cycle is None or not self.config.dol_use_cycle:
            erl_only = [p for p in pools if p.scope == "external"]
            return self._nearest_dol(erl_only, htf_bias, current_price)

        ranked = self._rank_dols(
            pools,
            htf_bias=htf_bias,
            cycle=cycle,
            htf_fib=htf_fib,
            current_price=current_price,
            killzone=killzone,
        )
        return ranked[0] if ranked else None


def _compact_dict(snap: MarketSnapshot) -> dict:
    def _ts(v):
        return str(v) if v is not None else None

    pools = [p.to_dict() for p in snap.all_pools[: snap.pools_to_llm]]
    dol = snap.nearest_dol.to_dict() if snap.nearest_dol else None
    dol_target = snap.dol_target.to_dict() if snap.dol_target else None

    ranked = [_ranked_dol_dict(p) for p in snap.ranked_dols[: snap.pools_to_llm]]

    return {
        "instrument": snap.instrument,
        "timestamp": _ts(snap.timestamp),
        "current_price": round(snap.current_price, 2),
        "current_killzone": snap.current_killzone,
        "current_session": snap.current_session,
        "session_levels": {k: (round(v, 2) if v is not None else None) for k, v in snap.session_levels.items()},
        "htf_bias": snap.htf_bias,
        "htf_bias_source": snap.htf_bias_source,
        "recent_swings": {tf: (lst[-1] if lst else None) for tf, lst in snap.recent_swings.items()},
        "market_structure": snap.market_structure,
        "premium_discount": {tf: (lst[-1] if lst else None) for tf, lst in snap.premium_discount.items()},
        "fib_levels": {tf: (_fib_dict(lst[-1]) if lst else None) for tf, lst in snap.fib_levels.items()},
        "fvgs": snap.fvgs,
        "ifvgs": snap.ifvgs,
        "order_blocks": snap.order_blocks,
        "breaker_blocks": snap.breaker_blocks,
        "volume_imbalances": snap.volume_imbalances,
        "opening_gaps": snap.opening_gaps,
        "recent_sweeps": snap.recent_sweeps,
        "displacements": snap.displacements,
        "recent_inducements": snap.recent_inducements,
        "po3_phase": snap.po3_phase,
        "all_pools": pools,
        "nearest_dol": dol,
        "liquidity_cycle": snap.liquidity_cycle,
        "ranked_dols": ranked,
        "dol_target": dol_target,
    }


def _ranked_dol_dict(p: LiquidityPool) -> dict:
    d = p.to_dict()
    return d


def _pool_type_for(level: float, current_price: float) -> str:
    return "bsl" if level > current_price else "ssl"


def _swing_dict(s: dict) -> dict:
    return {"type": s["type"], "price": round(s["price"], 2), "label": s.get("label")}


def _structure_dict(b: dict) -> dict:
    return {"type": b["type"], "break_price": round(b["break_price"], 2), "timestamp": str(b["timestamp"])}


def _pd_dict(p: dict) -> dict:
    return {
        "type": p["type"],
        "range_high": round(p["range_high"], 2),
        "range_low": round(p["range_low"], 2),
        "equilibrium": round(p["equilibrium"], 2),
        "ote_zone": [round(p["ote_zone"][0], 2), round(p["ote_zone"][1], 2)],
        "direction": p["direction"],
    }


def _fib_dict(fib: dict) -> dict:
    return {
        "direction": fib["direction"],
        "equilibrium": round(fib["equilibrium"], 2),
        "golden_pocket": [round(x, 2) for x in fib["golden_pocket"]],
        "ote_primary": round(fib["ote"]["primary"], 2),
        "ote_zone": [round(x, 2) for x in fib["ote"]["zone"]],
        "retracement_target": round(fib["retracement_target"], 2),
        "extensions": {k: round(v, 2) for k, v in fib["extensions"].items()},
        "premium_array": fib["premium_array"],
    }


def _fvg_dict(f: dict) -> dict:
    return {"type": f["type"], "top": round(f["top"], 2), "bottom": round(f["bottom"], 2), "timestamp": str(f["timestamp"])}


def _ifvg_dict(f: dict) -> dict:
    return {"type": f["type"], "top": round(f["top"], 2), "bottom": round(f["bottom"], 2), "timestamp": str(f["timestamp"])}


def _ob_dict(o: dict) -> dict:
    return {"type": o["type"], "top": round(o["top"], 2), "bottom": round(o["bottom"], 2), "mitigated": o.get("mitigated", False), "timestamp": str(o["timestamp"])}


def _breaker_dict(b: dict) -> dict:
    return {"type": b["type"], "timestamp": str(b["timestamp"])}


def _vi_dict(v: dict) -> dict:
    return {"type": v["type"], "top": round(v["top"], 2), "bottom": round(v["bottom"], 2), "timestamp": str(v["timestamp"])}


def _gap_dict(g: dict) -> dict:
    return {"type": g["type"], "top": round(g["top"], 2), "bottom": round(g["bottom"], 2), "timestamp": str(g["timestamp"])}


def _sweep_dict(s: dict) -> dict:
    return {"type": s["type"], "swept_level": round(s["swept_level"], 2), "timestamp": str(s["timestamp"])}


def _disp_dict(d: dict) -> dict:
    return {"type": d["type"], "strength": round(d.get("strength", 0.0), 2), "leaves_fvg": d.get("leaves_fvg", False), "timestamp": str(d["timestamp"])}


def _idm_dict(d: dict) -> dict:
    return {"type": d["type"], "induced_level": round(d["induced_level"], 2), "timestamp": str(d["timestamp"])}


def _po3_dict(p: dict) -> dict:
    return {"type": p["type"], "manipulation_direction": p.get("manipulation_direction"), "distribution_direction": p.get("distribution_direction"), "timestamp": str(p["timestamp"])}
