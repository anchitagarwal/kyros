"""snapshot.py — assemble a complete MarketSnapshot from a CandleWindow.

Runs every Phase 1 detector across all 5 timeframes and assembles a single
``MarketSnapshot`` dataclass with no LLM calls. The snapshot is the sole
input to the trigger engine and the LLM reasoning agent.

Key derivation rules (per the module layout spec):
  - htf_bias: detect_bos + detect_choch on 4h first; most recent confirmed
    break wins. Fall back to 1h. None if neither shows a confirmed break.
    htf_bias_source is populated iff htf_bias is not None.
  - all_pools: liquidity pools from equal H/L + prior levels + session levels.
    Sorted ascending by distance_points. confluence_count = pools within 0.1%.
  - nearest_dol: nearest unswept pool in htf_bias direction. None if no bias.
  - session_levels: all 15 keys; None for levels not yet formed.
  - fvgs/ifvgs/order_blocks/breaker_blocks/volume_imbalances: all 5 TFs,
    unmitigated/active only.
  - recent_sweeps/displacements/recent_inducements: all 5 TFs, last 10 only.
  - premium_discount/recent_swings: all 5 TFs.

Determinism: same window → identical snapshot (no wall-clock, no RNG).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo


def _parse_dt(ts: Any) -> datetime:
    """Parse a timestamp (datetime or ISO-8601 str) to an aware datetime."""
    if isinstance(ts, datetime):
        dt = ts
    else:
        dt = datetime.fromisoformat(str(ts))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_NY)
    return dt

from detectors.displacement import detect_displacement
from detectors.fair_value_gaps import detect_fvg, detect_ifvg
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

# Killzone windows (clock times in ET), matching sessions.py defaults.
_KILLZONES = [
    ("london_kz", time(2, 0), time(5, 0)),
    ("ny_am_kz", time(9, 30), time(11, 0)),
    ("ny_pm_kz", time(13, 30), time(15, 0)),
]


def _in_window(local_dt: datetime, start: time, end: time) -> bool:
    t = local_dt.time()
    if start <= end:
        return start <= t < end
    return t >= start or t < end


def _killzone_at(dt: datetime) -> str | None:
    """Return the killzone name containing ``dt`` (ET), or None."""
    local = dt.astimezone(_NY)
    for name, start, end in _KILLZONES:
        if _in_window(local, start, end):
            return name
    return None


def _session_at(dt: datetime) -> str | None:
    """Return the session name containing ``dt`` (ET), or None."""
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


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class LiquidityPool:
    """A single unswept liquidity level with proximity metadata."""

    level: float
    type: str  # "bsl" | "ssl" | "equal_highs" | "equal_lows" | "pdh" | "pdl" | "pwh" | "pwl"
    timeframe: str
    distance_points: float
    confluence_count: int
    swept: bool = False
    timestamp: Any = None

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "type": self.type,
            "timeframe": self.timeframe,
            "distance_points": round(self.distance_points, 2),
            "confluence_count": self.confluence_count,
            "swept": self.swept,
            "timestamp": str(self.timestamp) if self.timestamp is not None else None,
        }


@dataclass
class MarketSnapshot:
    """Complete market state at a point in time — the LLM's only input."""

    # Metadata
    instrument: str
    timestamp: datetime
    current_price: float
    # Session
    current_killzone: str | None
    current_session: str | None
    session_levels: dict[str, float | None]
    # HTF bias
    htf_bias: str | None  # "bullish" | "bearish" | None
    htf_bias_source: dict | None  # {timeframe, type, index, timestamp}
    recent_swings: dict[str, list[dict]]
    premium_discount: dict[str, list[dict]]
    # Entry structures (per TF, active/unmitigated only)
    fvgs: dict[str, list[dict]]
    ifvgs: dict[str, list[dict]]
    order_blocks: dict[str, list[dict]]
    breaker_blocks: dict[str, list[dict]]
    volume_imbalances: dict[str, list[dict]]
    opening_gaps: dict[str, list[dict]]
    # Triggers (per TF, last 10 only)
    recent_sweeps: dict[str, list[dict]]
    displacements: dict[str, list[dict]]
    recent_inducements: dict[str, list[dict]]
    # PO3
    po3_phase: dict[str, list[dict]]
    # DOL
    all_pools: list[LiquidityPool]
    nearest_dol: LiquidityPool | None

    def to_compact_dict(self) -> dict:
        """LLM payload: excludes raw candle lists, keeps summaries/levels only.

        Each detector's output is compressed — only active/unmitigated entries,
        capped at 5 most recent per timeframe, LLM-relevant fields only
        (level/zone, type, timestamp, mitigated flag — not candle indices).
        """
        return _compact_dict(self)


# ── Builder ───────────────────────────────────────────────────────────────────


class SnapshotBuilder:
    """Run all detectors across all timeframes and assemble a MarketSnapshot."""

    def __init__(self, instrument: str = "NQ", tz: str = TZ):
        self.instrument = instrument
        self.tz = tz

    def build(self, window, now: datetime | None = None) -> MarketSnapshot:
        """Build a snapshot from ``window`` (a CandleWindow).

        ``now`` defaults to the latest 1m candle timestamp.
        """
        # Gather candles per TF.
        candles_by_tf: dict[str, list[dict]] = {}
        for tf in TIMEFRAMES:
            try:
                candles_by_tf[tf] = window.to_list(tf)
            except KeyError:
                candles_by_tf[tf] = []

        # Determine "now" from the latest 1m candle if not given.
        if now is None:
            one_m = candles_by_tf.get("1m", [])
            now = one_m[-1]["timestamp"] if one_m else datetime.now(tz=_NY)
        now_dt = _parse_dt(now)
        current_price = candles_by_tf.get("1m", [{}])[-1].get("close", 0.0) if candles_by_tf.get("1m") else 0.0

        # Session + killzone.
        kz = _killzone_at(now_dt)
        sess = _session_at(now_dt)

        # Session levels (from 1m candles — a single trading day's worth).
        session_levels = detect_session_levels(candles_by_tf.get("1m", []), self.tz)

        # HTF bias: 4h first, then 1h.
        htf_bias, htf_bias_source = self._derive_htf_bias(candles_by_tf)

        # Per-TF detector runs.
        recent_swings: dict[str, list[dict]] = {}
        premium_discount: dict[str, list[dict]] = {}
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
                for d in (recent_swings, premium_discount, fvgs, ifvgs, order_blocks,
                          breaker_blocks, volume_imbalances, opening_gaps,
                          recent_sweeps, displacements, recent_inducements, po3_phase):
                    d[tf] = []
                continue

            swings = detect_swings(candles)
            recent_swings[tf] = [_swing_dict(s) for s in swings[-5:]]

            pd = detect_premium_discount(candles)
            premium_discount[tf] = [_pd_dict(p) for p in pd]

            # FVGs: keep all (active = not yet filled; detect_fvg emits all
            # confirmed gaps; we keep the most recent 5 for compactness).
            fvg_list = detect_fvg(candles)
            fvgs[tf] = [_fvg_dict(f) for f in fvg_list[-5:]]

            ifvgs[tf] = [_ifvg_dict(f) for f in detect_ifvg(candles)[-5:]]

            obs = detect_order_blocks(candles)
            # Active = unmitigated.
            order_blocks[tf] = [_ob_dict(o) for o in obs if not o.get("mitigated")][-5:]

            breaker_blocks[tf] = [_breaker_dict(b) for b in detect_breaker_blocks(candles)[-5:]]

            volume_imbalances[tf] = [_vi_dict(v) for v in detect_volume_imbalance(candles)[-5:]]

            opening_gaps[tf] = [_gap_dict(g) for g in detect_opening_gaps(candles, "day", self.tz)[-3:]]

            sweeps = detect_liquidity_sweeps(candles)
            recent_sweeps[tf] = [_sweep_dict(s) for s in sweeps[-10:]]

            disp = detect_displacement(candles)
            displacements[tf] = [_disp_dict(d) for d in disp[-10:]]

            idm = detect_inducement(candles)
            recent_inducements[tf] = [_idm_dict(d) for d in idm[-10:]]

            po3 = detect_power_of_three(candles, period="day", tz=self.tz)
            po3_phase[tf] = [_po3_dict(p) for p in po3[-3:]]

        # Liquidity pools + DOL (pass pre-computed session_levels to avoid a second scan).
        all_pools = self._build_pools(candles_by_tf, current_price, session_levels)
        nearest_dol = self._nearest_dol(all_pools, htf_bias, current_price)

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
        )

    # -- HTF bias -------------------------------------------------------------

    def _derive_htf_bias(
        self, candles_by_tf: dict[str, list[dict]]
    ) -> tuple[str | None, dict | None]:
        """Determine HTF bias from BOS/ChoCH on 4h, falling back to 1h.

        Takes the most recent confirmed break (BOS or ChoCH) on 4h. If none,
        falls back to 1h. Returns (bias, source) where bias is "bullish"/
        "bearish"/None and source is {timeframe, type, index, timestamp} or
        None. bias and source are always set together (or both None).
        """
        for tf in ("4h", "1h"):
            candles = candles_by_tf.get(tf, [])
            if not candles:
                continue
            bos = detect_bos(candles)
            choch = detect_choch(candles)
            # Merge and sort by break_index to find the most recent.
            events: list[dict] = []
            for b in bos:
                events.append({
                    "type": b["type"],
                    "index": b["break_index"],
                    "timestamp": b["timestamp"],
                })
            for c in choch:
                events.append({
                    "type": c["type"],
                    "index": c["break_index"],
                    "timestamp": c["timestamp"],
                })
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

    # -- Pools + DOL ----------------------------------------------------------

    def _build_pools(
        self,
        candles_by_tf: dict[str, list[dict]],
        current_price: float,
        session_levels: dict | None = None,
    ) -> list[LiquidityPool]:
        """Build all unswept liquidity pools, sorted by distance to price.

        Sources: equal highs/lows (detect_equal_levels), prior day/week levels
        (detect_prior_levels), and session highs/lows (detect_session_levels).
        confluence_count = number of other pools within 0.1% of this level.
        session_levels may be passed in to avoid a redundant detector call.
        """
        raw: list[tuple[float, str, str, Any]] = []  # (level, type, tf, timestamp)

        for tf in TIMEFRAMES:
            candles = candles_by_tf.get(tf, [])
            if not candles:
                continue
            # Equal highs/lows → BSL/SSL pools.
            for eq in detect_equal_levels(candles):
                ptype = "bsl" if eq["type"] == "equal_highs" else "ssl"
                raw.append((eq["level"], ptype, tf, eq["timestamp"]))
            # Prior day/week levels.
            for pl in detect_prior_levels(candles, "day", self.tz):
                ptype = "bsl" if pl["type"] in ("pdh", "pwh") else "ssl"
                raw.append((pl["level"], ptype, tf, pl["timestamp"]))
            for pl in detect_prior_levels(candles, "week", self.tz):
                ptype = "bsl" if pl["type"] in ("pdh", "pwh") else "ssl"
                raw.append((pl["level"], ptype, tf, pl["timestamp"]))

        # Session levels → BSL/SSL/reference pools.
        # _high → BSL; _low → SSL; _open → both (price draws to either side).
        sl = session_levels if session_levels is not None else detect_session_levels(candles_by_tf.get("1m", []), self.tz)
        for key, val in sl.items():
            if val is None:
                continue
            if key.endswith("_high"):
                raw.append((val, "bsl", "1m", None))
            elif key.endswith("_low"):
                raw.append((val, "ssl", "1m", None))
            elif key.endswith("_open"):
                raw.append((val, "bsl", "1m", None))
                raw.append((val, "ssl", "1m", None))

        if not raw:
            return []

        # Deduplicate by (level, type) keeping the first occurrence.
        seen: set[tuple] = set()
        unique: list[tuple[float, str, str, Any]] = []
        for level, ptype, tf, ts in raw:
            key = (round(level, 4), ptype)
            if key in seen:
                continue
            seen.add(key)
            unique.append((level, ptype, tf, ts))

        # Build pools with distance + confluence.
        pools: list[LiquidityPool] = []
        for level, ptype, tf, ts in unique:
            dist = abs(level - current_price)
            # confluence: count other pools within 0.1% of this level.
            band = abs(level) * 0.001
            conf = sum(
                1 for (l2, _t2, _tf2, _ts2) in unique
                if l2 is not level and abs(l2 - level) <= band
            )
            pools.append(LiquidityPool(
                level=level, type=ptype, timeframe=tf,
                distance_points=dist, confluence_count=conf,
                swept=False, timestamp=ts,
            ))

        # Sort ascending by distance_points.
        pools.sort(key=lambda p: p.distance_points)
        return pools

    def _nearest_dol(
        self, pools: list[LiquidityPool], htf_bias: str | None, current_price: float
    ) -> LiquidityPool | None:
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
        # pools are already sorted by distance; candidates inherit that order.
        return candidates[0]


# ── Compact serialization (LLM payload) ───────────────────────────────────────


def _compact_dict(snap: MarketSnapshot) -> dict:
    """Serialize a snapshot to a compact dict for the LLM.

    Excludes raw candle lists. Keeps counts, top-5 pools by distance, latest
    swing per TF, and at most 5 most-recent active entries per detector per TF.
    Only LLM-relevant fields: level/zone, type, timestamp, mitigated flag.

    Includes ALL detector summaries — opening_gaps is present so the LLM sees
    the full schema (no detector is silently starved).
    """
    def _ts(v):
        return str(v) if v is not None else None

    pools = [p.to_dict() for p in snap.all_pools[:5]]
    dol = snap.nearest_dol.to_dict() if snap.nearest_dol else None

    return {
        "instrument": snap.instrument,
        "timestamp": _ts(snap.timestamp),
        "current_price": round(snap.current_price, 2),
        "current_killzone": snap.current_killzone,
        "current_session": snap.current_session,
        "session_levels": {
            k: (round(v, 2) if v is not None else None)
            for k, v in snap.session_levels.items()
        },
        "htf_bias": snap.htf_bias,
        "htf_bias_source": snap.htf_bias_source,
        "recent_swings": {
            tf: (lst[-1] if lst else None) for tf, lst in snap.recent_swings.items()
        },
        "premium_discount": {
            tf: (lst[-1] if lst else None) for tf, lst in snap.premium_discount.items()
        },
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
    }


# ── Per-detector dict mappers (strip candle indices, keep LLM-relevant fields) ─


def _swing_dict(s: dict) -> dict:
    return {"type": s["type"], "price": round(s["price"], 2), "label": s.get("label")}


def _pd_dict(p: dict) -> dict:
    return {
        "type": p["type"],
        "range_high": round(p["range_high"], 2),
        "range_low": round(p["range_low"], 2),
        "equilibrium": round(p["equilibrium"], 2),
        "ote_zone": [round(p["ote_zone"][0], 2), round(p["ote_zone"][1], 2)],
        "direction": p["direction"],
    }


def _fvg_dict(f: dict) -> dict:
    return {
        "type": f["type"],
        "top": round(f["top"], 2),
        "bottom": round(f["bottom"], 2),
        "timestamp": str(f["timestamp"]),
    }


def _ifvg_dict(f: dict) -> dict:
    return {
        "type": f["type"],
        "top": round(f["top"], 2),
        "bottom": round(f["bottom"], 2),
        "timestamp": str(f["timestamp"]),
    }


def _ob_dict(o: dict) -> dict:
    return {
        "type": o["type"],
        "top": round(o["top"], 2),
        "bottom": round(o["bottom"], 2),
        "mitigated": o.get("mitigated", False),
        "timestamp": str(o["timestamp"]),
    }


def _breaker_dict(b: dict) -> dict:
    return {
        "type": b["type"],
        "timestamp": str(b["timestamp"]),
    }


def _vi_dict(v: dict) -> dict:
    return {
        "type": v["type"],
        "top": round(v["top"], 2),
        "bottom": round(v["bottom"], 2),
        "timestamp": str(v["timestamp"]),
    }


def _gap_dict(g: dict) -> dict:
    return {
        "type": g["type"],
        "top": round(g["top"], 2),
        "bottom": round(g["bottom"], 2),
        "timestamp": str(g["timestamp"]),
    }


def _sweep_dict(s: dict) -> dict:
    return {
        "type": s["type"],
        "swept_level": round(s["swept_level"], 2),
        "timestamp": str(s["timestamp"]),
    }


def _disp_dict(d: dict) -> dict:
    return {
        "type": d["type"],
        "strength": round(d["strength"], 2),
        "leaves_fvg": d.get("leaves_fvg", False),
        "timestamp": str(d["timestamp"]),
    }


def _idm_dict(d: dict) -> dict:
    return {
        "type": d["type"],
        "induced_level": round(d["induced_level"], 2),
        "timestamp": str(d["timestamp"]),
    }


def _po3_dict(p: dict) -> dict:
    return {
        "type": p["type"],
        "manipulation_direction": p.get("manipulation_direction"),
        "distribution_direction": p.get("distribution_direction"),
        "timestamp": str(p["timestamp"]),
    }
