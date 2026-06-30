"""candle_source.py — uniform candle feed (mock + replay), fully offline.

The trading loop consumes candles one "tick" at a time via ``next()``. Each
call returns a dict mapping timeframe → candle for every timeframe that
closed a new bar on that tick. ``is_done()`` signals exhaustion.

Design notes
------------
- MockCandleSource emits deterministic, scenario-driven candles for all 5
  timeframes. Each timeframe advances independently (one new bar per
  ``next()`` call) so windows fill at their natural rate. This keeps tests
  fast: a 500-bar 1m window fills in 500 calls, and the 4h window (60 bars)
  fills in 60 calls — well within test budgets.
- ReplayCandleSource loads a 1m OHLCV CSV and resamples to 5m/15m/1h/4h via
  pandas. It emits one aligned bar per timeframe per ``next()`` call,
  dropping partial trailing bars (left-closed, left-labeled convention).
  It also validates 1m timestamp contiguity and warns on gaps (weekends,
  holidays) so sparse higher-TF bars are surfaced, not silently produced.
- LOOKAHEAD SAFETY (ReplayCandleSource): a higher-timeframe bar is emitted
  ONLY once it has fully closed — i.e. when the 1m clock (``now``) reaches
  ``bar_open + tf_duration``. Emitting a bar at its open time would expose
  its fully-formed OHLC (high/low/close) before the period has elapsed,
  leaking future price into every downstream detector and the LLM. The 1m
  bar is the decision granularity: it is emitted at its own open timestamp
  (its OHLC is known because the bar just closed — the standard
  "process-closed-bars" model), and it defines ``now`` for the tick.
- Timestamps are timezone-aware America/New_York; DST handled by zoneinfo,
  never by manual UTC offsets.

No network, no broker, no API keys. Pure deterministic data.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

__all__ = [
    "CandleSource",
    "MockCandleSource",
    "ReplayCandleSource",
    "TIMEFRAMES",
    "TF_MINUTES",
]

# Canonical timeframe set, ordered coarse→fine.
TIMEFRAMES = ("4h", "1h", "15m", "5m", "1m")

# Minutes per timeframe — used for timestamp spacing in the mock source and
# for the higher-TF close-time emission gate in the replay source.
TF_MINUTES: dict[str, int] = {
    "4h": 240,
    "1h": 60,
    "15m": 15,
    "5m": 5,
    "1m": 1,
}

# pandas resample frequency strings. pandas 3.0 rejects bare "m" (ambiguous
# with month-end "ME"); minutes use "min", hours use "h".
_PANDAS_RULE: dict[str, str] = {
    "4h": "4h",
    "1h": "1h",
    "15m": "15min",
    "5m": "5min",
    "1m": "1min",
}

_NY = ZoneInfo("America/New_York")


def _candle(
    o: float,
    h: float,
    l: float,
    c: float,
    ts: datetime,
    vol: float = 1000.0,
) -> dict:
    """Build a single candle dict with a tz-aware ISO-8601 timestamp.

    The timestamp is serialized to an ISO-8601 string because the Phase 1
    detectors' ``_to_datetime``/``_to_epoch`` helpers accept int/float/str
    but NOT bare ``datetime`` objects (detectors are read-only, so we adapt
    at the source). The string carries the full offset, so tz-awareness and
    DST correctness are preserved.
    """
    return {
        "open": float(o),
        "high": float(h),
        "low": float(l),
        "close": float(c),
        "volume": float(vol),
        "timestamp": ts.isoformat(),
    }


# ── ABC ───────────────────────────────────────────────────────────────────────


class CandleSource(ABC):
    """Abstract candle feed: one aligned bar-set per ``next()`` call."""

    @abstractmethod
    def next(self) -> dict[str, dict] | None:
        """Return {timeframe: candle} for timeframes that closed a new bar.

        Returns ``None`` when the source is exhausted. A timeframe key is
        present only when that timeframe closed a new bar on this tick;
        ``"1m"`` is present on every non-None return.
        """

    @abstractmethod
    def is_done(self) -> bool:
        """True when no more candles remain."""


# ── MockCandleSource ──────────────────────────────────────────────────────────


class MockCandleSource(CandleSource):
    """Deterministic, scenario-driven candle feed for all 5 timeframes.

    Each ``next()`` call advances every timeframe by one bar (so all 5 keys
    are present on every call). This is the simplest contract that fills all
    windows at their natural rate and keeps tests fast.

    Scenarios:
        flat           — sideways noise around a base price; no structure.
        trending_up    — steady uptrend producing bullish BOS on HTFs.
        trending_down  — symmetric downtrend producing bearish BOS on HTFs.
        sweep_and_fvg  — a sweep → displacement → 5m FVG sequence.
        killzone_active— latest 1m timestamp inside a killzone, no triggers.

    Determinism: same ``(scenario, seed)`` → byte-identical candle sequence.
    ``trending_down`` is the exact mirror of ``trending_up`` (same noise,
    opposite drift), so the two are structurally symmetric.
    """

    SCENARIOS = ("flat", "trending_up", "trending_down", "sweep_and_fvg", "killzone_active")

    def __init__(self, scenario: str, seed: int = 0, n_bars: int = 500):
        if scenario not in self.SCENARIOS:
            raise ValueError(
                f"unknown scenario {scenario!r}; choose from {self.SCENARIOS}"
            )
        self.scenario = scenario
        self.seed = seed
        self.n_bars = n_bars
        # Pre-generate the full deterministic series for every timeframe.
        self._series: dict[str, list[dict]] = {
            tf: self._generate(tf) for tf in TIMEFRAMES
        }
        self._cursor = 0

    # -- generation -----------------------------------------------------------

    def _generate(self, tf: str) -> list[dict]:
        """Produce ``n_bars`` deterministic candles for ``tf``."""
        # Deterministic LCG (no external RNG dependency; fully reproducible).
        state = (self.seed * 1_000_003 + TF_MINUTES[tf] * 97 + 1) & 0xFFFFFFFF
        base = 20000.0
        bars: list[dict] = []
        # Start at 09:30 ET on a fixed Monday so killzone scenarios land
        # inside the NY AM killzone (09:30-11:00) for the first bars.
        start = datetime(2026, 6, 15, 9, 30, tzinfo=_NY)

        price = base
        for i in range(self.n_bars):
            ts = start + timedelta(minutes=TF_MINUTES[tf] * i)
            o, h, l, c = self._step(tf, i, price, state)
            # advance LCG state deterministically
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            bars.append(_candle(o, h, l, c, ts))
            price = c
        return bars

    def _step(self, tf: str, i: int, price: float, state: int) -> tuple[float, float, float, float]:
        """Return (open, high, low, close) for bar ``i`` of ``tf``."""
        sc = self.scenario
        # Normalised deterministic noise in [-1, 1).
        noise = ((state % 2000) - 1000) / 1000.0
        o = price

        if sc == "flat":
            drift = 0.0
            amp = 8.0
            c = o + drift + noise * amp
            h = max(o, c) + abs(noise) * amp * 0.5 + 0.5
            l = min(o, c) - abs(noise) * amp * 0.5 - 0.5
            return o, h, l, c
        elif sc == "trending_up":
            return self._trending_bar(i, price, direction=+1)
        elif sc == "trending_down":
            return self._trending_bar(i, price, direction=-1)
        elif sc == "sweep_and_fvg":
            return self._sweep_fvg_bar(tf, i, price)
        elif sc == "killzone_active":
            drift = 0.0
            amp = 2.0  # very tight, no structure
            c = o + drift + noise * amp
            h = max(o, c) + abs(noise) * amp * 0.5 + 0.5
            l = min(o, c) - abs(noise) * amp * 0.5 - 0.5
            return o, h, l, c
        else:  # pragma: no cover — guarded by __init__
            return o, o + 1, o - 1, o

    def _trending_bar(self, i: int, price: float, direction: int) -> tuple[float, float, float, float]:
        """Sawtooth trend: 3-bar impulse + 2-bar pullback, repeating.

        Produces clear swing highs/lows (strictly distinct pivots) so BOS
        fires in the trend direction on every timeframe. ``direction`` +1 =
        uptrend (bullish BOS), -1 = downtrend (bearish BOS). The two are
        exact mirrors: same magnitudes, opposite signs.
        """
        o = price
        phase = i % 5  # 5-bar cycle: 3 impulse + 2 pullback
        if direction == +1:
            if phase < 3:
                # Impulse up: close +15, tight wicks.
                c = o + 15.0
                h = c + 1.0
                l = o - 1.0
            else:
                # Pullback down: gap-open lower, close -10, high below prior close.
                o = o - 3.0
                c = o - 10.0
                h = o + 1.0
                l = c - 1.0
        else:  # direction == -1 (mirror)
            if phase < 3:
                # Impulse down: close -15, tight wicks.
                c = o - 15.0
                h = o + 1.0
                l = c - 1.0
            else:
                # Pullback up: gap-open higher, close +10, low above prior close.
                o = o + 3.0
                c = o + 10.0
                h = c + 1.0
                l = o - 1.0
        return o, h, l, c

    def _sweep_fvg_bar(self, tf: str, i: int, price: float) -> tuple[float, float, float, float]:
        """Controlled sweep → displacement → FVG sequence (noise-free).

        Produces a complete ICT 2022 setup detectable on all timeframes:
          - bars  0–14 : sawtooth uptrend (HH/HL structure → bullish BOS on HTF)
          - bar  15    : SSL sweep — low pierces the most recent HL, close back above
          - bars 16–17 : strong bullish displacement → bullish FVG
          - bars 18+   : mild uptrend continuation

        The 15-bar trending pre-phase ensures detect_bos fires on 4h/1h so
        htf_bias is set before the sweep/FVG trigger conditions appear.
        """
        # Phase 1 (bars 0-14): sawtooth uptrend — creates HH/HL swing structure
        # and bullish BOS that gives htf_bias = "bullish".
        if i < 15:
            return self._trending_bar(i, price, direction=+1)
        # Phase 2 (bar 15): SSL sweep — deep wick below the prior HL, close back above.
        if i == 15:
            c = price + 2.0
            h = price + 3.0
            l = price - 50.0
            return price, h, l, c
        # Phase 3 (bars 16-17): large bullish displacement candles → 5m FVG.
        # Body = 35 gives strength ~2.0 vs ATR, clearing the 1.5 threshold even
        # with the sweep bar's large range inflating the trailing window.
        if i in (16, 17):
            c = price + 35.0
            h = c + 1.0
            l = price - 1.0
            return price, h, l, c
        # Phase 4 (bars 18+): mild uptrend continuation.
        c = price + 3.0
        h = c + 1.0
        l = price - 1.0
        return price, h, l, c

    # -- feed interface -------------------------------------------------------

    def next(self) -> dict[str, dict] | None:
        if self._cursor >= self.n_bars:
            return None
        idx = self._cursor
        self._cursor += 1
        return {tf: self._series[tf][idx] for tf in TIMEFRAMES}

    def is_done(self) -> bool:
        return self._cursor >= self.n_bars


# ── ReplayCandleSource ────────────────────────────────────────────────────────


class ReplayCandleSource(CandleSource):
    """Replay a 1m OHLCV source (CSV or parquet), resampled to all 5 timeframes.

    The source must have columns: timestamp, open, high, low, close, volume.
    Files ending in ``.parquet`` are read with ``pd.read_parquet`` (the
    canonical DataLoader output); anything else is read as CSV. ``timestamp``
    is parsed and localized to America/New_York.

    Resampling uses left-closed, left-labeled bars (the bar's timestamp is its
    OPEN time). Partial trailing bars are dropped — only fully-formed bars are
    emitted. ``next()`` emits one aligned bar per timeframe per call, in
    chronological order of the 1m bars.

    LOOKAHEAD SAFETY: a higher-timeframe bar is emitted only once it has
    CLOSED. The 1m bar at the current index defines ``now`` (its open
    timestamp); a higher-TF bar whose period spans ``[bar_open, bar_close)``
    is emitted only when ``now >= bar_close`` (i.e. ``now >= bar_open +
    tf_duration``). This guarantees the bar's fully-formed OHLC is never
    exposed before its period has elapsed — no future price leaks into the
    snapshot, detectors, or LLM. The 1m bar itself is emitted at its own open
    (it is the decision granularity; its OHLC is known because it just closed).

    Gap validation: 1m timestamps are checked for contiguity. Intra-session
    gaps (missing 1m bars over weekends/holidays) are detected and a warning
    is emitted so sparse higher-TF bars are surfaced rather than silently
    produced. This does NOT alter the data — it only warns.
    """

    def __init__(self, path: str, tz: str = "America/New_York", validate_gaps: bool = True):
        import pandas as pd  # local import: keeps module import cheap

        self.tz = ZoneInfo(tz)
        # Read parquet (canonical DataLoader artifact) or CSV (raw export),
        # selected by extension so both feed the same replay path.
        if str(path).endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(self.tz)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Validate 1m timestamp contiguity before resampling. Gaps (weekends,
        # holidays, missing bars) would silently produce sparse higher-TF bars;
        # we surface them with a warning so callers know the resample is over a
        # non-contiguous calendar. This never raises — it only warns.
        if validate_gaps and len(df) > 1:
            self._validate_gaps(df)

        # Resample to each timeframe. label="left" → bar timestamp = open time.
        # closed="left" → [open, close). We drop the last (partial) bar per TF.
        self._bars: dict[str, list[dict]] = {}
        # Precomputed open/close datetimes per bar per TF — used by the
        # emission gate in next() to avoid re-parsing ISO strings in the hot
        # loop and to compare datetimes (robust across DST offsets).
        self._open_dts: dict[str, list[datetime]] = {}
        self._close_dts: dict[str, list[datetime]] = {}
        for tf in TIMEFRAMES:
            rule = _PANDAS_RULE[tf]
            agg = (
                df.set_index("timestamp")
                .resample(rule, label="left", closed="left")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
            )
            # Drop the final bar of each TF — it may be partial (incomplete).
            if len(agg) > 1:
                agg = agg.iloc[:-1]
            duration = timedelta(minutes=TF_MINUTES[tf])
            bars: list[dict] = []
            open_dts: list[datetime] = []
            close_dts: list[datetime] = []
            for ts, r in agg.iterrows():
                open_dt = ts.to_pydatetime()
                bars.append(
                    {
                        "open": float(r.open),
                        "high": float(r.high),
                        "low": float(r.low),
                        "close": float(r.close),
                        "volume": float(r.volume),
                        "timestamp": open_dt.isoformat(),
                    }
                )
                open_dts.append(open_dt)
                # A bar covers [open, open+duration); it closes at open+duration.
                close_dts.append(open_dt + duration)
            self._bars[tf] = bars
            self._open_dts[tf] = open_dts
            self._close_dts[tf] = close_dts

        # Cursor per timeframe; we emit the next unemitted bar for each TF
        # whose close time has been reached by the current 1m timestamp.
        self._cursors: dict[str, int] = {tf: 0 for tf in TIMEFRAMES}
        self._1m_idx = 0
        self._n_1m = len(self._bars["1m"])

    def _validate_gaps(self, df) -> None:
        """Warn on non-contiguous 1m timestamps.

        Compares each consecutive pair of 1m timestamps. A gap is any step
        that is not exactly 1 minute. This catches weekends, holidays, and
        missing bars. The warning names the gap boundaries and size so a
        caller can decide whether the resample is still meaningful.
        """
        ts = df["timestamp"]
        diffs = ts.diff().dropna()
        # A 1m series is contiguous when every step is exactly 1 minute.
        gaps = diffs[diffs != timedelta(minutes=1)]
        if not gaps.empty:
            n_gaps = len(gaps)
            # Report the first few gap boundaries for diagnostics.
            sample = []
            for idx in gaps.index[:3]:
                prev_ts = ts.iloc[idx - 1]
                gap_ts = ts.iloc[idx]
                sample.append(f"{prev_ts} → {gap_ts} ({diffs.iloc[idx-1]})")
            detail = "; ".join(sample) + ("; ..." if n_gaps > 3 else "")
            warnings.warn(
                f"ReplayCandleSource: {n_gaps} non-contiguous 1m gap(s) detected "
                f"in {len(ts)} bars. Higher-TF resampling will produce sparse "
                f"bars across these gaps. First gap(s): {detail}",
                stacklevel=3,
            )

    def next(self) -> dict[str, dict] | None:
        if self._1m_idx >= self._n_1m:
            return None
        # The 1m bar at the current index defines "now" (its open timestamp).
        now_dt = self._open_dts["1m"][self._1m_idx]
        out: dict[str, dict] = {}
        for tf in TIMEFRAMES:
            bars = self._bars[tf]
            cur = self._cursors[tf]
            if tf == "1m":
                # The 1m bar is the decision granularity: its OHLC is known
                # because it just closed. Emit it at its own open timestamp
                # (this is the finest resolution and defines "now"). In steady
                # state exactly one 1m bar is emitted per call.
                open_dts = self._open_dts[tf]
                while cur < len(bars) and open_dts[cur] <= now_dt:
                    out[tf] = bars[cur]
                    cur += 1
            else:
                # Higher timeframes: emit a bar ONLY once it has fully closed,
                # i.e. when now_dt >= bar_open + tf_duration (== bar close time
                # == next bar's open time). Emitting at bar_open would expose
                # the bar's fully-formed OHLC before its period has elapsed — a
                # lookahead bias leaking future price into every downstream
                # detector and the LLM. Gating on the close time guarantees no
                # future data is visible at the decision timestamp.
                close_dts = self._close_dts[tf]
                while cur < len(bars) and close_dts[cur] <= now_dt:
                    out[tf] = bars[cur]
                    cur += 1
            self._cursors[tf] = cur
        self._1m_idx += 1
        return out if out else None

    def is_done(self) -> bool:
        return self._1m_idx >= self._n_1m

    def __len__(self) -> int:
        """Total 1m ticks this source emits — exactly one per ``next()`` call.

        The replay drives off the 1m bars (one per tick until exhausted), so
        this is the precise iteration count callers can use to size a progress
        bar / ETA over a full replay.
        """
        return self._n_1m
