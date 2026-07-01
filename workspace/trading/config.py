"""config.py — TradingConfig: every currently-hardcoded trading knob in one place.

The ONLY permitted change to the semi-frozen trading layer is threading an
optional ``config: TradingConfig = TradingConfig()`` through SnapshotBuilder,
TriggerEngine, CooldownState, and validate_rr, replacing hardcoded literals
with config reads whose defaults equal today's literals. ``TradingConfig()``
with no arguments reproduces today's behavior byte-for-byte.

Phase 2B additions (additive, byte-preserving by default):
  - fib_* knobs feed detect_fibonacci (defaults equal the detector defaults).
  - irl_sources gates which internal-liquidity pool sources are emitted.
  - dol_use_cycle toggles cycle-aware DOL selection (augments _nearest_dol).
  - ranked_dols_to_llm controls how many scored DOL candidates are serialized.
  - dol/tf/role weight maps drive deterministic weighted DOL scoring.
  - ("fib_levels", 1) recency cap for the new per-TF fib_levels snapshot field.

All new knobs are added to config_hash()'s canonical dict.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

__all__ = ["TradingConfig"]


# ── Default values (audited per call site; equal today's literals) ────────────

_DEFAULT_RECENCY_CAPS = (
    ("breaker_blocks", 5),
    ("displacements", 10),
    ("fib_levels", 1),
    ("fvgs", 5),
    ("ifvgs", 5),
    ("market_structure", 10),
    ("opening_gaps", 3),
    ("order_blocks", 5),
    ("po3_phase", 3),
    ("recent_inducements", 10),
    ("recent_sweeps", 10),
    ("recent_swings", 5),
    ("volume_imbalances", 5),
)

_DEFAULT_KILLZONE_WINDOWS = (
    ("london_kz", ("02:00", "05:00")),
    ("ny_am_kz", ("09:30", "11:00")),
    ("ny_pm_kz", ("13:30", "15:00")),
)

_DEFAULT_SOFT_TRIGGER_ORDER = ("fvg", "ifvg", "sweep", "displacement")

_DEFAULT_SOFT_TRIGGER_TF_MAP = (
    ("displacement", ("5m", "1m")),
    ("fvg", ("5m", "15m")),
    ("ifvg", ("5m", "15m")),
    ("sweep", ("15m", "5m")),
)

# Phase 2B — Fibonacci grid defaults (equal detect_fibonacci's defaults).
_DEFAULT_FIB_RETRACEMENTS = (0.382, 0.5, 0.618, 0.66, 0.705, 0.79)
_DEFAULT_FIB_GOLDEN_POCKET = (0.618, 0.66)
_DEFAULT_FIB_OTE_GRID = (0.5, 0.62, 0.705, 0.79)
_DEFAULT_FIB_OTE_PRIMARY = 0.705
_DEFAULT_FIB_RETRACEMENT_TARGET = 0.382
_DEFAULT_FIB_EXTENSIONS = (-0.5, -1.0, -1.5, -2.0, -2.5)

_DEFAULT_IRL_SOURCES = ("fvg", "order_block", "equilibrium", "ote")

# Phase 2B continuation — weighted DOL scoring defaults.
# These are conservative, deterministic weights; Phase 3B can tune them.
_DEFAULT_DOL_WEIGHTS = (
    ("timeframe", 3.0),
    ("role", 2.0),
    ("cycle_align", 4.0),
    ("bias_align", 1.0),
    ("confluence", 0.5),
    ("pd_align", 1.0),
    ("clean_path", 2.0),
    ("proximity", 0.1),
    ("killzone", 0.2),
)

_DEFAULT_TF_WEIGHTS = (
    ("4h", 4.0),
    ("1h", 3.0),
    ("15m", 2.0),
    ("5m", 1.0),
    ("1m", 0.5),
)

_DEFAULT_ROLE_WEIGHTS = (
    ("equal", 4.0),
    ("prior", 3.0),
    ("session", 2.0),
    ("swing", 1.5),
    ("fvg_ce", 1.0),
    ("ob_ce", 1.0),
    ("equilibrium", 1.0),
    ("ote", 1.0),
    ("", 0.0),
)


@dataclass(frozen=True)
class TradingConfig:
    """Every currently-hardcoded trading knob."""

    rr_min: float = 1.0
    conviction_min: int = 40
    no_trade_cooldown_minutes: int = 5
    confluence_band_pct: float = 0.001
    recency_caps: tuple[tuple[str, int], ...] = _DEFAULT_RECENCY_CAPS
    pools_to_llm: int = 5
    htf_tf_order: tuple[str, ...] = ("4h", "1h")
    killzone_windows: tuple[tuple[str, tuple[str, str]], ...] = _DEFAULT_KILLZONE_WINDOWS
    soft_trigger_order: tuple[str, ...] = _DEFAULT_SOFT_TRIGGER_ORDER
    soft_trigger_tf_map: tuple[tuple[str, tuple[str, ...]], ...] = _DEFAULT_SOFT_TRIGGER_TF_MAP

    # ── Phase 2B: Fibonacci / IRL / cycle-aware DOL knobs (additive) ────────
    fib_retracements: tuple[float, ...] = _DEFAULT_FIB_RETRACEMENTS
    fib_golden_pocket: tuple[float, float] = _DEFAULT_FIB_GOLDEN_POCKET
    fib_ote_grid: tuple[float, ...] = _DEFAULT_FIB_OTE_GRID
    fib_ote_primary: float = _DEFAULT_FIB_OTE_PRIMARY
    fib_retracement_target: float = _DEFAULT_FIB_RETRACEMENT_TARGET
    fib_extensions: tuple[float, ...] = _DEFAULT_FIB_EXTENSIONS
    fib_anchor_lookback: int = 2
    irl_sources: tuple[str, ...] = _DEFAULT_IRL_SOURCES
    dol_use_cycle: bool = True

    # ── Phase 2B continuation: ranked DOLs + weights ───────────────────────
    ranked_dols_to_llm: int = 5
    dol_weights: tuple[tuple[str, float], ...] = _DEFAULT_DOL_WEIGHTS
    tf_weights: tuple[tuple[str, float], ...] = _DEFAULT_TF_WEIGHTS
    role_weights: tuple[tuple[str, float], ...] = _DEFAULT_ROLE_WEIGHTS

    # ── accessors (dict views for the trading layer) ────────────────────────

    def recency_caps_dict(self) -> dict[str, int]:
        return dict(self.recency_caps)

    def killzone_windows_list(self):
        out = []
        for name, (start_s, end_s) in self.killzone_windows:
            out.append((name, _parse_hhmm(start_s), _parse_hhmm(end_s)))
        return out

    def soft_trigger_tf_map_dict(self) -> dict[str, tuple[str, ...]]:
        return dict(self.soft_trigger_tf_map)

    def irl_sources_set(self) -> set[str]:
        return set(self.irl_sources)

    def dol_weights_dict(self) -> dict[str, float]:
        return dict(self.dol_weights)

    def tf_weights_dict(self) -> dict[str, float]:
        return dict(self.tf_weights)

    def role_weights_dict(self) -> dict[str, float]:
        return dict(self.role_weights)

    # ── stable hash ─────────────────────────────────────────────────────────

    def config_hash(self) -> str:
        canonical = {
            "rr_min": self.rr_min,
            "conviction_min": self.conviction_min,
            "no_trade_cooldown_minutes": self.no_trade_cooldown_minutes,
            "confluence_band_pct": self.confluence_band_pct,
            "recency_caps": dict(sorted(self.recency_caps)),
            "killzone_windows": dict(sorted(self.killzone_windows)),
            "pools_to_llm": self.pools_to_llm,
            "htf_tf_order": list(self.htf_tf_order),
            "soft_trigger_order": list(self.soft_trigger_order),
            "soft_trigger_tf_map": dict(sorted(self.soft_trigger_tf_map)),
            # Phase 2B additive knobs.
            "fib_retracements": list(self.fib_retracements),
            "fib_golden_pocket": list(self.fib_golden_pocket),
            "fib_ote_grid": list(self.fib_ote_grid),
            "fib_ote_primary": self.fib_ote_primary,
            "fib_retracement_target": self.fib_retracement_target,
            "fib_extensions": list(self.fib_extensions),
            "fib_anchor_lookback": self.fib_anchor_lookback,
            "irl_sources": list(self.irl_sources),
            "dol_use_cycle": self.dol_use_cycle,
            # Phase 2B continuation knobs.
            "ranked_dols_to_llm": self.ranked_dols_to_llm,
            "dol_weights": dict(sorted(self.dol_weights)),
            "tf_weights": dict(sorted(self.tf_weights)),
            "role_weights": dict(sorted(self.role_weights)),
        }
        blob = json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), default=_json_default
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def short_hash(self) -> str:
        return self.config_hash()[:12]


def _parse_hhmm(s: str):
    from datetime import time

    h, m = s.split(":")
    return time(int(h), int(m))


def _json_default(obj):
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, frozenset):
        return sorted(obj)
    return str(obj)
