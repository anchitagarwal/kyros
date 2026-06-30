"""config.py — TradingConfig: every currently-hardcoded trading knob in one place.

The ONLY permitted change to the semi-frozen trading layer is threading an
optional ``config: TradingConfig = TradingConfig()`` through SnapshotBuilder,
TriggerEngine, CooldownState, and validate_rr, replacing hardcoded literals
with config reads whose defaults equal today's literals. ``TradingConfig()``
with no arguments reproduces today's behavior byte-for-byte.

Design notes
------------
- Frozen dataclass: every field is an immutable type (tuple / scalar) so the
  dataclass is hashable and ``config_hash()`` is stable.
- Mappings (recency_caps, killzone_windows, soft_trigger_tf_map) are stored as
  ``tuple[tuple[str, V], ...]`` (sorted by key) — frozen and hashable. The
  trading layer converts to ``dict`` on access via ``dict(config.<field>)``.
- ``config_hash()`` canonicalizes (sorts mapping keys, normalizes numbers via
  json) then sha256 — deterministic across processes and Python hash-seed
  randomization (never uses builtin ``hash()``). Two logically-equal configs
  hash identically; any field change produces a different hash.

O0 resolution (conviction_min default):
  Conviction is enforced ONLY in the LLM system prompt today ("conviction < 40
  → no_trade"); the Python layer (validate_rr) never checked it. Auditing
  workspace/trade_traces.jsonl and every test fixture confirms NO taken trade
  has conviction < 40 (all are ≥ 60). Therefore a Python conviction gate with
  default 40 is byte-identical — a no-op for every current input. The gate
  activates only with a non-default conviction_min or when the LLM violates
  its own prompt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

__all__ = ["TradingConfig"]


# ── Default values (audited per call site; equal today's literals) ────────────

# recency_caps: snapshot field name → max entries kept (the [-N:] slices in
# SnapshotBuilder.build). Sorted by key for canonical hashing.
_DEFAULT_RECENCY_CAPS = (
    ("breaker_blocks", 5),
    ("displacements", 10),
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

# killzone_windows: name → (start, end) as "HH:MM" ET clock strings.
# Matches snapshot._KILLZONES (london_kz 02:00-05:00, ny_am_kz 09:30-11:00,
# ny_pm_kz 13:30-15:00). Sorted by name (== today's iteration order).
_DEFAULT_KILLZONE_WINDOWS = (
    ("london_kz", ("02:00", "05:00")),
    ("ny_am_kz", ("09:30", "11:00")),
    ("ny_pm_kz", ("13:30", "15:00")),
)

# soft_trigger_order: the priority order TriggerEngine evaluates soft triggers
# (first present wins). Matches today's dict insertion order.
_DEFAULT_SOFT_TRIGGER_ORDER = ("fvg", "ifvg", "sweep", "displacement")

# soft_trigger_tf_map: trigger name → ordered tuple of timeframes checked via
# ``or`` (bool(a.get(tf0) or a.get(tf1) ...)). Matches today's per-trigger
# ``get("5m") or get("15m")`` chains. Sorted by trigger name.
_DEFAULT_SOFT_TRIGGER_TF_MAP = (
    ("displacement", ("5m", "1m")),
    ("fvg", ("5m", "15m")),
    ("ifvg", ("5m", "15m")),
    ("sweep", ("15m", "5m")),
)


@dataclass(frozen=True)
class TradingConfig:
    """Every currently-hardcoded trading knob.

    ``TradingConfig()`` (no args) reproduces today's behavior exactly. Each
    field maps 1:1 to a literal audited in the trading layer.
    """

    # validate_rr: minimum risk-reward (today's ``if rr < 1.0``).
    rr_min: float = 1.0
    # validate_rr: minimum conviction (today enforced only in the LLM prompt
    # at 40; now also enforceable in Python — default 40 is byte-preserving
    # per O0). See module docstring.
    conviction_min: int = 40
    # CooldownState: no_trade cooldown in minutes (today's
    # NO_TRADE_COOLDOWN_MINUTES = 5).
    no_trade_cooldown_minutes: int = 5
    # SnapshotBuilder._build_pools: confluence band as a fraction of level
    # (today's ``abs(level) * 0.001``).
    confluence_band_pct: float = 0.001
    # SnapshotBuilder.build: per-detector recency caps (the [-N:] slices).
    recency_caps: tuple[tuple[str, int], ...] = _DEFAULT_RECENCY_CAPS
    # SnapshotBuilder._compact_dict: max pools sent to the LLM (today's [:5]).
    pools_to_llm: int = 5
    # SnapshotBuilder._derive_htf_bias: HTF timeframe order (today's ("4h","1h")).
    htf_tf_order: tuple[str, ...] = ("4h", "1h")
    # SnapshotBuilder._killzone_at: killzone windows (today's _KILLZONES).
    killzone_windows: tuple[tuple[str, tuple[str, str]], ...] = _DEFAULT_KILLZONE_WINDOWS
    # TriggerEngine.soft_triggers_present: evaluation priority order.
    soft_trigger_order: tuple[str, ...] = _DEFAULT_SOFT_TRIGGER_ORDER
    # TriggerEngine.soft_triggers_present: per-trigger timeframe or-chain.
    soft_trigger_tf_map: tuple[tuple[str, tuple[str, ...]], ...] = _DEFAULT_SOFT_TRIGGER_TF_MAP

    # ── accessors (dict views for the trading layer) ────────────────────────

    def recency_caps_dict(self) -> dict[str, int]:
        """recency_caps as a dict {field_name: cap}."""
        return dict(self.recency_caps)

    def killzone_windows_list(self):
        """killzone_windows as a list of (name, start_time, end_time) tuples.

        ``start_time``/``end_time`` are ``datetime.time`` objects, matching the
        shape SnapshotBuilder._killzone_at consumes.
        """
        out = []
        for name, (start_s, end_s) in self.killzone_windows:
            out.append((name, _parse_hhmm(start_s), _parse_hhmm(end_s)))
        return out

    def soft_trigger_tf_map_dict(self) -> dict[str, tuple[str, ...]]:
        """soft_trigger_tf_map as a dict {trigger: (tf, ...)}."""
        return dict(self.soft_trigger_tf_map)

    # ── stable hash ─────────────────────────────────────────────────────────

    def config_hash(self) -> str:
        """Deterministic sha256 over a canonical, sorted representation.

        Stable across processes and Python hash-seed randomization (never uses
        builtin ``hash()``). Mappings are sorted by key; numbers are normalized
        via json serialization. Two logically-equal configs hash identically;
        any field change produces a different hash.
        """
        canonical = {
            "rr_min": self.rr_min,
            "conviction_min": self.conviction_min,
            "no_trade_cooldown_minutes": self.no_trade_cooldown_minutes,
            "confluence_band_pct": self.confluence_band_pct,
            # mappings → sorted dict (key order canonicalized)
            "recency_caps": dict(sorted(self.recency_caps)),
            "killzone_windows": dict(sorted(self.killzone_windows)),
            "pools_to_llm": self.pools_to_llm,
            "htf_tf_order": list(self.htf_tf_order),
            "soft_trigger_order": list(self.soft_trigger_order),
            "soft_trigger_tf_map": dict(sorted(self.soft_trigger_tf_map)),
        }
        blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                          default=_json_default)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def short_hash(self) -> str:
        """First 12 hex chars of config_hash (for run-directory naming)."""
        return self.config_hash()[:12]


def _parse_hhmm(s: str):
    """Parse an 'HH:MM' string into a datetime.time."""
    from datetime import time
    h, m = s.split(":")
    return time(int(h), int(m))


def _json_default(obj):
    """JSON fallback for non-standard types (defensive; current fields are plain)."""
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, frozenset):
        return sorted(obj)
    return str(obj)
