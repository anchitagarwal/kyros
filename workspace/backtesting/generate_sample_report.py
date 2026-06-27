"""generate_sample_report.py — emit a sample backtest_report.md artifact.

This is a one-off helper (not a test) that exercises PerformanceReport with a
representative set of TradeTrace objects so that workspace/backtest_report.md
exists as a committed artifact for auditors. It uses NO live data, NO LLM, NO
broker — purely synthetic traces + the untrusted golden_alerts.json dataset.

Run:  uv run python workspace/backtesting/generate_sample_report.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure workspace/ is importable.
_WORKSPACE = Path(__file__).resolve().parent.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from backtesting.engine import TradeTrace
from backtesting.report import PerformanceReport


def _ts(days_ago: int, hour: int, minute: int = 0) -> str:
    """Build an ISO UTC timestamp for a sample trace."""
    base = datetime(2023, 8, 11, hour, minute, tzinfo=timezone.utc)
    return (base - timedelta(days=days_ago)).isoformat()


def _trace(timestamp, bias, model, killzone, trigger_reason, result, actual_rr):
    """Build a synthetic TradeTrace."""
    return TradeTrace(
        trace_id=f"{timestamp}_{bias}_{model}",
        timestamp=timestamp,
        instrument="NQ",
        killzone=killzone,
        trigger_reason=trigger_reason,
        snapshot_summary={"instrument": "NQ", "current_price": 15200.0},
        raw_llm_output='{"bias": "%s", "model": "%s"}' % (bias, model),
        alert={
            "bias": bias,
            "model": model,
            "entry_zone": [15195.0, 15205.0],
            "stop": 15150.0 if bias == "long" else 15250.0,
            "target": 15290.0 if bias == "long" else 15160.0,
        },
        rr_validated=True,
        outcome={
            "result": result,
            "actual_rr": actual_rr,
            "fill_price": 15200.0,
            "exit_price": 15290.0 if result == "win" else 15150.0,
            "candles_to_fill": 3 if result != "no_trade" else None,
            "candles_to_resolution": 12 if result in ("win", "loss") else None,
        },
    )


def main() -> None:
    # A representative mix: wins, losses, an expired, a no_fill, and a no_trade.
    traces = [
        _trace(_ts(0, 14, 5), "long", "2022", "ny_am_kz", "fvg", "win", 2.1),
        _trace(_ts(1, 13, 30), "short", "ifvg", "ny_am_kz", "sweep", "win", 1.8),
        _trace(_ts(2, 15, 0), "long", "unicorn", "ny_pm_kz", "displacement", "loss", -1.0),
        _trace(_ts(3, 14, 15), "long", "2022", "ny_am_kz", "fvg", "win", 2.4),
        _trace(_ts(4, 16, 0), "short", "breaker", "ny_pm_kz", "breaker", "loss", -1.0),
        _trace(_ts(5, 13, 45), "long", "silver_bullet", "ny_am_kz", "fvg", "expired", 0.3),
        _trace(_ts(6, 14, 30), "long", "2022", "ny_am_kz", "fvg", "no_fill", None),
        _trace(_ts(7, 15, 15), "long", "none", "ny_pm_kz", "fvg", "no_trade", None),
    ]

    report = PerformanceReport()
    golden_path = _WORKSPACE / "knowledge_base" / "golden_alerts.json"
    out_path = _WORKSPACE / "backtest_report.md"
    md = report.generate(traces, golden_alerts_path=golden_path, out_path=out_path)
    print(f"Wrote {out_path} ({len(md)} chars)")


if __name__ == "__main__":
    main()
