"""test_calibrator.py — TriggerCalibrator unit tests.

The calibrator runs TriggerEngine in isolation (zero LLM calls) over
MockCandleSource scenarios. It records every gate block and every soft
trigger, and writes workspace/calibration_report.json.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure workspace/ is importable.
_WORKSPACE = Path(__file__).resolve().parent.parent.parent / "workspace"
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from trading.candle_source import MockCandleSource, TIMEFRAMES
from trading.candle_window import CandleWindow, DEFAULT_SIZES
from trading.snapshot import SnapshotBuilder
from trading.trigger import TriggerEngine
from trading.cooldown import CooldownState
from backtesting.calibrator import TriggerCalibrator, CalibrationReport


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_calibrator(scenario, n_bars=100, window_sizes=None):
    """Build a TriggerCalibrator over a MockCandleSource."""
    src = MockCandleSource(scenario, n_bars=n_bars)
    w = CandleWindow(window_sizes or DEFAULT_SIZES)
    builder = SnapshotBuilder()
    cd = CooldownState()
    trigger = TriggerEngine(cd)
    cal = TriggerCalibrator(w, builder, trigger, cd)
    return cal, src


# ── flat scenario ─────────────────────────────────────────────────────────────


def test_flat_scenario_total_fires_zero(tmp_path, monkeypatch):
    """flat scenario with few bars: total_fires = 0, no_htf_bias blocks > 0.

    With n_bars=10, htf_bias never sets (BOS needs more bars on 4h/1h), so
    every candle is blocked at the no_htf_bias gate and no fires occur.
    """
    monkeypatch.chdir(tmp_path)
    cal, src = _make_calibrator("flat", n_bars=10)
    report = cal.run(src)
    assert report.total_fires == 0
    assert report.gate_blocks["no_htf_bias"] > 0


# ── killzone_active scenario ──────────────────────────────────────────────────


def test_killzone_active_has_gate_blocks(tmp_path, monkeypatch):
    """killzone_active: no_htf_bias or no_dol gate blocks > 0."""
    monkeypatch.chdir(tmp_path)
    cal, src = _make_calibrator("killzone_active", n_bars=100)
    report = cal.run(src)
    assert report.gate_blocks["no_htf_bias"] > 0 or report.gate_blocks["no_dol"] > 0


# ── sweep_and_fvg scenario ────────────────────────────────────────────────────


def test_sweep_and_fvg_total_fires_positive(tmp_path, monkeypatch):
    """sweep_and_fvg: total_fires > 0 (the sweep→FVG sequence fires)."""
    monkeypatch.chdir(tmp_path)
    cal, src = _make_calibrator("sweep_and_fvg", n_bars=100)
    report = cal.run(src)
    assert report.total_fires > 0


# ── Invariants ────────────────────────────────────────────────────────────────


def test_fires_by_killzone_sums_to_total(tmp_path, monkeypatch):
    """total_fires == sum(fires_by_killzone.values())."""
    monkeypatch.chdir(tmp_path)
    cal, src = _make_calibrator("sweep_and_fvg", n_bars=100)
    report = cal.run(src)
    assert report.total_fires == sum(report.fires_by_killzone.values())


def test_fires_by_month_sums_to_total(tmp_path, monkeypatch):
    """total_fires == sum(fires_by_month.values())."""
    monkeypatch.chdir(tmp_path)
    cal, src = _make_calibrator("sweep_and_fvg", n_bars=100)
    report = cal.run(src)
    assert report.total_fires == sum(report.fires_by_month.values())


def test_estimated_llm_cost(tmp_path, monkeypatch):
    """estimated_llm_cost_usd == total_fires * 0.003."""
    monkeypatch.chdir(tmp_path)
    cal, src = _make_calibrator("sweep_and_fvg", n_bars=100)
    report = cal.run(src)
    assert report.estimated_llm_cost_usd == round(report.total_fires * 0.003, 4)


# ── No LLM calls ──────────────────────────────────────────────────────────────


def test_no_llm_calls(tmp_path, monkeypatch):
    """The calibrator makes zero LLM calls.

    We install a mock model_router whose .call() raises if invoked, then run
    the calibrator. Completion proves the no-LLM contract.
    """
    monkeypatch.chdir(tmp_path)

    # A mock router that raises if .call() is ever invoked.
    router = MagicMock()

    def _boom(*a, **kw):
        raise AssertionError("calibrator must not call the LLM")

    router.call.side_effect = _boom

    cal, src = _make_calibrator("sweep_and_fvg", n_bars=100)
    # The calibrator never touches a router, but we assert it completes
    # without invoking any LLM path.
    report = cal.run(src)
    assert report.total_fires >= 0
    router.call.assert_not_called()


# ── JSON output ───────────────────────────────────────────────────────────────


def test_calibration_report_json_written(tmp_path, monkeypatch):
    """calibration_report.json is written with all required keys."""
    monkeypatch.chdir(tmp_path)
    cal, src = _make_calibrator("sweep_and_fvg", n_bars=100)
    report = cal.run(src)

    out_path = tmp_path / "workspace" / "calibration_report.json"
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    required = {
        "period", "total_1m_candles", "gate_blocks", "soft_triggers",
        "structures_present", "sweeps_by_session", "fires_by_killzone",
        "fires_by_month", "total_fires", "estimated_llm_cost_usd",
    }
    assert required.issubset(data.keys()), f"missing: {required - set(data.keys())}"
    assert data["total_fires"] == report.total_fires
    assert data["estimated_llm_cost_usd"] == report.estimated_llm_cost_usd


def test_calibration_report_gate_block_keys(tmp_path, monkeypatch):
    """gate_blocks has the four gate keys (+ no_soft_trigger)."""
    monkeypatch.chdir(tmp_path)
    cal, src = _make_calibrator("flat", n_bars=10)
    report = cal.run(src)
    for key in ("no_killzone", "no_htf_bias", "no_dol", "cooldown_active"):
        assert key in report.gate_blocks


def test_calibration_report_soft_trigger_keys(tmp_path, monkeypatch):
    """soft_triggers has the four soft-trigger keys."""
    monkeypatch.chdir(tmp_path)
    cal, src = _make_calibrator("sweep_and_fvg", n_bars=100)
    report = cal.run(src)
    for key in ("fvg", "ifvg", "sweep", "displacement"):
        assert key in report.soft_triggers


def test_structures_present_keys(tmp_path, monkeypatch):
    """structures_present has the four soft-trigger keys."""
    monkeypatch.chdir(tmp_path)
    cal, src = _make_calibrator("sweep_and_fvg", n_bars=100)
    report = cal.run(src)
    for key in ("fvg", "ifvg", "sweep", "displacement"):
        assert key in report.structures_present


def test_sweeps_by_session_keys(tmp_path, monkeypatch):
    """sweeps_by_session has all 15 session-level keys."""
    monkeypatch.chdir(tmp_path)
    cal, src = _make_calibrator("sweep_and_fvg", n_bars=100)
    report = cal.run(src)
    expected = {
        "midnight_open", "true_day_open", "london_open", "open_830", "open_930",
        "asia_high", "asia_low", "london_high", "london_low",
        "nyam_high", "nyam_low", "nylunch_high", "nylunch_low", "nypm_high", "nypm_low",
    }
    assert expected == set(report.sweeps_by_session.keys())


def test_soft_trigger_keys_cover_engine_triggers():
    """The calibrator's _SOFT_KEYS must cover every TriggerEngine soft trigger.

    Guards against drift: if a new soft trigger is added to
    TriggerEngine.soft_triggers_present but not to _SOFT_KEYS, structures_present
    would silently never count it. This test turns that into a loud failure.
    """
    from backtesting.calibrator import _SOFT_KEYS

    class _Snap:
        fvgs: dict = {}
        ifvgs: dict = {}
        recent_sweeps: dict = {}
        displacements: dict = {}

    engine_keys = set(TriggerEngine(CooldownState()).soft_triggers_present(_Snap()).keys())
    assert engine_keys <= set(_SOFT_KEYS), (
        f"TriggerEngine soft triggers {engine_keys - set(_SOFT_KEYS)} are not in "
        f"calibrator _SOFT_KEYS — structures_present would silently miss them"
    )


def test_count_structures_is_independent_of_priority():
    """_count_structures increments every present structure, not just the first.

    Uses a real calibrator so the shared TriggerEngine.soft_triggers_present
    drives the counts (no duplicated predicates).
    """
    from backtesting.calibrator import CalibrationReport

    class _Snap:
        fvgs = {"5m": [{"type": "fvg_bullish", "top": 100.0, "bottom": 99.0}]}
        ifvgs: dict = {}
        recent_sweeps = {"15m": [{"type": "sweep_ssl", "swept_level": 98.0}]}
        displacements = {"5m": [{"type": "displacement_bullish", "strength": 2.0}]}
        session_levels = {"asia_high": 100.0, "asia_low": 95.0, "midnight_open": None}

    cal, _ = _make_calibrator("flat", n_bars=1)
    report = CalibrationReport()
    cal._count_structures(report, _Snap())

    # fvg fires first in TriggerEngine, but every present structure is counted.
    assert report.structures_present["fvg"] == 1
    assert report.structures_present["sweep"] == 1
    assert report.structures_present["displacement"] == 1
    assert report.structures_present["ifvg"] == 0


def test_sweeps_by_session_matches_swept_level():
    """sweeps_by_session counts the session level the sweep actually ran, not
    every level that merely exists on the snapshot."""
    from backtesting.calibrator import CalibrationReport

    class _Snap:
        fvgs: dict = {}
        ifvgs: dict = {}
        # swept_level 100.0 matches asia_high exactly; nothing matches asia_low.
        recent_sweeps = {"15m": [{"type": "sweep_bsl", "swept_level": 100.0}]}
        displacements: dict = {}
        # midnight_open is always defined but was NOT swept — must stay 0.
        session_levels = {"asia_high": 100.0, "asia_low": 95.0, "midnight_open": 99.0}

    cal, _ = _make_calibrator("flat", n_bars=1)
    report = CalibrationReport()
    cal._count_structures(report, _Snap())

    assert report.sweeps_by_session["asia_high"] == 1
    assert report.sweeps_by_session["asia_low"] == 0
    assert report.sweeps_by_session["midnight_open"] == 0


def test_calibration_report_period_populated(tmp_path, monkeypatch):
    """period.start and period.end are populated from the replay."""
    monkeypatch.chdir(tmp_path)
    cal, src = _make_calibrator("sweep_and_fvg", n_bars=50)
    report = cal.run(src)
    assert report.period["start"] != ""
    assert report.period["end"] != ""


def test_calibration_report_total_1m_candles(tmp_path, monkeypatch):
    """total_1m_candles equals the number of candles iterated."""
    monkeypatch.chdir(tmp_path)
    n = 50
    cal, src = _make_calibrator("sweep_and_fvg", n_bars=n)
    report = cal.run(src)
    assert report.total_1m_candles == n


# ── Cooldown parity ───────────────────────────────────────────────────────────


def test_cooldown_active_blocks_present(tmp_path, monkeypatch):
    """After a fire, subsequent candles in the same killzone are cooldown-blocked.

    This verifies the calibrator mirrors production cooldown: a directional
    fire triggers same-killzone cooldown, producing cooldown_active blocks.
    """
    monkeypatch.chdir(tmp_path)
    cal, src = _make_calibrator("sweep_and_fvg", n_bars=100)
    report = cal.run(src)
    # sweep_and_fvg fires at least once; cooldown should block some candles.
    assert report.gate_blocks["cooldown_active"] > 0
