#!/usr/bin/env python
"""scripts/run_backtest.py — Phase 3A backtest runner.

Three stages, gated so you never spend on the LLM by accident:

  1. TriggerCalibrator — replays the whole period with NO LLM calls, maps the
     gate/soft-trigger distribution, and estimates the LLM cost.
  2. BacktestEngine    — drives the full Phase 2 reasoning stack (LLM in the
     loop). Each alert is appended to workspace/trade_alerts.jsonl the moment
     the LLM returns (crash-safe ledger), then outcomes are resolved offline
     into workspace/trade_traces.jsonl. Resumes automatically: timestamps
     already present in the ledger are skipped (no re-spend).
  3. PerformanceReport — aggregates the traces into workspace/backtest_report.md.

Usage:
    # Stage 1 only (free — estimate LLM cost before committing):
    uv run --env-file .env python scripts/run_backtest.py --calibrate-only

    # Full run (LLM calls — costs money; prompts for confirmation):
    uv run --env-file .env python scripts/run_backtest.py

    # Custom date range (inclusive of both bounds):
    uv run --env-file .env python scripts/run_backtest.py --start 2024-06-01 --end 2024-12-31

    # Non-interactive / resume an interrupted run (skips the confirm prompt):
    uv run --env-file .env python scripts/run_backtest.py --yes

Environment (.env):
    KYROS_DATA_BACKEND=csv                            (use the local CSV export)
    KYROS_CSV_PATH=workspace/data/nq_1min_data.csv    (path to that CSV)
    ZAI_API_KEY=...      (reasoning model from infrastructure.trading in .kyros_state.json; defaults to zai/glm-5.2)

Resume is automatic: the engine reads any existing trade_alerts.jsonl ledger and
skips already-processed alert timestamps, so a re-run continues where it left off
— even after a mid-replay crash, since the ledger is fsynced per alert. To start
clean, archive the old ledger first (per CLAUDE.md, never delete it):
    mv workspace/trade_alerts.jsonl artifacts/trade_alerts_$(date +%Y%m%d_%H%M%S).jsonl
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
# Mirror the codebase/test convention: top-level `trading.*` / `backtesting.*`
# (from workspace/) and `kyros.*` (from src/). Importing via `workspace.trading`
# would create a *second* module identity for AlertPayload et al. and break
# isinstance checks, so we use the same names the modules use internally.
ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT / "workspace"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

from backtesting.data_loader import DataLoader
from backtesting.calibrator import TriggerCalibrator
from backtesting.engine import BacktestEngine
from backtesting.outcome import OutcomeSimulator
from backtesting.report import PerformanceReport

from trading.candle_source import ReplayCandleSource
from trading.candle_window import CandleWindow
from trading.snapshot import SnapshotBuilder
from trading.trigger import TriggerEngine
from trading.cooldown import CooldownState
from trading.trading_loop import TradingLoop
from trading.reasoning_agent import LLMReasoningAgent

from kyros.core.model_router import ModelRouter
from kyros.core.agent_loader import KyrosAgentLoader

try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except ImportError:  # pragma: no cover - tqdm is a declared dependency
    _HAVE_TQDM = False

log = logging.getLogger("kyros.backtest")

# Kyros provider name → the env var holding its API key (for the pre-flight check).
_PROVIDER_ENV = {
    "zai": "ZAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
}


# ── logging ─────────────────────────────────────────────────────────────────


def setup_logging(verbose: bool, log_file: Path | None) -> None:
    """Configure root logging: console always, an optional file at DEBUG."""
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Drop handlers installed as a side effect of importing litellm et al.
    root.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s | %(message)s",
                                            datefmt="%H:%M:%S"))
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s | %(message)s"))
        root.addHandler(fh)
        log.info("Logging to %s", log_file)

    # Route warnings.warn(...) (e.g. ReplayCandleSource gap checks) through logging.
    logging.captureWarnings(True)
    # Tame noisy third-party loggers — we want our own progress, not theirs.
    for noisy in ("LiteLLM", "litellm", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── progress ────────────────────────────────────────────────────────────────


class ProgressCandleSource:
    """Wrap a CandleSource to advance a tqdm bar one tick per ``next()`` call.

    Drop-in for the calibrator/engine loops (they only call ``next()`` and
    ``is_done()``), so progress + ETA are reported without touching their
    internals. The bar's total comes from ``len(source)`` when available
    (exact for ReplayCandleSource); otherwise it runs in count-only mode. The
    bar is closed once the underlying source is exhausted.
    """

    def __init__(self, source, desc: str, enabled: bool = True):
        self._source = source
        self._bar = None
        if enabled and _HAVE_TQDM:
            try:
                total = len(source)
            except TypeError:
                total = None  # count-only (no ETA) for sources without __len__
            self._bar = tqdm(
                total=total, desc=desc, unit="candle", unit_scale=True,
                dynamic_ncols=True, leave=True, smoothing=0.05,
            )

    def next(self):
        candles = self._source.next()
        if self._bar is not None:
            if candles is None:
                self.close()
            else:
                self._bar.update(1)
        return candles

    def is_done(self) -> bool:
        done = self._source.is_done()
        if done:
            self.close()
        return done

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


# ── wiring ──────────────────────────────────────────────────────────────────


def build_reasoning_loop() -> tuple[TradingLoop, dict]:
    """Wire the full Phase 2 reasoning stack.

    Returns the TradingLoop (the engine consumes its window/builder/trigger/
    agent/cooldown) and the resolved model_engine dict (for the key pre-flight).
    The trigger and the loop share ONE CooldownState so the cooldown_active gate
    sees the same state the loop updates.
    """
    loader = KyrosAgentLoader(str(ROOT))
    router = ModelRouter()
    # Reasoning LLM comes from infrastructure.trading in .kyros_state.json,
    # falling back to the executor model if no trading block is configured.
    engine_cfg = loader.get_model_engine("trading", fallback_role="executor")
    config = {"model_engine": engine_cfg, "final_system_prompt": ""}

    cooldown = CooldownState()
    loop = TradingLoop(
        source=None,  # the engine drives the source itself
        window=CandleWindow(),
        builder=SnapshotBuilder(),
        trigger=TriggerEngine(cooldown),
        agent=LLMReasoningAgent(router, agent_config=config),
        cooldown=cooldown,
        output_path=str(ROOT / "workspace" / "alerts.jsonl"),
    )
    return loop, engine_cfg


def preflight_api_key(engine_cfg: dict) -> None:
    """Warn (don't fail) if the reasoning provider's API key is missing."""
    provider = engine_cfg.get("provider", "")
    env = _PROVIDER_ENV.get(provider)
    if env and not os.getenv(env):
        log.warning(
            "Reasoning provider is %r but %s is not set — every LLM call will "
            "fail and each alert will fall back to no_trade. Did you pass "
            "--env-file .env?", provider, env)


# ── stages ──────────────────────────────────────────────────────────────────


def run_calibration(source: ReplayCandleSource, progress: bool = True):
    """Stage 1: TriggerCalibrator (no LLM). Returns the CalibrationReport."""
    cooldown = CooldownState()
    calibrator = TriggerCalibrator(
        CandleWindow(), SnapshotBuilder(), TriggerEngine(cooldown), cooldown,
    )

    log.info("── Stage 1: trigger calibration (no LLM) ──")
    t0 = time.perf_counter()
    report = calibrator.run(ProgressCandleSource(source, "Stage 1 calibrating", progress))
    log.info("Calibration finished in %.1fs → workspace/calibration_report.json",
             time.perf_counter() - t0)

    log.info("  period               : %s → %s",
             report.period.get("start") or "?", report.period.get("end") or "?")
    log.info("  1m candles evaluated : %s", f"{report.total_1m_candles:,}")
    log.info("  trigger fires        : %s", f"{report.total_fires:,}")
    log.info("  est. LLM cost        : ~$%.2f", report.estimated_llm_cost_usd)

    log.info("  gate blocks (first failing gate per candle):")
    for gate, count in report.gate_blocks.items():
        pct = count / max(report.total_1m_candles, 1) * 100
        log.info("    %-16s %10s  (%4.1f%%)", gate, f"{count:,}", pct)
    log.info("  soft triggers fired:")
    for trig, count in report.soft_triggers.items():
        log.info("    %-16s %10s", trig, f"{count:,}")
    log.info("  fires by killzone:")
    for kz, count in report.fires_by_killzone.items():
        log.info("    %-16s %10s", kz, f"{count:,}")

    return report


def run_backtest(source: ReplayCandleSource, loop: TradingLoop, trace_path: Path,
                 progress: bool = True) -> list:
    """Stages 2+3: BacktestEngine (LLM in loop) then PerformanceReport."""
    engine = BacktestEngine(loop, OutcomeSimulator(), output_path=trace_path)

    log.info("── Stage 2: backtest engine (LLM in loop) ──")
    ledger_path = engine.alerts_path
    if ledger_path.exists():
        existing = sum(1 for line in ledger_path.open() if line.strip())
        if existing:
            log.info("Found %s existing alerts in %s — those timestamps will be "
                     "skipped (resume; no re-spend).", f"{existing:,}", ledger_path.name)
            log.warning("Existing alerts are INCLUDED in the report. To start "
                        "clean, archive the ledger first (see --help).")

    t0 = time.perf_counter()
    traces = engine.run(ProgressCandleSource(source, "Stage 2 backtesting", progress))
    dt = time.perf_counter() - t0
    log.info("Backtest finished in %.1fs (%.1f min). %s total traces → %s",
             dt, dt / 60.0, f"{len(traces):,}", trace_path)

    log.info("── Stage 3: performance report ──")
    report_path = ROOT / "workspace" / "backtest_report.md"
    markdown = PerformanceReport().generate(traces, out_path=report_path)
    log.info("Report written → %s", report_path)
    # Echo the headline metrics (bullet lines + section headers) to the log.
    for line in markdown.splitlines():
        s = line.strip()
        if s == "## Disclaimer":
            break
        if s.startswith("- ") or s.startswith("## ") or s.startswith("**System"):
            log.info("  %s", s)
    return traces


# ── cli ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kyros Phase 3A backtest runner")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="run Stage 1 only (no LLM, no spend)")
    parser.add_argument("--start", default="2024-01-01",
                        help="start date YYYY-MM-DD (inclusive; default 2024-01-01)")
    parser.add_argument("--end", default=date.today().isoformat(),
                        help="end date YYYY-MM-DD (inclusive; default today)")
    parser.add_argument("-y", "--yes", "--resume", dest="yes", action="store_true",
                        help="skip the spend-confirmation prompt (resume is automatic)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="DEBUG-level console logging")
    parser.add_argument("--log-file", default=None,
                        help="log-file path (default: workspace/logs/backtest_<ts>.log)")
    parser.add_argument("--no-log-file", action="store_true",
                        help="disable file logging")
    parser.add_argument("--no-progress", action="store_true",
                        help="disable the tqdm progress bar")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.no_log_file:
        log_file = None
    elif args.log_file:
        log_file = Path(args.log_file)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = ROOT / "workspace" / "logs" / f"backtest_{ts}.log"

    setup_logging(args.verbose, log_file)
    # Calibrator writes workspace/calibration_report.json relative to CWD, and the
    # csv backend resolves KYROS_CSV_PATH relative to CWD — pin both to the repo root.
    os.chdir(ROOT)

    log.info("Kyros Phase 3A backtest runner")
    log.info("repo root  : %s", ROOT)
    log.info("date range : %s → %s (inclusive)", args.start, args.end)
    backend = os.getenv("KYROS_DATA_BACKEND", "yfinance")
    log.info("data backend: %s", backend)
    if backend == "csv":
        log.info("csv path   : %s",
                 os.getenv("KYROS_CSV_PATH", "workspace/data/nq_1min_data.csv"))

    # ── Load + cache the canonical parquet ──────────────────────────────────
    log.info("Loading NQ 1m data...")
    t0 = time.perf_counter()
    try:
        data_path = DataLoader().load(args.start, args.end)
    except Exception:
        log.exception("Failed to load data — check KYROS_DATA_BACKEND / KYROS_CSV_PATH.")
        return 1
    log.info("Data ready in %.1fs → %s", time.perf_counter() - t0, data_path)

    # Show the bar only on an interactive terminal — when stderr is redirected
    # (piped, or captured to a file) tqdm would spam newlines, so fall back to
    # the plain per-stage timing logs instead.
    progress = not args.no_progress and sys.stderr.isatty()

    # ── Stage 1: calibration (validate gaps once, on this first pass) ────────
    cal = run_calibration(ReplayCandleSource(str(data_path), validate_gaps=True), progress)

    if args.calibrate_only:
        log.info("--calibrate-only set; stopping before any LLM calls.")
        log.info("Full run (est. ~$%.2f): drop --calibrate-only (add --yes to skip "
                 "the prompt).", cal.estimated_llm_cost_usd)
        return 0

    if cal.total_fires == 0:
        log.warning("0 trigger fires over this period — nothing for the LLM to "
                    "evaluate. Widen the date range or check the data. Exiting.")
        return 0

    # ── Build the reasoning stack + confirm spend ───────────────────────────
    loop, engine_cfg = build_reasoning_loop()
    log.info("reasoning model: %s/%s (temp=%s)",
             engine_cfg.get("provider"), engine_cfg.get("model"),
             engine_cfg.get("temperature"))
    preflight_api_key(engine_cfg)

    if not args.yes:
        if not sys.stdin.isatty():
            log.error("Non-interactive run without --yes; refusing to spend "
                      "~$%.2f on %s LLM calls. Re-run with --yes.",
                      cal.estimated_llm_cost_usd, f"{cal.total_fires:,}")
            return 1
        log.info("About to make ~%s LLM calls (est. ~$%.2f).",
                 f"{cal.total_fires:,}", cal.estimated_llm_cost_usd)
        try:
            answer = input("Proceed with the full backtest? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer != "y":
            log.info("Aborted. (Use --calibrate-only for the free path, or --yes "
                     "to skip this prompt.)")
            return 0

    # ── Stages 2+3: backtest + report (fresh source; gaps already reported) ──
    run_backtest(
        ReplayCandleSource(str(data_path), validate_gaps=False),
        loop,
        ROOT / "workspace" / "trade_traces.jsonl",
        progress,
    )
    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
