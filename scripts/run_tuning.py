#!/usr/bin/env python
"""scripts/run_tuning.py — Phase 3B offline tuning & walk-forward harness.

Two paths:

  DEFAULT (no --record): post-LLM tuning over an existing trace file.
    Loads a baseline trade_traces.jsonl → builds folds → runs walk-forward over
    the post-LLM grid → writes workspace/walkforward_report.md.
    ZERO LLM calls. No API key. Works fully offline.

  --record (Tier-2, cost-gated): record each PreLLMGrid config first, then tune
    over the union of recorded configs. Cost = n_configs × ONE full-span
    backtest (folds do NOT multiply cost — a config's LLM output is
    fold-independent, recorded once and sliced per fold by timestamp). Each
    config is recorded to workspace/tuning/runs/{config_hash}/ to prevent
    ledger collision; idempotent resume means a re-run skips already-recorded
    fires. The cost estimate is printed and confirmed BEFORE any spend
    (mirroring the Phase 3A spend gate).

Usage:
    # Default (free, offline): tune over an existing trace file.
    uv run python scripts/run_tuning.py --traces workspace/trade_traces.jsonl

    # Tier-2 (costs money): record pre-LLM variants first, then tune.
    uv run --env-file .env python scripts/run_tuning.py --record --yes

No broker, no IBKR, no live market data, no order placement.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# ── path setup (mirror run_backtest.py) ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT / "workspace"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

from trading.config import TradingConfig
from tuning.params import PreLLMGrid, default_post_params, param_grid, ALL, PostLLMParams
from tuning.walkforward import make_folds, run_walkforward
from tuning.report import WalkForwardReport

log = logging.getLogger("kyros.tuning")

# Per-config LLM cost per fire (matches the calibrator's $0.003).
_COST_PER_FIRE = 0.003


# ── trace loading ────────────────────────────────────────────────────────────


def load_traces(path: Path) -> list[dict]:
    """Load a trade_traces.jsonl file into a list of trace dicts."""
    if not path.exists():
        raise FileNotFoundError(f"trace file not found: {path}")
    traces = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            traces.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return traces


def load_trace_sets(traces_dir: Path) -> dict[str, list[dict]]:
    """Load per-config trace sets from runs/{config_hash}/trade_traces.jsonl.

    Returns {config_hash: traces}. Used by the Tier-2 path after recording.
    """
    trace_sets: dict[str, list[dict]] = {}
    if not traces_dir.exists():
        return trace_sets
    for sub in sorted(traces_dir.iterdir()):
        if not sub.is_dir():
            continue
        tf = sub / "trade_traces.jsonl"
        if tf.exists():
            trace_sets[sub.name] = load_traces(tf)
    return trace_sets


# ── default (free) path ──────────────────────────────────────────────────────


def run_default(
    traces_path: Path,
    train_days: int,
    test_days: int,
    step_days: int,
    min_trades: int,
    out_path: Path,
) -> str:
    """Default path: post-LLM tuning over an existing trace file. Zero LLM calls.

    Loads the baseline traces, builds folds, runs walk-forward over the
    post-LLM grid, and writes the report. No engine, no API key, no LLM.
    """
    log.info("Loading baseline traces from %s", traces_path)
    traces = load_traces(traces_path)
    log.info("Loaded %d traces", len(traces))

    baseline_cfg = TradingConfig()
    baseline_hash = baseline_cfg.config_hash()
    trace_sets = {baseline_hash: traces}

    return _tune_and_report(
        trace_sets, train_days, test_days, step_days, min_trades, out_path
    )


def _tune_and_report(
    trace_sets: dict[str, list[dict]],
    train_days: int,
    test_days: int,
    step_days: int,
    min_trades: int,
    out_path: Path,
) -> str:
    """Build folds on the baseline span, run walk-forward, write the report."""
    baseline_hash = TradingConfig().config_hash()
    baseline_traces = trace_sets.get(baseline_hash)
    if not baseline_traces:
        # Fall back to the first available set if baseline is absent.
        baseline_traces = next(iter(trace_sets.values()), [])

    log.info("Building folds (train=%dd, test=%dd, step=%dd)",
             train_days, test_days, step_days)
    folds = make_folds(baseline_traces, train_days, test_days, step_days)
    log.info("Produced %d folds", len(folds))

    # Post-LLM grid: baseline + a few conviction/rr variants. The baseline is
    # always included (default_post_params) for the apples-to-apples comparison.
    grid = _default_post_grid()

    log.info("Running walk-forward over %d grid points", len(grid))
    result = run_walkforward(trace_sets, folds, grid, min_trades=min_trades)

    log.info("Writing report → %s", out_path)
    md = WalkForwardReport.generate(result, out_path=str(out_path))
    return md


def _default_post_grid() -> list[PostLLMParams]:
    """The default post-LLM grid: baseline + conviction/rr variants.

    Kept small and deterministic. The baseline (default_post_params) is first
    so it is the deterministic tie-break winner when scores are equal.
    """
    return [
        default_post_params(),
        PostLLMParams(50, 1.0, ALL, ALL),
        PostLLMParams(60, 1.0, ALL, ALL),
        PostLLMParams(40, 1.5, ALL, ALL),
        PostLLMParams(40, 2.0, ALL, ALL),
    ]


# ── Tier-2 recording path (cost-gated) ───────────────────────────────────────


def estimate_recording_cost(configs: list[TradingConfig], data_path: Path) -> float:
    """Estimate the total LLM cost of recording all configs.

    Cost = sum over configs of (total_fires × $0.003), where total_fires is
    estimated by TriggerCalibrator (one full-span calibration per config, NO
    LLM calls). Folds do NOT multiply cost: a config's LLM output is
    fold-independent, recorded once over the whole span and sliced per fold.
    """
    from trading.candle_source import ReplayCandleSource
    from trading.candle_window import CandleWindow
    from trading.snapshot import SnapshotBuilder
    from trading.trigger import TriggerEngine
    from trading.cooldown import CooldownState
    from backtesting.calibrator import TriggerCalibrator

    total = 0.0
    for cfg in configs:
        # Each config gets its own builder/trigger/cooldown so the calibration
        # reflects that config's knobs (killzones, recency, soft triggers).
        cd = CooldownState(config=cfg)
        cal = TriggerCalibrator(
            CandleWindow(),
            SnapshotBuilder(config=cfg),
            TriggerEngine(cd, config=cfg),
            cd,
        )
        src = ReplayCandleSource(str(data_path), validate_gaps=False)
        report = cal.run(src)
        cost = report.total_fires * _COST_PER_FIRE
        log.info("  config %s: %d fires → ~$%.4f",
                 cfg.short_hash(), report.total_fires, cost)
        total += cost
    return total


def record_config(cfg: TradingConfig, data_path: Path, runs_dir: Path,
                  agent=None) -> list:
    """Record one config's traces over the full span (idempotent resume).

    Reuses BacktestEngine (idempotent resume via the alert ledger). Writes to
    runs/{config_hash}/trade_{alerts,traces}.jsonl so configs never collide.
    Returns the recorded traces.
    """
    from trading.candle_source import ReplayCandleSource
    from trading.candle_window import CandleWindow
    from trading.snapshot import SnapshotBuilder
    from trading.trigger import TriggerEngine
    from trading.cooldown import CooldownState
    from trading.trading_loop import TradingLoop
    from backtesting.engine import BacktestEngine
    from backtesting.outcome import OutcomeSimulator

    if agent is None:
        agent = _build_reasoning_agent()

    run_dir = runs_dir / cfg.short_hash()
    run_dir.mkdir(parents=True, exist_ok=True)
    traces_path = run_dir / "trade_traces.jsonl"

    cd = CooldownState(config=cfg)
    loop = TradingLoop(
        source=None,
        window=CandleWindow(),
        builder=SnapshotBuilder(config=cfg),
        trigger=TriggerEngine(cd, config=cfg),
        agent=agent,
        cooldown=cd,
        output_path=str(run_dir / "alerts.jsonl"),
    )
    engine = BacktestEngine(loop, OutcomeSimulator(), output_path=traces_path)
    src = ReplayCandleSource(str(data_path), validate_gaps=False)
    traces = engine.run(src)
    log.info("  config %s: recorded %d traces → %s",
             cfg.short_hash(), len(traces), traces_path)
    return traces


def _build_reasoning_agent():
    """Build the LLM reasoning agent for Tier-2 recording.

    Imported lazily so the default (free) path never imports the router/loader
    (and thus never requires an API key).
    """
    from trading.reasoning_agent import LLMReasoningAgent
    from kyros.core.model_router import ModelRouter
    from kyros.core.agent_loader import KyrosAgentLoader

    loader = KyrosAgentLoader(str(ROOT))
    router = ModelRouter()
    engine_cfg = loader.get_model_engine("trading", fallback_role="executor")
    config = {"model_engine": engine_cfg, "final_system_prompt": ""}
    return LLMReasoningAgent(router, agent_config=config)


def run_tier2(
    configs: list[TradingConfig],
    data_path: Path,
    runs_dir: Path,
    train_days: int,
    test_days: int,
    step_days: int,
    min_trades: int,
    out_path: Path,
    yes: bool,
) -> str:
    """Tier-2 path: cost-gated recording, then tune over the union of configs."""
    # ── 1. Cost estimate (no LLM; calibrator only) ──────────────────────────
    log.info("── Tier-2 cost estimate (calibrator only, no LLM) ──")
    for cfg in configs:
        log.info("  config %s", cfg.short_hash())
    total_cost = estimate_recording_cost(configs, data_path)
    log.info("Total estimated LLM cost: ~$%.4f (%d configs × full-span backtest)",
             total_cost, len(configs))

    # ── 2. Spend gate ───────────────────────────────────────────────────────
    if not yes:
        if not sys.stdin.isatty():
            log.error("Non-interactive run without --yes; refusing to spend "
                      "~$%.4f. Re-run with --yes.", total_cost)
            sys.exit(1)
        try:
            answer = input(f"Proceed with recording (~${total_cost:.4f})? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer != "y":
            log.info("Aborted. Zero spend. (Use the default path without --record "
                     "for the free post-LLM tuning.)")
            sys.exit(0)

    # ── 3. Record each config (idempotent resume) ───────────────────────────
    log.info("── Tier-2 recording (LLM in loop; idempotent resume) ──")
    for cfg in configs:
        record_config(cfg, data_path, runs_dir)

    # ── 4. Load all recorded trace sets → tune ──────────────────────────────
    trace_sets = load_trace_sets(runs_dir)
    # Ensure the baseline is present (it should be, if it was in the grid).
    baseline_hash = TradingConfig().config_hash()
    if baseline_hash not in trace_sets:
        log.warning("Baseline config not found in recorded runs; tuning over "
                    "available configs only.")

    return _tune_and_report(
        trace_sets, train_days, test_days, step_days, min_trades, out_path
    )


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kyros Phase 3B tuning runner")
    parser.add_argument("--traces", default=None,
                        help="path to a baseline trade_traces.jsonl (default path)")
    parser.add_argument("--traces-dir", default=None,
                        help="dir of per-config trace sets (runs/{hash}/) for the "
                             "default path over pre-recorded configs")
    parser.add_argument("--record", action="store_true",
                        help="enable Tier-2: record each PreLLMGrid config first "
                             "(cost-gated, LLM calls)")
    parser.add_argument("--data", default=None,
                        help="path to the 1m parquet for Tier-2 recording "
                             "(required with --record)")
    parser.add_argument("--train-days", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=2)
    parser.add_argument("--step-days", type=int, default=2)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--out", default="workspace/walkforward_report.md")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="skip the spend-confirmation prompt (Tier-2)")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    os.chdir(ROOT)

    out_path = Path(args.out)
    runs_dir = ROOT / "workspace" / "tuning" / "runs"

    if args.record:
        # ── Tier-2 path ────────────────────────────────────────────────────
        if not args.data:
            log.error("--record requires --data (path to the 1m parquet).")
            return 1
        configs = PreLLMGrid()  # default: baseline only; extend with presets here
        run_tier2(
            configs=configs,
            data_path=Path(args.data),
            runs_dir=runs_dir,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            min_trades=args.min_trades,
            out_path=out_path,
            yes=args.yes,
        )
    else:
        # ── Default (free) path ────────────────────────────────────────────
        if args.traces_dir:
            trace_sets = load_trace_sets(Path(args.traces_dir))
            if not trace_sets:
                log.error("No trace sets found in %s", args.traces_dir)
                return 1
            # Tune over pre-recorded configs without re-recording.
            md = _tune_and_report(
                trace_sets, args.train_days, args.test_days, args.step_days,
                args.min_trades, out_path,
            )
        else:
            traces_path = Path(args.traces or "workspace/trade_traces.jsonl")
            md = run_default(
                traces_path, args.train_days, args.test_days, args.step_days,
                args.min_trades, out_path,
            )
        log.info("Done. Report → %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
