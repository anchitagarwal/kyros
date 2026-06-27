## Problem Statement


Build the Phase 2 Agentic Reasoning Engine for Project Kyros: a non-stop Python
trading loop that ingests OHLCV candle data, runs ICT analysis via Phase 1
detectors, and calls an LLM to produce structured JSON trade alerts when market
conditions are worth reasoning about.

The system uses a mock/replay CandleSource for Phase 2 (no live data, no broker).
The LLM receives a pre-computed MarketSnapshot and outputs an AlertPayload.
Python validates R:R after every LLM call — alerts below 1:1 become no_trade
entries in the log.

Key architectural decisions already made:
- LLM-as-judge: all Phase 1 detectors run upfront, LLM synthesizes
- TriggerEngine gates LLM calls: killzone + HTF bias + DOL existence + cooldown
- DOL (draw on liquidity) drives TP, SL, and R:R
- ICT models the LLM recognizes: 2022, Unicorn, iFVG, Silver Bullet, Breaker
- Low hanging fruit: if no intermediate unswept liquidity blocks path to DOL
- Single model_router.call() per trigger — not call_agentic


## End Goal


A tested, runnable TradingLoop that:
1. Accepts a CandleSource (MockCandleSource or ReplayCandleSource)
2. Maintains a CandleWindow per timeframe (4h/1h/15m/5m/1m)
3. Builds a MarketSnapshot via SnapshotBuilder using Phase 1 detectors
4. Evaluates TriggerEngine on every new candle
5. Calls the LLM Reasoning Agent when triggered
6. Validates R:R (minimum 1:1) post-LLM call
7. Emits AlertPayload to JSON log file and stdout

Full test suite runnable with MockCandleSource — no live data or API keys needed.


## Constraints


HARD CONSTRAINTS:
- No broker, no IBKR, no order placement, no live data
- Phase 1 detectors (workspace/detectors/) are READ-ONLY — do not modify
- Reuse existing ModelRouter, ExecutorToolkit, KyrosAgentLoader
- Do NOT use call_agentic() for the trading agent — use call() only
- Python validates R:R post-LLM — never trust LLM arithmetic
- Alert output: JSON log + stdout only. No Telegram in Phase 2

ARCHITECTURE:
Timeframe stack:
  4h  → 60 candles  (weekly structure, HTF bias)
  1h  → 100 candles (daily structure, session context)
  15m → 200 candles (setup timeframe)
  5m  → 300 candles (entry timeframe)
  1m  → 500 candles (precision / MSS confirmation)

TriggerEngine — 4 hard gates (ALL must pass):
  1. current_killzone is not None
  2. htf_bias is not None (confirmed BOS/ChoCH on 4h or 1h)
  3. nearest_dol is not None (unswept opposing pool exists)
  4. cooldown clear (15 min since last LLM call)

TriggerEngine — soft triggers (ANY is sufficient):
  - Active unmitigated FVG on 5m
  - iFVG on 5m
  - Liquidity sweep on 15m in last 10 candles
  - Displacement on 5m in last 10 candles

DOL logic:
  - SnapshotBuilder delivers all_pools: all unswept opposing pools across
    all timeframes, sorted by proximity, with confluence_count
  - LLM selects target pool from all_pools and explains why in rationale
  - LLM considers: HTF significance, confluence, whether intermediate
    pools block path
  - Python validates selected target gives >= 1:1 R:R
  - TriggerEngine gate: any unswept opposing pool exists (binary, no threshold)

ICT SYSTEM PROMPT:
  - LLM identifies which model applies per alert:
      2022     → AMD structure: sweep → displacement → FVG retracement
      unicorn  → BOS displacement FVG + OB at same level
      ifvg     → filled FVG now acting as opposing S/R
      silver_bullet → 10:00-11:00 ET or 14:00-15:00 ET displacement FVG
      breaker  → failed OB flipped to opposing S/R
      none     → no_trade
  - DOL-first reasoning: enumerate all unswept pools, target nearest in bias
    direction after sweep. If intermediate unswept pool exists between entry
    and DOL, output no_trade with reason "intermediate liquidity in path"
  - Output structured JSON only — no prose outside the JSON block

ALERT OUTPUT SCHEMA:
{
  "bias":           "long | short | no_trade",
  "model":          "2022 | unicorn | ifvg | silver_bullet | breaker | none",
  "conviction":     0-100,
  "entry_zone":     [float, float],
  "stop":           float,
  "target":         float,
  "dol": {
    "level":        float,
    "type":         str,
    "timeframe":    str
  },
  "risk_reward":    float,
  "rationale":      str,
  "killzone":       str,
  "valid_until":    str,
  "no_trade_reason": str | null
}

INSTRUMENT: NQ only (/NQ=F) for Phase 2
