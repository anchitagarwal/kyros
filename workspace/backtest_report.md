# Kyros Backtest Report

**System prompt version:** `6874a38f`

## Overall Metrics

- Total traces: 24
- No-trade rate: 54.2% (13)
- Actionable alerts: 11
- Fill rate (filled / actionable): 81.8% (9/11)
- Win rate (of filled): 11.1% (1)
- Loss rate (of filled): 66.7% (6)
- Expired rate (of filled): 22.2% (2)
- Cancelled (target/stop hit before entry): 2
- Avg winning R: 2.16
- Avg losing R: -1.03
- Profit factor: 0.35
- Max drawdown (R): 4.09
- Expectancy per trade (R): -0.1677

## By Model Type

| Model | Fires | Win Rate |
|-------|-------|----------|
| 2022 | 6 | 0.0% |
| ifvg | 6 | 25.0% |
| none | 12 | 0.0% |

## By Killzone

| Killzone | Fires | Win Rate |
|----------|-------|----------|
| london_kz | 4 | 33.3% |
| ny_am_kz | 7 | 0.0% |
| ny_pm_kz | 13 | 0.0% |

## By Month

| Month | Fires | Win Rate |
|-------|-------|----------|
| 2026-06 | 24 | 11.1% |

## Golden Dataset Match

- Total directional golden entries (within backtest window): 3
- Matched (within ±15 min, same direction): 0
- Match rate: 0.0%

## Disclaimer

This backtest report is a SIMULATION on historical data. Past performance does not guarantee future results. Outcomes are resolved using the LLM's own entry_zone/stop/target with entry_mid fills (optimistic for gap fills) and conservative same-candle loss resolution. Slippage, commissions, and real-world execution friction are NOT modeled. Results may be optimistically biased: the LLM may have seen this period in its training data, so historical pattern recall cannot be distinguished from genuine edge. The golden_alerts.json dataset is untrusted community data matched for direction only — its claims carry no evidential weight. Do not use this report as the sole basis for any trading decision.
