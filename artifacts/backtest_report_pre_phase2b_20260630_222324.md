# Kyros Backtest Report

**System prompt version:** `b64522da`

## Overall Metrics

- Total traces: 179
- No-trade rate: 96.6% (173)
- Actionable alerts: 6
- Fill rate (filled / actionable): 83.3% (5/6)
- Win rate (of filled): 20.0% (1)
- Loss rate (of filled): 20.0% (1)
- Expired rate (of filled): 60.0% (3)
- Cancelled (target/stop hit before entry): 1
- Avg winning R: 2.41
- Avg losing R: -1.00
- Profit factor: 2.41
- Max drawdown (R): 1.00
- Expectancy per trade (R): 0.0079

## By Model Type

| Model | Fires | Win Rate |
|-------|-------|----------|
| 2022 | 3 | 0.0% |
| ifvg | 3 | 50.0% |
| none | 171 | 0.0% |
| silver_bullet | 1 | 0.0% |
| unicorn | 1 | 0.0% |

## By Killzone

| Killzone | Fires | Win Rate |
|----------|-------|----------|
| london_kz | 86 | 0.0% |
| ny_am_kz | 39 | 50.0% |
| ny_pm_kz | 54 | 0.0% |

## By Month

| Month | Fires | Win Rate |
|-------|-------|----------|
| 2026-06 | 179 | 20.0% |

## Golden Dataset Match

- Total directional golden entries (within backtest window): 3
- Matched (within ±15 min, same direction): 0
- Match rate: 0.0%

## Disclaimer

This backtest report is a SIMULATION on historical data. Past performance does not guarantee future results. Outcomes are resolved using the LLM's own entry_zone/stop/target with entry_mid fills (optimistic for gap fills) and conservative same-candle loss resolution. Slippage, commissions, and real-world execution friction are NOT modeled. Results may be optimistically biased: the LLM may have seen this period in its training data, so historical pattern recall cannot be distinguished from genuine edge. The golden_alerts.json dataset is untrusted community data matched for direction only — its claims carry no evidential weight. Do not use this report as the sole basis for any trading decision.
