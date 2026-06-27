# Kyros Backtest Report

**System prompt version:** `b64522da`

## Overall Metrics

- Total traces: 8
- No-trade rate: 12.5% (1)
- Actionable alerts: 7
- Fill rate (filled / actionable): 85.7% (6/7)
- Win rate (of filled): 50.0% (3)
- Loss rate (of filled): 33.3% (2)
- Expired rate (of filled): 16.7% (1)
- Avg winning R: 2.10
- Avg losing R: -1.00
- Profit factor: 3.15
- Max drawdown (R): 1.00
- Expectancy per trade (R): 0.5750

## By Model Type

| Model | Fires | Win Rate |
|-------|-------|----------|
| 2022 | 3 | 100.0% |
| breaker | 1 | 0.0% |
| ifvg | 1 | 100.0% |
| none | 1 | 0.0% |
| silver_bullet | 1 | 0.0% |
| unicorn | 1 | 0.0% |

## By Killzone

| Killzone | Fires | Win Rate |
|----------|-------|----------|
| ny_am_kz | 5 | 75.0% |
| ny_pm_kz | 3 | 0.0% |

## By Month

| Month | Fires | Win Rate |
|-------|-------|----------|
| 2023-08 | 8 | 50.0% |

## Golden Dataset Match

- Total directional golden entries: 645
- Matched (within ±15 min, same direction): 0
- Match rate: 0.0%

## Disclaimer

This backtest report is a SIMULATION on historical data. Past performance does not guarantee future results. Outcomes are resolved using the LLM's own entry_zone/stop/target with entry_mid fills (optimistic for gap fills) and conservative same-candle loss resolution. Slippage, commissions, and real-world execution friction are NOT modeled. Results may be optimistically biased: the LLM may have seen this period in its training data, so historical pattern recall cannot be distinguished from genuine edge. The golden_alerts.json dataset is untrusted community data matched for direction only — its claims carry no evidential weight. Do not use this report as the sole basis for any trading decision.
