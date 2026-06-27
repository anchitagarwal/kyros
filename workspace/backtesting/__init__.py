"""backtesting package — Phase 3A: calibration, outcome simulation, backtest engine.

All modules are offline: no broker, no IBKR, no live market data, no order
placement. Data sources are mock or replay (parquet/CSV). The LLM is the only
network dependency and is mocked in every test.
"""
