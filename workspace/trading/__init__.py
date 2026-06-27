"""trading package — Phase 2 trading loop and alert pipeline.

This package wires the Phase 1 detectors (read-only) into a live trading
pipeline: candle sources → sliding windows → snapshot builder → trigger
engine → LLM reasoning agent → validated alert → JSONL emit.

No broker, no IBKR, no live market data, no order placement. All sources are
mock or replay (offline).
"""
