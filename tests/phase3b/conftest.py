"""Shared fixtures for Phase 3B tests."""

import sys
from pathlib import Path

# Ensure workspace/ is importable (trading.*, backtesting.*, tuning.*).
_WORKSPACE = Path(__file__).resolve().parent.parent.parent / "workspace"
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))
