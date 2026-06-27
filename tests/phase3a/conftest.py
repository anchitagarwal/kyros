"""conftest.py — ensure workspace/ is on sys.path for `backtesting` + `trading` imports.

Mirrors tests/phase2/conftest.py. Only ADDS path entries; does not modify
existing test behavior.
"""

import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parent.parent.parent / "workspace"
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))
