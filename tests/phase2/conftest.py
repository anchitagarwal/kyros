"""Phase 2 test fixtures: ensure workspace is on sys.path for `trading` imports."""

import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parent.parent.parent / "workspace"
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))
