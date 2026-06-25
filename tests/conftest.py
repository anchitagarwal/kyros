"""pytest configuration for Kyros detector tests.

The detector package lives under ``workspace/detectors`` which is NOT on the
default ``pythonpath`` (pyproject sets ``pythonpath=["src"]``). We insert the
``workspace`` directory onto ``sys.path`` here so ``from detectors.X import Y``
resolves in the detector tests. This file only ADDS path entries; it does not
modify any existing test behavior.
"""

import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))
