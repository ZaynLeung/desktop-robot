"""让 ``code/`` 下的测试能 ``import dino_wbc``（同时保留 ``code/tests/conftest.py``）。"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
