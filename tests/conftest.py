"""pytest 公共设置。

两个作用：

1. 把仓库根加入 ``sys.path``，这样无需安装包也能 ``import desktop_robot``。
2. 提供 ``tmp_models`` fixture —— 统一在仓库内的 ``build/pytest-tmp/`` 下开临时目录。
   默认的 ``tmp_path`` 会落到系统临时目录，在某些受限环境（沙箱、只读 %TEMP%）下
   不可写；放在仓库内则总是可写，而且 ``build/`` 已经在 .gitignore 里。

pytest 和 ``python -m unittest`` 都会自动加载本文件。
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TMP_ROOT = ROOT / "build" / "pytest-tmp"


@pytest.fixture()
def tmp_models(request) -> Path:
    """一个干净的、位于仓库内的工作目录。"""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name)
    path = TMP_ROOT / safe_name
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
