"""``code/tests`` 的公共配置。

两件事：

1. 把 ``code/`` 插进 ``sys.path``，这样无论从仓库根目录还是别的目录跑 pytest，
   ``import dino_wbc`` 都能成功。
2. 提供一个「权限友好」的 ``tmp_path_factory``：pytest 9 默认用
   ``Path.mkdir(mode=0o700)`` 建临时目录，在 Windows + DSH 沙箱下这种目录会
   拒绝被列举（``WinError 5``），导致所有用 ``tmp_path`` 的用例报错。这里用
   默认权限在系统临时目录下重建同样的接口，行为与 pytest 一致。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class SafeTempPathFactory:
    """``TempPathFactory`` 的最小兼容实现（不设置 0o700 权限）。

    只实现 pytest 的 ``tmp_path`` / ``tmpdir`` 用到的接口：
    ``mktemp(basename, numbered=True)``、``getbasetemp()``、``_retention_policy``。
    目录放在系统临时目录下的 ``dino_wbc-pytest/<session>``，用默认权限创建，
    这样 Windows 沙箱下也能正常读写/列举。
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._retention_policy = "all"
        self._counter: dict[str, int] = {}

    def getbasetemp(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root

    def mktemp(self, basename: str, numbered: bool = True) -> Path:
        base = self.getbasetemp()
        if not numbered:
            path = base / basename
            path.mkdir(parents=True, exist_ok=True)
            return path
        number = self._counter.get(basename, 0)
        while True:
            path = base / f"{basename}{number}"
            try:
                path.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                number += 1
                continue
            self._counter[basename] = number + 1
            return path


@pytest.fixture(scope="session")
def tmp_path_factory() -> SafeTempPathFactory:
    """覆盖 pytest 内置同名 fixture，避开 0o700 权限在沙箱下的列举失败。"""
    import os

    base = Path(tempfile.gettempdir()) / "dino_wbc-pytest" / str(os.getpid())
    return SafeTempPathFactory(base)


@pytest.fixture(scope="session")
def code_dir() -> Path:
    """``code/`` 目录（registry.json 等资源以它为基准）。"""
    return CODE_DIR


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """仓库根目录。"""
    return REPO_ROOT


def load_model_or_skip():
    """加载默认模型；缺 mujoco / 模型文件时跳过当前用例。"""
    pytest.importorskip("mujoco")
    from dino_wbc import model_io

    try:
        return model_io.load()
    except FileNotFoundError as exc:  # pragma: no cover - 模型还没生成
        pytest.skip(f"模型文件不可用：{exc}")


@pytest.fixture(scope="session")
def model_layout():
    """默认模型与关节布局，供需要 MuJoCo 的用例复用。"""
    return load_model_or_skip()
