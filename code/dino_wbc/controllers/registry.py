"""控制器工厂：名字 → 控制器实例。

运行端 ``--controller`` 支持三种写法：

* ``scripted_stand`` / ``scripted_lift``：内置脚本控制器（无需 checkpoint）；
* ``stand`` / ``lift`` / ``standup`` / ``walk``：读 ``code/registry.json`` 里的
  checkpoint（阶段训练完成后由训练端写入）；
* 直接给 ``.pt`` / ``.pth`` 路径：加载任意 checkpoint。

checkpoint 缺失时抛 :class:`FileNotFoundError`，消息里附带训练命令提示。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import config
from .base import Controller
from .policy import PolicyController
from .scripted import HoldStandController, LiftFootController

#: ``code/`` 目录；registry.json 与 checkpoint 路径都以它为基准。
CODE_DIR = Path(__file__).resolve().parents[2]
#: registry 文件位置（测试里会用 monkeypatch 换成临时文件）。
REGISTRY_PATH = CODE_DIR / "registry.json"

#: 脚本控制器：名字 → 无参构造函数。
_SCRIPTED: dict[str, type[Controller]] = {
    "scripted_stand": HoldStandController,
    "scripted_lift": LiftFootController,
}

#: 训练命令提示（checkpoint 缺失时打印给用户）。
TRAIN_HINT = (
    "先训练该阶段再运行（训练端会写出 checkpoint 并更新 registry.json）：\n"
    "  E:\\conda_envs\\desktop-robot\\python.exe code\\train\\train.py --stage {stage}\n"
    "或者先用脚本控制器冒烟：--controller scripted_stand / scripted_lift"
)


class ControllerNotFoundError(FileNotFoundError):
    """名字/路径找不到对应控制器，或 checkpoint 文件缺失。"""


def _load_registry(path: Path | None = None) -> dict[str, Any]:
    """读 registry.json；文件缺失/损坏时给出可执行提示。"""
    target = Path(REGISTRY_PATH) if path is None else Path(path)
    if not target.exists():
        raise ControllerNotFoundError(
            f"找不到控制器注册表：{target}\n"
            "还没有任何训练好的 checkpoint。\n" + TRAIN_HINT.format(stage="stand")
        )
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControllerNotFoundError(
            f"控制器注册表无法解析：{target}\n  {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ControllerNotFoundError(
            f"控制器注册表格式错误（顶层必须是 JSON 对象）：{target}"
        )
    return payload


def _resolve_checkpoint(entry: dict[str, Any], checkpoint: str | None) -> Path:
    """registry 条目 → checkpoint 绝对路径（相对路径以 ``code/`` 为基准）。"""
    raw = checkpoint if checkpoint is not None else entry.get("checkpoint")
    if not raw:
        raise ControllerNotFoundError(
            "registry 条目里没有 checkpoint 字段；请重新训练或手动指定 --checkpoint"
        )
    target = Path(str(raw))
    if not target.is_absolute():
        target = (CODE_DIR / target).resolve()
    return target


def load_controller(
    name_or_path: str,
    *,
    expected_contract: str | None = None,
    checkpoint: str | None = None,
) -> Controller:
    """按名字 / registry 条目 / ``.pt`` 路径加载控制器。

    :param name_or_path: ``scripted_stand`` / ``scripted_lift`` / registry 名字 /
        checkpoint 路径；
    :param expected_contract: 当前模型契约哈希，policy 控制器加载时校验；
    :param checkpoint: 覆盖 registry 里记录的 checkpoint 路径。
    """
    key = str(name_or_path).strip()

    # 1) 内置脚本控制器
    scripted_cls = _SCRIPTED.get(key)
    if scripted_cls is not None:
        return scripted_cls()

    # 2) 直接给 checkpoint 路径
    path = Path(key)
    if path.suffix.lower() in {".pt", ".pth"}:
        target = Path(checkpoint) if checkpoint is not None else path
        if not target.is_absolute():
            target = (Path.cwd() / target).resolve()
        if not target.exists():
            raise ControllerNotFoundError(
                f"找不到策略 checkpoint：{target}\n"
                + TRAIN_HINT.format(stage="<stage>")
            )
        return PolicyController(
            target,
            expected_contract=expected_contract,
            name=f"policy@{target.stem}",
        )

    # 3) registry 名字
    registry = _load_registry()
    if key not in registry:
        options = sorted(set(_SCRIPTED) | set(registry))
        raise ControllerNotFoundError(
            f"未知控制器 {key!r}，可选：{options}\n"
            "也可以直接传 checkpoint 路径（*.pt）。"
        )
    entry = registry[key]
    if not isinstance(entry, dict):
        raise ControllerNotFoundError(f"registry 条目 {key!r} 格式错误：{entry!r}")

    target = _resolve_checkpoint(entry, checkpoint)
    if not target.exists():
        stage = str(entry.get("stage", key))
        raise ControllerNotFoundError(
            f"控制器 {key!r} 的 checkpoint 不存在：{target}\n"
            + TRAIN_HINT.format(stage=stage)
        )
    return PolicyController(
        target,
        expected_contract=expected_contract,
        name=f"{key}@{target.stem}",
    )
