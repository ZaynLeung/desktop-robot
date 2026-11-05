"""控制器层：统一接口 + 脚本控制器 + 策略控制器 + registry 工厂。

``BaseController`` / ``Controller``
    运行端唯一依赖的接口（:class:`~dino_wbc.controllers.base.Controller`）。

``HoldStandController`` / ``LiftFootController``
    不需要 checkpoint 的脚本控制器，用于冒烟、演示和动作尺度自检。

``PolicyController``
    加载 ``policy_io`` 保存的 ActorNet checkpoint，供已训练阶段使用。

``load_controller``
    运行端入口：名字 → 脚本控制器 / registry 条目 / ``.pt`` 路径。
"""

from .base import Controller
from .policy import PolicyController
from .registry import load_controller
from .scripted import HoldStandController, LiftFootController

__all__ = [
    "Controller",
    "HoldStandController",
    "LiftFootController",
    "PolicyController",
    "load_controller",
]
