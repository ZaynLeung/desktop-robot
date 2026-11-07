"""归一化动作 → 位置执行器目标（``ctrl``）。

模型是**位置伺服**：``ctrl`` 是站姿相对的关节目标角，MuJoCo 内部按
``kp·(ctrl − q) − kv·qd`` 出力并受 ``forcerange`` 限制。策略输出 18 维
``a ∈ [-1, 1]``，这里做尺度缩放并裁到关节范围。

另含 :class:`ActionFilter`：训练端和运行端共用的动作低通（EMA），用来压掉
策略输出的高频抖动；两端必须用同一个 α，否则训练/运行分布不一致。
"""

from __future__ import annotations

import numpy as np

from . import config
from .model_io import JointLayout


class ActionFilter:
    """一阶 EMA 动作滤波：``y = α·x + (1−α)·y_prev``。

    ``alpha >= 1`` 时等价于不滤波；``reset()`` 把内部状态清零（环境复位/重连时用）。
    """

    def __init__(self, alpha: float | None = None, num_joints: int = config.ACTION_DIM):
        a = config.ACTION_FILTER_ALPHA if alpha is None else float(alpha)
        self.alpha = float(np.clip(a, 0.0, 1.0))
        self.state = np.zeros(int(num_joints), dtype=np.float64)

    def reset(self, state: np.ndarray | None = None) -> None:
        if state is None:
            self.state[:] = 0.0
        else:
            self.state = np.asarray(state, dtype=np.float64).reshape(-1).copy()

    def update(self, raw: np.ndarray) -> np.ndarray:
        """输入策略原始动作，返回滤波后的动作（与输入同形状）。"""
        x = np.clip(np.asarray(raw, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if x.shape[0] != self.state.shape[0]:
            raise ValueError(
                f"动作维度应为 {self.state.shape[0]}，实际 {x.shape[0]}"
            )
        self.state = self.alpha * x + (1.0 - self.alpha) * self.state
        return self.state.copy()


def action_to_ctrl(
    action: np.ndarray,
    layout: JointLayout,
    *,
    scale: tuple[float, ...] = config.ACTION_SCALE,
) -> np.ndarray:
    """``action``（18 维，越界自动裁剪）→ ``ctrl``（18 维，关节范围内）。"""
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.shape[0] != layout.num_joints:
        raise ValueError(
            f"动作维度应为 {layout.num_joints}，实际 {values.shape[0]}"
        )
    target = np.clip(values, -1.0, 1.0) * np.asarray(scale, dtype=np.float64)
    return np.clip(target, layout.ranges[:, 0], layout.ranges[:, 1])


def ctrl_to_action(
    ctrl: np.ndarray,
    *,
    scale: tuple[float, ...] = config.ACTION_SCALE,
) -> np.ndarray:
    """逆映射，主要给调试/脚本控制器对齐用。"""
    return np.clip(
        np.asarray(ctrl, dtype=np.float64).reshape(-1)
        / np.asarray(scale, dtype=np.float64),
        -1.0,
        1.0,
    )
