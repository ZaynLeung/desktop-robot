"""归一化动作 → 位置执行器目标（``ctrl``）。

模型是**位置伺服**：``ctrl`` 是站姿相对的关节目标角，MuJoCo 内部按
``kp·(ctrl − q) − kv·qd`` 出力并受 ``forcerange`` 限制。策略输出 18 维
``a ∈ [-1, 1]``，这里做尺度缩放并裁到关节范围。
"""

from __future__ import annotations

import numpy as np

from . import config
from .model_io import JointLayout


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
