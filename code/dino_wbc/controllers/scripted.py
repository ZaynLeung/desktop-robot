"""无 checkpoint 的脚本控制器。

用途：

* ``scripted_stand``：零动作站姿，冒烟测试 / 演示渲染用（机器人靠位置伺服
  保持在站姿，短时间不会倒）。
* ``scripted_lift``：单脚正弦抬脚，用来肉眼检查抬脚方向、动作尺度与接触检测。

两者都不依赖 torch，缺依赖的机器上也能跑完整运行端。
"""

from __future__ import annotations

import numpy as np

from .. import config
from .base import Controller

#: 抬脚正弦的默认频率（Hz）。0.5 Hz ≈ 2 s 一个抬脚周期。
DEFAULT_LIFT_FREQ_HZ = 0.5
#: 抬脚动作幅度（归一化动作单位，≤ 1）。乘 ``ACTION_SCALE`` 后就是关节角幅度。
DEFAULT_LIFT_AMPLITUDE = 0.7


class HoldStandController(Controller):
    """零动作站姿控制器：所有关节目标 = 站姿（动作 0 = ctrl 0）。"""

    name = "scripted_stand"
    stage = "stand"

    def act(self, obs: np.ndarray) -> np.ndarray:
        self._check_obs(obs)
        return np.zeros(self.act_dim, dtype=np.float32)

    def describe(self) -> dict[str, object]:
        info = super().describe()
        info["policy"] = "hold-zero"
        return info


class LiftFootController(Controller):
    """按内部计时器做正弦抬脚（单脚）。

    只动目标脚的 ``hip_pitch`` / ``knee`` / ``ankle``，其余关节保持 0（站姿），
    因此动作空间里每个通道都在 ``[-1, 1]`` 内，乘 ``ACTION_SCALE`` 后天然不超过
    动作尺度。用 ``config.JOINT_NAMES.index`` 找关节下标，不写死顺序。

    抬脚波形用 ``0.5 * (1 - cos(2πft))``：始终 ≥ 0，等效于「抬起来 → 放下」，
    不会出现负半周把脚往地里踩的情况。
    """

    name = "scripted_lift"
    stage = "lift"

    #: 各关节相对抬脚波形的系数；符号按模型约定：
    #: 屈髋 = hip_pitch 正、屈膝 = knee 负、踝背屈 = ankle 正。
    _HIP_GAIN = 0.55
    _KNEE_GAIN = -0.75
    _ANKLE_GAIN = 0.45

    def __init__(
        self,
        *,
        target_foot: float = -1.0,
        frequency: float = DEFAULT_LIFT_FREQ_HZ,
        amplitude: float = DEFAULT_LIFT_AMPLITUDE,
        dt: float = 1.0 / config.CONTROL_HZ,
    ) -> None:
        if target_foot == 0.0:
            raise ValueError("target_foot 必须是 -1（左脚）或 +1（右脚）")
        if not 0.0 < float(amplitude) <= 1.0:
            raise ValueError("amplitude 必须在 (0, 1] 内（归一化动作尺度）")
        self.target_foot = 1.0 if float(target_foot) > 0.0 else -1.0
        self.frequency = float(frequency)
        self.amplitude = float(amplitude)
        self.dt = float(dt)
        self._time = 0.0

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def side(self) -> str:
        """目标脚侧：``"L"`` / ``"R"``。"""
        return "L" if self.target_foot < 0.0 else "R"

    def set_target_foot(self, target_foot: float) -> None:
        """运行端 ``L`` / ``R`` 键切换目标脚（-1 左脚 / +1 右脚）。"""
        if target_foot == 0.0:
            raise ValueError("target_foot 必须是 -1（左脚）或 +1（右脚）")
        new_side = "L" if float(target_foot) < 0.0 else "R"
        if new_side != self.side:
            # 换脚时把相位归零，避免一只脚刚从高点切到另一只脚。
            self._time = 0.0
        self.target_foot = 1.0 if float(target_foot) > 0.0 else -1.0

    def reset(self) -> None:
        self._time = 0.0

    # ------------------------------------------------------------------
    # 动作
    # ------------------------------------------------------------------
    def _joint_index(self, name: str) -> int:
        try:
            return config.JOINT_NAMES.index(name)
        except ValueError as exc:  # pragma: no cover - config 已固定 18 关节
            raise KeyError(f"config.JOINT_NAMES 里找不到关节 {name!r}") from exc

    def act(self, obs: np.ndarray) -> np.ndarray:
        self._check_obs(obs)
        # 0.5 * (1 - cos) 从 0 平滑升到 amplitude 再回 0，始终非负。
        wave = self.amplitude * 0.5 * (
            1.0 - np.cos(2.0 * np.pi * self.frequency * self._time)
        )
        action = np.zeros(self.act_dim, dtype=np.float64)
        side = self.side
        action[self._joint_index(f"hip_pitch_{side}")] = self._HIP_GAIN * wave
        action[self._joint_index(f"knee_{side}")] = self._KNEE_GAIN * wave
        action[self._joint_index(f"ankle_{side}")] = self._ANKLE_GAIN * wave
        self._time += self.dt
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def describe(self) -> dict[str, object]:
        info = super().describe()
        info.update(
            {
                "policy": "sine-lift",
                "target_foot": self.target_foot,
                "side": self.side,
                "frequency_hz": self.frequency,
                "amplitude": self.amplitude,
            }
        )
        return info
