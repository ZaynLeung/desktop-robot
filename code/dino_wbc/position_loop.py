"""外层位置环：位置误差 → walk 速度指令（内环由 walk 策略跟踪）。

低层 walk 策略只跟踪体坐标系 ``(vx, vy)``，本身没有"走到某个世界坐标"的
目标。这里用 P/I 控制器把世界系位置误差转成体坐标系速度指令，运行端每步
调用一次；它不改变观测/契约，也不需要重训低层策略。

设计要点：
* 体坐标系：walk 的 ``vx`` 是前进、``vy`` 是左侧；用当前 base yaw 做旋转。
* 死区：误差小于 ``deadband`` 时指令归零，避免在目标附近来回抖。
* 限幅：输出限制在 walk 策略训练过的速度范围内。
* 积分抗饱和：积分项单独限幅，死区时清零。
"""

from __future__ import annotations

import numpy as np

from . import config


class PositionLoop:
    """世界系位置目标 → 体坐标系速度指令。"""

    def __init__(
        self,
        target_xy: tuple[float, float] | np.ndarray | None = None,
        *,
        kp: float = config.POSITION_LOOP_KP,
        ki: float = config.POSITION_LOOP_KI,
        deadband: float = config.POSITION_LOOP_DEADBAND,
        max_vx: float = config.POSITION_LOOP_MAX_VX,
        max_vy: float = config.POSITION_LOOP_MAX_VY,
        i_clamp: float = config.POSITION_LOOP_I_CLAMP,
        dt: float = 1.0 / config.CONTROL_HZ,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.deadband = float(deadband)
        self.max_vx = float(max_vx)
        self.max_vy = float(max_vy)
        self.i_clamp = float(i_clamp)
        self.dt = float(dt)
        self.target_xy = np.zeros(2, dtype=np.float64)
        self.integral = np.zeros(2, dtype=np.float64)
        self.enabled = False
        if target_xy is not None:
            self.reset(target_xy)

    # ------------------------------------------------------------------
    # 目标/状态
    # ------------------------------------------------------------------
    def reset(self, target_xy: tuple[float, float] | np.ndarray) -> None:
        """设置目标并清积分；给定目标即启用。"""
        self.target_xy = np.asarray(target_xy, dtype=np.float64)[:2].copy()
        self.integral[:] = 0.0
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False
        self.integral[:] = 0.0

    def set_target(self, target_xy: tuple[float, float] | np.ndarray) -> None:
        self.reset(target_xy)

    # ------------------------------------------------------------------
    # 控制
    # ------------------------------------------------------------------
    def error(self, base_xy: np.ndarray) -> np.ndarray:
        """世界系位置误差（目标 − 当前）。"""
        return self.target_xy - np.asarray(base_xy, dtype=np.float64)[:2]

    def update(
        self, base_xy: np.ndarray, yaw_rad: float
    ) -> tuple[float, float]:
        """返回体坐标系速度指令 ``(vx, vy)``；未启用时返回 (0, 0)。"""
        if not self.enabled:
            return 0.0, 0.0
        err_world = self.error(base_xy)
        if float(np.linalg.norm(err_world)) < self.deadband:
            # 到目标附近：清积分，避免"越过目标后积分还推着走"。
            self.integral[:] = 0.0
            return 0.0, 0.0
        c = float(np.cos(yaw_rad))
        s = float(np.sin(yaw_rad))
        ex, ey = float(err_world[0]), float(err_world[1])
        err_body = np.array([c * ex + s * ey, -s * ex + c * ey])

        self.integral += err_body * self.dt
        self.integral[:] = np.clip(
            self.integral, -self.i_clamp, self.i_clamp
        )
        vx = self.kp * err_body[0] + self.ki * self.integral[0]
        vy = self.kp * err_body[1] + self.ki * self.integral[1]
        vx = float(np.clip(vx, -self.max_vx, self.max_vx))
        vy = float(np.clip(vy, -self.max_vy, self.max_vy))
        return vx, vy

    # ------------------------------------------------------------------
    def describe(self) -> dict:
        return {
            "enabled": self.enabled,
            "target_xy": self.target_xy.tolist(),
            "kp": self.kp,
            "ki": self.ki,
            "deadband": self.deadband,
            "max_vx": self.max_vx,
            "max_vy": self.max_vy,
        }
