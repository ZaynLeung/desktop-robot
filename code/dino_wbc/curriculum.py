"""课程推进：抬脚高度/行走速度按成功率爬坡，standup 按难度档位晋级。

每个 env 实例各自持有一份 :class:`Curriculum`（向量化训练时互不影响），
运行端不需要它——checkpoint 里已经固化了训练时的能力上限。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class CurriculumState:
    """可序列化/可打印的课程进度。"""

    stage: str
    level: float
    pose: str
    success_rate: float
    max_lift_h: float
    max_vx: float
    max_vy: float
    push_force: float = 0.0
    max_yaw: float = 0.0


class Curriculum:
    """按 episode 成败推进的简单课程。"""

    def __init__(self, stage: str, rng: np.random.Generator | None = None):
        self.stage = stage
        self.rng = rng if rng is not None else np.random.default_rng()
        self.level = 0.0  # lift / walk 的爬坡进度 ∈ [0, 1]
        self.pose_index = 0  # standup 的难度档位
        self.window: deque[bool] = deque(maxlen=config.CURRICULUM_WINDOW)
        self.lift_h_max = config.LIFT_H_RANGE[0]
        self.vx_max = config.WALK_VX_RANGE[0]
        self.vy_max = config.WALK_VY_RANGE[0]
        self.yaw_max = config.TURN_YAW_RANGE[0]
        push = config.STAGES[stage].push
        self.push_force = 0.0
        self.push_max = float(push.max_force) if push is not None else 0.0
        self._apply_level()

    # ------------------------------------------------------------------
    # 采样
    # ------------------------------------------------------------------
    def sample_lift_h(self) -> float:
        low, high = config.LIFT_H_RANGE
        return float(self.rng.uniform(low, max(low, self.lift_h_max)))

    def sample_lift_target(self) -> float:
        return -1.0 if self.rng.random() < 0.5 else 1.0

    def sample_velocity(self) -> tuple[float, float]:
        vx = float(self.rng.uniform(-self.vx_max, self.vx_max))
        vy = float(self.rng.uniform(-self.vy_max, self.vy_max))
        return vx, vy

    def sample_standup_pose(self) -> config.StandupPose:
        return config.STANDUP_POSES[self.pose_index]

    def sample_yaw(self) -> float:
        """turn 阶段采样一个绝对朝向指令（rad）。"""
        return float(self.rng.uniform(-self.yaw_max, self.yaw_max))

    def sample_push_force(self) -> float:
        """当前推力课程下的一次推力模长（N）；未启用或还未爬坡时返回 0。"""
        if self.push_max <= 0.0 or self.push_force <= 0.0:
            return 0.0
        return float(
            self.rng.uniform(config.PUSH_MIN_FORCE * self.push_force, self.push_force)
        )

    # ------------------------------------------------------------------
    # 推进
    # ------------------------------------------------------------------
    def on_episode(self, success: bool) -> None:
        self.window.append(bool(success))
        if self.stage == "standup":
            self._advance_standup()
        elif self.stage == "stand":
            # 站立阶段只推"轻推力度"课程：成功就加一点，失败就退一点。
            step = config.PUSH_RAMP_UP if success else -config.PUSH_RAMP_DOWN
            self.push_force = float(
                np.clip(self.push_force + step, 0.0, self.push_max)
            )
        elif self.stage in config.CURRICULUM:
            up = (
                config.CURRICULUM_TURN_UP
                if self.stage == "turn"
                else config.CURRICULUM_LIFT_UP
            )
            down = (
                config.CURRICULUM_TURN_DOWN
                if self.stage == "turn"
                else config.CURRICULUM_LIFT_DOWN
            )
            step = up if success else -down
            self.level = float(np.clip(self.level + step, 0.0, 1.0))
            self._apply_level()

    def _apply_level(self) -> None:
        if self.stage == "lift":
            cfg = config.CURRICULUM["lift"]["lift_h"]
            self.lift_h_max = cfg.start + (cfg.end - cfg.start) * self.level
        elif self.stage == "walk":
            cfg = config.CURRICULUM["walk"]
            self.vx_max = cfg["vx"].start + (cfg["vx"].end - cfg["vx"].start) * self.level
            self.vy_max = cfg["vy"].start + (cfg["vy"].end - cfg["vy"].start) * self.level
        elif self.stage == "turn":
            cfg = config.CURRICULUM["turn"]["yaw"]
            self.yaw_max = cfg.start + (cfg.end - cfg.start) * self.level

    def _advance_standup(self) -> None:
        if len(self.window) < self.window.maxlen:
            return
        rate = sum(self.window) / len(self.window)
        last = len(config.STANDUP_CURRICULUM) - 1
        if rate >= config.STANDUP_PROMOTE_SUCCESS and self.pose_index < last:
            self.pose_index += 1
            self.window.clear()
        elif rate <= config.STANDUP_DEMOTE_SUCCESS and self.pose_index > 0:
            self.pose_index -= 1
            self.window.clear()

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    @property
    def success_rate(self) -> float:
        return float(sum(self.window) / len(self.window)) if self.window else 0.0

    def state(self) -> CurriculumState:
        return CurriculumState(
            stage=self.stage,
            level=self.level,
            pose=config.STANDUP_POSES[self.pose_index].name,
            success_rate=self.success_rate,
            max_lift_h=self.lift_h_max,
            max_vx=self.vx_max,
            max_vy=self.vy_max,
            push_force=self.push_force,
            max_yaw=self.yaw_max,
        )
