"""80 维观测的构造（训练端与运行端共用）。

分块顺序（与 :mod:`dino_wbc.config` 的 ``OBS_*_DIM`` 一一对应）::

    base(10) | joints(36) | feet(6) | prev_action(18) | command(8) | phase(2)

命令向量内部用**物理单位**保存（m/s、m、±1），只在拼观测时缩放，
这样奖励函数可以直接拿它当目标值用。
"""

from __future__ import annotations

import numpy as np

from . import config
from .model_io import JointLayout
from .state import RobotState


def make_command(
    task: str,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    lift_h: float = 0.0,
    target_foot: float = 0.0,
) -> np.ndarray:
    """构造 8 维命令向量（物理单位）。"""
    command = np.zeros(config.COMMAND_DIM, dtype=np.float64)
    if task not in config.TASK_TO_INDEX:
        raise ValueError(f"未知任务 {task!r}，可选 {config.TASK_NAMES}")
    command[config.TASK_TO_INDEX[task]] = 1.0
    command[config.CMD_VX] = float(vx)
    command[config.CMD_VY] = float(vy)
    command[config.CMD_LIFT_H] = float(lift_h)
    command[config.CMD_TARGET_FOOT] = float(target_foot)
    return command


def target_foot_index(command: np.ndarray) -> int:
    """命令里的 ``target_foot``（-1 左 / +1 右）→ 数组下标。"""
    return 0 if command[config.CMD_TARGET_FOOT] < 0 else 1


def build_obs(
    state: RobotState,
    prev_action: np.ndarray,
    command: np.ndarray,
    phase_sin_cos: tuple[float, float],
    layout: JointLayout,
) -> np.ndarray:
    """按固定顺序拼出 80 维观测（float32）。"""
    base = np.empty(config.OBS_BASE_DIM, dtype=np.float64)
    base[0] = (state.base_pos[2] - layout.stand_base_z) * config.BASE_Z_SCALE
    base[1:4] = np.clip(state.gravity_b, -1.0, 1.0)
    base[4:7] = np.clip(
        state.base_lin_vel_b, -config.BASE_LIN_VEL_CLIP, config.BASE_LIN_VEL_CLIP
    )
    base[7:10] = np.clip(
        state.base_ang_vel_b, -config.BASE_ANG_VEL_CLIP, config.BASE_ANG_VEL_CLIP
    )

    joints = np.empty(config.OBS_JOINT_DIM, dtype=np.float64)
    joints[: config.NUM_JOINTS] = state.q
    joints[config.NUM_JOINTS :] = np.clip(
        state.qd * config.JOINT_VEL_SCALE,
        -config.JOINT_VEL_CLIP,
        config.JOINT_VEL_CLIP,
    )

    feet = np.empty(config.OBS_FOOT_DIM, dtype=np.float64)
    feet[0:2] = np.clip(
        state.foot_z * config.FOOT_Z_SCALE, 0.0, config.FOOT_Z_CLIP
    )
    feet[2:4] = state.contact.astype(np.float64)
    feet[4:6] = np.clip(
        state.foot_slip * config.FOOT_SLIP_SCALE, 0.0, config.FOOT_SLIP_CLIP
    )

    cmd = np.empty(config.OBS_COMMAND_DIM, dtype=np.float64)
    cmd[0:4] = command[0:4]
    cmd[4] = command[config.CMD_VX] * config.CMD_VEL_SCALE
    cmd[5] = command[config.CMD_VY] * config.CMD_VEL_SCALE
    cmd[6] = command[config.CMD_LIFT_H] * config.LIFT_H_SCALE
    cmd[7] = command[config.CMD_TARGET_FOOT]

    obs = np.concatenate(
        [
            base,
            joints,
            feet,
            np.clip(prev_action, -1.0, 1.0),
            cmd,
            np.asarray(phase_sin_cos, dtype=np.float64),
        ]
    )
    if obs.shape != (config.OBS_DIM,):  # pragma: no cover - 配置自检
        raise AssertionError(f"观测维度应为 {config.OBS_DIM}，实际 {obs.shape}")
    return obs.astype(np.float32)


class ObsBuilder:
    """有状态的观测构造器：保存上一步动作与步态时钟。"""

    def __init__(self, layout: JointLayout):
        self.layout = layout
        self.prev_action = np.zeros(layout.num_joints, dtype=np.float64)
        self.phase = 0.0

    def reset(self) -> None:
        self.prev_action[:] = 0.0
        self.phase = 0.0

    def push_action(self, action: np.ndarray) -> None:
        self.prev_action = np.clip(
            np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0
        )

    def build(
        self,
        state: RobotState,
        command: np.ndarray,
        *,
        gait: bool,
        dt: float = 1.0 / config.CONTROL_HZ,
    ) -> np.ndarray:
        if gait:
            self.phase = (
                self.phase + 2.0 * np.pi * config.GAIT_FREQ_HZ * dt
            ) % (2.0 * np.pi)
            phase = (float(np.sin(self.phase)), float(np.cos(self.phase)))
        else:
            phase = (0.0, 0.0)
        return build_obs(state, self.prev_action, command, phase, self.layout)
