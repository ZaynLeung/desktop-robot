"""三阶段奖励与终止判据。

奖励返回 ``(总奖励, 未加权分项)``：分项直接进 TensorBoard/info，
方便判断是"没学会"还是"奖励没调好"。

数值都是**起始值**，集中在 :mod:`dino_wbc.config` 的 ``STAGES[...].reward``，
调参时只改那里。
"""

from __future__ import annotations

import numpy as np

from . import config
from .model_io import JointLayout
from .state import RobotState

#: 势函数折扣（与 PPO 的 gamma 保持一致）。
POTENTIAL_GAMMA = 0.99

#: 高度奖励的容差（m）：偏离站姿高度 3 cm 时衰减到 1/e。
HEIGHT_SIGMA = 0.03
#: 抬脚高度的容差（m）。
LIFT_SIGMA = 0.010
#: 步态摆动足的期望离地高度容差（m）。
CLEARANCE_SIGMA = 0.010


# ----------------------------------------------------------------------
# 判据
# ----------------------------------------------------------------------
def is_fallen(state: RobotState) -> bool:
    """站立类任务的摔倒判据。"""
    return bool(
        state.base_pos[2] < config.FALL_BASE_Z
        or state.tilt_deg > config.FALL_TILT_DEG
    )


def is_standing(state: RobotState, layout: JointLayout) -> bool:
    """是否已回到"可视为站起来"的位形（standup 成功判据）。"""
    return bool(
        abs(state.base_pos[2] - layout.stand_base_z) <= config.SUCCESS_Z_TOL
        and state.tilt_deg <= config.SUCCESS_TILT_DEG
    )


def out_of_bounds(state: RobotState) -> bool:
    return bool(
        abs(state.base_pos[0]) > config.WORLD_HALF_EXTENT
        or abs(state.base_pos[1]) > config.WORLD_HALF_EXTENT
    )


def potential(state: RobotState, layout: JointLayout) -> float:
    """standup 的势函数：越高、越竖直、关节越接近站姿，值越大。"""
    height_term = 2.0 * (state.base_pos[2] / max(layout.stand_base_z, 1e-6))
    upright_term = 3.0 * float(np.clip(state.base_up[2], -1.0, 1.0))
    joint_term = -0.05 * float(np.linalg.norm(state.q))
    return height_term + upright_term + joint_term


# ----------------------------------------------------------------------
# 奖励分项
# ----------------------------------------------------------------------
def _common_terms(
    state: RobotState,
    action: np.ndarray,
    prev_action: np.ndarray,
    layout: JointLayout,
) -> dict[str, float]:
    height_error = (state.base_pos[2] - layout.stand_base_z) / HEIGHT_SIGMA
    return {
        "upright": state.uprightness,
        "height": float(np.exp(-height_error * height_error)),
        "alive": 1.0,
        "action_mag": float(np.mean(np.square(action))),
        "action_rate": float(np.mean(np.square(action - prev_action))),
        "base_vel": float(np.sum(np.square(state.base_lin_vel_b[:2]))),
        "yaw_rate": float(state.base_ang_vel_b[2] ** 2),
    }


def _lift_terms(
    state: RobotState,
    command: np.ndarray,
) -> dict[str, float]:
    target = 0 if command[config.CMD_TARGET_FOOT] < 0 else 1
    support = 1 - target
    wanted_h = float(command[config.CMD_LIFT_H])
    error = (state.foot_z[target] - wanted_h) / LIFT_SIGMA
    slip_error = state.foot_slip[support]
    return {
        "lift_height": float(np.exp(-error * error)),
        "support_contact": 1.0 if state.contact[support] else 0.0,
        "support_slip": float(np.exp(-20.0 * slip_error * slip_error)),
        "target_foot": float(target),
    }


def _walk_terms(state: RobotState, command: np.ndarray, phase: float) -> dict[str, float]:
    vx_err = state.base_lin_vel_b[0] - float(command[config.CMD_VX])
    vy_err = state.base_lin_vel_b[1] - float(command[config.CMD_VY])
    swing = 0 if np.sin(phase) >= 0.0 else 1
    clearance_error = (
        state.foot_z[swing] - config.WALK_FOOT_CLEARANCE
    ) / CLEARANCE_SIGMA
    contacts = int(state.contact.sum())
    return {
        "vel_track": float(np.exp(-4.0 * vx_err * vx_err - 6.0 * vy_err * vy_err)),
        "foot_clearance": float(np.exp(-clearance_error * clearance_error)),
        "contact_alt": 1.0 if contacts == 1 else (0.3 if contacts == 2 else 0.0),
        "heading": float(np.exp(-2.0 * state.base_ang_vel_b[2] ** 2)),
    }


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def compute_reward(
    stage: str,
    state: RobotState,
    action: np.ndarray,
    prev_action: np.ndarray,
    command: np.ndarray,
    layout: JointLayout,
    *,
    phase: float = 0.0,
    phi: float = 0.0,
    phi_prev: float | None = None,
    success: bool = False,
    fallen: bool = False,
) -> tuple[float, dict[str, float]]:
    """计算一步奖励并返回未加权分项。"""
    stage_cfg = config.STAGES.get(stage)
    if stage_cfg is None:
        raise ValueError(f"未知阶段 {stage!r}，可选 {tuple(config.STAGES)}")
    weights = stage_cfg.reward
    terms = _common_terms(state, action, prev_action, layout)

    if stage in ("lift", "walk"):
        terms.update(_lift_terms(state, command) if stage == "lift" else _walk_terms(state, command, phase))
    elif stage == "standup":
        if phi_prev is None:
            phi_prev = phi
        terms["progress"] = POTENTIAL_GAMMA * phi - phi_prev
        terms["time"] = 1.0
        terms["foot_contact"] = float(state.contact.sum()) / 2.0
    terms["success"] = 1.0 if success else 0.0
    terms["fall"] = 1.0 if fallen else 0.0

    total = 0.0
    for name, weight in weights.items():
        if name.startswith("_"):
            continue
        total += weight * terms.get(name, 0.0)
    return float(total), terms
