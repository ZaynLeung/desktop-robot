"""三阶段奖励与终止判据。

奖励返回 ``(总奖励, 未加权分项)``：分项直接进 TensorBoard/info，
方便判断是"没学会"还是"奖励没调好"。

诊断修复（2026-11 v2）：
* ``upright`` 改成**相对标称站姿的倾角**指数项（站姿=1），不再是原始 ``up_z``；
* 高度容差从 3 cm 收紧到 2 cm，新增 ``low_posture`` 惩罚，堵住"蹲着/趴着"的局部最优；
* 抬脚项只在 **stance gate** 成立时给（站在站姿高度附近、不歪、躯干不着地），
  并新增稠密线性项 ``lift_frac`` 与 ``target_ground`` 惩罚，解决 4 cm 时梯度消失；
* 躯干/头/臂/腿接触地板会终止（``body_contact``）；v2.2 起尾巴触地也会终止；
* v2.2 新增 ``heading``（绝对朝向）与 ``position``（绝对位置）奖励，
  让 stand 变成"原地 + 对朝向"的静态基座模型。
"""

from __future__ import annotations

import numpy as np

from . import config
from .model_io import JointLayout
from .state import RobotState, wrap_to_pi

#: 势函数折扣（与 PPO 的 gamma 保持一致）。
POTENTIAL_GAMMA = 0.99

#: 高度奖励的容差（m）：偏离站姿高度 2 cm 时衰减到 1/e。
HEIGHT_SIGMA = 0.02
#: 姿态奖励的容差（rad）：相对站姿倾角 0.2 rad（约 11.5°）时衰减到 1/e。
UPRIGHT_SIGMA = 0.2
#: 绝对朝向误差的容差（rad）：0.15 rad ≈ 8.6°。
HEADING_SIGMA = 0.15
#: 位置误差的容差（m）。
POSITION_SIGMA = 0.03
#: 抬脚高度的容差（m）。
LIFT_SIGMA = 0.015
#: 步态摆动足的期望离地高度容差（m）。
CLEARANCE_SIGMA = 0.010
#: "重心在支撑脚上方"的容差（m）。
COM_SIGMA = 0.05


# ----------------------------------------------------------------------
# 判据
# ----------------------------------------------------------------------
def is_fallen(state: RobotState) -> bool:
    """walk 用的通用摔倒判据（stand/lift 用更严格的阶段门限，见 env）。"""
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


def lift_stance_ok(state: RobotState, layout: JointLayout) -> bool:
    """抬脚奖励/课程的姿态门限。

    必须"站在站姿高度附近 + 基本竖直 + 躯干没有撑地"，抬脚项才有意义；
    否则策略会像 v1 那样趴在/蹲在地上蹭抬脚奖励。
    """
    return bool(
        (not state.body_contact)
        and (not state.tail_contact)
        and state.base_pos[2] > layout.stand_base_z - config.LIFT_STANCE_Z_DELTA
        and state.tilt_deg < config.LIFT_STANCE_TILT_DEG
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
    command: np.ndarray,
    target_xy: np.ndarray,
) -> dict[str, float]:
    height_error = (state.base_pos[2] - layout.stand_base_z) / HEIGHT_SIGMA
    tilt_error = state.tilt_rad / UPRIGHT_SIGMA
    heading_error = wrap_to_pi(float(command[config.CMD_YAW]) - state.yaw_rad)
    position_error = np.asarray(state.base_pos[:2]) - np.asarray(target_xy)
    low_posture = max(
        0.0,
        (layout.stand_base_z - config.LIFT_STANCE_Z_DELTA - state.base_pos[2])
        / config.LIFT_STANCE_Z_DELTA,
    )
    return {
        "alive": 1.0,
        "height": float(np.exp(-height_error * height_error)),
        "upright": float(np.exp(-tilt_error * tilt_error)),
        "heading": float(np.exp(-((heading_error / HEADING_SIGMA) ** 2))),
        "position": float(
            np.exp(-float(np.dot(position_error, position_error)) / POSITION_SIGMA**2)
        ),
        "base_vel": float(np.sum(np.square(state.base_lin_vel_b[:2]))),
        "yaw_rate": float(state.base_ang_vel_b[2] ** 2),
        "foot_contact": 1.0 if bool(state.contact.all()) else 0.0,
        "foot_slip": float(np.mean(np.exp(-20.0 * np.square(state.foot_slip)))),
        "action_mag": float(np.mean(np.square(action))),
        "action_rate": float(np.mean(np.square(action - prev_action))),
        "joint_posture": float(np.mean(np.square(state.q))),
        "low_posture": float(low_posture),
        "body_contact": 1.0 if state.body_contact else 0.0,
        "tail_contact": 1.0 if state.tail_contact else 0.0,
    }


def _lift_terms(
    state: RobotState,
    command: np.ndarray,
    *,
    gate: bool,
) -> dict[str, float]:
    """抬脚专属项；``gate`` 为 False 时全部返回 0（不给"趴着抬脚"任何奖励）。"""
    if not gate:
        return {
            "lift_height": 0.0,
            "lift_frac": 0.0,
            "target_ground": 0.0,
            "support_contact": 0.0,
            "support_slip": 0.0,
            "com_over_support": 0.0,
        }
    target = 0 if command[config.CMD_TARGET_FOOT] < 0 else 1
    support = 1 - target
    wanted = max(float(command[config.CMD_LIFT_H]), 1e-3)
    foot_z = float(state.foot_z[target])
    error = (foot_z - wanted) / LIFT_SIGMA
    support_slip = float(state.foot_slip[support])
    delta_xy = state.base_pos[:2] - state.foot_pos[support][:2]
    com_error = float(np.dot(delta_xy, delta_xy)) / (COM_SIGMA * COM_SIGMA)
    return {
        "lift_height": float(np.exp(-error * error)),
        "lift_frac": float(np.clip(foot_z / wanted, 0.0, 1.0)),
        "target_ground": 1.0 if bool(state.contact[target]) else 0.0,
        "support_contact": 1.0 if bool(state.contact[support]) else 0.0,
        "support_slip": float(np.exp(-20.0 * support_slip * support_slip)),
        "com_over_support": float(np.exp(-com_error)),
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
    target_xy: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """计算一步奖励并返回未加权分项。"""
    stage_cfg = config.STAGES.get(stage)
    if stage_cfg is None:
        raise ValueError(f"未知阶段 {stage!r}，可选 {tuple(config.STAGES)}")
    weights = stage_cfg.reward
    if target_xy is None:
        target_xy = np.zeros(2, dtype=np.float64)
    terms = _common_terms(state, action, prev_action, layout, command, target_xy)

    if stage == "lift":
        gate = (not fallen) and lift_stance_ok(state, layout)
        terms.update(_lift_terms(state, command, gate=gate))
    elif stage == "walk":
        terms.update(_walk_terms(state, command, phase))
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
