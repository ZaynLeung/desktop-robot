"""从 ``mjData`` 提取本体感知状态量。

训练端和运行端都调用 :func:`read_state`，因此"观测里到底有什么"只有一个实现。
所有量都在**世界系/躯干本体系**之间一次算清，观测层只负责缩放拼接。

坐标约定：x 前 / y 左 / z 上；躯干 body 的局部 z 轴是"背朝上"的方向。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .model_io import JointLayout

try:  # pragma: no cover
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError("dino_wbc.state 需要 mujoco") from exc


#: 世界"上"方向。
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

#: box 的 8 个顶点符号（用于算足底最低点）。
_CORNERS = np.array(
    [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
    dtype=np.float64,
)


@dataclass
class RobotState:
    """一帧的完整状态。数组都是 float64 / 固定形状。"""

    base_pos: np.ndarray  # (3,) 世界系
    base_quat: np.ndarray  # (4,) wxyz
    base_up: np.ndarray  # (3,) 躯干局部 z 轴的世界方向
    base_lin_vel_b: np.ndarray  # (3,) 本体系线速度
    base_ang_vel_b: np.ndarray  # (3,) 本体系角速度
    gravity_b: np.ndarray  # (3,) 本体系中的重力方向（站立时 ≈ (0,0,-1)）
    q: np.ndarray  # (N,) 站姿相对关节角
    qd: np.ndarray  # (N,)
    foot_pos: np.ndarray  # (2, 3) 足部几何中心，世界系
    foot_z: np.ndarray  # (2,) 足底最低点离地高度
    foot_slip: np.ndarray  # (2,) 足部水平速度模长
    contact: np.ndarray  # (2,) bool
    contact_force: np.ndarray  # (2,) 法向力 N
    #: 躯干/头/臂/大腿/小腿/膝接触地板（**不含足和尾巴**）。
    body_contact: bool
    #: 尾巴接触地板（允许，但会被轻微惩罚）。
    tail_contact: bool
    tilt_rad: float  # 相对标称站姿的倾角（站姿 = 0）
    time: float

    @property
    def tilt_deg(self) -> float:
        return float(np.degrees(self.tilt_rad))

    @property
    def uprightness(self) -> float:
        """1 = 完全竖直，0 = 侧躺，-1 = 倒立。"""
        return float(np.clip(0.5 * (1.0 + self.base_up[2]), -1.0, 1.0))


def base_z_stand(layout: JointLayout) -> float:
    return float(layout.stand_base_z)


def _geom_lowest_z(model, data, geom_id: int) -> float:
    """box 几何的最低顶点世界 z。"""
    half = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    mat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    world = pos + (_CORNERS * half) @ mat.T
    return float(world[:, 2].min())


def _foot_force(model, data, geom_id: int) -> float:
    """该 geom 参与的所有接触的法向力之和。"""
    total = 0.0
    force = np.zeros(6, dtype=np.float64)
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.geom1 != geom_id and contact.geom2 != geom_id:
            continue
        mujoco.mj_contactForce(model, data, index, force)
        total += float(force[0])
    return total


def _floor_contacts(model, data, layout: JointLayout) -> tuple[bool, bool]:
    """返回 ``(body_contact, tail_contact)``：地板与身体/尾巴的接触。

    足部接触由 :func:`read_state` 单独统计；这里只看"非足"的接触。
    尾巴单独区分：T-rex 尾巴作为配重/支撑是设计的一部分，允许接触，
    只给轻微惩罚；躯干/头/臂/腿着地则视为摔倒并终止。
    """
    floor = int(layout.floor_geom_id)
    if floor < 0:
        return False, False
    foot_ids = {gid for ids in layout.foot_geom_ids.values() for gid in ids}
    body = False
    tail = False
    force = np.zeros(6, dtype=np.float64)
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.geom1 != floor and contact.geom2 != floor:
            continue
        other = contact.geom2 if contact.geom1 == floor else contact.geom1
        if other in foot_ids:
            continue
        mujoco.mj_contactForce(model, data, index, force)
        if float(force[0]) <= config.CONTACT_FORCE_EPS:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
        if name.startswith("tail"):
            tail = True
        else:
            body = True
    return body, tail


def read_state(model, data, layout: JointLayout) -> RobotState:
    """把 ``mjData`` 压成一份 :class:`RobotState`。"""
    rot = np.asarray(data.xmat[layout.torso_body_id], dtype=np.float64).reshape(3, 3)
    up = rot[:, 2].copy()
    # 注意：标称站姿本身让躯干抬头 20°，所以「倾角」定义为**相对标称站姿**的
    # 偏差（站姿下 = 0），而不是相对世界 z 轴——否则站立时就会误判 20°。
    tilt = float(np.arccos(np.clip(float(up @ layout.stand_up), -1.0, 1.0)))

    qvel = np.asarray(data.qvel, dtype=np.float64)
    lin_vel_b = rot.T @ qvel[0:3]
    ang_vel_b = qvel[3:6].copy()

    q = np.asarray(data.qpos, dtype=np.float64)[layout.qposadr].copy()
    qd = qvel[layout.dofadr].copy()

    foot_pos = np.zeros((2, 3), dtype=np.float64)
    foot_z = np.zeros(2, dtype=np.float64)
    foot_slip = np.zeros(2, dtype=np.float64)
    contact_force = np.zeros(2, dtype=np.float64)
    vel6 = np.zeros(6, dtype=np.float64)

    for index, side in enumerate(("L", "R")):
        geoms = layout.foot_geom_ids[side]
        foot_pos[index] = np.mean(
            [np.asarray(data.geom_xpos[gid], dtype=np.float64) for gid in geoms], axis=0
        )
        foot_z[index] = min(_geom_lowest_z(model, data, gid) for gid in geoms)

        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            layout.foot_body_ids[side],
            vel6,
            0,  # 世界系
        )
        foot_slip[index] = float(np.linalg.norm(vel6[3:5]))

        contact_force[index] = sum(_foot_force(model, data, gid) for gid in geoms)

    body_contact, tail_contact = _floor_contacts(model, data, layout)

    return RobotState(
        base_pos=np.asarray(data.qpos[0:3], dtype=np.float64).copy(),
        base_quat=np.asarray(data.qpos[3:7], dtype=np.float64).copy(),
        base_up=up,
        base_lin_vel_b=lin_vel_b,
        base_ang_vel_b=ang_vel_b,
        gravity_b=rot.T @ (-WORLD_UP),
        q=q,
        qd=qd,
        foot_pos=foot_pos,
        foot_z=foot_z,
        foot_slip=foot_slip,
        contact=contact_force > config.CONTACT_FORCE_EPS,
        contact_force=contact_force,
        body_contact=body_contact,
        tail_contact=tail_contact,
        tilt_rad=tilt,
        time=float(data.time),
    )


# ----------------------------------------------------------------------
# 复位工具（训练/运行共用）
# ----------------------------------------------------------------------
def reset_to_stand(model, data, layout: JointLayout) -> None:
    """按 ``stand`` 关键帧复位（关节角与 ctrl 全 0 = 站姿）。"""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, layout.keyframe_name)
    if key_id < 0:  # pragma: no cover - build_layout 已校验
        raise RuntimeError(f"模型里找不到关键帧 {layout.keyframe_name!r}")
    mujoco.mj_resetDataKeyframe(model, data, key_id)


def set_joint_angles(data, layout: JointLayout, angles: dict[str, float]) -> None:
    for name, value in angles.items():
        data.qpos[layout.qposadr[layout.names.index(name)]] = float(value)


def set_base_pose(data, pos, quat) -> None:
    data.qpos[0:3] = np.asarray(pos, dtype=np.float64)
    data.qpos[3:7] = np.asarray(quat, dtype=np.float64)


def quat_from_roll_deg(roll_deg: float) -> np.ndarray:
    """绕世界 x 轴滚转的四元数（wxyz）。0=立姿，90=侧躺，180=仰卧。"""
    angle = np.radians(float(roll_deg)) * 0.5
    return np.array([np.cos(angle), np.sin(angle), 0.0, 0.0], dtype=np.float64)


def settle(
    model, data, layout: JointLayout, steps: int = config.SETTLE_STEPS
) -> None:
    """保持当前关节角、让接触先稳定下来。

    期间 ``ctrl`` 设成当前关节角，避免位置伺服把腿瞬间拉回站姿；
    仰卧/侧躺这类初始姿态尤其需要。
    """
    data.ctrl[:] = np.asarray(data.qpos, dtype=np.float64)[layout.qposadr]
    for _ in range(int(steps)):
        mujoco.mj_step(model, data)


def apply_standup_pose(model, data, layout: JointLayout, pose) -> None:
    """把机器人摆到 standup 课程的某个起始姿态，并让物理落到地面。

    训练环境的 reset 与运行端切换 standup 阶段共用这一个实现，保证
    "训练时看到的起始分布"和"演示时看到的一样"。
    """
    mujoco.mj_resetData(model, data)
    data.time = 0.0
    set_base_pose(
        data, (0.0, 0.0, pose.base_z_hint), quat_from_roll_deg(pose.roll_deg)
    )
    set_joint_angles(data, layout, pose.joints)
    mujoco.mj_forward(model, data)
    settle(model, data, layout, steps=max(config.SETTLE_STEPS, 40))
