"""双足机器人的正运动学与站立姿态求解。

只依赖 numpy，不依赖 MuJoCo，所以在没有安装 MuJoCo 的机器上也能跑测试。

坐标与链式关系（与 ``builder.py`` 生成的 MJCF 严格一致）：

    pelvis  --(hip_roll, hip_pitch)--  thigh  --(knee)--  shank  --(ankle)--  foot

髋部的两个关节锚在同一点（髋中心），在模型里同一个 body 上按 roll → pitch 顺序
复合；脚踝原点就是足部 body 的原点，足部几何相对脚踝还有 ``foot_center`` 偏移。
"""

from __future__ import annotations

import math

import numpy as np

from .config import RobotSpec, Vec3


def unit(v: Vec3) -> np.ndarray:
    """把向量归一化。"""
    a = np.asarray(v, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        raise ValueError(f"零向量无法作为关节轴: {v}")
    return a / n


def rot(axis: Vec3, angle: float) -> np.ndarray:
    """绕 ``axis`` 旋转 ``angle`` 弧度的旋转矩阵（Rodrigues 公式）。"""
    k = unit(axis)
    c, s = math.cos(angle), math.sin(angle)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) * c + s * kx + (1.0 - c) * np.outer(k, k)


def hip_offset(spec: RobotSpec, side: str) -> np.ndarray:
    """髋关节在骨盆坐标系里的位置。"""
    return np.array([0.0, spec.hip_y if side == "L" else -spec.hip_y, spec.hip_z])


def leg_fk(
    spec: RobotSpec, side: str, q: tuple[float, float, float, float] | list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """单腿正运动学。

    参数
    ----
    side : ``"L"`` 或 ``"R"``
    q    : ``(hip_roll, hip_pitch, knee, ankle)``，单位弧度

    返回
    ----
    ``(R, p)``：踝关节坐标系相对骨盆坐标系的旋转矩阵与位置。
    """
    if side not in ("L", "R"):
        raise ValueError(f"side 必须是 'L' 或 'R'，收到 {side!r}")
    roll, pitch, knee, ankle = q

    R = rot(spec.hip_roll.axis_for(side), roll) @ rot(spec.hip_pitch.axis, pitch)
    p = hip_offset(spec, side)

    p = p + R @ np.array([0.0, 0.0, -spec.thigh_len])
    R = R @ rot(spec.knee.axis, knee)

    p = p + R @ np.array([0.0, 0.0, -spec.shank_len])
    R = R @ rot(spec.ankle.axis, ankle)

    return R, p


def leg_q(spec: RobotSpec, q_by_name: dict[str, float], side: str) -> list[float]:
    """从"关节名 → 角度"字典里取出某一侧腿的四元组。"""
    return [
        q_by_name[f"{spec.hip_roll.name}_{side}"],
        q_by_name[f"{spec.hip_pitch.name}_{side}"],
        q_by_name[f"{spec.knee.name}_{side}"],
        q_by_name[f"{spec.ankle.name}_{side}"],
    ]


def foot_center(spec: RobotSpec, side: str, q: list[float]) -> np.ndarray:
    """足部几何中心在骨盆坐标系里的位置。"""
    R, p = leg_fk(spec, side, q)
    return p + R @ np.asarray(spec.foot_center, dtype=float)


def foot_corners(spec: RobotSpec, side: str, q: list[float]) -> np.ndarray:
    """足部 box 的 8 个顶点在骨盆坐标系里的位置，形状 ``(8, 3)``。"""
    R, p = leg_fk(spec, side, q)
    half = np.asarray(spec.foot_size, dtype=float) / 2.0
    center = p + R @ np.asarray(spec.foot_center, dtype=float)
    signs = np.array(
        [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=float,
    )
    return center + (signs * half) @ R.T


def foot_sole_z(spec: RobotSpec, side: str, q: list[float]) -> float:
    """足底最低点的高度（骨盆位于原点时）。"""
    return float(foot_corners(spec, side, q).min(axis=0)[2])


def solve_pelvis_z(spec: RobotSpec, stand: tuple[float, float, float] | None = None) -> float:
    """求让两脚足底恰好贴地 (z=0) 的骨盆高度。

    ``stand`` 为 ``(髋俯仰, 膝, 踝)``；默认用 ``spec.stand``。
    取两只脚 8 个顶点的最低点，因此对足部是倾斜的还是水平的都成立。
    """
    hip_pitch, knee, ankle = stand if stand is not None else spec.stand
    q = [0.0, hip_pitch, knee, ankle]
    lowest = min(foot_sole_z(spec, side, q) for side in ("L", "R"))
    return -lowest


def stand_qpos(spec: RobotSpec) -> list[float]:
    """标称站立姿态的完整 qpos（7 位自由关节 + 9 个关节角）。"""
    q = spec.stand_q()
    return [
        0.0,
        0.0,
        solve_pelvis_z(spec),
        1.0,
        0.0,
        0.0,
        0.0,
    ] + [q[name] for name in spec.joint_names]


def numeric_jacobian(
    spec: RobotSpec, side: str, q: list[float], eps: float = 1e-6
) -> np.ndarray:
    """踝关节位置的 3x4 数值雅可比（中心差分）。"""
    J = np.zeros((3, len(q)))
    for i in range(len(q)):
        dq = list(q)
        dq[i] += eps
        p_plus = leg_fk(spec, side, dq)[1]
        dq[i] -= 2.0 * eps
        p_minus = leg_fk(spec, side, dq)[1]
        J[:, i] = (p_plus - p_minus) / (2.0 * eps)
    return J
