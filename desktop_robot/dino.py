"""桌面恐龙（T-rex 造型）机器人的参数化 MuJoCo 模型生成器。

参数源在 :class:`DinoSpec` / :data:`DEFAULT_DINO_SPEC`，模型文件
``models/dino.xml`` 与 ``models/dino_scene.xml`` 是它的确定性生成物。

拓扑（11 个驱动关节，全部是矢状面俯仰）：::

    torso（浮动基座，freejoint: root）
    ├─ head          neck_pitch            ① 颈
    ├─ arm_L         shoulder_pitch_L      ② 左肩
    ├─ arm_R         shoulder_pitch_R      ③ 右肩
    ├─ thigh_L       hip_pitch_L           ④ 左髋
    │  └─ shank_L    knee_L                ⑥ 左膝
    │     └─ foot_L  ankle_L               ⑧ 左踝
    ├─ thigh_R       hip_pitch_R           ⑤ 右髋
    │  └─ shank_R    knee_R                ⑦ 右膝
    │     └─ foot_R  ankle_R               ⑨ 右踝
    └─ tail_1        tail1_pitch           ⑩ 尾部第 1 节
       └─ tail_2     tail2_pitch           ⑪ 尾部第 2 节

**每个连杆的长度都由对应 ``*_size`` 的相应分量决定**，改长度后关节/几何偏移
会自动跟着重算，不需要手工同步：

============================  ==========================================
连杆                          长度来源
============================  ==========================================
``thigh``                      ``thigh_size[2]``（膝关节在 z = −大腿长）
``shank``                      ``shank_size[2]``（踝关节在 z = −小腿长）
``arm``                        ``arm_size[2]``（臂几何中心在 z = −臂长/2）
``tail_1`` / ``tail_2``        ``tail1_size[0]`` / ``tail2_size[0]``
``foot`` / ``toe``             足 box 中心 ``foot_center``；前趾贴住足前端与足底
============================  ==========================================

躯干上的锚点（``neck_pos`` / ``shoulder_pos`` / ``hip_pos`` / ``tail1_pos``）
是显式字段：想改「关节在躯干上的位置」直接改这些偏移即可。

约定与仓库里的双足模型一致：x 前 / y 左 / z 上；俯仰轴统一 ``(0, -1, 0)``，
正角 = 子连杆向前摆（对朝后的尾巴，正角 = 尾尖下压）。标称站姿里躯干抬头
``base_pitch``，踝关节按「基座俯仰 + 髋 + 膝 + 踝 = 0」反解，保证足底水平。

python -m desktop_robot build-dino
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np

from .builder import MODELS_DIR, build_scene_for, fnum, half, vec
from .config import JointSpec, Vec3, joint_name
from .verifier import Report

#: 俯仰轴。与双足模型一致：正角 = 子连杆向前摆。
PITCH: Vec3 = (0.0, -1.0, 0.0)

#: 生成 XML 的表头（告诉读者这是生成物、去哪里改参数）。
_HEADER = """<?xml version="1.0" encoding="utf-8"?>
<!-- 由 desktop_robot.dino 自动生成，请勿手工修改。                  -->
<!-- 参数源：desktop_robot/dino.py 的 DEFAULT_DINO_SPEC。            -->
<!-- 重新生成：python -m desktop_robot build-dino                    -->
"""

_GREEN = "0.30 0.55 0.25 1"
_GREEN_LIMB = "0.24 0.47 0.20 1"
_TOE = "0.12 0.12 0.13 1"
_GREY = "0.55 0.56 0.58 1"

#: 关节轴承外壳（只做视觉，不参与质量和碰撞）：关节名 → (半径, 半长)。
#: 参考图里每个铰链都有一圈灰色轴承，加上它渲染出来才像那台机器人。
_CAPS: dict[str, tuple[float, float]] = {
    "neck_pitch": (0.016, 0.024),
    "shoulder_pitch": (0.014, 0.019),
    "hip_pitch": (0.016, 0.024),
    "knee": (0.014, 0.021),
    "ankle": (0.012, 0.017),
    "tail1_pitch": (0.012, 0.017),
    "tail2_pitch": (0.011, 0.015),
}


def _joint(
    name: str,
    rng: tuple[float, float],
    kp: float = 25.0,
    kv: float = 1.0,
    forcerange: float = 2.5,
) -> JointSpec:
    """构造一个单轴俯仰关节（轴固定为 ``(0, -1, 0)``）。"""
    return JointSpec(
        name=name, axis=PITCH, range=rng, kp=kp, kv=kv, forcerange=forcerange
    )


@dataclass(frozen=True)
class DinoSpec:
    """整机参数（全尺寸、米、弧度）。

    与 :class:`~desktop_robot.config.RobotSpec` 一样，**这里写的是 box 的全尺寸**，
    写进 MJCF 时再除以 2。改尺寸后重新 ``build-dino`` 即可，生成物与校验都跟着变。
    """

    name: str = "desktop_dino"

    # ---- 躯干：直接做浮动基座（图中没有腰关节）----
    torso_size: Vec3 = (0.30, 0.095, 0.085)

    # ---- 头：颈关节位置相对躯干中心；头几何中心相对颈关节 ----
    head_size: Vec3 = (0.100, 0.072, 0.055)
    neck_pos: Vec3 = (0.055, 0.0, 0.050)
    head_offset: Vec3 = (0.032, 0.0, 0.022)

    # ---- 臂：单段，无肘/手 ----
    arm_size: Vec3 = (0.045, 0.032, 0.075)
    shoulder_pos: Vec3 = (0.048, 0.058, 0.018)  # y 为左肩；右肩取反

    # ---- 腿 ----
    thigh_size: Vec3 = (0.048, 0.040, 0.090)
    shank_size: Vec3 = (0.044, 0.036, 0.075)
    foot_size: Vec3 = (0.085, 0.055, 0.032)
    toe_size: Vec3 = (0.028, 0.018, 0.026)
    hip_pos: Vec3 = (-0.085, 0.045, -0.030)
    foot_center: Vec3 = (0.012, 0.0, -0.016)  # 足 box 中心相对踝关节
    toe_y: float = 0.013  # 两个前趾中心相对足中线（旋转方向镜像 ±）
    toe_overlap: float = 0.006  # 前趾与足 box 在 x 方向的重叠量

    # ---- 尾巴 ----
    tail1_size: Vec3 = (0.085, 0.042, 0.036)
    tail2_size: Vec3 = (0.080, 0.036, 0.030)
    tail1_pos: Vec3 = (-0.068, 0.0, -0.020)

    # ---- 关节 ----
    neck: JointSpec = field(default_factory=lambda: _joint("neck_pitch", (-0.70, 0.70)))
    shoulder: JointSpec = field(
        default_factory=lambda: _joint("shoulder_pitch", (-1.20, 1.20))
    )
    hip: JointSpec = field(default_factory=lambda: _joint("hip_pitch", (-1.20, 1.20)))
    knee: JointSpec = field(default_factory=lambda: _joint("knee", (-1.60, 0.05)))
    ankle: JointSpec = field(
        default_factory=lambda: _joint("ankle", (-0.60, 0.80), forcerange=2.0)
    )
    tail1: JointSpec = field(default_factory=lambda: _joint("tail1_pitch", (-0.80, 0.80)))
    tail2: JointSpec = field(default_factory=lambda: _joint("tail2_pitch", (-0.80, 0.80)))

    # ---- 标称站姿（弧度）----
    # 躯干抬头 base_pitch；踝角由「足底水平」判据反解，保证脚掌贴地。
    base_pitch: float = math.radians(20.0)
    stand_neck: float = -0.25
    stand_shoulder: float = -0.15
    stand_hip: float = -0.30
    stand_knee: float = -0.10
    stand_tail1: float = 0.20
    stand_tail2: float = 0.12

    density: float = 800.0

    # ------------------------------------------------------------------
    # 长度（全部由尺寸分量派生，改尺寸后自动生效）
    # ------------------------------------------------------------------
    @property
    def thigh_len(self) -> float:
        return self.thigh_size[2]

    @property
    def shank_len(self) -> float:
        return self.shank_size[2]

    @property
    def arm_len(self) -> float:
        return self.arm_size[2]

    @property
    def tail1_len(self) -> float:
        return self.tail1_size[0]

    @property
    def tail2_len(self) -> float:
        return self.tail2_size[0]

    @property
    def toe_center(self) -> Vec3:
        """前趾几何中心相对踝关节（y 由 ``±toe_y`` 给出，这里写 0）。"""
        x = (
            self.foot_center[0]
            + self.foot_size[0] / 2.0
            + self.toe_size[0] / 2.0
            - self.toe_overlap
        )
        z = self.foot_center[2] + (self.toe_size[2] - self.foot_size[2]) / 2.0
        return (x, 0.0, z)

    @property
    def stand_ankle(self) -> float:
        """让足底保持水平（且与基座平行）的踝角。"""
        return -(self.base_pitch + self.stand_hip + self.stand_knee)

    # ------------------------------------------------------------------
    # 规模
    # ------------------------------------------------------------------
    def actuated(self) -> list[tuple[JointSpec, str | None]]:
        """按 MuJoCo qpos 顺序返回 (关节参数, 侧别)。

        顺序 = body 深度优先遍历 + body 内关节定义顺序：
        颈 → 左肩 → 右肩 → 左腿 3 个 → 右腿 3 个 → 尾 2 个。
        """
        return [
            (self.neck, None),
            (self.shoulder, "L"),
            (self.shoulder, "R"),
            (self.hip, "L"),
            (self.knee, "L"),
            (self.ankle, "L"),
            (self.hip, "R"),
            (self.knee, "R"),
            (self.ankle, "R"),
            (self.tail1, None),
            (self.tail2, None),
        ]

    @property
    def joint_names(self) -> list[str]:
        return [joint_name(joint, side) for joint, side in self.actuated()]

    @property
    def nq(self) -> int:
        return 7 + len(self.actuated())

    @property
    def nv(self) -> int:
        return 6 + len(self.actuated())

    @property
    def nu(self) -> int:
        return len(self.actuated())

    @property
    def n_bodies(self) -> int:
        """躯干 1 + 头 1 + 臂 2 + 每腿 3 ×2 + 尾 2。"""
        return 1 + 1 + 2 + 2 * 3 + 2

    @property
    def body_geoms(self) -> tuple[tuple[str, Vec3, float], ...]:
        """全部物理几何的 (名称, 全尺寸, 密度)，用于质量估算与校验。"""
        geoms: list[tuple[str, Vec3, float]] = [
            ("torso", self.torso_size, self.density),
            ("head", self.head_size, self.density),
            ("arm", self.arm_size, self.density),
            ("arm", self.arm_size, self.density),
            ("thigh", self.thigh_size, self.density),
            ("thigh", self.thigh_size, self.density),
            ("shank", self.shank_size, self.density),
            ("shank", self.shank_size, self.density),
            ("foot", self.foot_size, self.density),
            ("foot", self.foot_size, self.density),
            ("toe", self.toe_size, self.density),
            ("toe", self.toe_size, self.density),
            ("toe", self.toe_size, self.density),
            ("toe", self.toe_size, self.density),
            ("tail_1", self.tail1_size, self.density),
            ("tail_2", self.tail2_size, self.density),
        ]
        return tuple(geoms)

    def total_mass(self) -> float:
        """按 box 体积 × 密度估算总质量（与 MuJoCo 的算法一致）。"""
        return sum(
            size[0] * size[1] * size[2] * density for _, size, density in self.body_geoms
        )

    # ------------------------------------------------------------------
    # 站姿
    # ------------------------------------------------------------------
    def stand_q(self) -> dict[str, float]:
        """标称站姿的「关节名 → 角度」字典。"""
        q: dict[str, float] = {"neck_pitch": self.stand_neck}
        for side in ("L", "R"):
            q[f"shoulder_pitch_{side}"] = self.stand_shoulder
            q[f"hip_pitch_{side}"] = self.stand_hip
            q[f"knee_{side}"] = self.stand_knee
            q[f"ankle_{side}"] = self.stand_ankle
        q["tail1_pitch"] = self.stand_tail1
        q["tail2_pitch"] = self.stand_tail2
        return q


# ----------------------------------------------------------------------
# 默认机型：约 0.33 m 高、约 2.3 kg 的桌面恐龙
# ----------------------------------------------------------------------
DEFAULT_DINO_SPEC = DinoSpec()


# ----------------------------------------------------------------------
# 运动学：站立高度与足底贴地
# ----------------------------------------------------------------------
def _rot_pitch(angle: float) -> np.ndarray:
    """绕 ``(0, -1, 0)`` 转 ``angle`` 弧度的旋转矩阵（正角 = 前方抬头）。"""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def _corners(center, size: Vec3, rot: np.ndarray) -> np.ndarray:
    """box 的 8 个顶点：``center`` 在父坐标系，``rot`` 是父→box 的旋转。"""
    half_size = np.asarray(size, dtype=float) / 2.0
    signs = np.array(
        [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=float,
    )
    return np.asarray(center, dtype=float) + (signs * half_size) @ rot.T


def leg_boxes(
    spec: DinoSpec, side: str, hip: float, knee: float, ankle: float
) -> list[tuple[np.ndarray, Vec3, np.ndarray]]:
    """单腿 FK：返回足 box 与两个前趾的 ``(中心, 尺寸, 旋转)``，均在躯干坐标系。"""
    sign = 1.0 if side == "L" else -1.0
    rot = _rot_pitch(hip)
    p = np.array(
        (spec.hip_pos[0], sign * spec.hip_pos[1], spec.hip_pos[2]), dtype=float
    )

    p = p + rot @ np.array((0.0, 0.0, -spec.thigh_len))
    rot = rot @ _rot_pitch(knee)

    p = p + rot @ np.array((0.0, 0.0, -spec.shank_len))
    rot = rot @ _rot_pitch(ankle)

    foot_center = p + rot @ np.asarray(spec.foot_center, dtype=float)
    toe_center = np.asarray(spec.toe_center, dtype=float)
    boxes: list[tuple[np.ndarray, Vec3, np.ndarray]] = [
        (foot_center, spec.foot_size, rot)
    ]
    for toe_sign in (-1.0, 1.0):
        offset = np.array((toe_center[0], toe_sign * spec.toe_y, toe_center[2]))
        boxes.append((p + rot @ offset, spec.toe_size, rot))
    return boxes


def foot_contact_corners(
    spec: DinoSpec, side: str, hip: float, knee: float, ankle: float
) -> np.ndarray:
    """足 box + 两个前趾的全部顶点（躯干坐标系），形状 ``(24, 3)``。"""
    corners = [
        _corners(center, size, rot)
        for center, size, rot in leg_boxes(spec, side, hip, knee, ankle)
    ]
    return np.vstack(corners)


def sole_lowest_z(
    spec: DinoSpec,
    side: str,
    hip: float | None = None,
    knee: float | None = None,
    ankle: float | None = None,
) -> float:
    """足部最低点相对基座原点的高度（已含基座俯仰，未含基座高度）。"""
    hip = spec.stand_hip if hip is None else hip
    knee = spec.stand_knee if knee is None else knee
    ankle = spec.stand_ankle if ankle is None else ankle
    corners = foot_contact_corners(spec, side, hip, knee, ankle)
    world = corners @ _rot_pitch(spec.base_pitch).T
    return float(world[:, 2].min())


def solve_base_z(spec: DinoSpec) -> float:
    """让两只脚最低点恰好落在 z=0 的基座高度。"""
    lowest = min(sole_lowest_z(spec, side) for side in ("L", "R"))
    return -lowest


def base_quat(spec: DinoSpec) -> tuple[float, float, float, float]:
    """标称站姿的基座四元数 ``(qw, qx, qy, qz)``：绕 ``(0, -1, 0)`` 抬头。"""
    half_angle = spec.base_pitch / 2.0
    return (math.cos(half_angle), 0.0, -math.sin(half_angle), 0.0)


def stand_qpos(spec: DinoSpec) -> list[float]:
    """``stand`` 关键帧的完整 qpos（7 位自由关节 + 11 个关节角）。"""
    q = spec.stand_q()
    return [
        0.0,
        0.0,
        solve_base_z(spec),
        *base_quat(spec),
    ] + [q[name] for name in spec.joint_names]


def head_top_z(spec: DinoSpec) -> float:
    """标称站姿下头顶（最高点）离地高度。"""
    rot = _rot_pitch(spec.stand_neck)
    center = np.asarray(spec.neck_pos, dtype=float) + rot @ np.asarray(
        spec.head_offset, dtype=float
    )
    corners = _corners(center, spec.head_size, rot)
    world = corners @ _rot_pitch(spec.base_pitch).T
    return solve_base_z(spec) + float(world[:, 2].max())


# ----------------------------------------------------------------------
# MJCF 生成
# ----------------------------------------------------------------------
def _geom_attributes(
    name: str,
    size: Vec3,
    pos: Vec3,
    spec: DinoSpec,
    rgba: str,
    contype: int,
    conaffinity: int,
    friction: str = "0.9 0.02 0.001",
) -> dict[str, str]:
    """一个 box geom 的属性表（碰撞掩码规则与双足模型一致）。"""
    return {
        "name": name,
        "type": "box",
        "size": vec(half(size)),
        "pos": vec(pos),
        "density": fnum(spec.density),
        "rgba": rgba,
        "friction": friction,
        "contype": str(contype),
        "conaffinity": str(conaffinity),
    }


def _joint_attributes(joint: JointSpec, side: str | None) -> dict[str, str]:
    name = joint_name(joint, side)
    return {
        "name": name,
        "type": "hinge",
        "axis": vec(joint.axis_for(side)),
        "range": vec(joint.range),
        "limited": "true",
        "armature": fnum(joint.armature),
        "damping": fnum(joint.damping),
        "frictionloss": fnum(joint.frictionloss),
        "pos": vec((0.0, 0.0, 0.0)),
    }


def _actuator_attributes(joint: JointSpec, side: str | None) -> dict[str, str]:
    name = joint_name(joint, side)
    return {
        "name": f"act_{name}",
        "joint": name,
        "kp": fnum(joint.kp),
        "kv": fnum(joint.kv),
        "forcerange": f"{-joint.forcerange:g} {joint.forcerange:g}",
        "forcelimited": "true",
        "inheritrange": "1",
        "ctrllimited": "true",
    }


def _cap_attributes(joint: JointSpec, side: str | None) -> dict[str, str] | None:
    """关节轴承外壳（灰色圆柱，只做视觉）：没有登记尺寸的关节返回 ``None``。"""
    dims = _CAPS.get(joint.name)
    if dims is None:
        return None
    radius, half_len = dims
    return {
        "name": f"cap_{joint_name(joint, side)}",
        "type": "cylinder",
        "size": f"{fnum(radius)} {fnum(half_len)}",
        "pos": "0 0 0",
        # 圆柱默认沿 z 轴，转 90° 到 y 轴，和俯仰轴一致
        "quat": "0.7071068 0.7071068 0 0",
        "rgba": _GREY,
        "density": "0",
        "contype": "0",
        "conaffinity": "0",
    }


def _add_joint(body: ET.Element, joint: JointSpec, side: str | None) -> None:
    """在 body 原点挂一个铰链，再加上它的灰色轴承外壳（如果有）。"""
    ET.SubElement(body, "joint", _joint_attributes(joint, side))
    cap = _cap_attributes(joint, side)
    if cap is not None:
        ET.SubElement(body, "geom", cap)


def _add_leg(torso: ET.Element, spec: DinoSpec, side: str, sign: float) -> None:
    """挂一条腿：thigh → shank → foot（foot 上再挂两个前趾）。"""
    hip = (spec.hip_pos[0], sign * spec.hip_pos[1], spec.hip_pos[2])
    thigh = ET.SubElement(torso, "body", {"name": f"thigh_{side}", "pos": vec(hip)})
    _add_joint(thigh, spec.hip, side)
    ET.SubElement(
        thigh,
        "geom",
        _geom_attributes(
            f"thigh_geom_{side}",
            spec.thigh_size,
            (0.0, 0.0, -spec.thigh_len / 2.0),
            spec,
            _GREEN_LIMB,
            contype=2,
            conaffinity=1,
        ),
    )

    shank = ET.SubElement(
        thigh, "body", {"name": f"shank_{side}", "pos": vec((0.0, 0.0, -spec.thigh_len))}
    )
    _add_joint(shank, spec.knee, side)
    ET.SubElement(
        shank,
        "geom",
        _geom_attributes(
            f"shank_geom_{side}",
            spec.shank_size,
            (0.0, 0.0, -spec.shank_len / 2.0),
            spec,
            _GREEN_LIMB,
            contype=2,
            conaffinity=1,
        ),
    )

    foot = ET.SubElement(
        shank, "body", {"name": f"foot_{side}", "pos": vec((0.0, 0.0, -spec.shank_len))}
    )
    _add_joint(foot, spec.ankle, side)
    ET.SubElement(
        foot,
        "geom",
        _geom_attributes(
            f"foot_geom_{side}",
            spec.foot_size,
            spec.foot_center,
            spec,
            _GREEN_LIMB,
            contype=1,
            conaffinity=1,
            friction="1.2 0.02 0.001",
        ),
    )
    toe_x, _, toe_z = spec.toe_center
    for index, toe_sign in enumerate((-1.0, 1.0), start=1):
        ET.SubElement(
            foot,
            "geom",
            _geom_attributes(
                f"toe_geom_{side}{index}",
                spec.toe_size,
                (toe_x, toe_sign * spec.toe_y, toe_z),
                spec,
                _TOE,
                contype=1,
                conaffinity=1,
                friction="1.2 0.02 0.001",
            ),
        )


def build_dino(spec: DinoSpec = DEFAULT_DINO_SPEC) -> ET.Element:
    """构造机器人本体的 MJCF 根元素（不含地面/灯光）。"""
    root = ET.Element("mujoco", {"model": spec.name})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    world = ET.SubElement(root, "worldbody")

    torso = ET.SubElement(world, "body", {"name": "torso", "pos": "0 0 0"})
    ET.SubElement(torso, "freejoint", {"name": "root"})
    ET.SubElement(
        torso,
        "geom",
        _geom_attributes(
            "torso_geom",
            spec.torso_size,
            (0.0, 0.0, 0.0),
            spec,
            _GREEN,
            contype=2,
            conaffinity=1,
        ),
    )

    # ---- 头：颈关节在躯干上前方，头几何再向前上方偏一段 ----
    head = ET.SubElement(torso, "body", {"name": "head", "pos": vec(spec.neck_pos)})
    _add_joint(head, spec.neck, None)
    ET.SubElement(
        head,
        "geom",
        _geom_attributes(
            "head_geom",
            spec.head_size,
            spec.head_offset,
            spec,
            _GREEN,
            contype=2,
            conaffinity=1,
        ),
    )

    # ---- 两臂：单段，肩关节在躯干侧前方 ----
    for side, sign in (("L", 1.0), ("R", -1.0)):
        shoulder = (
            spec.shoulder_pos[0],
            sign * spec.shoulder_pos[1],
            spec.shoulder_pos[2],
        )
        arm = ET.SubElement(torso, "body", {"name": f"arm_{side}", "pos": vec(shoulder)})
        _add_joint(arm, spec.shoulder, side)
        ET.SubElement(
            arm,
            "geom",
            _geom_attributes(
                f"arm_geom_{side}",
                spec.arm_size,
                (0.0, 0.0, -spec.arm_len / 2.0),
                spec,
                _GREEN_LIMB,
                contype=2,
                conaffinity=1,
            ),
        )

    # ---- 两腿 ----
    _add_leg(torso, spec, "L", 1.0)
    _add_leg(torso, spec, "R", -1.0)

    # ---- 尾巴：两节，沿 −x 向后伸 ----
    tail1 = ET.SubElement(
        torso, "body", {"name": "tail_1", "pos": vec(spec.tail1_pos)}
    )
    _add_joint(tail1, spec.tail1, None)
    ET.SubElement(
        tail1,
        "geom",
        _geom_attributes(
            "tail1_geom",
            spec.tail1_size,
            (-spec.tail1_len / 2.0, 0.0, 0.0),
            spec,
            _GREEN,
            contype=2,
            conaffinity=1,
        ),
    )

    tail2 = ET.SubElement(
        tail1, "body", {"name": "tail_2", "pos": vec((-spec.tail1_len, 0.0, 0.0))}
    )
    _add_joint(tail2, spec.tail2, None)
    ET.SubElement(
        tail2,
        "geom",
        _geom_attributes(
            "tail2_geom",
            spec.tail2_size,
            (-spec.tail2_len / 2.0, 0.0, 0.0),
            spec,
            _GREEN,
            contype=2,
            conaffinity=1,
        ),
    )

    # ---- 执行器：每个驱动关节一个位置源 ----
    actuator = ET.SubElement(root, "actuator")
    for joint, side in spec.actuated():
        ET.SubElement(actuator, "position", _actuator_attributes(joint, side))

    # ---- 关键帧：标称站姿 ----
    stand = spec.stand_q()
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        {
            "name": "stand",
            "qpos": vec(stand_qpos(spec)),
            "ctrl": vec([stand[name] for name in spec.joint_names]),
        },
    )
    return root


def build_dino_scene(spec: DinoSpec = DEFAULT_DINO_SPEC) -> ET.Element:
    """构造恐龙场景：``<include file="dino.xml"/>`` + 地面/灯光/机位。

    除了通用场景自带的 ``front`` / ``iso`` 机位，再补一个 ``ref`` 机位：
    与参考图接近的右前上方 3/4 视角（远距离 + 窄视场，接近正交投影），
    方便出图和"模型 vs 参考图"的对照。
    """
    root = build_scene_for(
        "dino.xml",
        f"{spec.name}_scene",
        statistic_center="0 0 0.17",
        statistic_extent="0.5",
    )
    world = root.find("worldbody")
    if world is not None:
        ET.SubElement(
            world,
            "camera",
            {
                "name": "ref",
                "pos": "1.1 1.035 0.492",
                "xyaxes": "-0.682 0.731 0 -0.152 -0.142 0.978",
                "fovy": "18",
            },
        )
    return root


def _to_text(root: ET.Element) -> str:
    ET.indent(root, space="  ")
    return _HEADER + ET.tostring(root, encoding="unicode") + "\n"


def render_dino_xml(spec: DinoSpec = DEFAULT_DINO_SPEC) -> str:
    return _to_text(build_dino(spec))


def render_dino_scene_xml(spec: DinoSpec = DEFAULT_DINO_SPEC) -> str:
    return _to_text(build_dino_scene(spec))


def write_dino_models(
    spec: DinoSpec = DEFAULT_DINO_SPEC, out_dir: Path | str = MODELS_DIR
) -> list[Path]:
    """写出 ``dino.xml`` 与 ``dino_scene.xml``；内容没变就不重写。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, text in (
        ("dino.xml", render_dino_xml(spec)),
        ("dino_scene.xml", render_dino_scene_xml(spec)),
    ):
        path = out / filename
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8", newline="\n")
        written.append(path)
    return written


def summary(spec: DinoSpec = DEFAULT_DINO_SPEC) -> str:
    """给 CLI 用的一行摘要。"""
    return (
        f"{spec.name}: {spec.n_bodies} bodies, nq={spec.nq}, nv={spec.nv}, nu={spec.nu}, "
        f"mass≈{spec.total_mass():.3f} kg, height≈{head_top_z(spec):.3f} m"
    )


# ----------------------------------------------------------------------
# 静态校验
# ----------------------------------------------------------------------
def _check_duplicates(report: Report, names: list[str], kind: str) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            report.errors.append(f"{kind} 名字重复: {name}")
        seen.add(name)


def verify_dino_static(
    spec: DinoSpec = DEFAULT_DINO_SPEC, models_dir: Path | str = MODELS_DIR
) -> Report:
    """把生成出来的 ``dino*.xml`` 解析回来，与参数源逐项对照。"""
    report = Report()
    models = Path(models_dir)
    robot_path = models / "dino.xml"
    scene_path = models / "dino_scene.xml"
    for path in (robot_path, scene_path):
        if not path.exists():
            report.errors.append(f"缺少模型文件: {path}（先运行 python -m desktop_robot build-dino）")
    if not report.ok:
        return report

    try:
        robot_root = ET.parse(robot_path).getroot()
        scene_root = ET.parse(scene_path).getroot()
    except ET.ParseError as exc:  # pragma: no cover - 生成器正确时不会触发
        report.errors.append(f"XML 解析失败: {exc}")
        return report

    if robot_root.tag != "mujoco":
        report.errors.append(f"dino.xml 根元素应为 mujoco，实际为 {robot_root.tag}")
    if robot_root.get("model") != spec.name:
        report.errors.append(f"model 名不符: {robot_root.get('model')} != {spec.name}")

    compilers = robot_root.findall("compiler")
    if len(compilers) != 1:
        report.errors.append(f"compiler 元素应恰好 1 个，实际 {len(compilers)}")
    elif compilers[0].get("angle") != "radian":
        report.errors.append("compiler/angle 必须为 radian")

    for tag in ("worldbody", "actuator", "keyframe"):
        if robot_root.find(tag) is None:
            report.errors.append(f"dino.xml 缺少 <{tag}>")

    bodies = robot_root.findall(".//body")
    if len(bodies) != spec.n_bodies:
        report.errors.append(f"body 数量应为 {spec.n_bodies}，实际 {len(bodies)}")
    _check_duplicates(report, [b.get("name", "") for b in bodies], "body")

    freejoints = robot_root.findall(".//freejoint")
    if len(freejoints) != 1 or freejoints[0].get("name") != "root":
        report.errors.append("应存在且仅存在一个名为 root 的 freejoint")

    hinges = robot_root.findall(".//joint")
    actual_joints = [j.get("name", "") for j in hinges]
    if actual_joints != spec.joint_names:
        report.errors.append(
            "关节名/顺序与参数源不一致:\n"
            f"  期望 {spec.joint_names}\n  实际 {actual_joints}"
        )
    _check_duplicates(report, actual_joints, "joint")
    if len(hinges) != spec.nu:
        report.errors.append(f"关节数量应为 {spec.nu}，实际 {len(hinges)}")
    for joint in hinges:
        name = joint.get("name", "?")
        rng = joint.get("range")
        if not rng:
            report.errors.append(f"关节 {name} 缺少 range")
            continue
        lo, hi = (float(value) for value in rng.split())
        if not lo < hi:
            report.errors.append(f"关节 {name} 的 range 非法: [{lo}, {hi}]")
        axis = [float(value) for value in (joint.get("axis") or "0 0 1").split()]
        if abs(math.sqrt(sum(value * value for value in axis)) - 1.0) > 1e-9:
            report.errors.append(f"关节 {name} 的 axis 不是单位向量")

    actuators = robot_root.find("actuator")
    actuator_elems = list(actuators) if actuators is not None else []
    if len(actuator_elems) != spec.nu:
        report.errors.append(f"执行器数量应为 {spec.nu}，实际 {len(actuator_elems)}")
    _check_duplicates(report, [a.get("name", "") for a in actuator_elems], "actuator")
    joint_set = set(actual_joints)
    for act in actuator_elems:
        if act.get("joint") not in joint_set:
            report.errors.append(f"执行器 {act.get('name')} 指向不存在的关节 {act.get('joint')}")

    keys = robot_root.findall("keyframe/key")
    if len(keys) != 1 or keys[0].get("name") != "stand":
        report.errors.append("应存在且仅存在一个名为 stand 的关键帧")
    else:
        qpos = keys[0].get("qpos", "").split()
        ctrl = keys[0].get("ctrl", "").split()
        if len(qpos) != spec.nq:
            report.errors.append(f"stand 关键帧 qpos 长度应为 {spec.nq}，实际 {len(qpos)}")
        if len(ctrl) != spec.nu:
            report.errors.append(f"stand 关键帧 ctrl 长度应为 {spec.nu}，实际 {len(ctrl)}")
        if len(qpos) >= 3:
            z = float(qpos[2])
            expected = solve_base_z(spec)
            if abs(z - expected) > 1e-4:
                report.errors.append(f"stand 基座高度 {z:.6f} 与运动学解 {expected:.6f} 不符")

    # 接触件（足 + 前趾）1/1；关节轴承外壳只做视觉 0/0；其余连杆 2/1
    for geom in robot_root.findall(".//geom"):
        name = geom.get("name", "")
        mask = (geom.get("contype"), geom.get("conaffinity"))
        if name.startswith("cap_"):
            expected = ("0", "0")
        elif name.startswith("foot_geom_") or name.startswith("toe_geom_"):
            expected = ("1", "1")
        else:
            expected = ("2", "1")
        if mask != expected:
            report.errors.append(f"{name} 的碰撞掩码应为 {expected[0]}/{expected[1]}，实际为 {mask}")

    # 场景：include 本体、不能再写 compiler、必须有地板
    include = scene_root.find("include")
    if include is None or include.get("file") != "dino.xml":
        report.errors.append('dino_scene.xml 应当 <include file="dino.xml"/>')
    if scene_root.find("compiler") is not None:
        report.errors.append("dino_scene.xml 不应自带 <compiler>（会和 include 的冲突）")
    floor = scene_root.find(".//geom[@name='floor']")
    if floor is None:
        report.errors.append("dino_scene.xml 缺少 floor 地面")
    elif (floor.get("contype"), floor.get("conaffinity")) != ("1", "1"):
        report.errors.append("floor 的 contype/conaffinity 应为 1/1，否则脚踩不到地面")

    report.info["文件"] = f"{robot_path.name}, {scene_path.name}"
    report.info["规模"] = f"bodies={len(bodies)}, joints={len(hinges)}, actuators={len(actuator_elems)}"
    report.info["期望规模"] = f"nq={spec.nq}, nv={spec.nv}, nu={spec.nu}"
    report.info["总质量"] = f"≈ {spec.total_mass():.3f} kg"
    report.info["基座高度"] = f"{solve_base_z(spec):.4f} m"
    report.info["站立高度"] = f"{head_top_z(spec):.4f} m"
    return report
