"""桌面恐龙（T-rex 造型）机器人的参数化 MuJoCo 模型生成器。

参数源在 :class:`DinoSpec` / :data:`DEFAULT_DINO_SPEC`，模型文件
``models/dino.xml`` 与 ``models/dino_scene.xml`` 是它的确定性生成物。

拓扑（18 个驱动关节：俯仰 pitch 11 / 偏航 yaw 5 / 内外翻 roll 2）：::

    torso（浮动基座，freejoint: root）
    ├─ head          neck_pitch ①（上下） → neck_yaw ②（左右，轴 z）
    │                └─ 相机 cam_head（头部前视相机，随头一起转）
    ├─ arm_L         shoulder_pitch_L ③（上下）
    ├─ arm_R         shoulder_pitch_R ④（上下）
    ├─ thigh_L       hip_pitch_L ⑤（主髋，上下） → hip_roll_L ⑥（内外翻，轴 x）
    │  │                                        → hip_yaw_L ⑦（脚尖外旋，轴 z）
    │  └─ shank_L    knee_L ⑧
    │     └─ foot_L  ankle_L ⑨
    ├─ thigh_R       hip_pitch_R ⑩ → hip_roll_R ⑪ → hip_yaw_R ⑫
    │  └─ shank_R    knee_R ⑬
    │     └─ foot_R  ankle_R ⑭
    └─ tail_1        tail1_yaw ⑮（侧向，轴 z）
       └─ tail_2     tail2_pitch ⑯（上下）
          └─ tail_3  tail3_yaw ⑰（侧向，轴 z）
             └─ tail_4 tail4_pitch ⑱（上下）

每条腿都是 5 个关节：主髋 pitch → 内外翻 roll → 脚尖 yaw → 膝 → 踝。
「脚尖 yaw」是本版相对参考图新增的自由度，用来让脚掌相对腿转向。

**每个连杆的长度都由对应 ``*_size`` 的相应分量决定**，改长度后关节/几何偏移
会自动跟着重算，不需要手工同步：

============================  ==========================================
连杆                          长度来源
============================  ==========================================
``thigh``                      ``thigh_size[2]``（膝关节在 z = −大腿长）
``shank``                      ``shank_size[2]``（踝关节在 z = −小腿长）
``arm``                        ``arm_size[2]``（臂几何中心在 z = −臂长/2）
``tail_1`` … ``tail_4``        ``tail1_size[0]`` … ``tail4_size[0]``
``foot`` / ``toe``             足 box 中心 ``foot_center``；前趾贴住足前端与足底
============================  ==========================================

躯干上的锚点（``neck_pos`` / ``shoulder_pos`` / ``hip_pos`` / ``tail1_pos``）
是显式字段：想改「关节在躯干上的位置」直接改这些偏移即可。

坐标与轴向：x 前 / y 左 / z 上。

* 俯仰（上下）：轴 ``(0, -1, 0)``，正角 = 子连杆向前摆；对朝后的尾巴，正角 = 尾尖下压。
* 偏航（左右）：轴 ``(0, 0, 1)``，正角 = 从上往下看逆时针（头部向左、尾尖向右）。
* 内外翻：左腿轴 ``(1, 0, 0)``、右腿轴 ``(-1, 0, 0)``，两腿正角都表示外翻。
* 脚尖偏航：左腿轴 ``(0, 0, 1)``、右腿轴 ``(0, 0, -1)``（``JointSpec.mirror_z`` 镜像），
  **两腿正角都表示脚尖外旋**（左脚尖转向 +y、右脚尖转向 −y），控制器同样不用翻号。

轴写在关节所在 body 的局部坐标系里，会随上游关节一起转；同一个 body 里多个关节
按定义顺序复合，所以「髋 pitch → 内外翻 roll → 脚尖 yaw」「头 pitch → yaw」都按
这副顺序生效。

模型的**零位就是标称站姿**：每个 body 的 ``quat`` 烘了它所有关节的站姿角
（颈/肩/髋/膝/踝/尾），躯干 body 的 ``pos`` / ``quat`` 是站姿的基座高度与抬头角。
于是 ``qpos`` 全 0 时模型正好是标称站姿、脚底落在 z=0：``dino.xml`` /
``dino_scene.xml`` 一加载就站在地面上，不依赖"先套 stand 关键帧"。``stand``
关键帧仍然保留（关节角全 0、``ctrl`` 全 0），用于 ``mj_resetDataKeyframe`` 复位；
位置执行器的 ``ctrl=0`` 也就等于站姿。

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

#: 俯仰轴（上下）。与双足模型一致：正角 = 子连杆向前摆。
PITCH: Vec3 = (0.0, -1.0, 0.0)
#: 偏航轴（左右，从上往下看逆时针为正）。
YAW: Vec3 = (0.0, 0.0, 1.0)
#: 内外翻轴（前后方向）；右腿在 :meth:`JointSpec.axis_for` 里把 x 取反，
#: 于是两腿正角都表示外翻。
ROLL: Vec3 = (1.0, 0.0, 0.0)

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

#: 关节轴承外壳（只做视觉，不参与质量和碰撞）：关节基名 → (半径, 半长)。
#: 参考图里每个铰链都有一圈灰色轴承，加上它渲染出来才像那台机器人。
_CAPS: dict[str, tuple[float, float]] = {
    "neck_pitch": (0.016, 0.024),
    "neck_yaw": (0.014, 0.020),
    "shoulder_pitch": (0.014, 0.019),
    "hip_pitch": (0.016, 0.024),
    "hip_roll": (0.015, 0.022),
    "hip_yaw": (0.015, 0.022),
    "knee": (0.014, 0.021),
    "ankle": (0.012, 0.017),
    "tail1_yaw": (0.012, 0.018),
    "tail2_pitch": (0.011, 0.016),
    "tail3_yaw": (0.010, 0.014),
    "tail4_pitch": (0.009, 0.012),
}


def _joint(
    name: str,
    rng: tuple[float, float],
    axis: Vec3 = PITCH,
    kp: float = 25.0,
    kv: float = 1.0,
    forcerange: float = 2.5,
    mirror_x: bool = False,
    mirror_z: bool = False,
) -> JointSpec:
    """构造一个单轴铰链；默认是俯仰轴，可换成 yaw / roll。"""
    return JointSpec(
        name=name,
        axis=axis,
        range=rng,
        kp=kp,
        kv=kv,
        forcerange=forcerange,
        mirror_x=mirror_x,
        mirror_z=mirror_z,
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

    # ---- 头部相机：挂在 head body 上，随颈/头一起转（第一人称观测用）----
    # pos 相对 head body 原点，贴在头部前方脸上；xyaxes = (相机 x 轴, 相机 y 轴)，
    # 相机沿 −z 看：x = (0,−1,0)（画面右方 = −y）、y = (sin12°, 0, cos12°)，
    # 于是视轴 = (cos12°, 0, −sin12°)，即朝 +x 前方、相对头部下俯 12°。
    # 站姿下头部净俯仰约 +5.7°，所以相机世界视线约下俯 6.3°，能看到更多前方地面。
    head_camera_pos: Vec3 = (0.086, 0.0, 0.026)
    head_camera_xyaxes: str = "0 -1 0 0.2079 0 0.9781"
    head_camera_fovy: float = 75.0

    # ---- 臂：单段，无肘/手 ----
    arm_size: Vec3 = (0.045, 0.032, 0.075)
    shoulder_pos: Vec3 = (0.048, 0.058, 0.018)  # y 为左肩；右肩取反

    # ---- 腿 ----
    thigh_size: Vec3 = (0.048, 0.040, 0.050)
    shank_size: Vec3 = (0.044, 0.036, 0.045)
    foot_size: Vec3 = (0.085, 0.055, 0.032)
    toe_size: Vec3 = (0.028, 0.018, 0.026)
    hip_pos: Vec3 = (-0.085, 0.045, -0.030)
    foot_center: Vec3 = (0.012, 0.0, -0.016)  # 足 box 中心相对踝关节
    toe_y: float = 0.013  # 两个前趾中心相对足中线（旋转方向镜像 ±）
    toe_overlap: float = 0.006  # 前趾与足 box 在 x 方向的重叠量

    # ---- 尾巴：4 节，长度/截面都在这里改 ----
    tail1_size: Vec3 = (0.080, 0.070, 0.086)
    tail2_size: Vec3 = (0.075, 0.065, 0.074)
    tail3_size: Vec3 = (0.070, 0.061, 0.062)
    tail4_size: Vec3 = (0.060, 0.056, 0.050)
    tail1_pos: Vec3 = (-0.068, 0.0, -0.020)

    # ---- 关节 ----
    neck: JointSpec = field(default_factory=lambda: _joint("neck_pitch", (-0.70, 0.70)))
    neck_yaw: JointSpec = field(
        default_factory=lambda: _joint("neck_yaw", (-0.60, 0.60), axis=YAW)
    )
    shoulder: JointSpec = field(
        default_factory=lambda: _joint("shoulder_pitch", (-1.20, 1.20))
    )
    hip: JointSpec = field(default_factory=lambda: _joint("hip_pitch", (-1.20, 1.20)))
    hip_roll: JointSpec = field(
        default_factory=lambda: _joint(
            "hip_roll", (-0.35, 0.35), axis=ROLL, mirror_x=True
        )
    )
    hip_yaw: JointSpec = field(
        default_factory=lambda: _joint(
            "hip_yaw", (-0.60, 0.60), axis=YAW, mirror_z=True
        )
    )
    knee: JointSpec = field(default_factory=lambda: _joint("knee", (-1.60, 0.05)))
    ankle: JointSpec = field(
        default_factory=lambda: _joint("ankle", (-0.60, 0.80), forcerange=2.0)
    )
    tail1_yaw: JointSpec = field(
        default_factory=lambda: _joint("tail1_yaw", (-0.50, 0.50), axis=YAW)
    )
    tail2_pitch: JointSpec = field(
        default_factory=lambda: _joint("tail2_pitch", (-0.80, 0.50))
    )
    tail3_yaw: JointSpec = field(
        default_factory=lambda: _joint("tail3_yaw", (-0.50, 0.50), axis=YAW)
    )
    tail4_pitch: JointSpec = field(
        default_factory=lambda: _joint("tail4_pitch", (-0.80, 0.50))
    )

    # ---- 标称站姿（弧度）----
    # 躯干抬头 base_pitch；踝角由「足底水平」判据反解，保证脚掌贴地。
    # 新增的 yaw / roll 关节站姿角都是 0：零位即站姿的写法下它们不需要平移范围。
    base_pitch: float = math.radians(20.0)
    stand_neck: float = -0.25
    stand_neck_yaw: float = 0.0
    stand_shoulder: float = -0.15
    stand_hip: float = -0.30
    stand_hip_roll: float = 0.0
    stand_hip_yaw: float = 0.0
    stand_knee: float = -0.10
    stand_tail1_yaw: float = 0.0
    # v2.2：尾巴抬高、不再触地（实测最低点离地约 4.3 cm），尾巴不能作为支撑点。
    stand_tail2_pitch: float = -0.10
    stand_tail3_yaw: float = 0.0
    stand_tail4_pitch: float = -0.05

    density: float = 400.0

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
    def tail3_len(self) -> float:
        return self.tail3_size[0]

    @property
    def tail4_len(self) -> float:
        return self.tail4_size[0]

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
        颈 pitch → 颈 yaw → 左肩 → 右肩 →
        左腿 5 个（pitch → roll → yaw → knee → ankle）→ 右腿 5 个 → 尾 4 个。
        """
        return [
            (self.neck, None),
            (self.neck_yaw, None),
            (self.shoulder, "L"),
            (self.shoulder, "R"),
            (self.hip, "L"),
            (self.hip_roll, "L"),
            (self.hip_yaw, "L"),
            (self.knee, "L"),
            (self.ankle, "L"),
            (self.hip, "R"),
            (self.hip_roll, "R"),
            (self.hip_yaw, "R"),
            (self.knee, "R"),
            (self.ankle, "R"),
            (self.tail1_yaw, None),
            (self.tail2_pitch, None),
            (self.tail3_yaw, None),
            (self.tail4_pitch, None),
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
        """躯干 1 + 头 1 + 臂 2 + 每腿 3 ×2 + 尾 4。"""
        return 1 + 1 + 2 + 2 * 3 + 4

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
            ("tail_3", self.tail3_size, self.density),
            ("tail_4", self.tail4_size, self.density),
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
        q: dict[str, float] = {
            "neck_pitch": self.stand_neck,
            "neck_yaw": self.stand_neck_yaw,
            "tail1_yaw": self.stand_tail1_yaw,
            "tail2_pitch": self.stand_tail2_pitch,
            "tail3_yaw": self.stand_tail3_yaw,
            "tail4_pitch": self.stand_tail4_pitch,
        }
        for side in ("L", "R"):
            q[f"shoulder_pitch_{side}"] = self.stand_shoulder
            q[f"hip_pitch_{side}"] = self.stand_hip
            q[f"hip_roll_{side}"] = self.stand_hip_roll
            q[f"hip_yaw_{side}"] = self.stand_hip_yaw
            q[f"knee_{side}"] = self.stand_knee
            q[f"ankle_{side}"] = self.stand_ankle
        return q


# ----------------------------------------------------------------------
# 默认机型：约 0.31 m 高、约 2.0 kg 的桌面恐龙
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
    """单腿 FK：返回足 box 与两个前趾的 ``(中心, 尺寸, 旋转)``，均在躯干坐标系。

    只算矢状面内的三个 pitch 关节。站姿里髋内外翻 / 脚尖 yaw 的站姿角都是 0，
    在这里（用来反解站立高度）等价于单位旋转，所以不需要展开三维链路。
    """
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


def pitch_quat(angle: float) -> tuple[float, float, float, float]:
    """绕俯仰轴 ``(0, -1, 0)`` 转 ``angle`` 的四元数（MJCF 的 ``(w, x, y, z)``）。"""
    half_angle = angle / 2.0
    return (math.cos(half_angle), 0.0, -math.sin(half_angle), 0.0)


def base_quat(spec: DinoSpec) -> tuple[float, float, float, float]:
    """标称站姿的基座四元数 ``(qw, qx, qy, qz)``：绕 ``(0, -1, 0)`` 抬头。"""
    return pitch_quat(spec.base_pitch)


def _axis_quat(axis: Vec3, angle: float) -> tuple[float, float, float, float]:
    """绕任意轴转 ``angle`` 的四元数（MJCF 的 ``(w, x, y, z)``）。"""
    norm = math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
    s = math.sin(angle / 2.0) / norm
    return (math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s)


def _quat_mul(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """四元数乘法：``a * b`` 对应先转 ``b`` 再转 ``a`` 的旋转矩阵 ``R(a) R(b)``。"""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _body_quat(*axis_angles: tuple[Vec3, float]) -> str:
    """body 的静止姿态：把该 body 上所有关节的站姿旋转按定义顺序乘起来。

    ``axis`` 传该关节实际生效的轴（右腿内外翻要传镜像后的轴），
    这样 ``qpos=0`` 时模型正好是标称站姿。
    """
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    for axis, angle in axis_angles:
        quat = _quat_mul(quat, _axis_quat(axis, angle))
    return vec(quat)


def stand_qpos(spec: DinoSpec) -> list[float]:
    """``stand`` 关键帧的完整 qpos。

    基座取站姿的高度/抬头角，18 个关节角全 0——因为模型的零位已经是站姿了。
    """
    return [
        0.0,
        0.0,
        solve_base_z(spec),
        *base_quat(spec),
    ] + [0.0] * spec.nu


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


def _joint_attributes(
    joint: JointSpec, side: str | None, spec: DinoSpec
) -> dict[str, str]:
    """一个 hinge 的属性表。

    关节坐标的零位是**标称站姿**（站姿角烘在 body 的 ``quat`` 里），所以这里的
    ``range``（本来写的是物理角）要减去该关节的站姿角，换算到"相对站姿"的坐标。
    """
    name = joint_name(joint, side)
    stand = spec.stand_q()[name]
    low, high = joint.range
    return {
        "name": name,
        "type": "hinge",
        "axis": vec(joint.axis_for(side)),
        "range": vec((low - stand, high - stand)),
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
    attrs = {
        "name": f"cap_{joint_name(joint, side)}",
        "type": "cylinder",
        "size": f"{fnum(radius)} {fnum(half_len)}",
        "pos": "0 0 0",
        "rgba": _GREY,
        "density": "0",
        "contype": "0",
        "conaffinity": "0",
    }
    # 圆柱默认沿 z 轴；按关节轴把它转到 y（俯仰）或 x（内外翻），偏航沿用 z。
    axis = joint.axis_for(side)
    if abs(axis[1]) > 0.5:
        attrs["quat"] = "0.7071068 0.7071068 0 0"
    elif abs(axis[2]) < 0.5:
        attrs["quat"] = "0.7071068 0 0.7071068 0"
    return attrs


def _add_joint(
    body: ET.Element, joint: JointSpec, side: str | None, spec: DinoSpec
) -> None:
    """在 body 原点挂一个铰链，再加上它的灰色轴承外壳（如果有）。"""
    ET.SubElement(body, "joint", _joint_attributes(joint, side, spec))
    cap = _cap_attributes(joint, side)
    if cap is not None:
        ET.SubElement(body, "geom", cap)


def _add_leg(torso: ET.Element, spec: DinoSpec, side: str, sign: float) -> None:
    """挂一条腿：thigh → shank → foot（foot 上再挂两个前趾）。

    thigh body 上有三个关节，按「主髋 pitch → 内外翻 roll → 脚尖 yaw」的顺序；
    body 的 ``quat`` 烘的是这三个关节的站姿旋转，所以 qpos 全 0 时这条腿正好
    是标称站姿（髋 pitch = ``stand_hip``、内外翻 = ``stand_hip_roll``、
    脚尖 yaw = ``stand_hip_yaw``）。
    """
    hip = (spec.hip_pos[0], sign * spec.hip_pos[1], spec.hip_pos[2])
    thigh = ET.SubElement(
        torso,
        "body",
        {
            "name": f"thigh_{side}",
            "pos": vec(hip),
            "quat": _body_quat(
                (spec.hip.axis_for(side), spec.stand_hip),
                (spec.hip_roll.axis_for(side), spec.stand_hip_roll),
                (spec.hip_yaw.axis_for(side), spec.stand_hip_yaw),
            ),
        },
    )
    _add_joint(thigh, spec.hip, side, spec)
    _add_joint(thigh, spec.hip_roll, side, spec)
    _add_joint(thigh, spec.hip_yaw, side, spec)
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
        thigh,
        "body",
        {
            "name": f"shank_{side}",
            "pos": vec((0.0, 0.0, -spec.thigh_len)),
            "quat": _body_quat((spec.knee.axis_for(side), spec.stand_knee)),
        },
    )
    _add_joint(shank, spec.knee, side, spec)
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
        shank,
        "body",
        {
            "name": f"foot_{side}",
            "pos": vec((0.0, 0.0, -spec.shank_len)),
            "quat": _body_quat((spec.ankle.axis_for(side), spec.stand_ankle)),
        },
    )
    _add_joint(foot, spec.ankle, side, spec)
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

    # ---- 浮基座：body 的 pos/quat 就是 freejoint 的 qpos0，直接写成标称站姿，
    #      这样模型文件单独加载（不套 stand 关键帧）时也站在地面上 ----
    torso = ET.SubElement(
        world,
        "body",
        {
            "name": "torso",
            "pos": vec((0.0, 0.0, solve_base_z(spec))),
            "quat": vec(base_quat(spec)),
        },
    )
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

    # ---- 头：颈 pitch → 颈 yaw，头几何再向前上方偏一段 ----
    head = ET.SubElement(
        torso,
        "body",
        {
            "name": "head",
            "pos": vec(spec.neck_pos),
            "quat": _body_quat(
                (spec.neck.axis, spec.stand_neck),
                (spec.neck_yaw.axis, spec.stand_neck_yaw),
            ),
        },
    )
    _add_joint(head, spec.neck, None, spec)
    _add_joint(head, spec.neck_yaw, None, spec)
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
    # ---- 头部相机 cam_head：挂在 head body 上，随颈 pitch / yaw 一起转 ----
    ET.SubElement(
        head,
        "camera",
        {
            "name": "cam_head",
            "pos": vec(spec.head_camera_pos),
            "xyaxes": spec.head_camera_xyaxes,
            "fovy": fnum(spec.head_camera_fovy),
        },
    )

    # ---- 两臂：单段，肩关节在躯干侧前方 ----
    for side, sign in (("L", 1.0), ("R", -1.0)):
        shoulder = (
            spec.shoulder_pos[0],
            sign * spec.shoulder_pos[1],
            spec.shoulder_pos[2],
        )
        arm = ET.SubElement(
            torso,
            "body",
            {
                "name": f"arm_{side}",
                "pos": vec(shoulder),
                "quat": _body_quat(
                    (spec.shoulder.axis_for(side), spec.stand_shoulder)
                ),
            },
        )
        _add_joint(arm, spec.shoulder, side, spec)
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

    # ---- 尾巴：4 节，沿 −x 向后伸；关节依次 yaw → pitch → yaw → pitch ----
    tail1 = ET.SubElement(
        torso,
        "body",
        {
            "name": "tail_1",
            "pos": vec(spec.tail1_pos),
            "quat": _body_quat((spec.tail1_yaw.axis, spec.stand_tail1_yaw)),
        },
    )
    _add_joint(tail1, spec.tail1_yaw, None, spec)
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
        tail1,
        "body",
        {
            "name": "tail_2",
            "pos": vec((-spec.tail1_len, 0.0, 0.0)),
            "quat": _body_quat((spec.tail2_pitch.axis, spec.stand_tail2_pitch)),
        },
    )
    _add_joint(tail2, spec.tail2_pitch, None, spec)
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

    tail3 = ET.SubElement(
        tail2,
        "body",
        {
            "name": "tail_3",
            "pos": vec((-spec.tail2_len, 0.0, 0.0)),
            "quat": _body_quat((spec.tail3_yaw.axis, spec.stand_tail3_yaw)),
        },
    )
    _add_joint(tail3, spec.tail3_yaw, None, spec)
    ET.SubElement(
        tail3,
        "geom",
        _geom_attributes(
            "tail3_geom",
            spec.tail3_size,
            (-spec.tail3_len / 2.0, 0.0, 0.0),
            spec,
            _GREEN,
            contype=2,
            conaffinity=1,
        ),
    )

    tail4 = ET.SubElement(
        tail3,
        "body",
        {
            "name": "tail_4",
            "pos": vec((-spec.tail3_len, 0.0, 0.0)),
            "quat": _body_quat((spec.tail4_pitch.axis, spec.stand_tail4_pitch)),
        },
    )
    _add_joint(tail4, spec.tail4_pitch, None, spec)
    ET.SubElement(
        tail4,
        "geom",
        _geom_attributes(
            "tail4_geom",
            spec.tail4_size,
            (-spec.tail4_len / 2.0, 0.0, 0.0),
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

    # ---- 关键帧：标称站姿（关节角全 0，因为零位就是站姿）----
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        {
            "name": "stand",
            "qpos": vec(stand_qpos(spec)),
            "ctrl": vec([0.0] * spec.nu),
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
    expected_axes = {
        joint_name(joint, side): tuple(round(v, 9) for v in joint.axis_for(side))
        for joint, side in spec.actuated()
    }
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
        expected_axis = expected_axes.get(name)
        if expected_axis is not None and tuple(round(v, 9) for v in axis) != expected_axis:
            report.errors.append(
                f"关节 {name} 的 axis 应为 {expected_axis}，实际 {tuple(axis)}"
            )

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

    # ---- 头部相机：必须挂在 head body 里（这样它会随颈 / 头一起转）----
    head_body = robot_root.find(".//body[@name='head']")
    camera = head_body.find("camera[@name='cam_head']") if head_body is not None else None
    if camera is None:
        report.errors.append("dino.xml 的 head body 里缺少 cam_head 相机")
    else:
        actual_pos = [float(v) for v in (camera.get("pos") or "").split()]
        if len(actual_pos) != 3 or any(
            abs(a - b) > 1e-6 for a, b in zip(actual_pos, spec.head_camera_pos)
        ):
            report.errors.append(
                f"cam_head 的 pos 应为 {spec.head_camera_pos}，实际 {actual_pos}"
            )
        if camera.get("xyaxes") != spec.head_camera_xyaxes:
            report.errors.append(
                f"cam_head 的 xyaxes 应为 {spec.head_camera_xyaxes!r}，"
                f"实际 {camera.get('xyaxes')!r}"
            )
        fovy = camera.get("fovy")
        if fovy is None or abs(float(fovy) - spec.head_camera_fovy) > 1e-6:
            report.errors.append(f"cam_head 的 fovy 应为 {spec.head_camera_fovy}，实际 {fovy}")

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
    report.info["头部相机"] = "cam_head（head body）"
    return report
