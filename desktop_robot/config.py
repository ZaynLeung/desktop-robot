"""桌面双足机器人的唯一参数源。

约定
----
* 长度单位：米 (m)；角度单位：弧度 (rad)。
* 世界坐标系：x 向前、y 向左、z 向上（右手系）。
* 所有几何都是 box 原始体，没有外部网格资产，仓库里不出现二进制模型。
* 这里的值是 MuJoCo 模型的**唯一真相**：改这里 + 重新 build，就能同时更新
  ``models/biped.xml`` 与 ``models/scene.xml``。

俯仰关节的符号约定
------------------
髋俯仰 / 膝 / 踝的轴统一写成 ``(0, -1, 0)``，于是**正角度 = 子连杆向前摆**。
膝关节正角度表示"伸展"、负角度表示"屈膝"，因此正常站立时膝角为负值。
髋外展轴左腿 ``(1, 0, 0)``、右腿 ``(-1, 0, 0)``，两腿都是**正角度 = 外展**，
这样左右腿的控制量可以直接取相同符号，不需要在控制器里翻号。
髋偏航（恐龙机型）轴左腿 ``(0, 0, 1)``、右腿 ``(0, 0, -1)``（``mirror_z``），
两腿正角度都表示**脚尖外旋**，同样不需要翻号。
"""

from __future__ import annotations

from dataclasses import dataclass

Vec3 = tuple[float, float, float]


def _flip_x(v: Vec3) -> Vec3:
    """把向量的 x 分量取反（用于左右腿镜像）。"""
    return (-v[0], v[1], v[2])


def _flip_z(v: Vec3) -> Vec3:
    """把向量的 z 分量取反（用于左右腿偏航轴镜像）。"""
    return (v[0], v[1], -v[2])


@dataclass(frozen=True)
class JointSpec:
    """一个受驱动关节的参数。

    ``name`` 是基名；腿部关节在模型里会加 ``_L`` / ``_R`` 后缀，
    腰部关节（``side=None``）直接用基名。
    """

    name: str
    axis: Vec3
    range: tuple[float, float]
    kp: float
    kv: float
    forcerange: float
    armature: float = 0.005
    damping: float = 0.05
    frictionloss: float = 0.02
    mirror_x: bool = False
    mirror_z: bool = False

    def axis_for(self, side: str | None) -> Vec3:
        """返回指定侧的关节轴；右腿依次应用 ``mirror_x`` / ``mirror_z`` 镜像。

        两种镜像可以同时打开（例如需要同时翻转 x 与 z 的斜轴），
        默认都为 ``False``，所以对既有双足机型完全向后兼容。
        """
        if side != "R":
            return self.axis
        axis = self.axis
        if self.mirror_x:
            axis = _flip_x(axis)
        if self.mirror_z:
            axis = _flip_z(axis)
        return axis


@dataclass(frozen=True)
class RobotSpec:
    """整机参数。

    所有尺寸都是 **box 的全尺寸**（不是半尺寸）；写入 MJCF 时再除以 2。
    这样参数表和课程报告里列出的尺寸可以直接对照。
    """

    name: str

    # ---- 连杆几何：全尺寸 (x, y, z)，单位 m ----
    pelvis_size: Vec3
    torso_size: Vec3
    thigh_size: Vec3
    shank_size: Vec3
    foot_size: Vec3

    # ---- 关节锚点（相对各自父体原点）----
    hip_y: float  # 髋关节的 y 偏移（左右各 ±）
    hip_z: float  # 髋关节相对骨盆中心的 z 偏移（负值，在骨盆下方）
    waist_z: float  # 腰部偏航关节相对骨盆中心的 z 偏移（正值，在骨盆上方）
    torso_center_z: float  # 躯干几何中心相对腰部关节的 z 偏移
    foot_center: Vec3  # 足部几何中心相对踝关节的偏移

    # ---- 关节定义 ----
    waist_joint: JointSpec
    hip_roll: JointSpec
    hip_pitch: JointSpec
    knee: JointSpec
    ankle: JointSpec

    # ---- 材料与姿态 ----
    density: float = 800.0
    stand: tuple[float, float, float] = (0.06, -0.12, 0.06)  # (髋俯仰, 膝, 踝)

    # ------------------------------------------------------------------
    # 派生量
    # ------------------------------------------------------------------
    @property
    def thigh_len(self) -> float:
        """髋到膝的竖直距离。"""
        return self.thigh_size[2]

    @property
    def shank_len(self) -> float:
        """膝到踝的竖直距离。"""
        return self.shank_size[2]

    @property
    def leg_len(self) -> float:
        """髋到踝的竖直距离（伸直时）。"""
        return self.thigh_len + self.shank_len

    @property
    def leg_geoms(self) -> tuple[tuple[str, Vec3, float], ...]:
        """腿部三个连杆的 (名称, 全尺寸, 密度)，供质量估算与校验使用。"""
        return (
            ("thigh", self.thigh_size, self.density),
            ("shank", self.shank_size, self.density),
            ("foot", self.foot_size, self.density),
        )

    @property
    def body_geoms(self) -> tuple[tuple[str, Vec3, float], ...]:
        """全部连杆的 (名称, 全尺寸, 密度)，顺序与模型里的定义一致。"""
        return (
            ("pelvis", self.pelvis_size, self.density),
            ("torso", self.torso_size, self.density),
        ) + self.leg_geoms * 2

    @property
    def n_bodies(self) -> int:
        """body 数量：骨盆 + 躯干 + 左右各 3 段。"""
        return 2 + 2 * 3

    def actuated(self) -> list[tuple[JointSpec, str | None]]:
        """按 MuJoCo qpos 顺序返回 (关节参数, 侧别) 列表。

        顺序即模型世界体的定义顺序（深度优先）：
        ``waist_yaw`` → 左腿 4 个 → 右腿 4 个。
        """
        out: list[tuple[JointSpec, str | None]] = [(self.waist_joint, None)]
        for side in ("L", "R"):
            for joint in (self.hip_roll, self.hip_pitch, self.knee, self.ankle):
                out.append((joint, side))
        return out

    @property
    def joint_names(self) -> list[str]:
        """按 qpos 顺序排列的受驱动关节名（不含自由关节 ``root``）。"""
        return [joint_name(js, side) for js, side in self.actuated()]

    @property
    def nq(self) -> int:
        """广义坐标数量：自由关节 7 + 受驱动关节 9。"""
        return 7 + len(self.actuated())

    @property
    def nv(self) -> int:
        """广义速度数量：自由关节 6 + 受驱动关节 9。"""
        return 6 + len(self.actuated())

    @property
    def nu(self) -> int:
        """执行器数量。"""
        return len(self.actuated())

    def stand_q(self) -> dict[str, float]:
        """标称站立姿态的关节角字典（腰为 0，两腿用同一个 stand 三元组）。"""
        hip_pitch, knee, ankle = self.stand
        q: dict[str, float] = {self.waist_joint.name: 0.0}
        for side in ("L", "R"):
            q[f"{self.hip_roll.name}_{side}"] = 0.0
            q[f"{self.hip_pitch.name}_{side}"] = hip_pitch
            q[f"{self.knee.name}_{side}"] = knee
            q[f"{self.ankle.name}_{side}"] = ankle
        return q

    def total_mass(self) -> float:
        """按 box 体积 × 密度估算的总质量（与 MuJoCo 的算法一致）。"""
        return sum(
            size[0] * size[1] * size[2] * density for _, size, density in self.body_geoms
        )


def joint_name(joint: JointSpec, side: str | None) -> str:
    """把关节基名和侧别拼成模型里的关节名。"""
    return joint.name if side is None else f"{joint.name}_{side}"


def _joint(
    name: str,
    axis: Vec3,
    rng: tuple[float, float],
    kp: float,
    kv: float,
    forcerange: float,
    mirror_x: bool = False,
    mirror_z: bool = False,
) -> JointSpec:
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


# ----------------------------------------------------------------------
# 默认机型：约 0.31 m 高、约 1.5 kg 的桌面双足机器人
# ----------------------------------------------------------------------
DEFAULT_SPEC = RobotSpec(
    name="desktop_biped",
    pelvis_size=(0.09, 0.12, 0.04),
    torso_size=(0.08, 0.11, 0.09),
    thigh_size=(0.040, 0.040, 0.08),
    shank_size=(0.036, 0.036, 0.08),
    foot_size=(0.10, 0.05, 0.02),
    hip_y=0.035,
    hip_z=-0.02,
    waist_z=0.02,
    torso_center_z=0.045,
    foot_center=(0.02, 0.0, -0.01),
    waist_joint=_joint("waist_yaw", (0.0, 0.0, 1.0), (-1.57, 1.57), 25.0, 1.0, 2.5),
    hip_roll=_joint(
        "hip_roll", (1.0, 0.0, 0.0), (-0.35, 0.35), 25.0, 1.0, 2.0, mirror_x=True
    ),
    hip_pitch=_joint("hip_pitch", (0.0, -1.0, 0.0), (-0.9, 1.2), 25.0, 1.0, 2.5),
    knee=_joint("knee", (0.0, -1.0, 0.0), (-1.6, 0.05), 25.0, 1.0, 2.5),
    ankle=_joint("ankle", (0.0, -1.0, 0.0), (-0.6, 0.8), 25.0, 1.0, 2.0),
    density=800.0,
    stand=(0.06, -0.12, 0.06),
)
