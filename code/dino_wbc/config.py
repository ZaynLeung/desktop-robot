"""dino_wbc 的唯一参数源。

这里集中放「训练端和运行端都必须一致」的全部常量与默认值：

* 关节顺序（必须与 ``models/dino.xml`` 的深度优先遍历一致）
* 动作尺度、观测缩放/裁剪
* 阶段（stand / lift / standup / walk）的 episode 长度、reset 方式、奖励权重
* 终止/成功判据、课程范围、reset 噪声

改这里之前先看 ``code/README.md`` 的「调参入口」一节；模型尺寸改了要重新
生成 ``models/*.xml``，并重新训练（checkpoint 里有契约哈希会拒绝加载）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ----------------------------------------------------------------------
# 关节：顺序 = body 深度优先遍历 + body 内关节定义顺序
# ----------------------------------------------------------------------
#: 18 个驱动关节。加/删关节必须同步 ``desktop_robot/dino.py`` 的
#: ``DinoSpec.actuated()`` 与 ``models/dino.xml``；``model_io`` 会逐项校验。
JOINT_NAMES: tuple[str, ...] = (
    "neck_pitch",
    "neck_yaw",
    "shoulder_pitch_L",
    "shoulder_pitch_R",
    "hip_pitch_L",
    "hip_roll_L",
    "hip_yaw_L",
    "knee_L",
    "ankle_L",
    "hip_pitch_R",
    "hip_roll_R",
    "hip_yaw_R",
    "knee_R",
    "ankle_R",
    "tail1_yaw",
    "tail2_pitch",
    "tail3_yaw",
    "tail4_pitch",
)

NUM_JOINTS: int = len(JOINT_NAMES)
ACTION_DIM: int = NUM_JOINTS

#: 语义分组，奖励/调试时按组统计。
JOINT_GROUPS: dict[str, tuple[str, ...]] = {
    "head": ("neck_pitch", "neck_yaw"),
    "arm": ("shoulder_pitch_L", "shoulder_pitch_R"),
    "leg": (
        "hip_pitch_L",
        "hip_roll_L",
        "hip_yaw_L",
        "knee_L",
        "ankle_L",
        "hip_pitch_R",
        "hip_roll_R",
        "hip_yaw_R",
        "knee_R",
        "ankle_R",
    ),
    "tail": ("tail1_yaw", "tail2_pitch", "tail3_yaw", "tail4_pitch"),
}

#: 归一化动作 → 站姿相对目标角度的尺度（rad）。
#: 动作 ``a`` 在 [-1, 1]，目标角 = ``q_stand(0) + a * scale``，再裁到关节范围。
ACTION_SCALE_BY_JOINT: dict[str, float] = {
    "neck_pitch": 0.40,
    "neck_yaw": 0.40,
    "shoulder_pitch_L": 0.45,
    "shoulder_pitch_R": 0.45,
    "hip_pitch_L": 0.40,
    "hip_pitch_R": 0.40,
    "hip_roll_L": 0.25,
    "hip_roll_R": 0.25,
    "hip_yaw_L": 0.30,
    "hip_yaw_R": 0.30,
    "knee_L": 0.40,
    "knee_R": 0.40,
    "ankle_L": 0.25,
    "ankle_R": 0.25,
    "tail1_yaw": 0.35,
    "tail2_pitch": 0.35,
    "tail3_yaw": 0.35,
    "tail4_pitch": 0.35,
}
#: 按 ``JOINT_NAMES`` 顺序展开的尺度数组（tuple，避免在配置层引入 numpy）。
ACTION_SCALE: tuple[float, ...] = tuple(
    ACTION_SCALE_BY_JOINT[name] for name in JOINT_NAMES
)

# ----------------------------------------------------------------------
# 观测：80 维，固定解析缩放（不用 VecNormalize，保证脱离训练栈也能运行）
# ----------------------------------------------------------------------
OBS_DIM: int = 80

# 各块维度（顺序与 observation.build_obs 完全一致）
OBS_BASE_DIM = 10  # z 误差 / 投影重力(3) / 本体系线速度(3) / 本体系角速度(3)
OBS_JOINT_DIM = 2 * NUM_JOINTS  # q(18) + qd(18)
OBS_FOOT_DIM = 6  # 足高(2) / 接触(2) / 滑移(2)
OBS_ACTION_DIM = NUM_JOINTS  # 上一步动作
OBS_COMMAND_DIM = 8  # 任务 one-hot(4) + vx + vy + 抬脚高度 + 目标脚
OBS_PHASE_DIM = 2  # sin/cos 步态时钟

assert (
    OBS_BASE_DIM
    + OBS_JOINT_DIM
    + OBS_FOOT_DIM
    + OBS_ACTION_DIM
    + OBS_COMMAND_DIM
    + OBS_PHASE_DIM
    == OBS_DIM
), "观测分块维度之和必须等于 OBS_DIM"

#: 基座高度相对站姿的缩放（z 误差通常只有几厘米，放大 10 倍）。
BASE_Z_SCALE = 10.0
BASE_LIN_VEL_SCALE = 1.0
BASE_ANG_VEL_SCALE = 0.25  # qvel[3:6]，本体系
JOINT_VEL_SCALE = 0.05
FOOT_Z_SCALE = 10.0
FOOT_SLIP_SCALE = 10.0
CMD_VEL_SCALE = 10.0
LIFT_H_SCALE = 25.0

#: 观测裁剪（防止异常状态把策略推离训练分布）。
BASE_LIN_VEL_CLIP = 2.0
BASE_ANG_VEL_CLIP = 4.0
JOINT_VEL_CLIP = 20.0
FOOT_Z_CLIP = 0.25
FOOT_SLIP_CLIP = 2.0
FOOT_Z_OFFSET = 0.02  # 足底静止时足 body 原点离地约 16 mm

# ----------------------------------------------------------------------
# 指令向量
# ----------------------------------------------------------------------
#: 任务名 → one-hot 下标（顺序即 command[0:4]）。
TASK_NAMES: tuple[str, ...] = ("stand", "lift", "standup", "walk")
TASK_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(TASK_NAMES)}

CMD_STAND = 0
CMD_LIFT = 1
CMD_STANDUP = 2
CMD_WALK = 3
CMD_VX = 4
CMD_VY = 5
CMD_LIFT_H = 6
CMD_TARGET_FOOT = 7
COMMAND_DIM = OBS_COMMAND_DIM

#: 抬脚高度范围（m）；课程从下界往上推。
LIFT_H_RANGE: tuple[float, float] = (0.010, 0.040)
#: 行走速度范围（m/s）。没有腰 yaw，yaw 指令恒为 0，只做前后 + 侧向。
WALK_VX_RANGE: tuple[float, float] = (-0.10, 0.12)
WALK_VY_RANGE: tuple[float, float] = (-0.06, 0.06)

#: 步态时钟频率（Hz），只用于 walk 阶段的观测相位。
GAIT_FREQ_HZ = 1.25

# ----------------------------------------------------------------------
# 时间尺度
# ----------------------------------------------------------------------
#: 控制频率（Hz）。仿真步长 2 ms（scene 里的 <option>），10 步 = 20 ms。
CONTROL_HZ = 50.0
DECIMATION = 10
SIM_TIMESTEP = 0.002

# ----------------------------------------------------------------------
# 终止 / 成功判据
# ----------------------------------------------------------------------
FALL_BASE_Z = 0.10  # 基座低于此高度视为摔倒
FALL_TILT_DEG = 60.0  # 躯干倾角超过此值视为摔倒
SUCCESS_Z_TOL = 0.03  # standup 成功：基座高度误差
SUCCESS_TILT_DEG = 20.0
SUCCESS_HOLD_STEPS = 25  # 保持 0.5 s 才算站起来
STANDUP_TIMEOUT_STEPS = 400  # 8 s
WORLD_HALF_EXTENT = 2.0  # 4 m × 4 m 地面，滚出界算失败

#: 站姿基座高度由模型 ``stand`` 关键帧给出，不在配置里硬编码。
STAND_KEYFRAME = "stand"

# ----------------------------------------------------------------------
# reset
# ----------------------------------------------------------------------
RESET_Q_NOISE = 0.05  # rad，每个关节均匀噪声 ±
RESET_BASE_Z_NOISE = 0.005  # m
RESET_VEL_NOISE = 0.05  # m/s 与 rad/s
SETTLE_STEPS = 8  # reset 后先零控制稳定几步（16 ms），避免初始穿透

# ----------------------------------------------------------------------
# 阶段配置
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class StageConfig:
    """一个训练阶段的全部行为参数。"""

    name: str
    episode_steps: int
    reset_mode: str  # "stand" | "standup"
    reward: dict[str, float] = field(default_factory=dict)
    #: 该阶段是否推进步态时钟。
    gait: bool = False

    @property
    def duration_s(self) -> float:
        return self.episode_steps / CONTROL_HZ


STAGES: dict[str, StageConfig] = {
    "stand": StageConfig(
        name="stand",
        episode_steps=400,
        reset_mode="stand",
        reward={
            "upright": 1.0,
            "height": 1.0,
            "alive": 0.5,
            "action_mag": 0.02,
            "action_rate": 0.001,
            "base_vel": 0.5,
            "fall": -5.0,
        },
    ),
    "lift": StageConfig(
        name="lift",
        episode_steps=400,
        reset_mode="stand",
        reward={
            "upright": 1.0,
            "height": 1.0,
            "alive": 0.5,
            "lift_height": 2.0,
            "support_contact": 0.5,
            "support_slip": 0.3,
            "action_mag": 0.02,
            "action_rate": 0.001,
            "base_vel": 0.5,
            "fall": -5.0,
        },
    ),
    "standup": StageConfig(
        name="standup",
        episode_steps=STANDUP_TIMEOUT_STEPS,
        reset_mode="standup",
        reward={
            "progress": 5.0,
            "success": 5.0,
            "foot_contact": 0.2,
            "time": -0.02,
            "action_mag": 0.02,
            "fall": -5.0,
        },
    ),
    "walk": StageConfig(
        name="walk",
        episode_steps=1000,
        reset_mode="stand",
        gait=True,
        reward={
            "vel_track": 2.0,
            "upright": 0.6,
            "height": 0.6,
            "foot_clearance": 0.3,
            "contact_alt": 0.3,
            "heading": 0.3,
            "action_mag": 0.02,
            "action_rate": 0.002,
            "fall": -5.0,
        },
    ),
}

# ----------------------------------------------------------------------
# standup 起始姿态（课程）
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class StandupPose:
    """翻正起身的起始姿态。

    ``roll_deg`` 是绕世界 x 轴的滚转角：0 = 正常立姿（深蹲），90 = 侧躺，
    180 = 仰卧。``joints`` 是站姿相对关节角的初值；基座高度不在这里写死，
    环境 reset 时会先放到 ``base_z_hint`` 再让物理自己落到地面。
    """

    name: str
    roll_deg: float
    base_z_hint: float
    joints: dict[str, float] = field(default_factory=dict)


#: 深蹲：脚仍着地，是「起身」的入门难度。
#: 实测（settle 后）基座 z≈0.134 m、倾角≈15°，低于成功判据的 0.154 m，
#: 因此不会被误判成"已经站起来"。
_CROUCH_JOINTS = {
    "hip_pitch_L": 1.10,
    "hip_pitch_R": 1.10,
    "knee_L": -1.45,
    "knee_R": -1.45,
    "ankle_L": 0.50,
    "ankle_R": 0.50,
    "shoulder_pitch_L": -0.35,
    "shoulder_pitch_R": -0.35,
}

#: 侧躺/仰卧时把腿收起来，减少初始接触面积、方便翻滚。
_TUCK_JOINTS = {
    "hip_pitch_L": 0.90,
    "hip_pitch_R": 0.90,
    "knee_L": -1.20,
    "knee_R": -1.20,
    "ankle_L": 0.20,
    "ankle_R": 0.20,
    "hip_roll_L": 0.10,
    "hip_roll_R": 0.10,
    "shoulder_pitch_L": -0.40,
    "shoulder_pitch_R": -0.40,
    "tail2_pitch": 0.30,
    "tail4_pitch": 0.25,
}

STANDUP_POSES: tuple[StandupPose, ...] = (
    StandupPose("crouch", 0.0, 0.13, _CROUCH_JOINTS),
    StandupPose("side", 90.0, 0.09, _TUCK_JOINTS),
    StandupPose("supine", 180.0, 0.085, _TUCK_JOINTS),
)

#: 课程难度顺序：先深蹲，再侧躺，最后仰卧；成功率够了才放开下一档。
STANDUP_CURRICULUM: tuple[str, ...] = ("crouch", "side", "supine")
STANDUP_PROMOTE_SUCCESS = 0.70  # 连续成功率阈值
STANDUP_PROMOTE_WINDOW = 20  # 统计窗口（episode 数）
#: 低于这个比例就退回上一档（避免在某一档卡死）。
STANDUP_DEMOTE_SUCCESS = 0.30

# ----------------------------------------------------------------------
# 课程
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CurriculumRange:
    """随成功率从 ``start`` 线性推到 ``end`` 的参数范围。"""

    start: float
    end: float


CURRICULUM: dict[str, dict[str, CurriculumRange]] = {
    "lift": {
        "lift_h": CurriculumRange(LIFT_H_RANGE[0], LIFT_H_RANGE[1]),
    },
    "walk": {
        "vx": CurriculumRange(0.03, max(abs(v) for v in WALK_VX_RANGE)),
        "vy": CurriculumRange(0.015, max(abs(v) for v in WALK_VY_RANGE)),
    },
}
CURRICULUM_WINDOW = 20
CURRICULUM_SUCCESS = 0.60

# ----------------------------------------------------------------------
# 奖励/终止共用的判据
# ----------------------------------------------------------------------
#: 支撑/接触判据：足或前趾有接触力即视为接触。
CONTACT_FORCE_EPS = 1.0  # N，低于此值的接触忽略
#: walk 阶段期望的摆动足离地高度（m）与步频。
WALK_FOOT_CLEARANCE = 0.020

# ----------------------------------------------------------------------
# 运行时
# ----------------------------------------------------------------------
RENDER_WIDTH = 480
RENDER_HEIGHT = 360
RENDER_FPS = 30
FOLLOW_DISTANCE = 0.55
FOLLOW_AZIMUTH = 135.0
FOLLOW_ELEVATION = -25.0
FOLLOW_LOOKAT_ALPHA = 0.15  # 跟随相机 lookat 的指数平滑系数
HEAD_CAMERA = "cam_head"
