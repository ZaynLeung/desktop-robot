"""dino_wbc 的唯一参数源。

这里集中放「训练端和运行端都必须一致」的全部常量与默认值：

* 关节顺序（必须与 ``models/dino.xml`` 的深度优先遍历一致）
* 动作尺度、观测缩放/裁剪
* 阶段（stand / lift / standup / walk / turn）的 episode 长度、reset 方式、奖励权重
* 终止/成功判据、课程范围、reset 噪声

改这里之前先看 ``code/README.md`` 的「调参入口」一节；模型尺寸改了要重新
生成 ``models/*.xml``，并重新训练（checkpoint 里有契约哈希会拒绝加载）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

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
# 观测：85 维，固定解析缩放（不用 VecNormalize，保证脱离训练栈也能运行）
# ----------------------------------------------------------------------
OBS_DIM: int = 85

# 各块维度（顺序与 observation.build_obs 完全一致）
OBS_BASE_DIM = 13  # z 误差 / 重力(3) / 线速度(3) / 角速度(3) / 朝向误差 / 位置误差 xy
OBS_JOINT_DIM = 2 * NUM_JOINTS  # q(18) + qd(18)
OBS_FOOT_DIM = 6  # 足高(2) / 接触(2) / 滑移(2)
OBS_ACTION_DIM = NUM_JOINTS  # 上一步动作
OBS_COMMAND_DIM = 10  # 任务 one-hot(5) + vx + vy + 抬脚高度 + 目标脚 + 朝向指令
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
#: 绝对朝向误差/指令：除以 π 归一到 [-1, 1]。
HEADING_SCALE = 1.0 / math.pi
#: 位置误差（m）：放大 10 倍并裁剪到 ±1（对应 ±0.1 m）。
POSITION_SCALE = 10.0
POSITION_CLIP = 1.0

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
#: 任务名 → one-hot 下标（顺序即 command[0:5]）。
TASK_NAMES: tuple[str, ...] = ("stand", "lift", "standup", "walk", "turn")
TASK_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(TASK_NAMES)}

CMD_STAND = 0
CMD_LIFT = 1
CMD_STANDUP = 2
CMD_WALK = 3
CMD_TURN = 4
CMD_VX = 5
CMD_VY = 6
CMD_LIFT_H = 7
CMD_TARGET_FOOT = 8
CMD_YAW = 9
COMMAND_DIM = OBS_COMMAND_DIM

#: 绝对朝向指令范围（rad）：stand 固定 0，turn 课程从下界推到上界。
TURN_YAW_RANGE: tuple[float, float] = (0.30, math.pi)
#: turn 阶段判定"跟住朝向"的门限与保持步数。
TURN_HOLD_HEADING_DEG = 10.0
TURN_HOLD_POSITION = 0.05  # m
TURN_HOLD_STEPS = 25  # 0.5 s
#: turn 课程晋级/退档步长。
CURRICULUM_TURN_UP = 0.03
CURRICULUM_TURN_DOWN = 0.01

#: 抬脚高度范围（m）；课程从下界往上推。
LIFT_H_RANGE: tuple[float, float] = (0.010, 0.040)
#: 行走速度范围（m/s）：前后（vx 正=前进、负=倒车）+ 左右（vy 正=左、负=右）。
WALK_VX_RANGE: tuple[float, float] = (-0.10, 0.12)
WALK_VY_RANGE: tuple[float, float] = (-0.06, 0.06)

#: 步态时钟频率（Hz），只用于 walk 阶段的观测相位。
GAIT_FREQ_HZ = 1.25

# ---- walk v2 训练参数 ------------------------------------------------------
# 旧版 vel_track 用 exp(-4 e²)：1/e 误差 ~0.5 m/s，而指令只有 0.12 m/s，
# 奖励几乎饱和（命令 0.12、实际 0 也能拿 0.94），策略没有动力精确跟踪/刹车。
# 改成"速度误差 σ"形式，σ 与指令范围同量级。
WALK_VEL_SIGMA: tuple[float, float] = (0.05, 0.04)  # (vx, vy) 的 1/e 误差
#: 训练时每 1–3 s 换一次速度指令（含停车），让策略学会加速/减速/倒车/侧移切换。
WALK_COMMAND_HOLD_RANGE: tuple[int, int] = (50, 150)  # 控制步（20 ms/步）
#: 换指令时有多大概率直接给 (0, 0)，专门训练"停住"。
WALK_STOP_PROB = 0.20
#: 摆动脚触地惩罚的阈值（foot_z 低于它算没抬起来）。
WALK_SWING_CONTACT_Z = 0.005
#: walk 课程"成功"的平均 vel_track 阈值。新核对 5 cm/s 误差就掉到 0.37，
#: 所以 0.85 意味着平均跟踪误差约 2 cm/s；旧阈值 0.60 会让"站着不动"也过关。
WALK_SUCCESS_VEL_TRACK = 0.85

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
FALL_BASE_Z = 0.10  # walk 用的全局摔倒高度（stand / lift 用更严格的阶段门限）
FALL_TILT_DEG = 60.0  # walk 用的全局摔倒倾角
SUCCESS_Z_TOL = 0.03  # standup 成功：基座高度误差
SUCCESS_TILT_DEG = 20.0
SUCCESS_HOLD_STEPS = 25  # 保持 0.5 s 才算站起来
STANDUP_TIMEOUT_STEPS = 400  # 8 s
WORLD_HALF_EXTENT = 2.0  # 4 m × 4 m 地面，滚出界算失败

# ---- stand / lift 的严格门限 ------------------------------------------------
# 诊断结论：旧策略在 z=0.113 m、tilt≈48°、躯干撑地 211/400 步的情况下仍"存活"，
# 因为全局门限是 z<0.10 / tilt>60° 且不检查躯干接触。下面这组门限专门堵住该局部最优。
STAND_FALL_Z_DELTA = 0.06  # z < z_stand - 0.06 即终止
STAND_FALL_TILT_DEG = 40.0
LIFT_FALL_Z_DELTA = 0.06
LIFT_FALL_TILT_DEG = 40.0
LIFT_STANCE_Z_DELTA = 0.03  # 抬脚奖励/课程要求的姿态门限：z > z_stand - 0.03
LIFT_STANCE_TILT_DEG = 25.0
LIFT_HOLD_STEPS = 25  # 连续保持 0.5 s 才算"抬脚成功"（不是累计）
LIFT_HEIGHT_RATIO = 0.8  # 达到指令高度的 80% 即算达标步

# ---- 静态站立的绝对朝向 / 位置保持 ------------------------------------------
# 用户的 stand 策略实测 400 步朝向漂了 95–137°，因为观测/奖励里没有绝对朝向。
# 下面这组门限让"偏太多"的 rollout 提前终止，配合 heading/position 奖励项。
STAND_MAX_HEADING_ERROR_DEG = 30.0
STAND_MAX_POSITION_ERROR = 0.10  # m
LIFT_MAX_HEADING_ERROR_DEG = 30.0
LIFT_MAX_POSITION_ERROR = 0.15
TURN_MAX_POSITION_ERROR = 0.12
#: 朝向/位置违规需要连续多少步才终止（避免瞬时尖峰误杀）。
POSITION_GRACE_STEPS = 5

# ----------------------------------------------------------------------
# 外力轻推（stand 阶段的扰动课程）
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PushConfig:
    """周期性水平轻推，用于训练"被推一下也不倒"的站立策略。"""

    enabled: bool = False
    max_force: float = 0.0  # 课程上限（N）；实际力在 [0.5, 1.0]×上限 采样
    duration_steps: int = 5  # 5 × 20 ms = 0.1 s
    interval_steps: tuple[int, int] = (75, 150)  # 每 1.5–3 s 一次

PUSH_RAMP_UP = 0.2  # 成功一次 +0.2 N
PUSH_RAMP_DOWN = 0.1  # 失败一次 -0.1 N
PUSH_MIN_FORCE = 0.5  # 采样下限（× 当前上限）

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
    #: 阶段专属摔倒门限（None = 用全局 FALL_*）。
    fall_z_delta: float | None = None  # 相对站姿基座高度的下降量（m）
    fall_tilt_deg: float | None = None
    #: 躯干/头/臂/腿（不含尾巴）接触地板时是否终止。
    terminate_on_body_contact: bool = False
    #: 尾巴接触地板时是否终止（stand/lift/turn 为 True：尾巴不能当支撑点）。
    terminate_on_tail_contact: bool = False
    #: 绝对朝向误差/位置误差超过门限连续 ``heading_grace_steps`` 步就终止。
    max_heading_error_deg: float | None = None
    max_position_error_m: float | None = None
    heading_grace_steps: int = POSITION_GRACE_STEPS
    #: 外力轻推课程；None = 不施加。
    push: PushConfig | None = None

    @property
    def duration_s(self) -> float:
        return self.episode_steps / CONTROL_HZ


STAGES: dict[str, StageConfig] = {
    "stand": StageConfig(
        name="stand",
        episode_steps=400,
        reset_mode="stand",
        fall_z_delta=STAND_FALL_Z_DELTA,
        fall_tilt_deg=STAND_FALL_TILT_DEG,
        terminate_on_body_contact=True,
        terminate_on_tail_contact=True,
        max_heading_error_deg=STAND_MAX_HEADING_ERROR_DEG,
        max_position_error_m=STAND_MAX_POSITION_ERROR,
        push=PushConfig(
            enabled=True,
            max_force=5.0,
            duration_steps=5,
            interval_steps=(75, 150),
        ),
        reward={
            "alive": 0.3,
            "height": 1.5,
            "upright": 1.5,
            "heading": 2.0,
            "position": 1.5,
            "base_vel": 0.5,
            "yaw_rate": 0.2,
            "foot_contact": 0.3,
            "foot_slip": 0.3,
            "action_mag": 0.02,
            "action_rate": 0.001,
            "joint_posture": -0.05,
            "low_posture": -2.0,
            "body_contact": -5.0,
            "tail_contact": -0.2,
            "fall": -5.0,
        },
    ),
    "lift": StageConfig(
        name="lift",
        episode_steps=400,
        reset_mode="stand",
        fall_z_delta=LIFT_FALL_Z_DELTA,
        fall_tilt_deg=LIFT_FALL_TILT_DEG,
        terminate_on_body_contact=True,
        terminate_on_tail_contact=True,
        max_heading_error_deg=LIFT_MAX_HEADING_ERROR_DEG,
        max_position_error_m=LIFT_MAX_POSITION_ERROR,
        reward={
            "alive": 0.2,
            "height": 1.0,
            "upright": 1.0,
            "heading": 0.5,
            "position": 0.5,
            "base_vel": 0.5,
            "yaw_rate": 0.1,
            "foot_slip": 0.3,
            "action_mag": 0.02,
            "action_rate": 0.001,
            "joint_posture": -0.05,
            "low_posture": -2.0,
            "body_contact": -5.0,
            "tail_contact": -0.2,
            "fall": -5.0,
            # 抬脚专属项（只有 stance gate 成立时才给）
            "lift_height": 2.0,
            "lift_frac": 2.0,
            "target_ground": -0.5,
            "support_contact": 0.5,
            "support_slip": 0.3,
            "com_over_support": 0.2,
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
            # 速度跟踪（新核，见 WALK_VEL_SIGMA）
            "vel_track": 3.0,
            "upright": 0.6,
            "height": 0.6,
            # 抬脚/步态：抬高权重，惩罚拖脚和摆动脚触地
            "foot_clearance": 1.0,
            "contact_alt": 0.3,
            "swing_contact": -0.5,
            "stance_slip": 0.3,
            "foot_slip": 0.2,
            # 朝向/平滑/安全
            "heading": 0.5,
            "yaw_rate": 0.1,
            "action_mag": 0.02,
            "action_rate": 0.002,
            "fall": -5.0,
        },
    ),
    "turn": StageConfig(
        name="turn",
        episode_steps=500,
        reset_mode="stand",
        fall_z_delta=STAND_FALL_Z_DELTA,
        fall_tilt_deg=STAND_FALL_TILT_DEG,
        terminate_on_body_contact=True,
        terminate_on_tail_contact=True,
        # 不按朝向误差终止：转不到目标正是要学的；只限制别走位/别摔。
        max_position_error_m=TURN_MAX_POSITION_ERROR,
        reward={
            "alive": 0.3,
            "height": 1.5,
            "upright": 1.5,
            "heading": 3.0,
            "position": 1.0,
            "base_vel": 0.5,
            "yaw_rate": 0.1,
            "action_mag": 0.02,
            "action_rate": 0.001,
            "joint_posture": -0.05,
            "low_posture": -2.0,
            "body_contact": -5.0,
            "tail_contact": -0.2,
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
    "turn": {
        "yaw": CurriculumRange(TURN_YAW_RANGE[0], TURN_YAW_RANGE[1]),
    },
}
CURRICULUM_WINDOW = 20
CURRICULUM_SUCCESS = 0.60
#: lift 课程晋级/退档步长（比之前慢：必须稳定连续保持才允许加高）。
CURRICULUM_LIFT_UP = 0.03
CURRICULUM_LIFT_DOWN = 0.01

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
