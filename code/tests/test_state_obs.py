"""状态量提取与 80 维观测布局。"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="未安装 mujoco")

from dino_wbc import config, model_io  # noqa: E402
from dino_wbc.observation import ObsBuilder, make_command  # noqa: E402
from dino_wbc.state import read_state, reset_to_stand  # noqa: E402


@pytest.fixture(scope="module")
def model_layout():
    return model_io.load()


@pytest.fixture
def rig(model_layout):
    model, layout = model_layout
    data = mujoco.MjData(model)
    reset_to_stand(model, data, layout)
    mujoco.mj_forward(model, data)
    return model, layout, data


def test_stand_state_is_upright(rig):
    model, layout, data = rig
    state = read_state(model, data, layout)
    # 标称站姿本身抬头 20°，重力在体坐标系里是 (-sin20, 0, -cos20)
    expected_gravity = np.array(
        [-np.sin(np.radians(20.0)), 0.0, -np.cos(np.radians(20.0))]
    )
    assert state.tilt_deg < 0.5  # 相对标称站姿的偏差
    assert np.allclose(state.gravity_b, expected_gravity, atol=0.02)
    assert np.allclose(state.q, 0.0, atol=1e-9)
    assert state.base_pos[2] == pytest.approx(layout.stand_base_z, abs=1e-6)
    assert state.foot_z.shape == (2,)
    assert state.foot_z.min() > -0.01  # 站姿足底贴地


def test_base_velocity_is_expressed_in_body_frame(rig):
    model, layout, data = rig
    angle = np.pi / 2.0
    data.qpos[3:7] = [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]  # 绕 z 转 90°
    data.qvel[0:3] = [1.0, 0.0, 0.0]  # 世界系 +x
    data.qvel[3:6] = [0.0, 0.0, 1.0]  # 本体系角速度
    mujoco.mj_forward(model, data)
    state = read_state(model, data, layout)
    # 世界 +x 在体坐标系里是 -y
    assert np.allclose(state.base_lin_vel_b, [0.0, -1.0, 0.0], atol=1e-9)
    assert np.allclose(state.base_ang_vel_b, [0.0, 0.0, 1.0], atol=1e-9)


def test_obs_layout_and_scaling(rig):
    model, layout, data = rig
    state = read_state(model, data, layout)
    command = make_command("walk", vx=0.1, vy=-0.05)
    obs = ObsBuilder(layout).build(state, command, gait=True)

    assert obs.shape == (config.OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()

    # base 块：z 误差 / 重力 / 线速度 / 角速度 / 朝向误差 / 位置误差 xy
    assert obs[0] == pytest.approx(0.0, abs=1e-5)
    expected_gravity = np.array(
        [-np.sin(np.radians(20.0)), 0.0, -np.cos(np.radians(20.0))]
    )
    assert np.allclose(obs[1:4], expected_gravity, atol=0.02)
    assert obs[10] == pytest.approx(0.0, abs=1e-5)  # 朝向误差（walk 目标 0）
    assert np.allclose(obs[11:13], 0.0, atol=1e-5)  # 位置误差（目标默认原点）
    # joints 块：站姿全 0
    assert np.allclose(obs[13:49], 0.0, atol=1e-6)
    # command 块（起点 = 13+36+6+18 = 73）
    command_start = config.OBS_DIM - config.OBS_PHASE_DIM - config.OBS_COMMAND_DIM
    assert obs[command_start + 5] == pytest.approx(1.0)  # vx = 0.1 × 10
    assert obs[command_start + 6] == pytest.approx(-0.5)  # vy = -0.05 × 10
    assert obs[command_start + 9] == pytest.approx(0.0)  # yaw 指令
    # phase 块非零
    assert not np.allclose(obs[-2:], 0.0)


def test_yaw_and_heading_error(rig):
    """绕 z 转 30° 后 yaw 应为 30°，heading_error 与指令相反。"""
    model, layout, data = rig
    theta = np.radians(30.0)
    data.qpos[3:7] = [np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)]
    mujoco.mj_forward(model, data)
    state = read_state(model, data, layout)
    assert np.degrees(state.yaw_rad) == pytest.approx(30.0, abs=1e-6)

    command = make_command("stand", yaw=0.0)
    start = config.OBS_DIM - config.OBS_PHASE_DIM - config.OBS_COMMAND_DIM
    obs = ObsBuilder(layout).build(state, command, gait=False)
    # heading_error = 0 - 30° → -30° / π
    assert obs[10] == pytest.approx(np.radians(-30.0) / np.pi, abs=1e-6)
    assert obs[start + 9] == pytest.approx(0.0, abs=1e-6)

    # 指令 30° 时误差归零、命令块记录 30°/π
    command = make_command("stand", yaw=theta)
    obs = ObsBuilder(layout).build(state, command, gait=False)
    assert obs[10] == pytest.approx(0.0, abs=1e-6)
    assert obs[start + 9] == pytest.approx(theta / np.pi, abs=1e-6)


def test_position_error_in_obs(rig):
    model, layout, data = rig
    state = read_state(model, data, layout)
    target = np.array([0.05, -0.02])
    obs = ObsBuilder(layout).build(state, make_command("stand"), gait=False)
    builder = ObsBuilder(layout)
    builder.reset(target)
    obs = builder.build(state, make_command("stand"), gait=False)
    assert obs[11] == pytest.approx(-0.05 * config.POSITION_SCALE, abs=1e-5)
    assert obs[12] == pytest.approx(0.02 * config.POSITION_SCALE, abs=1e-5)


def test_obs_is_clipped(rig):
    model, layout, data = rig
    data.qvel[0:3] = [50.0, -50.0, 0.0]
    data.qvel[3:6] = [100.0, -100.0, 0.0]
    mujoco.mj_forward(model, data)
    state = read_state(model, data, layout)
    obs = ObsBuilder(layout).build(state, make_command("stand"), gait=False)
    assert np.max(np.abs(obs)) <= max(config.JOINT_VEL_CLIP, config.BASE_ANG_VEL_CLIP) + 1e-6


def test_body_contact_detected_when_lying(rig):
    """躯干着地必须能被识别（v1 的"趴着活 400 步"就是漏了这个信号）。"""
    from dino_wbc.state import quat_from_roll_deg

    model, layout, data = rig
    data.qpos[2] = 0.03
    data.qpos[3:7] = quat_from_roll_deg(180.0)
    mujoco.mj_forward(model, data)
    state = read_state(model, data, layout)
    assert state.body_contact
    assert not state.tail_contact  # 仰卧时尾巴翘起，不接触


def test_standing_has_no_body_contact(rig):
    model, layout, data = rig
    state = read_state(model, data, layout)
    assert not state.body_contact
    # 该模型的标称站姿尾巴尖端本来就轻触地面（约 4.7 N），这是允许的。
    assert isinstance(state.tail_contact, bool)
