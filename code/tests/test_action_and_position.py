"""动作滤波 / 位置环的回归测试（对应抖动 + 位置控制改进）。"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="未安装 mujoco")

from dino_wbc import config  # noqa: E402
from dino_wbc.action import ActionFilter  # noqa: E402
from dino_wbc.env import DinoWbcEnv  # noqa: E402
from dino_wbc.observation import make_command  # noqa: E402
from dino_wbc.position_loop import PositionLoop  # noqa: E402
from dino_wbc.rewards import compute_reward  # noqa: E402
from dino_wbc.state import read_state  # noqa: E402


# ----------------------------------------------------------------------
# 动作滤波
# ----------------------------------------------------------------------
def test_action_filter_ema_and_reset():
    filt = ActionFilter(alpha=0.5, num_joints=3)
    filt.reset()
    out = filt.update(np.array([1.0, 1.0, 1.0]))
    assert np.allclose(out, [0.5, 0.5, 0.5])
    out = filt.update(np.array([1.0, 1.0, 1.0]))
    assert np.allclose(out, [0.75, 0.75, 0.75])
    filt.reset()
    assert np.allclose(filt.state, 0.0)

    passthrough = ActionFilter(alpha=1.0, num_joints=3)
    raw = np.array([-1.0, 0.2, 0.9])
    assert np.allclose(passthrough.update(raw), raw)


def test_env_applies_action_filter_to_prev_action():
    env = DinoWbcEnv("stand", seed=0)
    env.reset(seed=0)
    action = np.ones(config.ACTION_DIM, dtype=np.float32)
    env.step(action)
    expected = np.full(config.ACTION_DIM, config.ACTION_FILTER_ALPHA)
    assert np.allclose(env.obs_builder.prev_action, expected)
    assert np.allclose(env.action_filter.state, expected)
    env.close()


def test_action_jerk_term():
    env = DinoWbcEnv("stand", seed=1)
    env.reset(seed=1)
    state = read_state(env.model, env.data, env.layout)
    command = make_command("stand")
    zeros = np.zeros(config.ACTION_DIM)
    ones = np.ones(config.ACTION_DIM)
    _, terms = compute_reward(
        "stand",
        state,
        zeros,
        ones,
        command,
        env.layout,
        target_xy=env._target_xy,
        prev_prev_action=zeros,
    )
    env.close()
    # mean((0 - 2*1 + 0)^2) = 4
    assert terms["action_jerk"] == pytest.approx(4.0)


# ----------------------------------------------------------------------
# 外层位置环
# ----------------------------------------------------------------------
def test_position_loop_gains_and_deadband():
    loop = PositionLoop()
    loop.reset((0.2, 0.0))
    vx, vy = loop.update(np.array([0.0, 0.0]), 0.0)
    assert vx == pytest.approx(config.POSITION_LOOP_MAX_VX)
    assert abs(vy) < 1e-12

    loop.reset((0.0, 0.0))
    vx, vy = loop.update(np.array([0.010, 0.0]), 0.0)
    assert vx == 0.0 and vy == 0.0, "死区内应输出 0"


def test_position_loop_rotates_to_body_frame():
    loop = PositionLoop()
    loop.reset((0.2, 0.0))
    # 机器人朝 +y（yaw=90°），世界系 +x 目标在它的"右侧"→ vy 为负
    vx, vy = loop.update(np.array([0.0, 0.0]), np.pi / 2.0)
    assert abs(vx) < 1e-9
    assert vy == pytest.approx(-config.POSITION_LOOP_MAX_VY)


def test_position_loop_clamps_and_disables():
    loop = PositionLoop()
    loop.reset((10.0, 10.0))
    vx, vy = loop.update(np.array([0.0, 0.0]), 0.0)
    assert vx == pytest.approx(config.POSITION_LOOP_MAX_VX)
    assert vy == pytest.approx(config.POSITION_LOOP_MAX_VY)

    loop.disable()
    assert loop.update(np.array([0.0, 0.0]), 0.0) == (0.0, 0.0)


def test_position_loop_integral_accumulates():
    loop = PositionLoop(kp=0.0, ki=1.0)
    loop.reset((0.2, 0.0))
    v1, _ = loop.update(np.array([0.0, 0.0]), 0.0)
    v2, _ = loop.update(np.array([0.0, 0.0]), 0.0)
    assert v2 > v1 > 0.0
    # 积分限幅
    for _ in range(1000):
        loop.update(np.array([0.0, 0.0]), 0.0)
    assert np.all(np.abs(loop.integral) <= config.POSITION_LOOP_I_CLAMP + 1e-9)


def test_position_loop_clear_integral_on_deadband():
    loop = PositionLoop()
    loop.reset((0.2, 0.0))
    for _ in range(10):
        loop.update(np.array([0.0, 0.0]), 0.0)
    assert np.any(loop.integral != 0.0)
    loop.update(np.array([0.2, 0.0]), 0.0)  # 到目标
    assert np.allclose(loop.integral, 0.0)
