"""v2.2 新增能力的回归测试：绝对朝向/位置保持、尾巴离地、turn 阶段。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="未安装 mujoco")

from dino_wbc import config  # noqa: E402
from dino_wbc.curriculum import Curriculum  # noqa: E402
from dino_wbc.env import DinoWbcEnv  # noqa: E402
from dino_wbc.observation import make_command  # noqa: E402
from dino_wbc.rewards import compute_reward  # noqa: E402
from dino_wbc.state import read_state  # noqa: E402


def _zeros():
    return np.zeros(config.ACTION_DIM, dtype=np.float32)


def _yaw_quat(deg: float) -> list[float]:
    angle = np.radians(deg) * 0.5
    return [float(np.cos(angle)), 0.0, 0.0, float(np.sin(angle))]


def test_stand_reset_has_tail_clear():
    """新站姿尾巴抬高：reset 时尾巴不应触地。"""
    env = DinoWbcEnv("stand", seed=0)
    _, info = env.reset(seed=0)
    env.close()
    assert info["tail_contact"] is False
    assert info["body_contact"] is False


def test_tail_contact_terminates():
    env = DinoWbcEnv("stand", seed=1)
    env.reset(seed=1)
    idx2 = env.layout.names.index("tail2_pitch")
    idx4 = env.layout.names.index("tail4_pitch")
    env.data.qpos[env.layout.qposadr[idx2]] = 0.55
    env.data.qpos[env.layout.qposadr[idx4]] = 0.50
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, _, info = env.step(_zeros())
    env.close()
    assert terminated
    assert info["tail_contact"]


def test_heading_drift_terminates():
    env = DinoWbcEnv("stand", seed=2)
    env.reset(seed=2)
    env.data.qpos[3:7] = _yaw_quat(45.0)
    mujoco.mj_forward(env.model, env.data)
    terminated = False
    for _ in range(12):
        _, _, terminated, truncated, info = env.step(_zeros())
        if terminated or truncated:
            break
    env.close()
    assert terminated
    assert abs(info["heading_error_deg"]) > config.STAND_MAX_HEADING_ERROR_DEG


def test_position_drift_terminates():
    env = DinoWbcEnv("stand", seed=3)
    env.reset(seed=3)
    env.data.qpos[0] += 0.25
    mujoco.mj_forward(env.model, env.data)
    terminated = False
    for _ in range(12):
        _, _, terminated, truncated, info = env.step(_zeros())
        if terminated or truncated:
            break
    env.close()
    assert terminated
    assert info["position_error"] > config.STAND_MAX_POSITION_ERROR


def test_turn_stage_reset_and_step():
    env = DinoWbcEnv("turn", seed=4)
    obs, info = env.reset(seed=4)
    assert obs.shape == (config.OBS_DIM,)
    assert info["command"][config.CMD_TURN] == 1.0
    assert abs(info["yaw_cmd_deg"]) <= np.degrees(config.TURN_YAW_RANGE[0]) + 1e-6
    obs, reward, terminated, truncated, info = env.step(_zeros())
    env.close()
    assert np.isfinite(obs).all()
    assert np.isfinite(reward)
    assert "heading_error_deg" in info


def test_turn_curriculum_ramps_yaw():
    curriculum = Curriculum("turn", rng=np.random.default_rng(0))
    start = curriculum.yaw_max
    assert start == pytest.approx(config.TURN_YAW_RANGE[0])
    for _ in range(20):
        curriculum.on_episode(True)
    assert curriculum.yaw_max > start
    assert curriculum.yaw_max <= config.TURN_YAW_RANGE[1] + 1e-9
    assert abs(curriculum.sample_yaw()) <= curriculum.yaw_max + 1e-9
    assert "max_yaw" in curriculum.state().__dict__


def test_heading_and_position_reward_terms():
    env = DinoWbcEnv("stand", seed=5)
    env.reset(seed=5)
    # reset 现在会随机挪开基座（位置恢复训练）；先把机器人放回目标点再断言。
    env.data.qpos[0:2] = env._target_xy
    mujoco.mj_forward(env.model, env.data)
    state = read_state(env.model, env.data, env.layout)
    command = make_command("stand", yaw=0.0)
    action = np.zeros(config.ACTION_DIM)
    _, terms = compute_reward(
        "stand",
        state,
        action,
        action,
        command,
        env.layout,
        target_xy=env._target_xy,
    )
    assert terms["heading"] > 0.99
    assert terms["position"] > 0.99

    env.data.qpos[3:7] = _yaw_quat(45.0)
    env.data.qpos[0] += 0.20
    mujoco.mj_forward(env.model, env.data)
    moved = read_state(env.model, env.data, env.layout)
    _, terms = compute_reward(
        "stand",
        moved,
        action,
        action,
        command,
        env.layout,
        target_xy=env._target_xy,
    )
    env.close()
    assert terms["heading"] < 0.01
    assert terms["position"] < 0.01


def test_stand_success_requires_heading_and_position():
    """把朝向/位置项人为清零时，stand 不应判成功。"""
    env = DinoWbcEnv("stand", seed=6)
    env.reset(seed=6)
    env._height_term_sum = 400.0
    env._heading_term_sum = 0.0
    env._position_term_sum = 400.0
    env._step_count = 400
    assert not env._episode_outcome(False)
    env._heading_term_sum = 400.0
    assert env._episode_outcome(False)
    env.close()


def test_standup_and_walk_stages_still_build():
    for stage in ("standup", "walk"):
        env = DinoWbcEnv(stage, seed=7)
        obs, _ = env.reset(seed=7)
        env.close()
        assert obs.shape == (config.OBS_DIM,)
