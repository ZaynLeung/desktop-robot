"""环境：三阶段 reset/step、确定性、动作限幅、跌倒终止。"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="未安装 mujoco")

from dino_wbc import config  # noqa: E402
from dino_wbc.env import DinoWbcEnv  # noqa: E402


@pytest.mark.parametrize("stage", tuple(config.STAGES))
def test_reset_and_step(stage):
    env = DinoWbcEnv(stage, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (config.OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert np.isfinite(info["base_z"])

    obs, reward, terminated, truncated, info = env.step(
        np.zeros(config.ACTION_DIM, dtype=np.float32)
    )
    assert obs.shape == (config.OBS_DIM,)
    assert np.isfinite(obs).all()
    assert np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert info["stage"] == stage
    env.close()


def test_reset_is_deterministic_given_seed():
    first = DinoWbcEnv("lift", seed=7)
    second = DinoWbcEnv("lift", seed=7)
    obs_a, _ = first.reset(seed=7)
    obs_b, _ = second.reset(seed=7)
    assert np.allclose(obs_a, obs_b)
    assert np.allclose(first.command, second.command)
    first.close()
    second.close()


def test_action_is_clipped_to_joint_ranges():
    env = DinoWbcEnv("stand", seed=1)
    env.reset(seed=1)
    env.step(np.full(config.ACTION_DIM, 100.0, dtype=np.float32))
    ctrl = np.asarray(env.data.ctrl)
    assert np.all(ctrl >= env.layout.ranges[:, 0] - 1e-9)
    assert np.all(ctrl <= env.layout.ranges[:, 1] + 1e-9)
    env.close()


def test_falling_terminates_the_episode():
    env = DinoWbcEnv("stand", seed=2)
    env.reset(seed=2)
    env.data.qpos[2] = 0.02
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, _, info = env.step(
        np.zeros(config.ACTION_DIM, dtype=np.float32)
    )
    assert terminated
    assert info["fallen"]
    env.close()


def test_standup_starts_from_crouch():
    env = DinoWbcEnv("standup", seed=3)
    _, info = env.reset(seed=3)
    assert env.curriculum.state().pose == "crouch"
    # 深蹲档位仍然是接近竖直的，只是低
    assert info["tilt_deg"] < 40.0
    assert info["base_z"] < env.layout.stand_base_z
    env.close()


def test_command_can_be_overridden():
    env = DinoWbcEnv("walk", seed=4)
    env.reset(seed=4)
    from dino_wbc.observation import make_command

    env.set_command(make_command("walk", vx=0.08, vy=-0.02))
    assert env.command[config.CMD_VX] == pytest.approx(0.08)
    assert env.command[config.CMD_VY] == pytest.approx(-0.02)
    env.close()
