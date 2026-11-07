"""walk v2 的回归测试：速度核、方向采样、命令切换、固定指令评测。"""

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


# ----------------------------------------------------------------------
# 速度核
# ----------------------------------------------------------------------
def test_walk_velocity_kernel_is_sharp():
    env = DinoWbcEnv("walk", seed=0)
    env.reset(seed=0)
    state = read_state(env.model, env.data, env.layout)
    command = make_command("walk", vx=0.10, vy=0.0)
    action = np.zeros(config.ACTION_DIM)

    def term_with(vx: float, vy: float) -> float:
        moved = replace(
            state,
            base_lin_vel_b=np.array([vx, vy, 0.0]),
            body_contact=False,
        )
        _, terms = compute_reward(
            "walk", moved, action, action, command, env.layout
        )
        return terms["vel_track"]

    env.close()
    assert term_with(0.10, 0.0) > 0.99
    # 旧核 exp(-4·0.05²)=0.99，新核应明显掉下来
    assert term_with(0.05, 0.0) < 0.45
    assert term_with(-0.05, 0.0) < 0.45


def test_walk_terms_include_swing_and_stance_quality():
    env = DinoWbcEnv("walk", seed=0)
    env.reset(seed=0)
    state = read_state(env.model, env.data, env.layout)
    command = make_command("walk", vx=0.05, vy=0.0)
    action = np.zeros(config.ACTION_DIM)
    _, terms = compute_reward(
        "walk", state, action, action, command, env.layout, target_xy=env._target_xy
    )
    env.close()
    for key in ("vel_track", "foot_clearance", "contact_alt", "swing_contact", "stance_slip"):
        assert key in terms


def test_walk_stop_gates_gait_and_rewards_holding():
    """零指令时关闭抬脚/交替接触奖励，改为奖励站住；非零指令时不出现停车项。"""
    env = DinoWbcEnv("walk", seed=8)
    env.reset(seed=8)
    state = read_state(env.model, env.data, env.layout)
    # reset 后还有残余速度；停车项要测"已经站住"的情况，先清零速度。
    from dataclasses import replace

    state = replace(state, base_lin_vel_b=np.zeros(3, dtype=np.float64))
    action = np.zeros(config.ACTION_DIM)

    def terms(vx: float, vy: float) -> dict[str, float]:
        command = make_command("walk", vx=vx, vy=vy)
        _, value = compute_reward(
            "walk",
            state,
            action,
            action,
            command,
            env.layout,
            target_xy=env._target_xy,
        )
        return value

    stop = terms(0.0, 0.0)
    moving = terms(0.05, 0.0)
    env.close()

    assert stop["foot_clearance"] == 0.0
    assert stop["contact_alt"] == 0.0
    assert stop["stop_vel"] > 0.5
    assert stop["stop_hold"] > 0.9
    assert "stop_contact" in stop

    assert moving["foot_clearance"] > 0.0
    assert moving["stop_vel"] == 0.0
    assert moving["stop_contact"] == 0.0
    assert moving["stop_hold"] == 0.0


# ----------------------------------------------------------------------
# 方向采样：前进 / 倒车 / 左 / 右 / 停车都要覆盖
# ----------------------------------------------------------------------
def test_walk_command_covers_all_directions():
    curriculum = Curriculum("walk", rng=np.random.default_rng(0))
    curriculum.vx_max = config.WALK_VX_RANGE[1]
    curriculum.vy_max = config.WALK_VY_RANGE[1]
    samples = [curriculum.sample_velocity() for _ in range(2000)]
    vx = np.array([s[0] for s in samples])
    vy = np.array([s[1] for s in samples])
    assert (vx > 0.01).any(), "没有前进"
    assert (vx < -0.01).any(), "没有倒车"
    assert (vy > 0.005).any(), "没有左移"
    assert (vy < -0.005).any(), "没有右移"
    assert ((np.abs(vx) < 1e-9) & (np.abs(vy) < 1e-9)).any(), "没有停车"
    assert np.all(vx >= config.WALK_VX_RANGE[0] - 1e-9)
    assert np.all(vx <= config.WALK_VX_RANGE[1] + 1e-9)
    assert np.all(vy >= config.WALK_VY_RANGE[0] - 1e-9)
    assert np.all(vy <= config.WALK_VY_RANGE[1] + 1e-9)


def test_walk_command_changes_during_episode():
    env = DinoWbcEnv("walk", seed=1)
    env.reset(seed=1)
    first = env.command.copy()
    changed = False
    for _ in range(400):
        before = env.command.copy()
        env.step(_zeros())
        if not np.allclose(before, env.command):
            changed = True
            break
    env.close()
    assert changed, "walk 训练命令在 400 步内没有换过"


def test_walk_set_command_locks_scheduler():
    env = DinoWbcEnv("walk", seed=2)
    env.reset(seed=2)
    fixed = make_command("walk", vx=-0.06, vy=0.03)
    env.set_command(fixed)
    for _ in range(400):
        env.step(_zeros())
    env.close()
    assert np.allclose(env.command, fixed), "手动指令被随机换向覆盖了"


# ----------------------------------------------------------------------
# 评测固定指令
# ----------------------------------------------------------------------
def test_evaluate_walk_fixed_command():
    from dino_wbc.evaluate import evaluate

    class _ZeroWalk:
        stage = "walk"

        def reset(self):
            pass

        def act(self, obs):
            return np.zeros(config.ACTION_DIM, dtype=np.float32)

        def close(self):
            pass

    result = evaluate(
        _ZeroWalk(),
        episodes=1,
        max_steps=40,
        walk_vx=-0.05,
        walk_vy=0.02,
    )
    summary = result["summary"]
    assert "mean_vx" in summary and "mean_vy" in summary
    assert np.isfinite(summary["mean_vx"])
    assert np.isfinite(summary["mean_vy"])
