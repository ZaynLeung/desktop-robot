"""v2 修复的回归测试：严格终止、连续抬脚计数、外力轻推、评测指标。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="未安装 mujoco")

from dino_wbc import config  # noqa: E402
from dino_wbc.curriculum import Curriculum  # noqa: E402
from dino_wbc.env import DinoWbcEnv  # noqa: E402
from dino_wbc.observation import make_command  # noqa: E402
from dino_wbc.state import read_state  # noqa: E402


def _zeros():
    return np.zeros(config.ACTION_DIM, dtype=np.float32)


# ----------------------------------------------------------------------
# 站立：严格终止不再允许"趴着活"
# ----------------------------------------------------------------------
def test_stand_survives_zero_action_without_body_contact():
    env = DinoWbcEnv("stand", seed=0)
    env.reset(seed=0)
    body_steps = 0
    terminated = False
    truncated = False
    for _ in range(400):
        _, _, terminated, truncated, info = env.step(_zeros())
        body_steps += int(info["body_contact"])
        if terminated or truncated:
            break
    env.close()
    assert body_steps == 0
    assert not terminated
    assert truncated


def test_excessive_tilt_terminates():
    env = DinoWbcEnv("stand", seed=1)
    env.reset(seed=1)
    angle = np.pi / 2.0  # 侧翻 90°（世界 x 轴）
    env.data.qpos[3:7] = [np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0]
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, _, info = env.step(_zeros())
    env.close()
    assert terminated
    assert info["fallen"]


def test_body_contact_terminates():
    env = DinoWbcEnv("stand", seed=5)
    env.reset(seed=5)
    env.data.qpos[2] = 0.03
    env.data.qpos[3:7] = [0.0, 1.0, 0.0, 0.0]  # 仰卧
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, _, info = env.step(_zeros())
    env.close()
    assert terminated
    assert info["body_contact"] or info["fallen"]


# ----------------------------------------------------------------------
# 抬脚：必须连续保持，瞬时尖峰不算
# ----------------------------------------------------------------------
def test_lift_hold_requires_consecutive_steps():
    env = DinoWbcEnv("lift", seed=2)
    env.reset(seed=2)
    env._command = make_command("lift", lift_h=0.02, target_foot=-1.0)
    state = read_state(env.model, env.data, env.layout)
    raised = replace(
        state,
        base_pos=np.array([0.0, 0.0, env.layout.stand_base_z]),
        tilt_rad=0.0,
        body_contact=False,
        foot_z=np.array([0.02, 0.0]),
        contact=np.array([False, True]),
    )
    dropped = replace(raised, foot_z=np.array([0.0, 0.0]))

    for _ in range(config.LIFT_HOLD_STEPS - 1):
        env._track_curriculum_metrics(raised, {}, False)
    assert env._lift_best_streak == config.LIFT_HOLD_STEPS - 1
    env._track_curriculum_metrics(dropped, {}, False)  # 中断，连续计数清零
    assert env._lift_streak == 0
    assert not env._episode_outcome(False)

    for _ in range(config.LIFT_HOLD_STEPS):
        env._track_curriculum_metrics(raised, {}, False)
    env._lift_best_streak = config.LIFT_HOLD_STEPS
    assert env._episode_outcome(False)
    env.close()


# ----------------------------------------------------------------------
# 外力轻推
# ----------------------------------------------------------------------
def test_push_is_applied_then_cleared():
    env = DinoWbcEnv("stand", seed=3)
    env.reset(seed=3)
    env.curriculum.push_force = 3.0  # 模拟已爬坡的课程（首集默认 0 N）
    torso = env.layout.torso_body_id
    applied = False
    for _ in range(250):
        _, _, terminated, truncated, info = env.step(_zeros())
        if info["pushed"]:
            applied = True
            assert np.any(np.abs(env.data.xfrc_applied[torso, :3]) > 0.0)
            break
        if terminated or truncated:
            break
    assert applied, "250 步内没有触发一次轻推"
    for _ in range(7):
        env.step(_zeros())
    assert np.allclose(env.data.xfrc_applied[torso, :3], 0.0)
    env.close()


def test_stand_push_curriculum_ramps_and_resets():
    curriculum = Curriculum("stand", rng=np.random.default_rng(0))
    assert curriculum.push_force == 0.0
    for _ in range(10):
        curriculum.on_episode(True)
    assert curriculum.push_force > 0.0
    for _ in range(200):
        curriculum.on_episode(False)
    assert curriculum.push_force == 0.0


def test_curriculum_state_has_push_force():
    curriculum = Curriculum("stand", rng=np.random.default_rng(0))
    assert "push_force" in curriculum.state().__dict__


# ----------------------------------------------------------------------
# 奖励：蹲着时抬脚项必须为 0
# ----------------------------------------------------------------------
def test_lift_terms_are_gated_when_crouched():
    from dino_wbc.rewards import compute_reward, lift_stance_ok

    env = DinoWbcEnv("lift", seed=4)
    env.reset(seed=4)
    state = read_state(env.model, env.data, env.layout)
    crouched = replace(
        state, base_pos=np.array([0.0, 0.0, env.layout.stand_base_z - 0.05])
    )
    command = make_command("lift", lift_h=0.02, target_foot=-1.0)
    action = np.zeros(config.ACTION_DIM)
    _, terms = compute_reward(
        "lift", crouched, action, action, command, env.layout
    )
    env.close()
    assert not lift_stance_ok(crouched, env.layout)
    assert terms["lift_height"] == 0.0
    assert terms["lift_frac"] == 0.0
    assert terms["support_contact"] == 0.0
    assert terms["low_posture"] > 0.0


# ----------------------------------------------------------------------
# 评测指标
# ----------------------------------------------------------------------
def test_evaluate_reports_v2_metrics():
    from dino_wbc.controllers.scripted import HoldStandController
    from dino_wbc.evaluate import evaluate

    result = evaluate(HoldStandController(), episodes=1, max_steps=30)
    summary = result["summary"]
    for key in (
        "success_rate",
        "body_contact_rate",
        "tail_contact_rate",
        "support_contact_rate",
        "mean_target_foot_z",
        "max_target_foot_z",
        "lift_hold_success_rate",
        "heading_error_mean_deg",
        "heading_error_max_deg",
        "position_error_mean",
        "position_error_max",
        "yaw_drift_deg",
        "yaw_hold_success_rate",
    ):
        assert key in summary, key
        assert np.isfinite(summary[key])
