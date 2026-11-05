"""奖励/终止判据的符号与量级。"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="未安装 mujoco")

from dino_wbc import config, model_io  # noqa: E402
from dino_wbc.observation import make_command  # noqa: E402
from dino_wbc.rewards import (  # noqa: E402
    compute_reward,
    is_fallen,
    is_standing,
    potential,
)
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
    return model, layout, data, read_state(model, data, layout)


def test_standing_is_not_fallen(rig):
    _, layout, _, state = rig
    assert not is_fallen(state)
    assert is_standing(state, layout)
    assert potential(state, layout) > 0.0


def test_lying_down_is_fallen(rig):
    model, layout, data, _ = rig
    reset_to_stand(model, data, layout)
    data.qpos[2] = 0.08
    mujoco.mj_forward(model, data)
    state = read_state(model, data, layout)
    assert is_fallen(state)
    assert not is_standing(state, layout)


def test_stand_reward_is_positive_when_standing(rig):
    model, layout, data, state = rig
    action = np.zeros(config.ACTION_DIM)
    reward, terms = compute_reward(
        "stand", state, action, action, make_command("stand"), layout
    )
    assert reward > 0.0
    assert terms["upright"] > 0.95
    assert terms["height"] > 0.99


def test_fall_penalty_is_negative(rig):
    model, layout, data, state = rig
    action = np.zeros(config.ACTION_DIM)
    reward, _ = compute_reward(
        "walk",
        state,
        action,
        action,
        make_command("walk", vx=0.1),
        layout,
        fallen=True,
    )
    assert reward < 0.0


def test_lift_reward_prefers_target_foot_up(rig):
    model, layout, data, state = rig
    action = np.zeros(config.ACTION_DIM)
    command = make_command("lift", lift_h=0.03, target_foot=-1.0)
    reward_stand, _ = compute_reward("lift", state, action, action, command, layout)

    elevated = state
    elevated.foot_z = state.foot_z.copy()
    elevated.foot_z[0] = 0.03
    elevated.contact = np.array([False, True])
    reward_lift, terms = compute_reward("lift", elevated, action, action, command, layout)
    assert terms["lift_height"] > 0.9
    assert reward_lift > reward_stand


def test_standup_progress_rewards_getting_up(rig):
    model, layout, data, state = rig
    action = np.zeros(config.ACTION_DIM)
    low = potential(state, layout)
    reward, terms = compute_reward(
        "standup",
        state,
        action,
        action,
        make_command("standup"),
        layout,
        phi=low + 0.5,
        phi_prev=low,
    )
    assert terms["progress"] > 0.0
