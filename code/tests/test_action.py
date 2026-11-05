"""动作映射：零动作=站姿、越界裁剪、关节限幅。"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="未安装 mujoco")

from dino_wbc import config, model_io  # noqa: E402
from dino_wbc.action import action_to_ctrl, ctrl_to_action  # noqa: E402


@pytest.fixture(scope="module")
def layout():
    _, layout = model_io.load()
    return layout


def test_zero_action_is_stand_pose(layout):
    ctrl = action_to_ctrl(np.zeros(config.ACTION_DIM), layout)
    assert np.allclose(ctrl, 0.0)


def test_action_is_scaled(layout):
    action = np.ones(config.ACTION_DIM)
    ctrl = action_to_ctrl(action, layout)
    expected = np.clip(
        np.asarray(config.ACTION_SCALE), layout.ranges[:, 0], layout.ranges[:, 1]
    )
    assert np.allclose(ctrl, expected)


def test_action_is_clipped_to_joint_ranges(layout):
    for sign in (1.0, -1.0):
        action = np.full(config.ACTION_DIM, sign * 10.0)
        ctrl = action_to_ctrl(action, layout)
        assert np.all(ctrl >= layout.ranges[:, 0] - 1e-12)
        assert np.all(ctrl <= layout.ranges[:, 1] + 1e-12)


def test_action_dim_is_checked(layout):
    with pytest.raises(ValueError):
        action_to_ctrl(np.zeros(config.ACTION_DIM + 1), layout)


def test_roundtrip(layout):
    # 用足够小的动作，保证不会碰到关节限位（knee 上限只有 0.15 rad）
    action = np.linspace(-0.2, 0.2, config.ACTION_DIM)
    ctrl = action_to_ctrl(action, layout)
    back = ctrl_to_action(ctrl)
    assert np.allclose(back, action, atol=1e-12)
