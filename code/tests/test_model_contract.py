"""模型契约：关节顺序/规模/限位/头部相机/契约哈希。"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="未安装 mujoco")

from dino_wbc import config, model_io  # noqa: E402


@pytest.fixture(scope="module")
def loaded():
    return model_io.load()


def test_scale_matches_config(loaded):
    model, layout = loaded
    assert config.NUM_JOINTS == 18
    assert model.nq == 7 + config.NUM_JOINTS == 25
    assert model.nv == 6 + config.NUM_JOINTS == 24
    assert model.nu == config.NUM_JOINTS == 18
    assert layout.num_joints == config.NUM_JOINTS


def test_joint_order_and_ranges(loaded):
    _, layout = loaded
    assert layout.names == config.JOINT_NAMES
    for name, (low, high) in zip(layout.names, layout.ranges):
        assert low < high, name
        # 零位就是站姿：每个关节的范围都必须包含 0
        assert low <= 0.0 <= high, name


def test_hip_yaw_is_mirrored(loaded):
    model, layout = loaded
    l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_yaw_L")
    r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_yaw_R")
    assert np.allclose(model.jnt_axis[l_id], [0, 0, 1])
    assert np.allclose(model.jnt_axis[r_id], [0, 0, -1])


def test_head_camera_is_bound_to_head(loaded):
    model, layout = loaded
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, config.HEAD_CAMERA)
    assert cam_id >= 0, "models/*.xml 里缺少 cam_head，请重新 build-dino"
    assert model.cam_bodyid[cam_id] == layout.head_body_id


def test_stand_keyframe(loaded):
    _, layout = loaded
    assert layout.stand_base_z == pytest.approx(0.184144, abs=1e-4)
    assert np.allclose(layout.stand_qpos[7:], 0.0)


def test_contract_hash_is_deterministic(loaded):
    model, layout = loaded
    first = model_io.contract_hash(model, layout)
    second = model_io.contract_hash(model, layout)
    assert first == second
    assert len(first) == 16
